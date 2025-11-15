# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

MODEL_PATH = '/HOME/scw6fgn/run/VideoMamba-main/videomamba/models/'
_MODELS = {
    # "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    # "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    # "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_k400_f16_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_k400_f32_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_k400_f16_res224.pth"),
}

"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python Implementation of the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
We provide both biased estimator and unbiased estimators (unbiased estimator is used in the paper)
"""


# import torch
# import torch.nn as nn


def to_numpy(x):
    """convert Pytorch tensor to numpy array
    """
    return x.clone().detach().cpu().numpy()


class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.

    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: \frac{1}{m (m - 3)} \bigg[ tr (\tilde K \tilde L) + \frac{1^\top \tilde K 1 1^\top \tilde L 1}{(m-1)(m-2)} - \frac{2}{m-2} 1^\top \tilde K \tilde L 1 \bigg].
        where \tilde K and \tilde L are related to K and L by the diagonal entries of \tilde K_{ij} and \tilde L_{ij} are set to zero.

    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """

    def __init__(self, sigma_x, sigma_y=None, algorithm='unbiased',
                 reduction=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if algorithm == 'biased':
            self.estimator = self.biased_estimator
        elif algorithm == 'unbiased':
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError('invalid estimator: {}'.format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)

        N = len(input1)

        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation.
    """

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)


class MinusRbfHSIC(RbfHSIC):
    """`Minus'' RbfHSIC for the max'' optimization.
    """

    def forward(self, input1, input2, **kwargs):
        return -self.estimator(input1, input2)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# 方法二
class Uncertain():
    def __init__(self, alpha):
        super().__init__()
        self.views = 2
        self.num_classes = 8 # 记得修改类别数
        self.alpha = alpha # alpha shape: [batch_size, 2, num_classes]

    def DS_Combin_two(self):
        alpha = self.alpha  # [batch_size, 2, num_classes]

        # 计算 S, E, b, u
        S = torch.sum(alpha, dim=-1, keepdim=True) + 1e-10 # [batch_size, 2, 1]  在计算中避免除以零，添加极小值epsilon。
        E = alpha - 1  # [batch_size, 2, num_classes]
        # b = E / S.expand(E.shape)  # [batch_size, 2, num_classes]
        u = self.num_classes / S  # [batch_size, 2, 1]
        # print("u shape:", u.shape)
        # print(f"u value:", u)
        return u  # 形状为 [batch_size, 2, 1]

    def compute_u(self):
        # alpha = self.alpha + 1  # 确保 alpha 不小于 1
        return self.DS_Combin_two()   # ？self.DS_Combin_two()？


def GAP(r1, r2, alpha):
    """
    r1: [batch_size, feature_dim]
    r2: [batch_size, feature_dim]
    alpha: [batch_size, 2, num_classes]
    """
    u = Uncertain(alpha).compute_u()  # [batch_size, 2, 1]

    u1 = u[:, 0, :]  # [batch_size, 1]
    u2 = u[:, 1, :]  # [batch_size, 1]
    sum_u = 2 - (u1 + u2)  # [batch_size, 1]
    print("sum_u shape:", sum_u.shape)
    print("sum_u value:", sum_u)

    # 计算系数
    coeff1 = (1 - u1) / sum_u  # [batch_size, 1]
    coeff2 = (1 - u2) / sum_u  # [batch_size, 1]

    # 打印系数的形状和值
    print("coeff1 shape:", coeff1.shape)
    print("coeff1 value:", coeff1)
    print("coeff2 shape:", coeff2.shape)
    print("coeff2 value:", coeff2)

    # 计算最终融合特征
    r = coeff1 * r1 + coeff2 * r2  # [batch_size, feature_dim]
    # r shape: torch.Size([8, 384])
    print("r shape:", r.shape)
    return r

# 方法三对应
class Uncertain3(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.views = 2
        self.embed_dim = embed_dim  # 使用特征维度代替 num_classes

    def compute_u(self, features):
        # features: [batch_size, 2, embed_dim]
        # 将特征值转为非负（例如取绝对值或 softplus）
        features = F.softplus(features)  # 确保非负
        S = torch.sum(features, dim=-1, keepdim=True) + 1e-10  # [batch_size, 2, 1]
        u = self.embed_dim / S  # 不确定性，[batch_size, 2, 1]
        return u

def GAP3(r1, r2, features):
    """
    r1: [batch_size, embed_dim]
    r2: [batch_size, embed_dim]
    features: [batch_size, 2, embed_dim] (features_view1 和 features_view2 堆叠)
    """
    uncertain = Uncertain3(r1.shape[-1])  # embed_dim 作为参数
    u = uncertain.compute_u(features)  # [batch_size, 2, 1]

    u1 = u[:, 0, :]  # [batch_size, 1]
    u2 = u[:, 1, :]  # [batch_size, 1]
    sum_u = 2 - (u1 + u2)  # [batch_size, 1]

    # 计算系数
    coeff1 = (1 - u1) / sum_u  # [batch_size, 1]
    coeff2 = (1 - u2) / sum_u  # [batch_size, 1]

    # 融合特征
    r = coeff1 * r1 + coeff2 * r2  # [batch_size, embed_dim]
    return r



# 方法四
from math import log2

class Uncertain4(nn.Module):
    def __init__(self, num_classes, uncertainty_prior=0.2):
        super().__init__()
        self.views = 2
        self.num_classes = num_classes
        self.uncertainty_prior = uncertainty_prior  # 默认不确定性分配比例
        self.hypotheses = [f'A{i + 1}' for i in range(num_classes)] + ['A_all']  # 假设：单一类别 + 联合假设

    def create_bpa(self, logits):
        """
        根据logits生成BPA
        logits: [batch_size, num_classes]
        """
        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
        bpa = torch.zeros(batch_size, len(self.hypotheses), device=logits.device)

        # 单一类别BPA
        for i in range(self.num_classes):
            bpa[:, i] = probs[:, i] * (1 - self.uncertainty_prior)

        # 联合假设BPA（不确定性）
        bpa[:, -1] = self.uncertainty_prior  # A_all

        return bpa  # [batch_size, num_hypotheses]

    def deng_entropy(self, bpa):
        """
        计算Deng熵，量化BPA的不确定性
        bpa: [batch_size, num_hypotheses]
        """
        entropy = torch.zeros(bpa.shape[0], device=bpa.device)
        for i, h in enumerate(self.hypotheses):
            mass = bpa[:, i]
            cardinality = self.num_classes if h == 'A_all' else 1
            mask = mass > 0
            entropy[mask] -= mass[mask] * torch.log2(mass[mask] / cardinality)
        return entropy  # [batch_size]

    def dempster_rule(self, bpa1, bpa2):
        """
        Dempster融合规则
        bpa1, bpa2: [batch_size, num_hypotheses]
        """
        batch_size = bpa1.shape[0]
        fused_bpa = torch.zeros_like(bpa1)
        conflict = torch.zeros(batch_size, device=bpa1.device)

        for i, h1 in enumerate(self.hypotheses):
            for j, h2 in enumerate(self.hypotheses):
                intersection = self.get_intersection(h1, h2)
                if intersection == 'empty':
                    conflict += bpa1[:, i] * bpa2[:, j]
                else:
                    idx = self.hypotheses.index(intersection)
                    fused_bpa[:, idx] += bpa1[:, i] * bpa2[:, j]

        # 归一化
        norm_factor = 1 - conflict
        mask = norm_factor > 0
        fused_bpa[mask] /= norm_factor[mask].unsqueeze(-1)

        return fused_bpa, conflict  # [batch_size, num_hypotheses], [batch_size]

    def get_intersection(self, h1, h2):
        """
        计算两个假设的交集
        """
        if h1 == 'A_all' or h2 == 'A_all':
            return h1 if h2 == 'A_all' else h2
        return h1 if h1 == h2 else 'empty'

    def belief_function(self, bpa, hypothesis_idx):
        """
        计算指定假设的信念值
        bpa: [batch_size, num_hypotheses]
        """
        belief = torch.zeros(bpa.shape[0], device=bpa.device)
        for i, h in enumerate(self.hypotheses):
            if i <= hypothesis_idx or h == 'A_all':
                belief += bpa[:, i]
        return belief


def GAP4(r1, r2, logits1, logits2, num_classes):
    """
    r1: [batch_size, feature_dim]
    r2: [batch_size, feature_dim]
    logits1, logits2: [batch_size, num_classes]
    """
    uncertain = Uncertain4(num_classes=num_classes)

    # 生成BPA
    bpa1 = uncertain.create_bpa(logits1)  # [batch_size, num_hypotheses]
    bpa2 = uncertain.create_bpa(logits2)  # [batch_size, num_hypotheses]

    # 计算不确定性（Deng熵）
    entropy1 = uncertain.deng_entropy(bpa1)  # [batch_size]
    entropy2 = uncertain.deng_entropy(bpa2)  # [batch_size]
    print(f"视角1不确定性（Deng熵）: {entropy1.mean().item():.4f}")
    print(f"视角2不确定性（Deng熵）: {entropy2.mean().item():.4f}")

    # Dempster融合
    fused_bpa, conflict = uncertain.dempster_rule(bpa1, bpa2)
    print(f"冲突系数K: {conflict.mean().item():.4f}")
    print(f"融合后BPA: {fused_bpa.mean(dim=0).detach().cpu().numpy()}")  # 添加detach()


    # 使用不确定性倒数作为权重
    weight1 = 1 / (entropy1 + 1e-10)  # [batch_size]
    weight2 = 1 / (entropy2 + 1e-10)  # [batch_size]
    sum_weight = weight1 + weight2
    coeff1 = (weight1 / sum_weight).unsqueeze(-1)  # [batch_size, 1]
    coeff2 = (weight2 / sum_weight).unsqueeze(-1)  # [batch_size, 1]

    print(f"coeff1 value: {coeff1.mean().item():.4f}")
    print(f"coeff2 value: {coeff2.mean().item():.4f}")

    # 融合特征
    r = coeff1 * r1 + coeff2 * r2  # [batch_size, feature_dim]
    print(f"r shape: {r.shape}")

    return r

class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1,
            num_frames=8,
            fc_drop_rate=0.,
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,

            # hsic
            hsic_algorithm='unbiased',
            auto_sigma=True,

    ):
        factory_kwargs = {"device": device, "dtype": dtype}  # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        # HSIC参数设置
        self.hsic_algorithm = hsic_algorithm
        self.auto_sigma = auto_sigma  # 是否自动计算sigma
        self.base_sigma = 1.0  # 默认基准值
        self.hsic_criterion = None  # 延迟初始化

    def _init_hsic(self, sample_input):
        """用第一批数据自动计算sigma"""
        print("_init_hsic")
        with torch.no_grad():
            # 提取第一批数据的特征
            features = self.forward_features(sample_input)
            X = features  # 取CLS token

            # 计算中位数距离
            dists = torch.cdist(X, X).flatten()
            valid_dists = dists[dists > 0]  # 过滤零距离（自身）
            median_dist = torch.median(valid_dists).item()

            # 设置sigma为距离中位数的1/2
            self.base_sigma = max(median_dist / 2, 1e-6)  # 防止除零

        # 初始化HSIC准则
        self.hsic_criterion = RbfHSIC(
            sigma_x=self.base_sigma,
            algorithm=self.hsic_algorithm
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        print("x shape:",x.shape)
        x = self.patch_embed(x)
        # print("补丁嵌入后:", x.shape)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        # print("位置嵌入后:", x.shape)
        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        # print("重排后:", x.shape)
        # if x.size(1) == 8: # 同时数据集改185和192行//2,降低静态帧率
        #     # print("静态片段用8")
        #     new_temporal_dim = 8
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, new_temporal_dim, self.embed_dim).to(self.device, non_blocking=True))
        # else:
        #     # print("动态片段用16")
        #     new_temporal_dim = 16
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, new_temporal_dim, self.embed_dim).to(self.device, non_blocking=True))


        # current_temporal_dim = x.size(1)
        # if current_temporal_dim != self.temporal_pos_embedding.size(1):
        #     self.temporal_pos_embedding = nn.Parameter(
        #         torch.zeros(1, current_temporal_dim, self.embed_dim).to(x.device)
        #     )
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        return hidden_states[:, 0, :]

    def forward(self, x1, x2=None, inference_params=None,  hsic=False):
        features_view1 = self.forward_features(x1, inference_params)  # features_view1 shape: torch.Size([8, 384])
        print("features_view1 shape:",features_view1.shape)
        if x2 is not None:

            features_view2 = self.forward_features(x2, inference_params)# features_view2 shape: torch.Size([8, 384])
            # alpha1 = self.head(self.head_drop(features_view1)) # alpha1 shape: torch.Size([8, 10])
            # # print("alpha1 shape:", alpha1.shape)
            # alpha2 = self.head(self.head_drop(features_view2))
            # alpha1 = torch.relu(self.head(self.head_drop(features_view1)))  # 使用ReLU确保非负
            # alpha2 = torch.relu(self.head(self.head_drop(features_view2)))
    # 方法二：
    #         print("证据融合")
    #         alpha1 = self.head(self.head_drop(features_view1))
    #         alpha2 = self.head(self.head_drop(features_view2))
    #         alpha3= F.softplus(alpha1)
    #         alpha4 = F.softplus(alpha2)
    #         alphas = torch.stack([alpha3+1, alpha4+1], dim=1)  # [batch_size, 2, num_classes]
    #         # print("alphas shape:", alphas.shape)
    #         # fused_cls = GAP(features_view1, features_view2, alphas)
    #         # print("fused_cls shape:", fused_cls.shape)
    #         # x = self.head(self.head_drop(fused_cls))
    #         x = GAP(alpha1, alpha2, alphas)
    #         alpha1 = alpha3
    #         alpha2 = alpha4
    # 方法三：
    #         # 直接用 features_view1 和 features_view2 作为证据量
    #         print("证据融合 with raw features")
    #         features = torch.stack([features_view1, features_view2], dim=1)  # [batch_size, 2, embed_dim]
    #         fused_cls = GAP3(features_view1, features_view2, features)  # [batch_size, embed_dim]
    #         x = self.head(self.head_drop(fused_cls))  # [batch_size, num_classes]
    #         alpha1 = self.head(self.head_drop(features_view1))
    #         alpha2 = self.head(self.head_drop(features_view2))
    #         alpha1 = F.softplus(alpha1) # edl去掉
    #         alpha2 = F.softplus(alpha2) # edl去掉
    # 方法四：
    #         print("D-S证据融合")
    #         logits1 = self.head(self.head_drop(features_view1))  # [batch_size, num_classes]
    #         logits2 = self.head(self.head_drop(features_view2))  # [batch_size, num_classes]
    #
    #         fused_cls = GAP4(features_view1, features_view2, logits1, logits2, self.num_classes)  # [batch_size, embed_dim]
    #         x = self.head(self.head_drop(fused_cls))  # [batch_size, num_classes]
    #         # probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]

    # 方法一：
            print("平均融合")
            fused_cls = (features_view1 + features_view2) / 2
            x = self.head(self.head_drop(fused_cls))
    #         # x shape: torch.Size([8, 10])

            if hsic:
                # 首次运行时自动初始化HSIC
                if self.hsic_criterion is None and self.auto_sigma:
                    self._init_hsic(x1)  # 用第一个batch初始化

                # 计算HSIC损失（带安全机制）
                if self.hsic_criterion is not None:
                    loss_hsic = self.hsic_criterion(features_view1, features_view2)
                else:
                    loss_hsic = torch.tensor(0.0, device=x1.device)  # 避免未初始化错误

                # loss_hsic = self.hsic_criterion(cls_token_view1, cls_token_view2)  # 计算HSIC损失
                return x, loss_hsic
            else:
                # 方法二：
                # return x,alpha1,alpha2
                # 方法三

                # 方法四
                # return x, logits1, logits2


                # baseline
                return x

        else:
            print("单视角")
            # features_view1 = self.forward_features(x1, inference_params)
            x = self.head(self.head_drop(features_view1))
            return x

    # def forward(self, x1, x2, inference_params=None):
    #     x = self.forward_features(x1, x2, inference_params)
    #     # 选择每个视角的 cls_token
    #     x1_cls = x[:, :self.num_patches, :]
    #     x2_cls = x[:, self.num_patches:2 * self.num_patches, :]
    #     # 将两个视角的 cls_token 拼接在一起
    #     x = torch.cat([x1_cls, x2_cls], dim=1)
    #     x = self.head(self.head_drop(x))
    #     return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cuda')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cuda')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=576,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cuda')
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 64
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time() - s)
