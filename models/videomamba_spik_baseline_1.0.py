import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
import torch.utils.checkpoint as checkpoint
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from einops import rearrange
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
__all__ = ['spikformer']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_bn = nn.BatchNorm1d(hidden_features)
        # self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        # self.fc2_bn = nn.BatchNorm1d(out_features)
        # self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        T,B,C,N = x.shape
        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,self.c_hidden,N).contiguous()
        # x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        # x = self.fc2_lif(x)
        return x

class SSM_Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
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
        #if self.fused_add_norm:
        #    assert RMSNorm is not None, "RMSNorm import fails"
        #    assert isinstance(
        #        self.norm, (nn.LayerNorm, RMSNorm)
        #    ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
        if self.fused_add_norm:
        # 去掉对RMSNorm存在性的强制检查，只验证当前norm的类型
            assert isinstance(
                self.norm, (nn.LayerNorm,)  # 只保留LayerNorm（如果确实不用RMSNorm）
            ), "Only LayerNorm is supported for fused_add_norm"


    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states.shape
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            #fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            fused_add_norm_fn = layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            ) # hidden_states [1, 1569, 576]
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states.shape
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

#     def forward(self, x):
#         x = x + self.attn(x)
#         x = x + (self.mlp(x))
#         return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_channels=3, embed_dim=768, dropout=0.1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )
        self.proj_bn = nn.BatchNorm3d(embed_dim)
        self.act = nn.ReLU(inplace=True)
        self.proj_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        #self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
    def forward(self, x):
        x = self.proj(x)
        x = self.proj_bn(x)
        x = self.act(x)
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * T, C, H * W).transpose(1, 2)
        x = self.proj_linear(x)
        x = self.dropout(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.fc2_bn = nn.BatchNorm1d(out_features * 4)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc1_bn(x)
        # x = self.fc1_lif(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.fc2_bn(x)
        # x = self.fc2_lif(x)
        x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.qk_bn = nn.BatchNorm1d(dim )
        self.qk_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
       
        qk = self.qk_lif(self.qk_bn(self.qk(x) )).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        # qk = self.qk_bn(qk)
        
        # qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 空间增强：深度可分离卷积（CPE）
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # 归一化层
        self.norm1 = norm_layer(dim)  # 图像特征归一化
        self.norm2 = norm_layer(dim)  # FFN前归一化
        self.norm3 = norm_layer(dim)  # 残差融合前归一化
        self.cls_norm = norm_layer(dim)  # CLS融合后归一化

        # 特征投影与激活
        self.in_proj = nn.Linear(dim, dim)  # 输入投影
        self.act_proj = nn.Linear(dim, dim)  # 激活投影
        self.out_proj = nn.Linear(dim, dim)  # 输出投影
        self.feat_downsample = nn.Linear(dim*2, dim)
        self.act = nn.SiLU()  # 激活函数

        # 核心组件
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 深度卷积
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    
    def forward(self, x):
        B, total_len, C = x.shape  # total_len = 1 (CLS) + H*W (图像特征)
        H, W = self.input_resolution
        img_len = H * W  # 图像特征长度（不含CLS）

        # 1. 分离CLS token与图像特征
        cls_token = x[:, 0:1, :]  # [B, 1, C]
        img_feat = x[:, 1:, :]    # [B, H*W, C]

        # 2. 图像特征空间增强（CPE1）+ 构建残差shortcut（含CLS）
        img_feat = img_feat + self.cpe1(
            img_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        ).flatten(2).permute(0, 2, 1)  # 转回 [B, H*W, C]
        shortcut = torch.cat([cls_token, img_feat], dim=1)  # [B, 1+H*W, C]

        # 3. 双向特征构建（正序+反序）
        img_feat_norm = self.norm1(img_feat)
        img_feat_rev = img_feat_norm.flip(1)  # 图像特征反序
        img_feat_bidir = torch.cat([img_feat_norm, img_feat_rev], dim=1)  # [B, 2*H*W, C]

        # 4. 激活门控计算
        act_gate = self.act(self.act_proj(img_feat_bidir))  # [B, 2*H*W, C]

        # 5. 深度卷积增强 + 特征分离
        img_feat_proj = self.in_proj(img_feat_bidir).view(B, 2*H, W, C)  # reshape适配卷积
        img_feat_conv = self.act(
            self.dwc(img_feat_proj.permute(0, 3, 1, 2))  # [B, C, 2H, W]
        ).permute(0, 2, 3, 1).view(B, 2*img_len, C)  # 转回 [B, 2*H*W, C]
        
        feat_pos, feat_rev = img_feat_conv.chunk(2, dim=1)  # 分离正序/反序图像特征

        # 6. 拼接CLS + 线性注意力
        feat_pos_with_cls = torch.cat([cls_token, feat_pos], dim=1)  # [B, 1+H*W, C]
        feat_rev_with_cls = torch.cat([cls_token, feat_rev], dim=1)  # [B, 1+H*W, C]
        
        feat_pos_attn = self.attn(feat_pos_with_cls)
        feat_rev_attn = self.attn(feat_rev_with_cls)

        # 7. 反序特征翻转 + CLS融合
        feat_rev_attn_flipped = feat_rev_attn.flip(1)  # 反序特征翻转回正序
        cls_fused = self.cls_norm(feat_pos_attn[:, 0:1, :] + feat_rev_attn_flipped[:, 0:1, :])

        # 8. 图像特征融合 + 激活门控应用
        img_feat_attn_pos = feat_pos_attn[:, 1:, :]
        img_feat_attn_rev = feat_rev_attn_flipped[:, 1:, :]
        img_feat_fused = torch.cat([img_feat_attn_pos, img_feat_attn_rev], dim=1)  # [B, 2*H*W, C]
        img_feat_out = self.out_proj(img_feat_fused * act_gate)  # 门控筛选

        # 维度缩减
        img_feat_out = img_feat_out.view(B, img_len, 2*C)  # [B, H×W, 2*C]（适配Linear的输入维度）
        img_feat_out = self.feat_downsample(img_feat_out)  # [B, C, H×W]（维度缩减）

        # 9. 残差连接（含CLS）
        out_with_cls = torch.cat([cls_fused, img_feat_out], dim=1)
        out_res = shortcut + self.drop_path(out_with_cls)

        # 10. 第二次空间增强（CPE2）+ 残差融合
        cls_fused_new = out_res[:, 0:1, :]
        img_feat_res = out_res[:, 1:, :]
        
        img_feat_cpe2 = self.cpe2(
            img_feat_res.reshape(B, H, W, C).permute(0, 3, 1, 2)
        ).flatten(2).permute(0, 2, 1)
        img_feat_final = img_feat + self.norm2(img_feat_cpe2)

        # 11. 拼接CLS + FFN
        final_feat = torch.cat([cls_fused_new, img_feat_final], dim=1)
        final_feat = final_feat + self.drop_path(self.mlp(self.norm3(final_feat)))

        return final_feat

    def forward(self, x):
        
        
        # x = self.out_bn(x)
        # x = self.out_lif(x)
        xz = self.dim2(xz.permute(0,2,1)).permute(0, 2,1)
        xz = torch.cat([cls_token, xz], dim=1)
        xz = shortcut + self.drop_path(xz)

        cls_token = xz[:, 0:1, :]

        xz = torch.cat([cls_token, xz], dim=1)

        # FFN
        xz = xz + self.drop_path(self.mlp(self.norm2(xz)))
        return xz

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x

class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        x.shape
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x.shape
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=False,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    #norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)
    block = SSM_Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



class SpatialPriorSpatioTemporalEmbedding(nn.Module):
    def __init__(self, 
                 embed_dim,          # 总嵌入维度（需与模型特征维度一致）
                 H_patch, W_patch,   # 每帧图像的块行数和列数（如14x14块）
                 max_num_frames=16,  # 最大帧数（时序维度上限）
                 spatial_weight=0.7  # 空间嵌入占比（0.5~1.0，越大空间信息越突出）
                ):
        super().__init__()
        self.embed_dim = embed_dim
        self.H_patch = H_patch  # 每帧的空间块行数（如H=224，patch_size=16 → H_patch=14）
        self.W_patch = W_patch  # 每帧的空间块列数
        self.max_num_frames = max_num_frames
        self.spatial_dim = int(embed_dim * spatial_weight)  # 空间嵌入维度（占主导）
        if self.spatial_dim % 2 != 0:
            self.spatial_dim += 1  # 确保空间维度为偶数，便于行列嵌入均分
        self.temporal_dim = embed_dim - self.spatial_dim     # 时序嵌入维度（轻量化）
        
        # --------------------------
        # 1. 空间位置嵌入（高维度，精细编码）
        # --------------------------
        # 生成空间坐标 (i,j)：每块在帧内的行列索引（如0~13行，0~13列）
        self.spatial_pos = self._generate_spatial_grid()  # [H_patch*W_patch, 2]，存储(i,j)坐标
        
        # 空间嵌入：用可学习参数编码(i,j)的绝对位置和相对关系
        # 先分别编码行索引i和列索引j，再融合（增强空间结构感知）
        self.spatial_embed_i = nn.Embedding(H_patch, self.spatial_dim // 2)  # 行嵌入
        self.spatial_embed_j = nn.Embedding(W_patch, self.spatial_dim // 2)  # 列嵌入
        
        # --------------------------
        # 2. 时序位置嵌入（低维度，轻量化）
        # --------------------------
        # 时序嵌入：仅编码帧的绝对位置（简洁设计，避免覆盖空间信息）
        self.temporal_embed = nn.Embedding(max_num_frames, self.temporal_dim)

        # 初始化参数（空间嵌入更精细，用更小的初始化标准差）
        nn.init.trunc_normal_(self.spatial_embed_i.weight, std=0.01)
        nn.init.trunc_normal_(self.spatial_embed_j.weight, std=0.01)
        nn.init.trunc_normal_(self.temporal_embed.weight, std=0.02)

    def _generate_spatial_grid(self):
        """生成每帧内所有块的空间坐标 (i,j)，形状 [H_patch*W_patch, 2]"""
        i = torch.arange(self.H_patch)  # 行索引：0 ~ H_patch-1
        j = torch.arange(self.W_patch)  # 列索引：0 ~ W_patch-1
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')  # 生成网格 [H_patch, W_patch]
        spatial_pos = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)  # [H*W, 2]
        return spatial_pos  # 例如14x14块 → [196, 2]

    def forward(self, B, T):
        """
        生成时空位置嵌入，形状 [B, T*N, embed_dim]，其中 N=H_patch*W_patch
        
        Args:
            B: 批量大小
            T: 当前视频的帧数（时序长度）
        Returns:
            spatio_temporal_emb: 时空位置嵌入，[B, T*N, embed_dim]
        """
        N = self.H_patch * self.W_patch  # 每帧的块数（空间序列长度）
        
        # --------------------------
        # 1. 生成空间嵌入 [1, N, spatial_dim]
        # --------------------------
        # 从坐标(i,j)获取行嵌入和列嵌入
        i = self.spatial_pos[:, 0].to(self.spatial_embed_i.weight.device)  # [N]
        j = self.spatial_pos[:, 1].to(self.spatial_embed_j.weight.device)  # [N]
        embed_i = self.spatial_embed_i(i)  # [N, spatial_dim//2]
        embed_j = self.spatial_embed_j(j)  # [N, spatial_dim//2]
        spatial_emb = torch.cat([embed_i, embed_j], dim=-1)  # [N, spatial_dim]
        # 扩展到所有时间步和批次：同一空间位置在不同帧共享空间嵌入
        spatial_emb = spatial_emb.unsqueeze(0).repeat(T, 1, 1)  # [T, N, spatial_dim]

        # --------------------------
        # 2. 生成时序嵌入 [1, T, temporal_dim]
        # --------------------------
        t = torch.arange(T, device=self.temporal_embed.weight.device)  # [T]
        temporal_emb = self.temporal_embed(t)  # [T, temporal_dim]
        # 扩展到所有空间块：同一时间步的所有块共享时序嵌入
        temporal_emb = temporal_emb.unsqueeze(1).repeat(1, N, 1)  # [T, N, temporal_dim]

        # --------------------------
        # 3. 融合时空嵌入（空间为主）
        # --------------------------
        # 拼接空间嵌入（高维）和时序嵌入（低维）
        spatio_temporal_emb = torch.cat([spatial_emb, temporal_emb], dim=-1)  # [T, N, embed_dim]
        # 调整形状为 [T*N, embed_dim]，并扩展到批量维度
        spatio_temporal_emb = spatio_temporal_emb.reshape(T*N, self.embed_dim)  # [T*N, embed_dim]
        spatio_temporal_emb = spatio_temporal_emb.unsqueeze(0).repeat(B, 1, 1)  # [B, T*N, embed_dim]

        return spatio_temporal_emb




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



# 方法二
class Uncertain():
    def __init__(self, alpha):
        super().__init__()
        self.views = 2
        self.num_classes = 10 # 记得修改类别数
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
    # print("sum_u shape:", sum_u.shape)
    # print("sum_u value:", sum_u)

    # 计算系数
    coeff1 = (1 - u1) / sum_u  # [batch_size, 1]
    coeff2 = (1 - u2) / sum_u  # [batch_size, 1]

    # 打印系数的形状和值
    # print("coeff1 shape:", coeff1.shape)
    # print("coeff1 value:", coeff1)
    # print("coeff2 shape:", coeff2.shape)
    # print("coeff2 value:", coeff2)

    # 计算最终融合特征
    r = coeff1 * r1 + coeff2 * r2  # [batch_size, feature_dim]
    # r shape: torch.Size([8, 384])
    # print("r shape:", r.shape)
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


class SpatialBranch(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=8, in_channels=3, batch_size=8,
                 embed_dims=256, num_heads=8, mlp_ratios=4.0, num_classes=300,num_frames=16,
                 qkv_bias=True, depths=2, sr_ratios=1, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # 空间分支主要处理空间信息
        self.patch_embed = PatchEmbed(
            img_size=img_size_h,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dims,
        )
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        # 初始化嵌入模块
        H_patch = img_size_h // patch_size
        W_patch = img_size_w // patch_size
        self.st_embed = SpatialPriorSpatioTemporalEmbedding(
            embed_dim=embed_dims,
            H_patch=H_patch,
            W_patch=W_patch,
            max_num_frames=num_frames,
        )

        # MLLA层配置
        self.num_layers = len([depths]) if isinstance(depths, int) else len(depths)
        self.mlla_depths = [depths] if isinstance(depths, int) else depths
        self.mlla_num_heads = [num_heads]
        self.mlp_ratio = mlp_ratios
        self.mlla_norm_layer = norm_layer

        # 构建MLLA层
        patches_resolution = [img_size_h // patch_size, img_size_w // patch_size]
        self.mlla_dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.mlla_depths))]  # 固定drop_path_rate为0.1
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dims * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=self.mlla_depths[i_layer],
                num_heads=self.mlla_num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=0.,
                drop_path=self.mlla_dpr[sum(self.mlla_depths[:i_layer]):sum(self.mlla_depths[:i_layer + 1])],
                norm_layer=self.mlla_norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False
            )
            self.layers.append(layer)

        # 空间分支输出投影
        self.proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, x):
        # x形状: [B, C, T, H, W]    
        x = self.patch_embed(x) # 经过PatchEmbed处理 [B*T, N, C]

        BT ,N , C = x.shape
        x = x.view(self.batch_size, self.num_frames, N, C).flatten(1, 2) # [B, T*N, C]

        # 生成时空嵌入
        emb = self.st_embed(self.batch_size, self.num_frames)  # [B, T*N, embed_dim]  

        # 融合嵌入（特征 + 时空位置信息）
        x = x + emb  # 空间信息主导，时序信息辅助'

        x = x.reshape(-1, N, C)  # 恢复批次维度 [B*T, N, C]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B*T, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # [B*T, N+1, C]

        # 经过MLLA层  input: [B*T, N+1, C]
        for layer in self.layers:   
            x = layer(x)
        # 恢复批次维度并投影
        x = x.reshape(-1, self.batch_size, x.shape[1], x.shape[2]).mean(0)  # 时间维度平均 [B, N, C]
        return self.proj(x)

class TemporalBranch(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=3, embed_dims=256,
                 num_frames=16, num_classes=300, ssm_embed_dims=256, batch_size=8,
                 rms_norm=True, residual_in_fp32=True, fused_add_norm=False, bimamba=True):
        super().__init__()
        # 时间分支主要处理时序信息，复用原有PatchEmbed和Mamba模块
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dims,
        )

        self.batch_size = batch_size
        # Mamba配置
        self.ssm_embed_dims = ssm_embed_dims
        self.num_frames = num_frames
        ssm_dpr = [x.item() for x in torch.linspace(0, 0.1, 12)]  # 固定drop_path_rate为0.1
        ssm_inter_dpr = [0.0] + ssm_dpr
        factory_kwargs = {"device": None, "dtype": None}

        # 构建Mamba层
        self.mamba_layers = nn.ModuleList(
            [
                create_block(
                    ssm_embed_dims,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=ssm_inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(1)  # 保持1层Mamba
            ]
        )

        #self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(ssm_embed_dims, eps=1e-5)
        # 直接使用 nn.LayerNorm，不再考虑 rms_norm 参数
        self.norm_f = nn.LayerNorm(ssm_embed_dims, eps=1e-5)
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # 时间分支输出投影
        self.proj = nn.Linear(ssm_embed_dims, ssm_embed_dims)

    def forward(self, x, inference_params=None):
        # x形状: [B, C, T, H, W]
        x = self.patch_embed(x)  # 经过PatchEmbed处理 [B*T, N, C]
        BT ,N , C = x.shape
        x = x.view(self.batch_size, self.num_frames, N, C).flatten(1, 2) # [B, T*N, C]
        x.shape
        # 时序位置嵌入
        temporal_pos_emb = self._get_temporal_pos_emb(self.batch_size, x.shape[1])
        temporal_pos_emb.shape
        x = x + temporal_pos_emb
        
        # 经过Mamba层
        residual = None
        hidden_states = x  # 恢复批次维度 [B, T*N, C]

        for layer in self.mamba_layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        # 最终归一化
        if not self.fused_add_norm:
            residual = residual + self.drop_path(hidden_states) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
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

        return self.proj(hidden_states)

    def _get_temporal_pos_emb(self, B, N):
        # 生成时序位置嵌入 [B, N, C]
        temporal_pos = torch.arange(self.num_frames, device=self.norm_f.weight.device)
        temporal_emb = nn.functional.one_hot(temporal_pos, num_classes=self.ssm_embed_dims).float()
        temporal_emb = temporal_emb.unsqueeze(1).repeat(1, N // self.num_frames, 1)  # 每个时间步对应相同空间位置
        return temporal_emb.reshape(1, -1, self.ssm_embed_dims).repeat(B, 1, 1)


class DualStreamSpikMamba(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=8, in_channels=3,
                 num_classes=300, embed_dims=256, num_heads=8, mlp_ratios=4.0,
                 qkv_bias=True, depths=2, sr_ratios=1, num_frames=16, batch_size=8,              
                 ssm_embed_dims=256, rms_norm=True, residual_in_fp32=True, 
                 fused_add_norm=False, bimamba=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # 初始化空间分支和时间分支（保持不变）
        self.spatial_branch = SpatialBranch(
            img_size_h=img_size_h, img_size_w=img_size_w, patch_size=patch_size,
            in_channels=in_channels, embed_dims=embed_dims, num_heads=num_heads,
            mlp_ratios=mlp_ratios, num_classes=num_classes, qkv_bias=qkv_bias,batch_size=batch_size,
            depths=depths, sr_ratios=sr_ratios, norm_layer=norm_layer,num_frames=num_frames
        )

        self.temporal_branch = TemporalBranch(
            img_size=img_size_h, patch_size=patch_size, in_chans=in_channels,
            embed_dims=embed_dims, num_frames=num_frames, num_classes=num_classes,
            ssm_embed_dims=ssm_embed_dims, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm, bimamba=bimamba, batch_size=batch_size
        )

        # 交叉注意力融合层（时间特征→Query，空间特征→Key/Value）
        # 1. 特征归一化
        self.temporal_norm = norm_layer(ssm_embed_dims)  # 时间特征（Query）归一化
        self.spatial_norm = norm_layer(embed_dims)  # 空间特征（Key/Value）归一化

        # 2. 交叉注意力模块（需保证Query与Key/Value的维度一致）
        # 若空间特征维度（embed_dims）与时间特征维度（ssm_embed_dims）不同，需投影
        self.key_proj = nn.Linear(embed_dims, ssm_embed_dims)  # Key投影到Query维度
        self.value_proj = nn.Linear(embed_dims, ssm_embed_dims)  # Value投影到Query维度
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ssm_embed_dims,  # 与Query维度（时间特征）一致
            num_heads=num_heads,
            batch_first=True  # 输入形状为[B, seq_len, C]
        )

        # 3. 融合后处理
        self.fusion_proj = nn.Linear(ssm_embed_dims, embed_dims)  # 调整维度
        self.classifier = nn.Linear(embed_dims, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        # 输入x形状: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # 1. 提取分支特征
        # 空间分支输入: [B, C, T, H, W] → 输出: [B, T, N, C1=embed_dims]
        spatial_feat = self.spatial_branch(x)

        # 时间分支输入: [B, C, T, H, W] → 输出: [B, T, N, C2=ssm_embed_dims]
        temporal_feat = self.temporal_branch(x)

        # 2. 时空合并 # [B, T*N, C1=embed_dims]
        spatial_feat = spatial_feat.flatten(1, 2)  
        temporal_feat = temporal_feat.flatten(1, 2) 

        # 3. 交叉注意力融合（时间特征Query → 空间特征Key/Value）
        # 归一化
        query = self.temporal_norm(temporal_feat)  # [B, T*N, C2=ssm_embed_dims]
        key = self.spatial_norm(spatial_feat)  # [B, T*N, C1=embed_dims]
        value = key  # Key和Value共享空间特征

        # 投影Key/Value到Query维度（若维度不同）
        key = self.key_proj(key)  # [B, T*N, C2=ssm_embed_dims]
        value = self.value_proj(value)  # [B, T*N, C2=ssm_embed_dims]

        # 交叉注意力计算：时间特征查询空间特征中的关键信息
        attn_output, _ = self.cross_attn(query, key, value)  # [B, T*N, C2]

        # 残差连接（以时间特征为基础，融合空间信息）
        fused_feat = temporal_feat + attn_output  # [B, T*N, C2]

        # 4. 分类头
        fused_feat = self.fusion_proj(fused_feat)  # [B, T*N, embed_dims]
        return self.classifier(fused_feat.mean(1))  # 平均序列维度


    def forward(self, x1, x2=None, inference_params=None,  hsic=False):
        features_view1 = self.forward_features(x1)  # features_view1 shape: torch.Size([8, 384])
        # print("features_view1 shape:",features_view1.shape)
        return features_view1

# 注册模型
@register_model
def dual_stream_spikmamba(pretrained=False, **kwargs):
    model = DualStreamSpikMamba(
        img_size_h=128, img_size_w=128, patch_size=8, embed_dims=256,
        num_heads=8, mlp_ratios=4.0, in_channels=3, num_classes=300,
        qkv_bias=True, num_frames=16, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=2, sr_ratios=1, ssm_embed_dims=256, **kwargs
    )
    model.default_cfg = _cfg()
    return model




# 测试代码
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import os
    import os

    model = DualStreamSpikMamba(
        img_size_h=128, img_size_w=128, patch_size=8, embed_dims=256,
        num_heads=8, mlp_ratios=4.0, in_channels=3, num_classes=300,batch_size=8,
        qkv_bias=True, num_frames=16, depths=2, sr_ratios=1, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).cuda()

    input_tensor1 = torch.randn(8, 3, 16, 128, 128).cuda()  # [B, C, T, H, W]
    input_tensor2 = torch.randn(8, 3, 16, 128, 128).cuda()  # [B, C, T, H, W]
    output = model(input_tensor1, input_tensor2)
    print(f"Output shape: {output.shape}")  # 应为 [16, 300]