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

from spikingjelly.clock_driven.neuron import MultiStepLIFNode

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
        T, B, C, N = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        # x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        # x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)

        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, N))

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + (self.mlp(x))
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=3, embed_dims=64, name="1"):
        super().__init__()
        self.name = name
        self.image_size = [img_size_h, img_size_w]
        tuple_patch_size = to_2tuple(patch_size)
        self.num_patch_size = patch_size
        self.patch_size = tuple_patch_size
        self.C = in_channels
        self.embed_dims = embed_dims
        self.H, self.W = self.image_size[0] // tuple_patch_size[0], self.image_size[1] // tuple_patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        # self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        # self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        if self.name == "spikmamba":
            x = x.permute(0, 2, 1, 3, 4)
        else:
            x = x
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.proj_conv(x)  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        # x = self.proj_conv3(x)
        # x = self.proj_bn3(x).reshape(T, B, -1, 16, 16).contiguous()
        # x = self.proj_lif3(x).flatten(0, 1).contiguous()
        # x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0, 1)
        x = x + x_rpe
        if self.name == "spikmamba":
            x = x.reshape(T, -1, B, (H // self.num_patch_size), (H // self.num_patch_size)).contiguous()
        else:
            x = x.reshape(B, self.embed_dims, -1, (H // self.num_patch_size), (H // self.num_patch_size)).contiguous()

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768, num_frames=8):
        super().__init__()
        tuple_img_size = to_2tuple(img_size)
        tuple_patch_size = to_2tuple(patch_size)
        num_patches = (tuple_img_size[1] // tuple_patch_size[1]) * (tuple_img_size[0] // tuple_patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, tuple_patch_size[0], tuple_patch_size[1]),
            stride=(kernel_size, tuple_patch_size[0], tuple_patch_size[1])
        )
        self.proj_bn = nn.BatchNorm1d(int(1024))
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        x = self.proj(x)  #
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

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
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

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

        self.qk_bn = nn.BatchNorm1d(dim * 4)
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

        qk = self.qk_lif(self.qk_bn(self.qk(x))).reshape(b, n, 2, c).permute(2, 0, 1, 3)
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

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)

        self.in_proj = nn.Linear(dim, dim)
        # self.in_bn = nn.BatchNorm1d(dim * 4 )
        # self.in_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.to2 = nn.Linear(dim, dim * 2)
        self.act_proj = nn.Linear(dim, dim)
        # self.act_bn = nn.BatchNorm1d(dim * 4 )
        # self.act_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)

        self.dim2 = nn.Linear(2048, 1024)
        self.out_proj = nn.Linear(dim, dim)
        # self.out_bn = nn.BatchNorm1d(dim * 4 )
        # self.out_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution

        # B, L, C = x.shape
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # x = x.flatten(0, 1)
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        # x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        # x2 = self.to2(x)
        # xf, xb = torch.chunk(x2, 2, dim=2)
        xb = x.flip(1)
        xz = torch.cat([x, xb], dim=1)

        xz = self.act_proj(xz)
        # act_res = self.act(self.act_lif(self.act_bn(x)))
        act_res = self.act(self.act_proj(xz))

        # x = self.in_lif(self.in_bn(self.in_proj(x)))
        xz = self.in_proj(xz).view(B, L * 2, C)
        # x = self.in_bn(x)
        # x = self.in_lif(x)
        xz = xz.view(B, H * 2, W, C)

        xz = self.act(self.dwc(xz.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L * 2, C)
        xf, xb = xz.chunk(2, dim=1)
        # Linear Attention
        xf = self.attn(xf)
        xb = self.attn(xb)
        #
        xz = torch.cat([xf, xb.flip(1)], dim=1)

        # x = self.out_lif(self.out_bn(self.out_proj(x * act_res)))

        xz = self.out_proj(xz * act_res)
        # x = self.out_bn(x)
        # x = self.out_lif(x)
        xz = self.dim2(xz.permute(0, 2, 1)).permute(0, 2, 1)
        xz = shortcut + self.drop_path(xz)
        xz = x + self.cpe2(xz.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

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
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1,
                      groups=int(out_channels * ratio), norm=None),
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
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
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


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=384, img_size_w=384, patch_size=16, in_channels=3, num_classes=300,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8,
                 kernel_size=1,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], batch_size=32
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.mlp_ratio = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.num_frame = num_frames
        self.sps_patch_embed = SPS(img_size_h=img_size_h,
                                   img_size_w=img_size_w,
                                   patch_size=patch_size,
                                   in_channels=in_channels,
                                   embed_dims=embed_dims,
                                   name="spikeformer")

        self.patch_embed = PatchEmbed(
            img_size=img_size_h, patch_size=patch_size,
            kernel_size=1,
            in_chans=3, embed_dim=embed_dims,
            num_frames=self.num_frame
        )

        self.imgae_size = img_size_h
        self.patch_size = patch_size

        num_patches = self.patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", self.patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        self.mlla_norm_layer = nn.LayerNorm
        self.embed_dims = embed_dims
        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dims))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, (num_frames) // kernel_size, self.embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)
        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)
        patches_resolution = [img_size_h // patch_size, img_size_w // patch_size]
        self.layers = nn.ModuleList()
        self.mlla_depths = [2]
        self.mlla_num_heads = [8]
        self.num_layers = len(self.mlla_depths)

        self.mlla_img_size = to_2tuple(128)
        self.mlla_patch_size = to_2tuple(4)
        patches_resolution = [self.mlla_img_size[0] // self.mlla_patch_size[0],
                              self.mlla_img_size[1] // self.mlla_patch_size[1]]
        self.mlla_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.mlla_depths))]

        self.batch_size = batch_size

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.mlla_depths[i_layer],
                               num_heads=self.mlla_num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=self.mlla_dpr[
                                         sum(self.mlla_depths[:i_layer]):sum(self.mlla_depths[:i_layer + 1])],
                               norm_layer=self.mlla_norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        # print(self.layers)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        # a = x
        x = self.patch_embed(x)  # T B L C
        # # for blk in block:
        # #     x = blk(x)
        # x = self.sps_patch_embed(x) # 128 64 8 32 32

        # B, C, T, H, W = x.shape #   8，256，8，32，32

        # x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) # [128,1024,64]

        cls_token = self.cls_token.expand(x.shape[0], -1,
                                          -1)  # stole cls_tokens impl from Phil Wang, thanks [8, 1, 576]
        x = torch.cat((cls_token, x), dim=1)  # [1024, 65, 192]
        x = x + self.pos_embed  # pos_embed [1, 197, 576] x [8, 197, 576]
        cls_tokens = x[:self.batch_size, :1, :]  # cls_tokens [1,1,576]
        x = x[:, 1:]  # [8, 196, 576]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=int(self.batch_size / 2), t=self.num_frame)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=int(self.batch_size / 2), t=self.num_frame)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        x = x.reshape(self.imgae_size // self.patch_size, -1, self.embed_dims).contiguous()
        # 32 8192 256
        x = x.reshape(-1, (self.imgae_size // self.patch_size) * (self.imgae_size // self.patch_size),
                      self.embed_dims).contiguous()
        for layer in self.layers:
            x = layer(x)
        x = x
        x = x.view(int(x.size(0) / self.num_frame), x.size(1), -1, self.embed_dims)
        return x.mean(2)

    def forward(self, x1, x2=None):
        features_view1 = self.forward_features(x1)  # features_view1 shape: torch.Size([8, 384])
        print("features_view1 shape:", features_view1.shape)
        if x2 is not None:

            features_view2 = self.forward_features(x2)  # features_view2 shape: torch.Size([8, 384])
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
#     model = VisionMamba(
#         patch_size=16,
#         embed_dim=384,
#         depth=24,
#         rms_norm=True,
#         residual_in_fp32=True,
#         fused_add_norm=True,
#         **kwargs
#     )
#     model.default_cfg = _cfg()
#     if pretrained:
#         print('load pretrained weights')
#         state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cuda')
#         load_state_dict(model, state_dict, center=True)
#     return model
#
# @register_model
# def spikformer_mlla_spik3(pretrained=False, **kwargs):
    model = Spikformer(
        img_size_h=224, img_size_w=224,patch_size=16, embed_dims=384, num_heads=8 , mlp_ratios=4,
        in_channels=3, num_classes=8, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2 , sr_ratios=1,num_frames=16,
        **kwargs
    )
    model.default_cfg = _cfg()
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
