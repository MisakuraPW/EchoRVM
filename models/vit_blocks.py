"""Small ViT building blocks used by EchoRMAE."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0, separate_qv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias and not separate_qv_bias)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias and separate_qv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias and separate_qv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        bias = self.qkv.bias
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, nq, c = query.shape
        nk = context.shape[1]
        q = self.q(query).reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(b, nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(b, nq, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-5,
        separate_qv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, separate_qv_bias=separate_qv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossBlock(nn.Module):
    """Cross-attention, MLP, then self-attention integration block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.cross(self.norm_q(x), self.norm_ctx(context)))
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        x = x + self.drop_path(self.self_attn(self.norm_self(x)))
        return x
