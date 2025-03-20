import torch
import torch.nn as nn

from torch import  Tensor

from typing import Union

from timm.models.layers import DropPath
from .adapter import Adapter


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        # q = self.q_proj(x)
        # k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        # v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        # q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_probs = self.attn_drop(attn_weights)
        # attn_output = torch.bmm(attn_probs, v)

        # attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)
        # attn_output = attn_output.reshape(B, N, C)

        # x = self.proj(attn_output)
        # x = self.proj_drop(x)

        # return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None, use_dinov2=False):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if use_dinov2:
            self.ls1 = LayerScale(dim, init_values=1e-5)
            self.ls2 = LayerScale(dim, init_values=1e-5)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

        if config.ffn_adapt:
            self.adaptmlp = Adapter(self.config, dropout=0.0, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )

    def forward(self, x, use_adapter = True):
        x = x + self.ls1(self.drop_path(self.attn(self.norm1(x))))
        # x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.config.ffn_adapt and self.config.ffn_option == 'parallel' and use_adapter:
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.ls2(self.drop_path(self.mlp_drop(self.fc2(x))))
        # x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        # x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt and use_adapter:
            if self.config.ffn_option == 'sequential':
                x = self.adaptmlp(x)
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)
        x = residual + x
        return x