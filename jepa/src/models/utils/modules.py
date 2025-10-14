# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MLP_custom(nn.Module):
    def __init__(
        self,
        in_features,
        # hidden_features=None,
        # out_features=None,
        reduction_ratio,
        act_layer=nn.GELU,
        drop=0.,
        
    ):
        super().__init__()
        # self.in_features = in_features
        # self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(in_features, int(in_features * reduction_ratio))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(in_features * reduction_ratio), int(in_features * (reduction_ratio**2)))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_value = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, mask=None): # x: [B, N, C], mask: [B, N]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None,:].expand(B, self.num_heads, N, N)
        if self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(backends=torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                
                x = F.scaled_dot_product_attention(q, k, v, 
                                                   dropout_p=self.attn_drop_value if self.training else 0.0,
                                                   attn_mask=mask, # True -> consider to compute attention, False -> NOT consider to compute
                                                   scale=self.scale)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            if mask:
                attn = attn.masked_fill(~mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0., # for MLP and projection layer
        attn_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_sdpa=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            use_sdpa=use_sdpa,
            proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        q_k_v_dim=None,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim if q_k_v_dim is None else q_k_v_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int((dim if q_k_v_dim is None else q_k_v_dim) *2), bias=qkv_bias)
        self.q_k_v_dim = q_k_v_dim if q_k_v_dim is not None else dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_value = attn_drop
        # if q_k_v_dim is not None and q_k_v_dim != dim:
        self.proj = nn.Linear(dim if q_k_v_dim is None else q_k_v_dim, dim)
        # else:
        #     self.proj = nn.Identity()
            
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, q, x,mask=None,return_xattn=False): # q: [B, n, C], x: [B, N, C], mask: [B, 1=q_seq_len, 1=key_seq_len, N]
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, self.q_k_v_dim // self.num_heads).permute(0, 2, 1, 3) # (batch_size, num_heads, query_len, feature_dim_per_head)
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.q_k_v_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None,:].expand(B, self.num_heads, n, N)
        if self.use_sdpa and not return_xattn:
            with torch.nn.attention.sdpa_kernel(backends=torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                # if mask is not None:
                #     print("Debug")
                q = F.scaled_dot_product_attention(q, k, v,
                                                   dropout_p=self.attn_drop_value if self.training else 0.0,
                                                   attn_mask=mask, # True -> consider to compute attention, False -> NOT consider to compute
                                                   scale=self.scale)
                xattn = None
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                xattn = xattn.masked_fill(~mask, float('-inf'))
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            xattn = self.attn_drop(xattn)
            q = (xattn @ v)
        q = q.transpose(1, 2).reshape(B, n, self.q_k_v_dim)  # (batch_size, query_len, feature_dim)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q, xattn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        q_k_v_dim=None,
        mlp_ratio=4.,
        drop=0.0,
        attn_drop=0.0,
        residual_drop=0.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path_mode='row',
        custom_mlp=False,
        use_sdpa=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, 
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    q_k_v_dim=q_k_v_dim,
                                    use_sdpa=use_sdpa,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if not custom_mlp:
            self.mlp = MLP(in_features=dim, 
                           hidden_features=mlp_hidden_dim, 
                           act_layer=act_layer, drop=drop)
        else:
            self.mlp = MLP_custom(in_features=dim, 
                                  reduction_ratio=mlp_ratio, 
                                  act_layer=act_layer, 
                                  drop=drop)
        self.residual_drop = StochasticDepth(residual_drop,drop_path_mode) if residual_drop > 0.0 else nn.Identity()

    def forward(self, q, x, mask=None, return_xattn=False):
        y,xattn = self.xattn(q, self.norm1(x), mask=mask, return_xattn=return_xattn)
        q = self.residual_drop(q) + y
        q = self.mlp(self.norm2(q))
        return q, xattn
