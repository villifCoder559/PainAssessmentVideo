# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import math

import torch
import torch.nn as nn
import os
import sys

if os.path.abspath('jepa') not in sys.path:
    sys.path.append(os.path.abspath('jepa'))
    
from src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from src.utils.tensors import trunc_normal_
from jepa.src.models.utils import pos_embs

class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        q_k_v_dim=768,  # This is the dimension of the query, key, and value vectors
        num_heads=12,
        num_cross_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        mlp_dropout=0.0,
        attn_dropout=0.0,
        residual_dropout=0.0,
        qkv_bias=True,
        complete_block=True,
        use_sdpa=False,
        cross_block_after_transformers=False,
        custom_mlp=False,
        # grid_size_pos=None # [T, S, S] -> S depends on the model (Base,Giant,etc), while T is input dependent
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        # self.pos_enc = pos_enc
        # self.grid_size_pos = grid_size_pos
        self.cross_block_after_transformers = cross_block_after_transformers
        # self.pos_enc_tensor = None
        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_cross_heads,
                q_k_v_dim=q_k_v_dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=mlp_dropout,
                attn_drop=attn_dropout,
                residual_drop=residual_dropout,
                use_sdpa=use_sdpa,
                custom_mlp=custom_mlp,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_cross_heads,
                attn_drop=attn_dropout,
                q_k_v_dim=q_k_v_dim,
                proj_drop=mlp_dropout,
                use_sdpa=use_sdpa,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    use_sdpa=use_sdpa,
                    drop=mlp_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=norm_layer)
                for i in range(depth-1)])
            print(f'Using {len(self.blocks)} transformer blocks')

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask,return_xattn=False): # x: [B, T, C], mask: [B, T] (use for attention mask)
        if self.cross_block_after_transformers:
            if self.blocks is not None:
                for blk in self.blocks:
                    x = blk(x,mask)
            q = self.query_tokens.repeat(len(x), 1, 1)
            q,xattn = self.cross_attention_block(q, x, mask, return_xattn)
            return q,xattn
        else:
            # Original implementation
            q = self.query_tokens.repeat(len(x), 1, 1)
            q,xattn = self.cross_attention_block(q=q, x=x, mask=mask, return_xattn=return_xattn)
            if self.blocks is not None:
                for blk in self.blocks:
                    q = blk(q) # [B, num_queries, C] -> No mask needed, q is the query vector
            return q,xattn

class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        num_cross_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        attn_dropout=0.0,
        residual_dropout=0.0,
        dropout_mlp=0.0,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        pos_enc=False,
        use_sdpa=False,
        cross_block_after_transformers=False,
        grid_size_pos=None,
        num_queries=1,
        agg_method='mean'
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_cross_heads=num_cross_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            mlp_dropout=dropout_mlp,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_sdpa=use_sdpa,
            grid_size_pos=grid_size_pos,
            cross_block_after_transformers=cross_block_after_transformers
        )
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pos_enc = pos_enc
        self.pos_enc_tensor = None
        self.total_grid_area = grid_size_pos[1]*grid_size_pos[2]
        self.grid_size_pos = grid_size_pos
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)
        self.linear.reset_parameters()

    def forward(self, x, key_padding_mask=None,return_xattn=False): # x: [B, T, C], key_padding_mask: [B, T] (use for attention mask)
        raise NotImplementedError('AttentiveClassifier is changed. Use AttentiveHeadJEPA instead.')
        if self.pos_enc:
            if self.pos_enc_tensor is None or self.pos_enc_tensor.shape[0] != x.size(1):
                if x.size(1) % self.total_grid_area != 0:
                    raise ValueError(f'Input length {x.size(1)} is not divisible by batch size {x.size(0)}')
                self.pos_enc_tensor = pos_embs.get_1d_sincos_pos_embed_torch(embed_dim=x.size(2),
                                                                           grid_size=x.size(1)//self.total_grid_area).to(x.device)
                # self.pos_enc_tensor = pos_embs.get_3d_sincos_pos_embed_torch(embed_dim=x.size(2),
                #                                                              grid_depth=x.size(1)//(self.total_grid_area), # Considering same length for all dimensions!
                #                                                              grid_size=self.grid_size_pos[1]).to(x.device)
            x = x + self.pos_enc_tensor
        x,xattn = self.pooler(x,key_padding_mask,return_xattn) 
        x = x.squeeze(1) # # [B, num_queries=1, C] -> [B, C]
        x = self.linear(x)
        return x, xattn
    
    def _initialize_weights(self,init_type='default'):
        if init_type == 'default':
            trunc_normal_(self.pooler.query_tokens, std=self.pooler.init_std)
            self.pooler.apply(self.pooler._init_weights)
            self.pooler._rescale_blocks()
            self.linear.reset_parameters()
        else:
            raise NotImplementedError(f'Initialization method {init_type} not implemented')

