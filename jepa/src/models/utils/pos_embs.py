# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

def get_3d_sincos_pos_embed_torch(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    Args:
        embed_dim (int): embedding dimension.
        grid_size (int): height and width of the grid.
        grid_depth (int): depth of the grid.
        cls_token (bool): whether to prepend a class token embedding.
        uniform_power (bool): flag for uniform power scaling.
    Returns:
        pos_embed (Tensor): position embeddings of shape [grid_depth * grid_size * grid_size, embed_dim] 
                            or [1 + grid_depth * grid_size * grid_size, embed_dim] if cls_token is True.
    """
    # Create coordinate ranges in torch; using torch.float32 for float type.
    gd = torch.arange(grid_depth, dtype=torch.float32)
    gh = torch.arange(grid_size, dtype=torch.float32)
    gw = torch.arange(grid_size, dtype=torch.float32)
    
    # Use torch.meshgrid to create the coordinate grids.
    # Here, we order the inputs as (grid_depth, grid_size, grid_size) so that:
    # - grid_d varies along dimension 0,
    # - grid_h along dimension 1,
    # - grid_w along dimension 2.
    grid_d, grid_h, grid_w = torch.meshgrid(gd, gh, gw, indexing="ij")
    
    # Flatten the coordinate grids into vectors.
    grid_d = grid_d.reshape(-1)
    grid_h = grid_h.reshape(-1)
    grid_w = grid_w.reshape(-1)
    
    # Determine the number of dimensions used for each axis.
    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        # Use math.ceil to compute the required dims.
        dim_each = int(np.ceil(embed_dim / 6) * 2)
        h_embed_dim = w_embed_dim = d_embed_dim = dim_each

    # Compute 1D sin-cos embeddings for each axis.
    # Here you must supply a torch-version of the helper function.
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(h_embed_dim, grid_h)  # expected shape: [N, h_embed_dim]
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(w_embed_dim, grid_w)  # expected shape: [N, w_embed_dim]
    emb_d = get_1d_sincos_pos_embed_from_grid_torch(d_embed_dim, grid_d)  # expected shape: [N, d_embed_dim]
    
    # Concatenate along the feature dimension.
    pos_embed = torch.cat([emb_d, emb_h, emb_w], dim=1)
    pos_embed = pos_embed[:, :embed_dim]
    
    # Optionally prepend a zero vector for the cls token.
    if cls_token:
        cls_token_embed = torch.zeros((1, embed_dim), dtype=pos_embed.dtype, device=pos_embed.device)
        pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
    
    return pos_embed

def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    grid_depth,
    cls_token=False,
    uniform_power=False
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)  # order of meshgrid is very important for indexing as [d,h,w]
    
    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    GPU-optimized version of grid embedding helper
    embed_dim: output dimension for each position
    pos: Tensor of positions to be encoded (already on GPU)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1.0 / 10000.0 ** omega
    
    out = torch.outer(pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    
    return torch.cat([emb_sin, emb_cos], dim=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_torch(embed_dim, grid_size, cls_token=False, device='cpu'):
    """
    GPU-optimized 1D sinusoidal positional embedding
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = torch.arange(grid_size, dtype=torch.float32, device=device)
    pos_embed = get_1d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([
            torch.zeros(1, embed_dim, device=device), 
            pos_embed
        ], dim=0)
    return pos_embed


