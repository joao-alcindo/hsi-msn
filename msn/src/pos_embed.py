# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

np.float = float 

import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# HSI MAE 
# get_3d_sincos_pos_embed: https://github.com/Ryan21wy/HSIMAE/blob/master/Models.py
# --------------------------------------------------------




def get_3d_sincos_pos_embed(embed_dim, t_size, h_size, w_size, cls_token=False, scale_t=None):
    assert embed_dim % 4 == 0, "embed_dim precisa ser divisível por 4"
    embed_dim_spatial = embed_dim // 2
    embed_dim_temporal = embed_dim // 2

    # Spatial (H, W)
    grid_h = np.arange(h_size, dtype=np.float32)
    grid_w = np.arange(w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, H, W)
    grid = np.stack(grid, axis=0)       # (2, H, W)
    grid = grid.reshape([2, 1, h_size, w_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)  # (H*W, D/2)

    # Temporal (T)
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, scale=scale_t)  # (T, D/2)

    # Expand para combinar tempo e espaço
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]           # (T, 1, D/2)
    pos_embed_temporal = np.tile(pos_embed_temporal, (1, h_size*w_size, 1))  # (T, H*W, D/2)

    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]             # (1, H*W, D/2)
    pos_embed_spatial = np.tile(pos_embed_spatial, (t_size, 1, 1))      # (T, H*W, D/2)

    # Juntar temporal + espacial
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # (T, H*W, D)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # (T*H*W, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return torch.tensor(pos_embed, dtype=torch.float, requires_grad=False).unsqueeze(0)  # (1, num_patches, D)


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_rect(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size_h: int, grid height
    grid_size_w: int, grid width
    return:
    pos_embed: [grid_size_h * grid_size_w, embed_dim] or [1 + grid_size_h * grid_size_w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # [2, H, W]
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size_h, grid_size_w])

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed




def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed