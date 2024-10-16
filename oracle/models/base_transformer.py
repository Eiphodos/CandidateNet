"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from oracle.models.utils import init_weights
from oracle.models.blocks import Block

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        dropout=0.1,
        drop_path_rate=0.0,
        channels=1,
        split_ratio=4
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio

        # Pos Emb
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches, d_model)
        )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # Split layers
        self.splits = nn.ModuleList(
            [nn.Linear(d_model // self.split_ratio, d_model) for i in range(n_layers)]
        )


        trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def divide_tokens(self, tokens, oracle_labels, patches_scale_coords, lvl):
        #oracle_labels = oracle_labels.view(-1)
        ind_to_split = oracle_labels == 0
        ind_to_keep = oracle_labels == 1

        patches_at_curr_res = patches_scale_coords[patches_scale_coords[:, 0] == lvl]
        low_res_tokens_to_keep = tokens[patches_scale_coords[:, 0] != lvl]
        oracle_labels_at_curr_res = oracle_labels[:, patches_at_curr_res[:, 1], patches_at_curr_res[:, 2]]

        ind_to_split = oracle_labels_at_curr_res == 0
        ind_to_keep = oracle_labels_at_curr_res == 1

        tokens_to_split = tokens[:, ind_to_split, :]
        tokens_to_keep = tokens[:, ind_to_keep, :]


        return tokens_to_split

    def forward(self, im, oracle_labels):
        B, _, H, W = im.shape
        PS = self.patch_size
        x = self.patch_embed(im)
        x = x + self.pos_embed

        patches_coords = torch.meshgrid(torch.arange(0, H // PS), torch.arange(0, H // PS), indexing='ij')
        patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
        patches_coords = patches_coords.permute(1, 2, 0)
        patches_coords = patches_coords.view(-1, 2)

        scale_lvl = torch.tensor([[0]] * self.patch_embed.num_patches)
        patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)

        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
            x_to_keep, x_to_split = self.divide_tokens(x, oracle_labels, patches_scale_coords, blk_idx)
            x_splitted = rearrange(x_to_split, 'b n (d s) -> b (n s) d', s=self.split_ratio)
            x_splitted = self.splits[blk_idx](x_splitted)
            x = torch.cat([x_to_keep, x_splitted], dim=1)

        return x