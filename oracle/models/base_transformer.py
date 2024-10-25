"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from oracle.models.utils import init_weights
from oracle.models.blocks import Block

from oracle.image_utils import get_1d_coords_scale_from_h_w_ps, convert_1d_index_to_2d, convert_2d_index_to_1d

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

    def divide_tokens_coords_on_scale(self, tokens, patches_scale_coords, curr_scale):
        indx_curr_scale = patches_scale_coords[:, 0] == curr_scale
        indx_old_scales = patches_scale_coords[:, 0] != curr_scale 
        coords_at_curr_scale = patches_scale_coords[indx_curr_scale]
        coords_at_older_scales = patches_scale_coords[indx_old_scales]
        tokens_at_curr_scale = tokens[:, indx_curr_scale, :]
        tokens_at_older_scale = tokens[:, indx_old_scales, :]

        return tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales


    def divide_tokens_to_split_and_keep(self, tokens_at_curr_scale, patches_scale_coords_curr_scale, split_choices_curr_scale):
        indx_for_choice = patches_scale_coords_curr_scale[:, 1]
        choices_for_scale = split_choices_curr_scale[indx_for_choice]
        tokens_to_keep = tokens_at_curr_scale[:, choices_for_scale == 1, :]
        tokens_to_split = tokens_at_curr_scale[:, choices_for_scale == 0, :]
        coords_to_keep = patches_scale_coords_curr_scale[choices_for_scale == 1]
        coords_to_split = patches_scale_coords_curr_scale[choices_for_scale == 0]

        return tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep

    def split_tokens(self, tokens_to_split, curr_scale):
        x_splitted = rearrange(tokens_to_split, 'b n (s d) -> b n s d', s=self.split_ratio)
        x_splitted = self.splits[curr_scale](x_splitted)
        x_splitted = rearrange(x_splitted, 'b n s d -> b (n s) d', s=self.split_ratio)
        return x_splitted

    def split_coords(self, coords_to_split, patch_size, curr_scale):
        new_scale = curr_scale + 1
        new_coord_ratio = self.split_ratio // 2
        two_d_coords = convert_1d_index_to_2d(coords_to_split[:, 1], patch_size)
        a = torch.stack([two_d_coords[:, 0] * new_coord_ratio, two_d_coords[:, 1] * new_coord_ratio])
        b = torch.stack([two_d_coords[:, 0] * new_coord_ratio, two_d_coords[:, 1] * new_coord_ratio + 1])
        c = torch.stack([two_d_coords[:, 0] * new_coord_ratio + 1, two_d_coords[:, 1] * new_coord_ratio])
        d = torch.stack([two_d_coords[:, 0] * new_coord_ratio + 1, two_d_coords[:, 1] * new_coord_ratio + 1])

        new_coords_2dim = torch.stack([a, b, c, d]).permute(2, 0, 1)
        new_coords_2dim = rearrange(new_coords_2dim, 'n s c -> (n s) c', s=self.split_ratio, c=2)
        new_coords_1dim = convert_2d_index_to_1d(new_coords_2dim, patch_size * 2).unsqueeze(1).long()

        scale_lvl = torch.tensor([[new_scale]] * new_coords_1dim.shape[0]).to('cuda').long()
        patches_scale_coords = torch.cat([scale_lvl, new_coords_1dim], dim=1)

        return patches_scale_coords
         

    def split_input(self, tokens, oracle_labels, patches_scale_coords, curr_scale, patch_size):
        tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales = self.divide_tokens_coords_on_scale(tokens, patches_scale_coords, curr_scale)
        tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep = self.divide_tokens_to_split_and_keep(tokens_at_curr_scale, coords_at_curr_scale, oracle_labels.view(-1))
        tokens_after_split = self.split_tokens(tokens_to_split, curr_scale)
        coords_after_split = self.split_coords(coords_to_split, patch_size, curr_scale)

        all_tokens = torch.cat([tokens_at_older_scale, tokens_to_keep, tokens_after_split], dim=1)
        all_coords = torch.cat([coords_at_older_scales, coords_to_keep, coords_after_split], dim=0)

        return all_tokens, all_coords


    def forward(self, im, oracle_labels):
        B, _, H, W = im.shape
        PS = self.patch_size
        patched_im_size = H // PS
        x = self.patch_embed(im)
        x = x + self.pos_embed

        scale = 0
        patches_scale_coords = get_1d_coords_scale_from_h_w_ps(H, W, PS, scale).to('cuda')

        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
            print("Current total number of tokens in layer {}: {}".format(blk_idx, x.shape[1]))
            for s in range(blk_idx + 1):
                indx_scale = patches_scale_coords[:, 0] == s
                coords_at_scale = patches_scale_coords[indx_scale]
                print("Current number of tokens at scale {} in layer {}: {}".format(s, blk_idx, len(coords_at_scale)))
            if blk_idx < len(self.blocks) - 1: 
                ol =  oracle_labels[blk_idx]
                x, patches_scale_coords = self.split_input(x, ol, patches_scale_coords, blk_idx, patched_im_size)
                PS /= 2
                patched_im_size *= 2

        return x, patches_scale_coords