"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from methods.metaloss.models.utils import init_weights
from methods.metaloss.models.blocks import Block, DownSampleConvBlock, OverlapDownSample

from methods.metaloss.image_utils import get_1d_coords_scale_from_h_w_ps, convert_1d_index_to_2d, convert_2d_index_to_1d

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


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dim_list = [channels] + [embed_dim]*(n_layers-1)
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], embed_dim)
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.conv_layers(im).flatten(2).transpose(1, 2)
        return x




class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_blocks,
        dim,
        n_heads,
        dim_ff,
        dropout=0.0,
        drop_path_rate=0.0,
        ):
        super().__init__()

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, dim_ff, dropout, dpr[i]) for i in range(n_blocks)]
        )

    def forward(self, x):
        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        n_heads,
        dropout=0.0,
        drop_path_rate=0.0,
        channels=1,
        split_ratio=4,
        n_scales=2
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding(
            image_size,
            patch_size,
            d_model[0],
            channels,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        # Pos Embs
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, d_model[0]))
        self.rel_pos_embs = nn.ParameterList([nn.Parameter(torch.randn(1, self.split_ratio, d_model[i])) for i in range(n_scales - 1)])
        self.scale_embs = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model[i])) for i in range(n_scales - 1)])


        # transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(n_layers[i], d_model[i], n_heads[i], d_model[i]*4, dropout, drop_path_rate) for i in range(len(n_layers))]
        )

        # Downsamplers
        self.downsamplers = nn.ModuleList([nn.Linear(d_model[i], d_model[i + 1]) for i in range(len(n_layers) - 1)])

        # Split layers
        self.splits = nn.ModuleList(
            [nn.Linear(d_model[i], d_model[i] * self.split_ratio) for i in range(len(n_layers))]
        )

        # Metaloss predictions
        self.metalosses = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model[i], d_model[i]),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(d_model[i], affine=True),
            nn.Linear(d_model[i], 1)) for i in range(len(n_layers))])

        self.high_res_patchers = nn.ModuleList(
            [nn.Conv2d(channels, d_model[i - 1], kernel_size=patch_size // (2 ** i), stride=patch_size // (2 ** i)) for i in
             range(1, len(n_layers))])


        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def divide_tokens_to_split_and_keep(self, tokens_at_curr_scale, patches_scale_coords_curr_scale, curr_scale):
        k_split = tokens_at_curr_scale.shape[1] // self.split_ratio
        k_keep = tokens_at_curr_scale.shape[1] - k_split
        pred_meta_loss = self.metalosses[curr_scale](tokens_at_curr_scale).squeeze(2)
        tkv, tki = torch.topk(pred_meta_loss, k=k_split, dim=1, sorted=False)
        bkv, bki = torch.topk(pred_meta_loss, k=k_keep, dim=1, sorted=False, largest=False)

        batch_indices_k = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_keep)
        batch_indices_s = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_split)

        tokens_to_keep = tokens_at_curr_scale[batch_indices_k, bki]
        tokens_to_split = tokens_at_curr_scale[batch_indices_s, tki]
        coords_to_keep = patches_scale_coords_curr_scale[bki].squeeze(0)
        coords_to_split = patches_scale_coords_curr_scale[tki].squeeze(0)

        return tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss


    def divide_tokens_coords_on_scale(self, tokens, patches_scale_coords, curr_scale):
        indx_curr_scale = patches_scale_coords[:, 0] == curr_scale
        indx_old_scales = patches_scale_coords[:, 0] != curr_scale 
        coords_at_curr_scale = patches_scale_coords[indx_curr_scale]
        coords_at_older_scales = patches_scale_coords[indx_old_scales]
        tokens_at_curr_scale = tokens[:, indx_curr_scale, :]
        tokens_at_older_scale = tokens[:, indx_old_scales, :]

        return tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales
    

    def split_tokens(self, tokens_to_split, curr_scale):
        x_splitted = self.splits[curr_scale](tokens_to_split)
        x_splitted = rearrange(x_splitted, 'b n (s d) -> b n s d', s=self.split_ratio)
        x_splitted = x_splitted + self.rel_pos_embs[curr_scale] + self.scale_embs[curr_scale]
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
        new_coords_1dim = convert_2d_index_to_1d(new_coords_2dim, patch_size * 2).unsqueeze(1).int()

        scale_lvl = torch.tensor([[new_scale]] * new_coords_1dim.shape[0]).to('cuda').int()
        patches_scale_coords = torch.cat([scale_lvl, new_coords_1dim], dim=1)

        return patches_scale_coords

    def add_high_res_features(self, tokens, coords, curr_scale, image):
        patched_im = self.high_res_patchers[curr_scale](image)
        patched_im = rearrange(patched_im, 'b c h w -> b (h w) c')
        patched_im = patched_im[:, coords]
        tokens = tokens + patched_im

        return tokens



    def split_input(self, tokens, patches_scale_coords, curr_scale, patch_size, im):
        tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales = self.divide_tokens_coords_on_scale(tokens, patches_scale_coords, curr_scale)
        meta_loss_coords = coords_at_curr_scale[:,1]
        tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss = self.divide_tokens_to_split_and_keep(tokens_at_curr_scale, coords_at_curr_scale, curr_scale)
        tokens_after_split = self.split_tokens(tokens_to_split, curr_scale)
        coords_after_split = self.split_coords(coords_to_split, patch_size, curr_scale)

        tokens_after_split = self.add_high_res_features(tokens_after_split, coords_after_split[:, 1], curr_scale, im)

        all_tokens = torch.cat([tokens_at_older_scale, tokens_to_keep, tokens_after_split], dim=1)
        all_coords = torch.cat([coords_at_older_scales, coords_to_keep, coords_after_split], dim=0)

        return all_tokens, all_coords, pred_meta_loss, meta_loss_coords


    def forward(self, im):
        B, _, H, W = im.shape
        PS = self.patch_size
        patched_im_size = H // PS
        x = self.patch_embed(im)
        x = x + self.pos_embed

        patches_scale_coords = get_1d_coords_scale_from_h_w_ps(H, W, PS, 0).to('cuda')
        meta_losses = []
        meta_loss_coords = []
        for l_idx in range(len(self.layers)):
            x = self.layers[l_idx](x)
            #print("Current total number of tokens in layer {}: {}".format(blk_idx, x.shape[1]))
            '''
            for s in range(blk_idx + 1):
                indx_scale = patches_scale_coords[:, 0] == s
                coords_at_scale = patches_scale_coords[indx_scale]
                print("Current number of tokens at scale {} in layer {}: {}".format(s, blk_idx, len(coords_at_scale)))
            '''
            if l_idx < self.n_scales - 1: 
                x, patches_scale_coords, meta_loss, meta_loss_coord = self.split_input(x, patches_scale_coords, l_idx, patched_im_size, im)
                PS /= 2
                patched_im_size *= 2
                x = self.downsamplers[l_idx](x)
                meta_losses.append(meta_loss)
                meta_loss_coords.append(meta_loss_coord)

        return x, patches_scale_coords, meta_losses, meta_loss_coords