"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import math
import torch.nn as nn

from methods.metaloss.models.utils import init_weights
from methods.metaloss.models.blocks import Block

from methods.metaloss.image_utils import patches_to_images, convert_1d_patched_index_to_2d_org_index

from einops import rearrange


class SegmentationTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_encoder,
        n_heads,
        n_query_tokens,
        dropout=0.0,
        drop_path_rate=0.0,
        n_cls=3,
        split_ratio=4,
        n_scales=2
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_encoder = d_encoder
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_cls = n_cls
        self.scale = d_model ** -0.5
        self.n_query_tokens = n_query_tokens

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_model*4, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, self.n_query_tokens, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(self.n_query_tokens)


        minimum_resolution = image_size[0] // (patch_size // 2**(n_scales-1))
        upsampling_ratio = image_size[0] // minimum_resolution
        up_proj = []
        for i in range(int(math.log(upsampling_ratio, 2))):
            up_proj.append(nn.Conv2d(self.n_query_tokens, self.n_query_tokens, kernel_size=3, stride=1, padding=1))
            up_proj.append(nn.InstanceNorm2d(self.n_query_tokens))
            up_proj.append(nn.LeakyReLU())
            up_proj.append(nn.ConvTranspose2d(self.n_query_tokens, self.n_query_tokens, kernel_size=2, stride=2))
            up_proj.append(nn.InstanceNorm2d(self.n_query_tokens))
            up_proj.append(nn.LeakyReLU())
        self.conv_up_proj = nn.ModuleList(up_proj)

        self.conv_out_proj = nn.Conv2d(self.n_query_tokens, self.n_cls, kernel_size=3, stride=1, padding=1)

        self.n_patches = minimum_resolution ** 2
        self.pos_embed = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.apply(init_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


    def forward(self, x, patch_code):
        H, W = self.image_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        x = x + self.pos_embed(patch_code.float())
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, : -self.n_query_tokens], x[:, -self.n_query_tokens :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.unsqueeze(1)
        masks = rearrange(masks, "b h w c -> b c h w")

        max_scale = patch_code[:, 0].max()
        new_coord_PS = self.patch_size // 2**max_scale
        new_GS = H // new_coord_PS
        org_patch_codes_list = []
        for scale in range(0, max_scale + 1):
            indx_curr_scale = patch_code[:, 0] == scale
            coords_curr_scale = patch_code[indx_curr_scale, 1]
            scale_curr_scale = patch_code[indx_curr_scale, 0]
            org_patch_codes = convert_1d_patched_index_to_2d_org_index(coords_curr_scale, H, self.patch_size, scale, new_coord_PS).cuda()
            org_patch_codes = torch.cat([scale_curr_scale.unsqueeze(1), org_patch_codes], dim=1)
            org_patch_codes_list.append(org_patch_codes)
        patch_code_org = torch.cat(org_patch_codes_list, dim=0).unsqueeze(0)

        masks = patches_to_images(masks, patch_code_org, (new_GS, new_GS))
        #masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        #masks = self.conv_up_proj(masks)
        #masks = self.out_norm(masks)
        #masks = F.leaky_relu(masks)
        for cup in self.conv_up_proj:
            masks = cup(masks)
        masks = self.conv_out_proj(masks)

        return masks