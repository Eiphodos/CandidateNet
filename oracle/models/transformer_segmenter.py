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


class SegmentationTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_encoder,
        d_ff,
        n_heads,
        dropout=0.1,
        drop_path_rate=0.0,
        n_cls=3,
        split_ratio=4
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_encoder = d_encoder
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_cls = n_cls
        self.scale = d_model ** -0.5

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


    def forward(self, x, im_size, patch_code):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)

        

        return masks