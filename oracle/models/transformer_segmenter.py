import torch
import torch.nn as nn

from oracle.models.transformer_encoder import VisionTransformer
from oracle.models.transformer_decoder import SegmentationTransformer


class MultiResSegmenter(nn.Module):
    def __init__(self,         
        image_size=(512, 512),
        patch_size=128,
        channels=1,
        n_layers_encoder=4,
        d_encoder=64,
        d_ff_encoder=256,
        n_heads_encoder=4,
        n_layers_decoder=4,
        d_decoder=64,
        d_ff_decoder=256,
        n_heads_decoder=4,
        n_cls=3,
        split_ratio=4,
        n_scales=2) -> None:
        super().__init__()

        self.encoder = VisionTransformer(image_size=image_size,
                    patch_size=patch_size,
                    channels=channels,
                    n_layers=n_layers_encoder,
                    d_model=d_encoder,
                    d_ff=d_ff_encoder,
                    n_heads=n_heads_encoder,
                    split_ratio=split_ratio,
                    n_scales=n_scales)

        self.decoder = SegmentationTransformer(image_size=image_size,
                          patch_size=patch_size,
                          n_layers=n_layers_decoder,
                          d_model=d_decoder,
                          d_encoder=d_encoder,
                          d_ff=d_ff_decoder,
                          n_heads=n_heads_decoder,
                          n_cls=n_cls,
                          split_ratio=split_ratio,
                          n_scales=n_scales)
        

    def forward(self, im, oracle_labels):
        enc_out, patches_scale_coords = self.encoder(im, oracle_labels)
        #print(enc_out.shape)
        dec_out = self.decoder(enc_out, patches_scale_coords)

        return dec_out, enc_out, patches_scale_coords

        

        
