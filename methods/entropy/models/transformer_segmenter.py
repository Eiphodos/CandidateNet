import torch.nn as nn

from methods.metaloss.models.transformer_encoder import VisionTransformer
from methods.metaloss.models.transformer_decoder import SegmentationTransformer
from methods.metaloss.models.transformer_decoder_simple import SimpleMaskDecoder


class MultiResSegmenter(nn.Module):
    def __init__(self,         
        image_size=(512, 512),
        patch_size=128,
        channels=1,
        n_layers_encoder=[2, 2],
        d_encoder=[128, 64],
        n_heads_encoder=[16, 8],
        n_layers_decoder=4,
        d_decoder=64,
        n_heads_decoder=4,
        n_cls=3,
        split_ratio=4,
        n_scales=2,
        decoder_type='advanced') -> None:
        super().__init__()

        self.encoder = VisionTransformer(image_size=image_size,
                    patch_size=patch_size,
                    channels=channels,
                    n_layers=n_layers_encoder,
                    d_model=d_encoder,
                    n_heads=n_heads_encoder,
                    split_ratio=split_ratio,
                    n_scales=n_scales)

        if decoder_type == 'advanced':
            self.decoder = SegmentationTransformer(image_size=image_size,
                            patch_size=patch_size,
                            n_layers=n_layers_decoder,
                            d_model=d_decoder,
                            d_encoder=d_encoder[-1],
                            n_heads=n_heads_decoder,
                            n_cls=n_cls,
                            split_ratio=split_ratio,
                            n_scales=n_scales)
        elif decoder_type == 'simple':
            self.decoder = SimpleMaskDecoder(image_size=image_size,
                    patch_size=patch_size,
                    n_layers=n_layers_decoder,
                    d_model=d_decoder,
                    d_encoder=d_encoder[-1],
                    n_heads=n_heads_decoder,
                    n_cls=n_cls)

    def forward(self, im):
        enc_out, patches_scale_coords = self.encoder(im)
        #print(enc_out.shape)
        dec_out = self.decoder(enc_out, patches_scale_coords)

        return dec_out, enc_out, patches_scale_coords

        

        
