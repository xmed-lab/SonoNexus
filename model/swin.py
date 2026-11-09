"""Shared SonoNexus architecture for pretraining, inference and transfer."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# Keep standard Swin-B stage depths; only the patch projection is replaced.
ENCODER_DEPTHS = (2, 2, 18, 2)
ENCODER_CHANNELS = (128, 256, 512, 1024)
EMBED_PATCH_SIZE = 2
RECON_PATCH_SIZE = 8


def build_encoder(img_size=224, in_channels=3):
    """Build the pretraining encoder without allocating a reconstruction decoder."""
    if img_size % 16:
        raise ValueError("Image size must be divisible by the encoder output stride (16).")
    # Preserve the original checkpoint-producing construction exactly: create
    # patch4 first, then replace only the projection. Do not recompute grids,
    # stage resolutions or window/shift settings when loading existing weights.
    encoder = timm.create_model(
        "swin_base_patch4_window7_224.ms_in22k",
        pretrained=False, features_only=True, img_size=img_size,
        in_chans=in_channels, depths=ENCODER_DEPTHS,
    )
    encoder.patch_embed.proj = nn.Conv2d(in_channels, 128, kernel_size=2, stride=2)
    return encoder


def info_nce_loss(feature1, feature2, temperature=0.07):
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)
    logits = feature1 @ feature2.T / temperature
    return F.cross_entropy(logits, torch.arange(len(feature1), device=logits.device))


class VisionUlt(nn.Module):
    def __init__(self, in_channels=3, img_size=224, patch_size=8, temperature=0.07):
        super().__init__()
        if patch_size != RECON_PATCH_SIZE:
            raise ValueError("Stage-3 reconstruction requires an 8x8 output patch.")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temperature = temperature
        self.model = build_encoder(img_size, in_channels)
        self.decoder_blocks = nn.Sequential(*[
            Block(512, 8, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(4)
        ])
        self.decoder_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, in_channels * patch_size ** 2, bias=True),
        )
        self.merge = nn.Sequential(nn.Linear(ENCODER_CHANNELS[-1], 768))

    def decode(self, stage3):
        batch, height, width, dim = stage3.shape
        tokens = stage3.reshape(batch, height * width, dim)
        pixels = self.decoder_fc(self.decoder_blocks(tokens))
        return self.unpatchify(pixels, height, width)

    def reconstruct(self, x, mask):
        return self.decode(self.model(x * (1 - mask))[2])

    def forward(self, x, mask, masked_view=None):
        # The masked branch reconstructs its own augmentation; CL pairs the views.
        # Omitting masked_view preserves the original full/masked input interface.
        target = x if masked_view is None else masked_view
        full_features = self.model(x)
        masked_features = self.model(target * (1 - mask))
        reconstruction = self.decode(masked_features[2])
        loss_mask, loss_visible = self.forward_reconloss(target, reconstruction, mask)
        # Average tokens are the global/class representations used for CL.
        full_global = self.merge(full_features[3].mean(dim=(1, 2)))
        masked_global = self.merge(masked_features[3].mean(dim=(1, 2)))
        loss_cl = info_nce_loss(full_global, masked_global, self.temperature)
        return reconstruction, loss_mask, loss_visible, loss_cl

    @staticmethod
    def forward_reconloss(x, reconstruction, mask):
        error = (x - reconstruction).abs().mean(dim=1)
        masked = mask[:, 0]
        visible = 1 - masked
        return (
            (error * masked).sum() / masked.sum().clamp_min(1),
            (error * visible).sum() / visible.sum().clamp_min(1),
        )

    def unpatchify(self, x, height, width):
        x = x.reshape(x.shape[0], height, width, self.patch_size, self.patch_size, self.in_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_channels, height * self.patch_size, width * self.patch_size)
