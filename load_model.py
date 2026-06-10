import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class VisionUlt(nn.Module):
    """Inference/checkpoint-loading version of SonoNexus.

    This mirrors the architecture used by the provided `load_model.py` snippet:
    a Swin-Base feature extractor with custom depths, the MAE reconstruction
    decoder, and the `merge` projection used by feature visualization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.model = timm.create_model(
            "swin_base_patch4_window7_224.ms_in22k",
            pretrained=False,
            depths=(2, 2, 18, 10),
            features_only=True,
        )
        self.model.patch_embed.proj = nn.Conv2d(3, 128, kernel_size=2, stride=2)

        out_channels = in_channels * patch_size ** 2
        decoder_depth = 4
        self.decoder_blocks = nn.Sequential(*[
            Block(512, 8, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.decoder_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, out_channels, bias=True),
        )
        self.merge = nn.Sequential(
            nn.Linear(768, 768),
        )

    def forward(self, x, mask):
        x_mask = x * (1 - mask)

        f_mask = self.model(x_mask)

        f3_mask = f_mask[3]
        batch_size, self.h, self.w, dim = f3_mask.size()
        f3_mask = f3_mask.reshape(batch_size, self.h * self.w, dim)
        _ = self.merge(f3_mask)

        f2_mask = f_mask[2]
        batch_size, self.h, self.w, dim = f2_mask.size()
        f2_mask = f2_mask.reshape(batch_size, self.h * self.w, dim)

        f_up_mask = self.decoder_fc(self.decoder_blocks(f2_mask))
        return self.unpatchify(f_up_mask)

    def unpatchify(self, x):
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], 3, self.h * self.patch_size, self.w * self.patch_size))


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_pretrained_model(checkpoint, device=None, strict=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionUlt().to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=strict)
        print(
            f"Loaded checkpoint: {checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Load/convert a SonoNexus pretrained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output state_dict path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_pretrained_model(args.checkpoint, device=device, strict=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved converted SonoNexus state_dict to {output}")


if __name__ == "__main__":
    main()
