"""Load the shared pretraining model, or export/load its encoder alone."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from model.swin import VisionUlt as PretrainingVisionUlt, build_encoder


class VisionUlt(PretrainingVisionUlt):
    """Reconstruction-only interface with exactly the pretraining parameters."""
    def forward(self, x, mask):
        return self.reconstruct(x, mask)


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def read_checkpoint(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a model state_dict.")
    return clean_state_dict(payload)


def load_pretrained_encoder(checkpoint=None, device=None, img_size=224):
    """Ignore reconstruction weights, but require every encoder weight to match."""
    encoder = build_encoder(img_size=img_size)
    if checkpoint:
        state_dict = read_checkpoint(checkpoint)
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {
                key[len("model."):]: value
                for key, value in state_dict.items() if key.startswith("model.")
            }
        encoder.load_state_dict(state_dict, strict=True)
        print(f"Loaded complete pretrained encoder: {checkpoint}")
    return encoder.to(device or torch.device("cpu"))


def load_pretrained_model(checkpoint, device=None, strict=True, img_size=224):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionUlt(img_size=img_size)
    if checkpoint:
        incompatible = model.load_state_dict(read_checkpoint(checkpoint), strict=strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint does not match the pretraining architecture: {incompatible}")
        print(f"Loaded complete pretraining model: {checkpoint}")
    return model.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser(description="Load/export a SonoNexus pretraining checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder-only", action="store_true", help="Export just the pretrained encoder")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    loader = load_pretrained_encoder if args.encoder_only else load_pretrained_model
    model = loader(args.checkpoint, device=torch.device(args.device), img_size=args.img_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved {'encoder' if args.encoder_only else 'pretraining model'} weights to {output}")


if __name__ == "__main__":
    main()
