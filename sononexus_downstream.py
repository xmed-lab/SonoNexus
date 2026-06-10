import random
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_model import VisionUlt, clean_state_dict


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class LoRALinear(nn.Module):
    """Trainable low-rank adapter around a frozen Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base = base
        freeze_module(self.base)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.dropout(self.lora_a(x))) * self.scale


def _matches_lora_target(module_name: str, target_keywords: Sequence[str]) -> bool:
    module_name = module_name.lower()
    return any(keyword.lower() in module_name for keyword in target_keywords)


def inject_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_keywords: Sequence[str] = ("qkv", "proj", "fc1", "fc2", "reduction"),
    prefix: str = "",
) -> int:
    """Recursively replace selected frozen Linear layers with LoRA adapters."""
    injected = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches_lora_target(full_name, target_keywords):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            injected += 1
        else:
            injected += inject_lora(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_keywords=target_keywords,
                prefix=full_name,
            )
    return injected


def load_frozen_lora_backbone(
    checkpoint: Optional[str],
    device: torch.device,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_targets: Sequence[str] = ("qkv", "proj", "fc1", "fc2", "reduction"),
) -> nn.Module:
    """Load SonoNexus, freeze the pre-trained backbone, then attach LoRA."""
    mae = VisionUlt()

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = mae.load_state_dict(clean_state_dict(state_dict), strict=False)
        print(
            f"Loaded checkpoint: {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print("No checkpoint was provided; using randomly initialized SonoNexus weights.")

    backbone = mae.model
    freeze_module(backbone)
    injected = inject_lora(
        backbone,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_keywords=lora_targets,
    )
    print(f"Frozen SonoNexus backbone with {injected} LoRA Linear adapters.")
    return backbone.to(device)


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in module.parameters() if param.requires_grad)


def count_trainable_parameters(module: nn.Module) -> Tuple[int, int]:
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    total = sum(param.numel() for param in module.parameters())
    return trainable, total


def feature_channels(encoder: nn.Module, fallback: int = 1024) -> int:
    if hasattr(encoder, "feature_info"):
        try:
            return encoder.feature_info.channels()[-1]
        except Exception:
            return fallback
    return fallback


def multi_scale_channels(encoder: nn.Module, fallback=(128, 256, 512, 1024)):
    if hasattr(encoder, "feature_info"):
        try:
            return tuple(encoder.feature_info.channels())
        except Exception:
            return fallback
    return fallback


def to_nchw(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, got shape {tuple(feature.shape)}")
    if feature.shape[-1] > feature.shape[1]:
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


def pooled_backbone_feature(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    feature = to_nchw(encoder(images)[-1])
    return F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)


class SonoNexusLoRAClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_outputs: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        in_features = feature_channels(encoder)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_outputs),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = pooled_backbone_feature(self.encoder, images)
        return self.head(features)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(16, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SonoNexusLoRAUNetSegmenter(nn.Module):
    """Frozen SonoNexus + LoRA adapters + UNet-like multi-scale decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 1,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
    ):
        super().__init__()
        self.encoder = encoder
        channels = multi_scale_channels(encoder)
        if len(channels) < 4:
            raise ValueError(f"Expected at least 4 encoder feature scales, got {channels}")

        c1, c2, c3, c4 = channels[-4:]
        d4, d3, d2, d1 = decoder_channels

        self.center = ConvBlock(c4, d4)
        self.up3 = UpBlock(d4, c3, d3)
        self.up2 = UpBlock(d3, c2, d2)
        self.up1 = UpBlock(d2, c1, d1)
        self.refine = ConvBlock(d1, d1)
        self.seg_head = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = [to_nchw(feature) for feature in self.encoder(images)]
        f1, f2, f3, f4 = features[-4:]

        x = self.center(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.refine(x)
        logits = self.seg_head(x)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == targets).float().mean().item()


def binary_dice_from_logits(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (union + eps)).mean().item()
