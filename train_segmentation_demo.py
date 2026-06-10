import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sononexus_downstream import (
    SonoNexusLoRAUNetSegmenter,
    binary_dice_from_logits,
    count_trainable_parameters,
    load_frozen_lora_backbone,
    seed_everything,
    trainable_parameters,
)


class DummySegmentationDataset(Dataset):
    """Replace this with paired ultrasound images and segmentation masks."""

    def __init__(self, samples: int, num_classes: int, img_size: int):
        self.samples = samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        image = torch.randn(3, self.img_size, self.img_size)
        if self.num_classes == 1:
            mask = (torch.rand(1, self.img_size, self.img_size) > 0.65).float()
        else:
            mask = torch.randint(0, self.num_classes, size=(self.img_size, self.img_size)).long()
        return image, mask


def build_train_loader(args):
    dataset = DummySegmentationDataset(args.dummy_samples, args.num_classes, args.img_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )


def segmentation_loss(logits, masks, num_classes):
    if num_classes == 1:
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        probs = torch.sigmoid(logits)
        intersection = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
        return bce + dice_loss
    return F.cross_entropy(logits, masks)


def train_one_epoch(model, loader, optimizer, device, epoch, num_classes):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(loader, desc=f"segmentation epoch {epoch}", ncols=110)
    for step, (images, masks) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = segmentation_loss(logits, masks, num_classes)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if num_classes == 1:
            running_dice += binary_dice_from_logits(logits.detach(), masks)
            pbar.set_postfix(loss=f"{running_loss / step:.4f}", dice=f"{running_dice / step:.3f}")
        else:
            pbar.set_postfix(loss=f"{running_loss / step:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="SonoNexus LoRA downstream segmentation demo")
    parser.add_argument("--checkpoint", default="", help="Path to a SonoNexus pretraining checkpoint")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dummy-samples", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/segmentation_lora_unet_demo.pt")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_frozen_lora_backbone(
        args.checkpoint or None,
        device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = SonoNexusLoRAUNetSegmenter(encoder, num_classes=args.num_classes).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    loader = build_train_loader(args)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, loader, optimizer, device, epoch, args.num_classes)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, output)
    print(f"Saved segmentation LoRA-UNet demo checkpoint to {output}")


if __name__ == "__main__":
    main()
