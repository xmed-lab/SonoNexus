import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sononexus_downstream import (
    SonoNexusLoRAClassifier,
    count_trainable_parameters,
    load_frozen_lora_backbone,
    multilabel_accuracy,
    seed_everything,
    trainable_parameters,
)


class DummyDiagnosisDataset(Dataset):
    """Replace this with disease labels from your clinical diagnosis dataset."""

    def __init__(self, samples: int, num_labels: int, img_size: int):
        self.samples = samples
        self.num_labels = num_labels
        self.img_size = img_size

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        image = torch.randn(3, self.img_size, self.img_size)
        labels = torch.randint(0, 2, size=(self.num_labels,)).float()
        return image, labels


def build_train_loader(args):
    dataset = DummyDiagnosisDataset(args.dummy_samples, args.num_labels, args.img_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(loader, desc=f"diagnosis epoch {epoch}", ncols=110)
    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += multilabel_accuracy(logits.detach(), labels)
        pbar.set_postfix(loss=f"{running_loss / step:.4f}", micro_acc=f"{running_acc / step:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="SonoNexus LoRA downstream multi-label diagnosis demo")
    parser.add_argument("--checkpoint", default="", help="Path to a SonoNexus pretraining checkpoint")
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dummy-samples", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/diagnosis_lora_demo.pt")
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
    model = SonoNexusLoRAClassifier(encoder, num_outputs=args.num_labels).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    loader = build_train_loader(args)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, loader, optimizer, device, epoch)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, output)
    print(f"Saved diagnosis LoRA demo checkpoint to {output}")


if __name__ == "__main__":
    main()
