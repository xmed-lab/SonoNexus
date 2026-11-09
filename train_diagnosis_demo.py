import argparse

from finetune_training import add_training_arguments, fit_downstream

import torch
from torch.utils.data import DataLoader, Dataset

from sononexus_downstream import (
    SonoNexusLoRAClassifier,
    count_trainable_parameters,
    load_frozen_lora_backbone,
    seed_everything,
)


class DummyDiagnosisDataset(Dataset):
    """Replace this with disease labels from your clinical diagnosis dataset."""

    def __init__(self, samples: int, num_labels: int, img_size: int, seed: int = 2026):
        self.seed = seed
        self.samples = samples
        self.num_labels = num_labels
        self.img_size = img_size

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn(3, self.img_size, self.img_size, generator=generator)
        labels = ((torch.arange(self.num_labels) + index) % 2).float()
        return image, labels


def build_train_loader(args):
    dataset = DummyDiagnosisDataset(args.dummy_samples, args.num_labels, args.img_size, args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )


def build_val_loader(args):
    """Replace with a patient-disjoint held-out validation set, never the test set."""
    dataset = DummyDiagnosisDataset(args.dummy_val_samples, args.num_labels, args.img_size, args.seed + 1_000_000)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


def parse_args():
    parser = argparse.ArgumentParser(description="SonoNexus LoRA downstream multi-label diagnosis demo")
    parser.add_argument("--num-labels", type=int, default=4)
    add_training_arguments(parser, "outputs/diagnosis_lora_demo.pt")
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
        img_size=args.img_size,
    )
    model = SonoNexusLoRAClassifier(encoder, num_outputs=args.num_labels).to(device)
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    fit_downstream(model, build_train_loader(args), build_val_loader(args),
                   device, args, task="diagnosis")



if __name__ == "__main__":
    main()
