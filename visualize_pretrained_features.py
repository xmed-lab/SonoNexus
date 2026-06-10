import argparse
import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from load_model import VisionUlt, load_pretrained_model
from sononexus_downstream import IMAGENET_MEAN, IMAGENET_STD, seed_everything, to_nchw


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def use_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


class UltrasoundImageFolder(Dataset):
    def __init__(self, image_dir: str, img_size: int):
        self.root = Path(image_dir)
        self.paths = sorted(
            path for path in self.root.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images were found under {self.root}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        label = path.parent.name if path.parent != self.root else "unlabeled"
        return self.transform(image), label, str(path)


def denormalize(batch: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    images = (batch * std + mean).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    return (images * 255).astype(np.uint8)


def generate_block_mask(
    batch_size: int,
    img_size: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    total = grid_h * grid_w
    masked = max(1, int(total * mask_ratio))
    masks = torch.zeros(batch_size, 1, grid_h, grid_w, device=device)
    for index in range(batch_size):
        selected = torch.randperm(total, device=device)[:masked]
        masks[index, 0].view(-1)[selected] = 1.0
    return F.interpolate(masks, size=(img_size, img_size), mode="nearest")


def similarity_heatmap(features: torch.Tensor, image_size: Tuple[int, int], anchor: str) -> torch.Tensor:
    fmap = to_nchw(features)
    if anchor == "avg":
        query = fmap.mean(dim=(2, 3), keepdim=True)
    elif anchor == "max":
        query = torch.amax(fmap, dim=(2, 3), keepdim=True)
    else:
        raise ValueError("anchor must be 'avg' or 'max'")

    heatmap = F.cosine_similarity(fmap, query, dim=1).unsqueeze(1)
    heatmap = F.interpolate(heatmap, size=image_size, mode="bilinear", align_corners=False)
    return heatmap.squeeze(1)


def colorize_heatmap(heatmap: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    return (cmap(heatmap)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    color = colorize_heatmap(heatmap)
    return ((1 - alpha) * image + alpha * color).clip(0, 255).astype(np.uint8)


def save_gallery(
    output_path: Path,
    images: torch.Tensor,
    masks: torch.Tensor,
    recon: torch.Tensor,
    heatmaps: Sequence[torch.Tensor],
    anchors: Sequence[str],
    paths: Sequence[str],
) -> None:
    originals = denormalize(images)
    recon_images = denormalize(recon)
    masked_images = (originals * (1.0 - masks.permute(0, 2, 3, 1).cpu().numpy())).astype(np.uint8)

    rows = min(images.shape[0], 6)
    cols = 3 + len(anchors)
    use_plot_style()
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    fig.patch.set_facecolor("#f8fafc")
    titles = ["Original", "Masked input", "Reconstruction"] + [f"{anchor.upper()} token map" for anchor in anchors]

    for row in range(rows):
        panels: List[np.ndarray] = [originals[row], masked_images[row], recon_images[row]]
        for heatmap in heatmaps:
            panels.append(overlay_heatmap(originals[row], heatmap[row].detach().cpu().numpy()))

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[col], fontsize=12, weight="bold", color="#0f172a")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.1)
                spine.set_edgecolor("#e2e8f0")

        axes[row, 0].set_ylabel(Path(paths[row]).name[:28], fontsize=10, color="#475569")

    fig.suptitle("SonoNexus Pretrained Feature Visualization", fontsize=18, weight="bold", color="#0f172a")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pca_2d(embeddings: torch.Tensor) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    q = min(2, centered.shape[0], centered.shape[1])
    _, _, vectors = torch.pca_lowrank(centered, q=q)
    coords = (centered @ vectors[:, :q]).cpu().numpy()
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords[:, :2]


def save_embedding_plot(output_path: Path, coords: np.ndarray, labels: Sequence[str]) -> None:
    unique_labels = sorted(set(labels))
    palette = plt.get_cmap("tab20")

    use_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for index, label in enumerate(unique_labels):
        mask = np.array([item == label for item in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=54,
            alpha=0.82,
            color=palette(index % 20),
            edgecolors="#ffffff",
            linewidths=0.7,
            label=label,
        )

    ax.set_title("SonoNexus Feature Embedding PCA", fontsize=18, weight="bold", color="#0f172a")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_embedding_csv(output_path: Path, paths: Sequence[str], labels: Sequence[str], coords: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "pc1", "pc2"])
        for path, label, coord in zip(paths, labels, coords):
            writer.writerow([path, label, float(coord[0]), float(coord[1])])


def extract_embeddings(model: VisionUlt, loader: DataLoader, device: torch.device, max_images: int):
    embeddings = []
    labels = []
    paths = []
    seen = 0
    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(loader, desc="extract embeddings", ncols=100):
            images = images.to(device)
            features = to_nchw(model.merge(model.model(images)[-1]))
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            embeddings.append(pooled.cpu())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
            seen += images.shape[0]
            if seen >= max_images:
                break
    embeddings = torch.cat(embeddings, dim=0)[:max_images]
    return embeddings, labels[:max_images], paths[:max_images]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SonoNexus pretrained reconstruction and features")
    parser.add_argument("--image-dir", required=True, help="Folder of ultrasound images. Parent folders are used as labels.")
    parser.add_argument("--checkpoint", default="", help="Path to a SonoNexus pretraining checkpoint")
    parser.add_argument("--output-dir", default="outputs/feature_visualization")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--anchor", choices=("avg", "max", "both"), default="both")
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    dataset = UltrasoundImageFolder(args.image_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = load_pretrained_model(args.checkpoint or None, device=device, strict=False)

    images, labels, paths = next(iter(loader))
    images = images.to(device)
    masks = generate_block_mask(images.shape[0], args.img_size, args.patch_size, args.mask_ratio, device)

    with torch.no_grad():
        recon = model(images, masks)
        masked_features = model.merge(model.model(images * (1.0 - masks))[-1])

    anchors = ["avg", "max"] if args.anchor == "both" else [args.anchor]
    heatmaps = [similarity_heatmap(masked_features, (args.img_size, args.img_size), anchor) for anchor in anchors]
    save_gallery(output_dir / "feature_gallery.png", images, masks, recon, heatmaps, anchors, paths)

    embedding_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    embeddings, all_labels, all_paths = extract_embeddings(model, embedding_loader, device, args.max_images)
    coords = pca_2d(embeddings)
    save_embedding_plot(output_dir / "embedding_pca.png", coords, all_labels)
    save_embedding_csv(output_dir / "embedding_pca.csv", all_paths, all_labels, coords)

    print(f"Saved visualization gallery to {output_dir / 'feature_gallery.png'}")
    print(f"Saved embedding PCA plot to {output_dir / 'embedding_pca.png'}")
    print(f"Saved embedding coordinates to {output_dir / 'embedding_pca.csv'}")


if __name__ == "__main__":
    main()
