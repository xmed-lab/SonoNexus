<div align="center">
  <h1>SonoNexus</h1>
  <h3>A Universal Foundation Model for Sensor-Agnostic Ultrasound Imaging</h3>

  <p>
    <b>Unified ultrasound representation</b> ·
    <b>Masked image reconstruction</b> ·
    <b>Downstream clinical transfer</b>
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#pre-training">Pre-training</a> ·
    <a href="#downstream-demos">Downstream Demos</a> ·
    <a href="#feature-visualization">Feature Visualization</a>
  </p>
</div>

<div align="center">
  <img src="./imgs/Figure 1-dataset1.png" width="78%" alt="SonoNexus dataset overview">
</div>

## Highlights

SonoNexus is a foundation model for ultrasound images across heterogeneous scanners, hospitals, organs, and clinical tasks. It is designed as a sensor-agnostic representation learner: a single pre-trained model can be transferred to view classification, disease diagnosis, organ segmentation, and detection-style downstream workflows.

| Component | Status | Entry point |
| --- | --- | --- |
| MAE-style ultrasound pre-training | Available | `main_mae_cnn.py`, `train_mae_cnn.py` |
| Mask generation and augmentation | Available | `dataset_mae_cnn.py` |
| SonoNexus backbone | Available | `model/swin.py` |
| Pre-trained checkpoint loader | Available | `load_model.py` |
| LoRA downstream utilities | Available | `sononexus_downstream.py` |
| Downstream classification demo | Available | `train_classification_demo.py` |
| Downstream diagnosis demo | Available | `train_diagnosis_demo.py` |
| Downstream segmentation demo | Available | `train_segmentation_demo.py` |
| Feature and reconstruction visualization | Available | `visualize_pretrained_features.py` |

## News

- **2025-11**: Initial SonoNexus project setup.
- **2026-06**: Added downstream training demos and a polished feature visualization toolkit.

## Overview

SonoNexus is built around two central ideas:

1. **Sono-21M scale**: a large ultrasound corpus with 21.14M images, 20 major organ types, 10 mainstream sensor models, and multi-center acquisition across 17 hospitals.
2. **Self-supervised transfer**: masked ultrasound reconstruction and contrastive regularization encourage robust anatomical features that can transfer across scanners and tasks.

<div align="center">
  <img src="./imgs/pipe.jpg" width="58%" alt="SonoNexus training pipeline">
</div>

## Quick Start

Install the usual PyTorch vision stack first. The code relies on `torch`, `torchvision`, `timm`, `tqdm`, `matplotlib`, `Pillow`, `numpy`, `pyyaml`, `wandb`, and `tensorboard`.

```bash
python main_mae_cnn.py \
  -data /path/to/pretraining/images \
  --batch_size 64 \
  --epochs 300 \
  --imgsize 224 \
  --gpus 0
```

The pre-training dataset is expected to be an image folder. `dataset_mae_cnn.py` recursively scans common image extensions and creates random block masks for MAE reconstruction.

## Pre-training

The pre-training model is implemented in `model/swin.py`.

- **Backbone**: Swin Transformer from `timm`, configured as a feature extractor.
- **Objective 1**: masked image reconstruction on ultrasound patches.
- **Objective 2**: contrastive alignment between full-image and masked-image global features.
- **Training loop**: `train_mae_cnn.py` with TensorBoard/W&B logging and reconstruction previews.

Pre-trained model weights can be downloaded from:
[Google Drive checkpoint folder](https://drive.google.com/drive/folders/1Ff7fdXIGN9nsWqf9la0Tc7kYRKxFds04?usp=drive_link)

For downstream transfer and visualization, use the checkpoint-loading architecture in `load_model.py`. It matches the released inference model with `depths=(2, 2, 18, 10)`, the `ReLU + Linear` reconstruction head, and the `merge` projection for feature maps.

```bash
python load_model.py \
  --checkpoint /path/to/timm_model_99.pth \
  --output /path/to/epochs.pth
```

<div align="center">
  <img src="./imgs/visualization_similarity1.png" width="36%" alt="SonoNexus reconstruction visualization">
  <img src="./imgs/visualization_similarity.png" width="36%" alt="SonoNexus similarity visualization">
</div>

## Downstream Demos

The repository now provides three clean demo trainers in the project root. Each downstream script follows the same transfer recipe:

- load the SonoNexus pre-trained model
- freeze the pre-trained backbone weights
- inject trainable LoRA adapters into Swin linear layers
- train only LoRA parameters and the task-specific head/decoder

Dataset definitions are intentionally lightweight: each script contains a `build_train_loader(args)` function and a dummy dataset so the training loop can be smoke-tested immediately. Replace that function with your real dataset when ready.

### 1. Fetal View / General Classification

Use this for mutually exclusive classes such as fetal ultrasound view classification or organ category classification.

```bash
python train_classification_demo.py \
  --checkpoint /path/to/sononexus_pretrained.pth \
  --num-classes 5 \
  --lora-rank 8 \
  --batch-size 8 \
  --epochs 2
```

Loss: `CrossEntropyLoss`. Trainable modules: LoRA adapters plus the classification head.

<div align="center">
  <img src="./imgs/vc.jpg" width="52%" alt="Fetal view classification">
</div>

### 2. Disease Diagnosis

Use this for binary or multi-label disease diagnosis. The demo head produces `num_labels` logits and trains with `BCEWithLogitsLoss`.

```bash
python train_diagnosis_demo.py \
  --checkpoint /path/to/sononexus_pretrained.pth \
  --num-labels 4 \
  --lora-rank 8 \
  --batch-size 8 \
  --epochs 2
```

Loss: `BCEWithLogitsLoss`. Trainable modules: LoRA adapters plus the multi-label diagnosis head.

<div align="center">
  <img src="./imgs/ds.jpg" width="52%" alt="Disease classification">
</div>

### 3. Organ / Lesion Segmentation

Use this for binary masks or multi-class masks. The demo extracts multi-scale features from the frozen SonoNexus backbone and feeds them into a UNet-like decoder with skip connections.

```bash
python train_segmentation_demo.py \
  --checkpoint /path/to/sononexus_pretrained.pth \
  --num-classes 1 \
  --lora-rank 8 \
  --batch-size 4 \
  --epochs 2
```

Loss:

- `BCEWithLogitsLoss + Dice loss` for `--num-classes 1`
- `CrossEntropyLoss` for `--num-classes > 1`

Trainable modules: LoRA adapters plus the UNet-like decoder. The frozen pre-trained model supplies four Swin feature scales.

<div align="center">
  <img src="./imgs/os.jpg" width="52%" alt="Organ segmentation">
</div>

## Feature Visualization

`visualize_pretrained_features.py` provides a complete visualization workflow for pre-trained SonoNexus checkpoints:

- masked-input reconstruction preview
- average-pooled token similarity map
- max-pooled token similarity map
- PCA projection of global image embeddings
- CSV export of embedding coordinates

```bash
python visualize_pretrained_features.py \
  --image-dir /path/to/ultrasound/images \
  --checkpoint /path/to/sononexus_pretrained.pth \
  --output-dir outputs/feature_visualization \
  --anchor both \
  --max-images 128
```

Outputs:

| File | Description |
| --- | --- |
| `feature_gallery.png` | Original, masked input, reconstruction, and similarity overlays |
| `embedding_pca.png` | 2D PCA view of SonoNexus global features |
| `embedding_pca.csv` | Image path, folder label, PC1, and PC2 |

If images are arranged as `root/class_name/image.png`, parent folder names are used as labels in the PCA plot. If images are placed directly under one folder, they are treated as `unlabeled`.

## Supported Clinical Tasks

| Task | Typical head | Demo |
| --- | --- | --- |
| Fetal ultrasound view classification | frozen backbone + LoRA + pooled classifier | `train_classification_demo.py` |
| Disease classification / diagnosis | frozen backbone + LoRA + multi-label head | `train_diagnosis_demo.py` |
| Organ or lesion segmentation | frozen backbone + LoRA + multi-scale UNet-like decoder | `train_segmentation_demo.py` |
| Anatomical structure detection | detector head on SonoNexus features | planned |

<div align="center">
  <img src="./imgs/dt.jpg" width="52%" alt="Anatomical structure detection">
</div>

## Repository Layout

```text
SonoNexus-main/
├── dataset_mae_cnn.py                    # Pre-training image loader and mask generator
├── main_mae_cnn.py                       # Pre-training launch script
├── train_mae_cnn.py                      # Pre-training loop and logging
├── load_model.py                         # Released checkpoint loader/converter
├── sononexus_downstream.py               # Frozen backbone, LoRA, downstream heads/decoder
├── train_classification_demo.py          # LoRA multi-class classification demo
├── train_diagnosis_demo.py               # LoRA multi-label diagnosis demo
├── train_segmentation_demo.py            # LoRA multi-scale UNet-like segmentation demo
├── visualize_pretrained_features.py      # Reconstruction and feature visualization
├── model/
│   ├── swin.py                           # SonoNexus MAE backbone
│   ├── mednext.py                        # MedNeXt encoder experiments
│   └── vqvae.py
└── imgs/                                 # README figures
```

## Notes for Custom Datasets

To connect private datasets, edit only the `build_train_loader(args)` function in the corresponding downstream demo. The model expects normalized RGB tensors shaped `[B, 3, H, W]`. For ultrasound grayscale images, convert to RGB by channel repetition or `PIL.Image.convert("RGB")` before normalization.

Recommended normalization is the ImageNet convention already used by pre-training:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## License

This project is released under the Apache 2.0 License.
