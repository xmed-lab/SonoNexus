<div style="display: flex; align-items: center; justify-content: center;">
  <!--<img src="Hulu-Med.png" width="50" style="margin-right: 15px; flex-shrink: 0;">-->
  <h1 style="margin: 0; text-align: left;">
    SonoNexus: A Universal Foundation Model for Sensor-Agnostic Ultrasound Imaging
  </h1>
</div>



## 🔥 News
- **[2025-11]** Setup the GitHub project of SonoNexus!!!

## 📖 Overview

**Hulu-Med** SonoNexus is a foundation model-powered sensing system that acts as a **hardware-agnostic Rosetta Stone** for interpreting images across the entire sensor landscape. It is built upon two cornerstone contributions. First, we construct **Sono-21M**, the largest and most diverse ultrasound dataset to date, comprising 21.14 million images of 20 major organ types. Purposefully curated from 10 distinct mainstream sensor models across 17 hospitals. Second, we developed SonoNexus via a self-supervised learning strategy, enabling seamless performance across a broad spectrum of devices and downstream clinical applications.

<div align="center">
<img src="./imgs/Figure 1-dataset1.png" width="70%">
</div>


## 📊 Pre-Training towards Unified Representation for US Imaging

Here, we provide the inference codes to show the effectivenss of the [pre-trained models] on reconstruct the masked US images and capture the discriminative features.
<div style="display: flex; justify-content: center;">
  <img src="./imgs/visualization_similarity1.png" style="width:34%; margin-right:1%;">
  <img src="./imgs/visualization_similarity.png" style="width:34%;">
</div>

Detailed feature visualization and image inference codes are define in test_model.py. To calculate the activation maps, we provide two query anchors, including max-pooled token and average-pooled token among patch tokens.

```Python
import torch
import torch.nn.functional as F
from model.swin import VisionUlt
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataset_mae_cnn import get_data

# 设置设备
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 辅助函数
# ==========================================

def denormalize(img_tensor):
    """
    将 ImageNet 标准化的 tensor 转回 0-255 的 numpy array (H, W, C)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    return (img * 255).astype(np.uint8)

def compute_similarity_heatmap(feats, img_size, pool_type='avg'):
    """
    计算特征图与全局特征的余弦相似度热力图
    
    Args:
        feats: [B, H, W, C] 输入特征
        img_size: (Target_H, Target_W) 原图尺寸
        pool_type: 'avg' (平均池化) 或 'max' (最大池化)
    """
    B, H, W, C = feats.shape
    
    # 1. 计算全局特征向量 (Global Feature Vector)
    # 在空间维度 (H, W) 即维度 1 和 2 上进行池化
    if pool_type == 'avg':
        # [B, H, W, C] -> [B, 1, 1, C]
        global_feat = feats.mean(dim=(1, 2), keepdim=True)
    elif pool_type == 'max':
        # [B, H, W, C] -> [B, C] -> [B, 1, 1, C]
        # torch.amax 支持多维度 max
        global_feat = torch.amax(feats, dim=(1, 2), keepdim=True)
    else:
        raise ValueError("pool_type must be 'avg' or 'max'")
        
    # 2. 计算余弦相似度
    # feats:       [B, H, W, C]
    # global_feat: [B, 1, 1, C]
    # F.cosine_similarity 会自动广播，沿着 dim=-1 (通道) 计算
    similarity_map = F.cosine_similarity(feats, global_feat, dim=-1) # 结果: [B, H, W]
    
    # 3. 上采样到原图尺寸
    # 插值需要 [B, C, H, W] 格式，这里 C=1
    similarity_map = similarity_map.unsqueeze(1) # [B, 1, H, W]
    similarity_map = F.interpolate(similarity_map, size=img_size, mode='bilinear', align_corners=False)
    similarity_map = similarity_map.squeeze(1)   # [B, Target_H, Target_W]
    
    return similarity_map

def apply_heatmap_overlay(img_rgb, heatmap_tensor):
    """
    将热力图叠加到原图上
    """
    # 转为 numpy
    heatmap_np = heatmap_tensor.cpu().detach().numpy()
    
    # 归一化 (Min-Max) 到 0-1
    # 余弦相似度范围通常在 [-1, 1]，我们需要将其映射到可视化范围
    heatmap_np = heatmap_np - np.min(heatmap_np)
    heatmap_np = heatmap_np / (np.max(heatmap_np) + 1e-8)
    
    # 转换为伪彩色
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
    
    return heatmap_color, overlay

# ==========================================
# 2. 模型加载
# ==========================================

path = "path_for_downloaded_pth_file"

model = VisionUlt().to(device)
checkpoint = torch.load(path, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Model loaded successfully.")

dataloader = get_data(data_root="path_for_your_test_images", batch_size=4)

# ==========================================
# 3. 主循环与可视化
# ==========================================

for data in dataloader:
    image, mask, _ = data
    image = image.to(device)
    mask = mask.to(device)
    
    # 推理
    with torch.no_grad():
        image_recon = model(image, mask)
        # 获取特征: [B, H, W, C]
        feats = model.model(image * (1 - mask))[3]
        feats = model.merge(feats)
    
    print(f"Feats shape: {feats.shape}") 

    mse = ((image_recon - image) ** 2).mean()
    print(f"mse is {mse}")

    # 准备可视化数据
    batch_size = image.shape[0]
    img_h, img_w = image.shape[2], image.shape[3]
    
    # --- 核心修改：计算相似度热力图 ---
    # 您可以选择 pool_type='avg' 或 'max'
    heatmaps_resized = compute_similarity_heatmap(feats, (img_h, img_w), pool_type='max')
    
    # 创建画布
    fig, axs = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1: axs = axs[None, :]
    
    for i in range(batch_size):
        # 1. 原始图片
        img_orig = denormalize(image[i])
        img_recon = denormalize(image_recon[i])
        
        # 2. Masked Image
        mask_np = mask[i].permute(1, 2, 0).cpu().detach().numpy()
        img_masked = img_orig * (1 - mask_np)
        img_masked = img_masked.astype(np.uint8)
        
        # 3. Similarity Heatmap & Overlay
        heatmap_vis, overlay_vis = apply_heatmap_overlay(img_orig, heatmaps_resized[i])
        
        # --- 绘图 ---
        axs[i, 0].imshow(img_orig)
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(img_masked)
        axs[i, 1].set_title("Masked Input")
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(img_recon)
        axs[i, 2].set_title("Recon Image")
        axs[i, 2].axis('off')
        
        axs[i, 3].imshow(overlay_vis)
        axs[i, 3].set_title("Overlay")
        axs[i, 3].axis('off')

    plt.tight_layout()
    save_path = "visualization_similarity.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    break
```


**If ones are willing to pre-train SonoNexus on in-house datasets, please refer**:

### 1. Data Preparation

  <div align="center">
  <img src="./imgs/pipe.jpg" width="50%">
  </div>
The training and testing datasets are defined in ./dataset_mae_cnn.py, with the data pre-processing augmentation pipeline and masking strategy.
Our in-house pre-trained data consists of a large-scale dataset of **21,140,761** covering **20
major organs**, enabling comprehensive model training and evaluation, collected from 10 types of ultrasound equipment/sensors.

### 2. Model Architecture

The model is in ./model/swin.py, including the model definition, masked image reconstruction loss and contrastive loss.

### 3. Training Pipeline

The training process is in ./train_mae_cnn.py and the running file is ./main_mae_cnn.py



## 📋 Supported Tasks

- ✅ Fetal ultrasound view classification
  <div align="center">
  <img src="./imgs/vc.jpg" width="50%">
  </div>
- ✅ Organ segmentation
  <div align="center">
  <img src="./imgs/os.jpg" width="50%">
  </div>
- ✅ Anatomical structure detection
  <div align="center">
  <img src="./imgs/dt.jpg" width="50%">
  </div>
- ✅ Disease classification
  <div align="center">
  <img src="./imgs/ds.jpg" width="50%">
  </div>

When pre-trained period is finised, ones can easily transfer the model into diverse down-stream tasks for US images. In the main paper, we focus on four tasks, inclduing fetal ultrasound view classification, organ segmentation, anatomical structure detection and disease classification.






## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).

---
