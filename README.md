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
<img src="./imgs/Figure 1-dataset1.png" width="100%">
</div>


## 📊 Training

### Data Preparation

The training and testing datasets are defined in ./dataset_mae_cnn.py, with the data pre-processing and augmentation pipeline.

### Model Architecture

The model is in ./model/swin.py, including the model definition, masked image reconstruction loss and contrastive loss.

### Training Pipeline

The training process is in ./train_mae_cnn.py



## 📋 Supported Tasks

- ✅ Fetal ultrasound view classification
- ✅ Organ segmentation
- ✅ Anatomical structure detection
- ✅ Disease classification



## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).

---
