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


## 📊 Pre-Training

Here, we provide the inference codes to show the effectivenss of the [pre-trained models] on reconstruct the masked US images and capture the discriminative features.
<div style="display: flex; justify-content: center;">
  <img src="./imgs/visualization_similarity1.png" style="width:34%; margin-right:1%;">
  <img src="./imgs/visualization_similarity.png" style="width:35%;">
</div>


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
