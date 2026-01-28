# Deep Learning Image Inpainting with Attention U-Net

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![License](https://img.shields.io/badge/License-MIT-green)

A computer vision project that reconstructs missing regions in landscape images using a custom **Attention U-Net** architecture. This repository contains the complete training pipeline and an interactive Flask web application for real-time inference.

![Inpainting Results](assets/visualization.png)
*Figure 1: Model performance on the validation set. Columns: Masked Input, Binary Mask, Model Output, Ground Truth.*

## Project Overview
The goal of this project is to perform **Image Inpainting**—filling in missing pixels in a way that is semantically consistent with the surrounding context.

Standard U-Nets often struggle with localized reconstruction because skip connections transfer too much background noise. To solve this, I implemented **Attention Gates** in the decoder, which filter the skip connections to focus only on relevant features during upsampling.

## Key Features
* **Attention U-Net Architecture:** Custom implementation of U-Net with Attention Gates to suppress irrelevant regions in skip connections.
* **Perceptual Loss (VGG16):** Combines **L1 Loss** (for pixel accuracy) with **VGG Perceptual Loss** (using a pre-trained VGG16) to ensure texture and structural consistency.
* **Interactive Web UI:** A Flask-based frontend where users can upload images, manually draw masks on a canvas, and see results instantly.
* **Robust Data Pipeline:** Dynamic mask generation during training to prevent overfitting to specific hole shapes.

## Technical Architecture

### 1. The Model
The network takes a 4-channel input (RGB Image + 1-channel Mask).
* **Encoder:** Convolutional blocks extracting deep spatial features.
* **Attention Gates:** Learn to weight the skip connections, highlighting salient features while suppressing background noise before concatenation.
* **Decoder:** Upsamples the features to reconstruct the original resolution.

### 2. Loss Function
The model is trained using a composite loss function:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{\text{L1}} + \lambda_2 \mathcal{L}_{\text{perceptual}}$$

* **L1 Loss:** Minimizes the pixel-wise difference between the output and ground truth.
* **Perceptual Loss:** Computes the Euclidean distance between feature maps extracted from a frozen VGG16 network, ensuring the *style* and *texture* match the original image.

## Dataset & Preprocessing
* **Source:** [LHQ-1024 Dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024) (Nature/Landscapes).
* **Resolution:** Images are resized to $256 \times 256$ and normalized to range $[-1, 1]$.

### Dynamic Mask Generation
To prevent the model from overfitting to a fixed hole shape or position, masks are generated dynamically during training. As shown in **Figure 2** below, the model pipeline consists of three parts:
1. **Ground Truth:** The original, uncorrupted image.
2. **Mask:** A randomly generated binary mask (white rectangle indicates the missing area).
3. **Masked Input:** The input fed to the network, where the masked region is zeroed out.

![Data Preprocessing](assets/sample_inpainting.png)
*Figure 2: The Training Triplet. The network receives the **Masked Image** (Right) and the **Mask** (Center) as inputs and attempts to reconstruct the **Original Image** (Left).*

### Author
Mikołaj Dybizbański

