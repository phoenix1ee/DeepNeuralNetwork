# Lab 9: Multi-Scale Convolutional Neural Networks & Transfer Learning for Facial Image Classification

## Overview
This project tackles binary human facial image classification to predict male vs. female portraits ("Venus vs. Mars"). Working with face portrait imagery of dimensions $218 \times 178 \times 3$ (CelebA resolution), the pipeline combines a **custom multi-scale Convolutional Neural Network (CNN)**, **transfer learning via MobileNetV2**, and **horizontal-flip test-time augmentation (TTA)** to achieve a robust **94.40% classification accuracy**.

* **Primary Notebook**: [9_VenusMars_final.ipynb](9_VenusMars_final.ipynb)
* **Task**: Binary Facial Image Classification (Male vs. Female / Mars vs. Venus)
* **Input Dimensions**: $218 \times 178 \times 3$ RGB Face Images (resized to $224 \times 224$)
* **Primary Evaluation Metric**: Classification Accuracy
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `numpy`, `pandas`

---

## Computer Vision Architecture

```
                     Input Image (218 × 178 × 3)
                                  │
                       Aspect-Ratio Resizing
                           (224 × 224 × 3)
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
 [Custom Multi-Scale CNN]                       [Pretrained Backbone]
Conv2D (32, 5×5, stride 3)                           MobileNetV2
          │                                     (GlobalAveragePooling)
Conv2D (64, 3×3, stride 2)                                │
          │                                               │
Conv2D (128, 3×3, stride 1) + BN + MaxPool               │
          │                                               │
Conv2D (256, 2×2, stride 1) + BN + Dropout(0.25)          │
          │                                               │
    Global Pooling                                        │
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                          Concatenated Features
                                  │
                          Dense (64) + BatchNorm
                                  │
                           Dropout (0.25)
                                  │
                           Output (Sigmoid)
```

1. **Custom Multi-Scale CNN**:
   * Uses decreasing receptive fields: starts with $5 \times 5$ convolutions (stride 3) to extract coarse facial structural geometry and head contours, transitioning down to $3 \times 3$ and $2 \times 2$ convolutions to resolve localized facial features (eyes, nose, jawline, and hairline).
   * Incorporates **Batch Normalization** and **MaxPooling2D** on deeper blocks to stabilize internal feature distributions and reduce spatial dimensions without loss of salient visual cues.
2. **Transfer Learning Concatenation (MobileNetV2)**:
   * A frozen MobileNetV2 backbone provides general visual semantic representations, fused directly with domain-specific representations learned by the custom CNN.
3. **Test-Time Augmentation (TTA)**:
   * Human facial morphology exhibits strong horizontal symmetry. At inference time, predictions are generated for both the original image and its horizontal reflection, averaging the outputs:
     $$\hat{y}_{\text{final}} = \frac{1}{2} \left( f(x) + f(\text{flip}_{\text{horizontal}}(x)) \right)$$
   * This reduces variance and smooths predictions on ambiguous boundary cases.

---

## Experimental Validation & Results

* **Validation Strategy**: 4-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`).
* **Batch Size / Training**: Batch size 192, 30 epochs, Adam optimizer with learning rate $\eta = 0.002$.

| Model Configuration | Mean Accuracy | Std Dev |
| :--- | :--- | :--- |
| Custom Multi-Scale CNN (32 $\to$ 64 $\to$ 128 $\to$ 256) + MobileNetV2 Concatenation + TTA | **94.40%** | **±0.0025** |

---

## Files
* [9_VenusMars_final.ipynb](9_VenusMars_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
