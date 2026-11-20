# Lab 5: Wide & Deep Neural Networks for E-Commerce Purchase Intent Prediction

## Overview
This project predicts whether an online shopping browsing session will culminate in a purchase transaction (`Rev = 1`). Web analytics datasets present a fundamental modeling challenge: they require simultaneously **memorizing** high-value specific feature combinations (e.g., specific visitor types shopping during holiday months on weekends) and **generalizing** across continuous behavioral signals (e.g., session durations, bounce rates, and page values). To address this, we implement a **Wide & Deep Neural Network** in TensorFlow/Keras inspired by Google's recommendation architecture (Cheng et al., 2016).

* **Primary Notebook**: [5_Shop_final.ipynb](5_Shop_final.ipynb)
* **Task**: Binary Classification (Shopper Conversion / Revenue)
* **Target Variable**: `Rev` (0: No Purchase, 1: Purchase)
* **Primary Evaluation Metrics**: Accuracy, F1-Score
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `pandas`, `numpy`

---

## Domain Feature Engineering (`ShopFeatureEngineer`)
Raw session metrics are processed through customized transformations designed to highlight purchasing intent:
1. **Page Value Thresholding & Non-Linear Mapping**:
   * `Val_is_zero`: Binary indicator flagging whether Google Analytics recorded non-zero revenue potential.
   * `Val_log1p`: Log-transformed page value ($\log(1 + \text{PgVal})$) to compress exponential value outliers.
2. **Bounce vs. Exit Dynamics**:
   * `Bnc_to_Ext_ratio_log`: Log ratio of bounce rate to exit rate ($\log(1 + \frac{\text{BncRt}}{\text{ExtRt} + \epsilon})$).
   * `Ext_minus_Bnc`: Direct difference highlighting pages where users intentionally leave versus immediate single-page drop-offs.
3. **Session Engagement & Pace**:
   * Average duration per page visited (`Avg_Prd_Dur`, `Avg_Adm_Dur`, `Avg_Inf_Dur`).
   * Total accumulated session duration and total pages visited across Administrative, Informational, and Product categories.
4. **Special Day & Skew Mitigation**:
   * Binary flags for special holiday proximity (`SpclDay_B`) and logarithmic compression of highly skewed page counts and dwell times.

---

## Model Architecture: Wide & Deep Neural Network

```
                   Input Features
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   [Wide Inputs]                   [Deep Inputs]
  Explicit Crosses            Continuous     Categoricals
 (Mo×VstTp, Wkd×VstTp)         Features       (Mo, VstTp...)
         │                        │                │
         │                        │          Entity Embeddings
         │                        │                │
         │                        └───────┬────────┘
         │                                ▼
         │                           Dense MLP
         │                       (BatchNorm, Dropout)
         │                                │
         └───────────────┬────────────────┘
                         ▼
                  Concat Layer
                         │
                 Output (Sigmoid)
```

1. **Wide Linear Branch (Memorization)**:
   * Explicit sparse interaction crosses: `Month × VisitorType`, `Weekend × VisitorType`, and `Month × Weekend`.
   * Directly learns linear weights for specific event combinations, enabling memorization of high-converting niche cohorts.
2. **Deep MLP Branch (Generalization)**:
   * **Categorical Entity Embeddings**: Distinct trainable embedding layers learn dense, continuous semantic representations for high-cardinality factors.
   * **Continuous Pipeline**: Scaled continuous features concatenated with embeddings pass through fully-connected layers with Batch Normalization and Dropout regularization to prevent co-adaptation.
3. **Joint Training**:
   * The wide and deep representations are fused and mapped to a calibrated sigmoid probability output trained end-to-end via binary cross-entropy.

---

## Experimental Validation & Results
* **Validation Strategy**: 5-Seed Stratified Train/Validation splits (`train_size = 0.8`).
* **Performance**:
  * Consistently achieves **~94.5%+ holdout accuracy** with strong F1-score balance, effectively capturing conversion probability while managing severe class imbalance.

---

## Files
* [5_Shop_final.ipynb](5_Shop_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
