# Lab 3: Group-Aware Acoustic Phoneme Classification & Multi-Model Ensembling

## Overview
This project addresses multi-class phoneme recognition from acoustic speech signals. Speech recognition datasets present a classic machine learning vulnerability: multiple acoustic frames originate from the same speaker, introducing latent vocal tract characteristics that cause naive cross-validation to leak identity and overestimate generalization. This lab implements a **group-aware validation strategy**, **speaker normalization preprocessing (Cepstral Mean Subtraction)**, and an **ensemble combining four distinct classifier families**.

* **Primary Notebook**: [Phonemes_final.ipynb](Phonemes_final.ipynb)
* **Task**: Multiclass Classification (5 Phoneme Classes)
* **Target Variable**: Phoneme Label `g`
* **Grouping Column**: `SpkrID` (Speaker Identifier)
* **Primary Evaluation Metrics**: Accuracy, Macro F1-Score
* **Core Framework**: `scikit-learn`, `xgboost`, `numpy`, `pandas`

---

## Leakage Prevention & Preprocessing

### 1. Speaker-Aware Split (`StratifiedGroupKFold`)
* Splitting speech data randomly across rows places frames from the same speaker in both training and test folds. The model learns speaker identification rather than phoneme acoustics.
* We employ **`StratifiedGroupKFold`** with `groups = df['SpkrID']`, ensuring:
  1. Zero speaker overlap between training and holdout validation folds.
  2. Balanced preservation of phoneme class proportions across folds.

### 2. Cepstral Mean Subtraction (`CMSTransformer`)
* To strip individual speaker vocal tract coloration from acoustic features, we apply per-speaker group normalization:
  $$\tilde{x}_i = x_i - \mu_{\text{speaker}}$$
  with optional variance scaling:
  $$\tilde{x}_i = \frac{x_i - \mu_{\text{speaker}}}{\sigma_{\text{speaker}} + \epsilon}$$
* This normalizes spectral envelopes and forces classifiers to rely strictly on phonetic formant variations.

---

## Multi-Model Ensembling Architecture
Rather than relying on a single model family, four fundamentally distinct inductive biases are trained and blended:

| Estimator | Inductive Bias & Role | Preprocessing Pipeline |
| :--- | :--- | :--- |
| **Linear Discriminant Analysis (LDA)** | Linear decision boundaries under shared Gaussian covariance assumptions | `CMSTransformer` + `StandardScaler` |
| **XGBClassifier (XGBoost)** | Non-linear axis-aligned gradient boosted decision trees | Label encoding + `CMSTransformer` |
| **Support Vector Classifier (SVC)** | Maximum-margin classification in kernel Hilbert space (RBF kernel) | `CMSTransformer` + `StandardScaler` |
| **Random Forest Classifier (RF)** | High-entropy bagging of decorrelated decision trees | `CMSTransformer` |

### Strategic Subsampling & Blending:
* Base pipelines are trained on diverse stratified subsets with varying random seeds.
* Inference aggregates soft-voting class probabilities across the ensemble:
  $$P(\hat{y} = c \mid x) = \frac{1}{M} \sum_{m=1}^M P_m(y = c \mid x)$$
* Ensembling dampens individual estimator variance and reconciles boundary disagreements on ambiguous phoneme transitions.

---

## Files
* [Phonemes_final.ipynb](Phonemes_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
