# Deep Neural Networks & Applied Machine Learning Portfolio

A comprehensive collection of end-to-end machine learning and deep learning lab projects. Spanning classical machine learning, domain-specific feature engineering, hybrid Wide & Deep tabular architectures, unsupervised representation learning, computer vision, and multi-scale temporal sequence forecasting.

---

## Repository Architecture & Lab Index

Each lab is self-contained within its own directory, featuring a dedicated `README.md` detailing problem formulations, engineering innovations, and empirical results, along with a standalone research notebook.

| Lab | Project Directory | Core Notebook | Domain & Problem Type | Key Architecture & Techniques | Primary Metric / Result |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | [1-diamond](1-diamond/README.md) | [diamond_v2.ipynb](1-diamond/diamond_v2.ipynb) | **Tabular Regression**<br>Diamond Valuation | Quantile-binned `SegmentedLinearRegressor` along facet ratio ($x/\text{table}$) with global fallback and polynomial features. | Lowest MAE with localized linear bounds |
| **02** | [2-stellar](2-stellar/README.md) | [stellar_final.ipynb](2-stellar/stellar_final.ipynb) | **Multiclass Classification**<br>Astronomical Spectra | Redshift-partitioned Support Vector Classifier (`SVC`), Color-Redshift Interactions (CRI), and Normalized Flux Ratios (NFR). | **~0.97 Accuracy & F1**<br>(drastic $\mathcal{O}(N^2)$ speedup) |
| **03** | [3-Phonemes](3-Phonemes/README.md) | [Phonemes_final.ipynb](3-Phonemes/Phonemes_final.ipynb) | **Acoustic Classification**<br>Speech Phonemes | `StratifiedGroupKFold` split on `SpkrID`, Cepstral Mean Subtraction (`CMSTransformer`), and blended ensemble (LDA, XGBoost, SVC, RF). | Leak-free speaker generalization |
| **05** | [5-Shop-NN](5-Shop-NN/README.md) | [5_Shop_final.ipynb](5-Shop-NN/5_Shop_final.ipynb) | **Binary Classification**<br>E-Commerce Conversion | Wide & Deep Neural Network (`build_wide_and_deep`), sparse categorical cross memorization, dense entity embeddings, engagement dynamics. | **~94.5%+ Accuracy**<br>Balanced Precision/Recall |
| **06** | [6-Bacteria](6-Bacteria/README.md) | [6_Bacteria_v4.ipynb](6-Bacteria/6_Bacteria_v4.ipynb) | **Genomic Classification**<br>Bacterial 10-mer DNA | Deep MLP with Input Batch Normalization, wide skip connections, and unsupervised Gaussian Mixture Model (GMM) posterior features. | **99.47% Accuracy**<br>**0.9877 Macro F1** |
| **07** | [7-gene](7-gene/README.md) | [7_Genomics_final.ipynb](7-gene/7_Genomics_final.ipynb) | **Genomic Classification**<br>Complex Genetic Sequences | K-mer extraction (dinucleotides & trinucleotides), boundary overlap decomposition, and multi-branch late-fusion neural network. | **0.8940 CV** on difficult overlapping boundary |
| **08** | [8-radon](8-radon/README.md) | [8_Radon_final.ipynb](8-radon/8_Radon_final.ipynb) | **Spatial Regression**<br>Indoor Environmental Radon | Domain feature engineering (log transforms, inverse measurement error weighting, spatial $(x, y)$ coordinates), Deep Keras Regressor. | Minimized Out-of-Fold MSE |
| **09** | [9-VenusMars](9-VenusMars/README.md) | [9_VenusMars_final.ipynb](9-VenusMars/9_VenusMars_final.ipynb) | **Computer Vision**<br>Facial Classification (Male vs. Female) | Multi-Scale CNN (decreasing receptive fields $5\times5 \to 2\times2$) concatenated with pretrained MobileNetV2 + Test-Time Augmentation (TTA). | **94.40% Mean Accuracy**<br>($\pm 0.0025$ on 4-Fold CV) |
| **10** | [10-Crypto](10-Crypto/README.md) | [10Crypto_final.ipynb](10-Crypto/10Crypto_final.ipynb) | **Sequence Forecasting**<br>Continuous 24,431-Step Crypto Returns | Cross-Resolution Multi-Scale Ensemble ($K=100$ macro + $K=50$ intraday), 1D-CNN + BiGRU, PCHIP cubic splines, differentiable Pearson loss. | **0.6783 Mean CV PCC**<br>|

> *Note: Lab 4 is intentionally omitted in this curriculum sequence.*

---

## Technical Progression Roadmap

The projects follow a deliberate architectural progression from classical feature engineering to state-of-the-art deep architectures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Classical Machine Learning & Domain-Informed Feature Engineering        │
│    • Lab 1: Geometric ratios, polynomial interaction, segmented regressors  │
│    • Lab 2: Astrophysics flux ratios, redshift partitioning, SVM scaling    │
│    • Lab 3: Speaker leakage prevention, CMSTransformer, multi-model ensemble│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Deep Tabular & Biological Representation Learning                       │
│    • Lab 5: Google Wide & Deep networks, feature crosses, entity embeddings │
│    • Lab 6: Deep MLPs, input batch normalization, unsupervised GMM priors   │
│    • Lab 7: K-mer tokenization, overlap isolation, multi-branch late fusion │
│    • Lab 8: Heteroskedastic error weighting, spatial coordinate regression   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Spatial & Temporal Deep Learning (Vision & Sequence Forecasting)         │
│    • Lab 9: Multi-scale CNNs, transfer learning, test-time augmentation     │
│    • Lab 10: 1D-CNN + BiGRU, multi-resolution resampling, PCHIP splines,    │
│             differentiable sequence Pearson correlation loss                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure & Data

* Datasets are omitted due to licensing and copyright restrictions.
* Each lab folder contains a self-contained `README.md` explaining the methodology and results, alongside the final reproducible research notebook (`*.ipynb`).

---

## Tech Stack & Dependencies

* **Language**: Python 3
* **Deep Learning Frameworks**: TensorFlow / Keras
* **Machine Learning & Scientific Computing**: Scikit-Learn, XGBoost, SciPy, NumPy, Pandas
* **Data Visualization**: Matplotlib, Seaborn

---

## Notes & Curriculum Context
Developed as part of a Deep Neural Network and Applied Machine Learning curriculum aligned with concepts from Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
