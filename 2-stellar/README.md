# Lab 2: Segmented Support Vector Classification for Astronomical Objects

## Overview
This project classifies astronomical celestial observations into three distinct cosmological categories: **Galaxy (`GALAXY`)**, **Star (`STAR`)**, and **Quasar (`QSO`)**. Drawing on data from the Sloan Digital Sky Survey (SDSS), the pipeline incorporates domain-guided astrophysical feature engineering (color differences, flux ratios, and angular object size) and implements a **Segmented Support Vector Classifier** partitioned along the spectroscopic redshift dimension.

* **Primary Notebook**: [stellar_final.ipynb](stellar_final.ipynb)
* **Task**: Multiclass Classification (3 Classes: GALAXY, STAR, QSO)
* **Target Variable**: Celestial Object Class
* **Primary Evaluation Metrics**: Accuracy, Macro F1-Score
* **Core Framework**: `scikit-learn`, `numpy`, `pandas`

---

## Astrophysical Feature Engineering
Exploratory data analysis (EDA) revealed that celestial bodies follow starkly distinct photometric energy distributions across wavelength bands and distance scales:
1. **Photometric Color Differences (`FeatureCreator`)**:
   * Measures spectral flux slopes via inter-band magnitude differences: $u-g, g-r, r-i, i-z$.
2. **Color-Redshift Interaction (CRI, `ColorRedshiftInteractionAdder`)**:
   * Multiplies color indices with spectroscopic redshift: $\text{CRI} = (r - i) \times \text{redshift}$, capturing how cosmological expansion red-shifts emission spectra.
3. **Normalized Flux Ratios (NFR, `NormalizedFluxRatioAdder`)**:
   * Converts magnitude differences back into linear flux ratios using Pogson's relation:
     $$\text{NFR} = 10^{-0.4 (m_1 - m_2)}$$
4. **Composite CRI $\times$ NFR (`CRINFRComposer`)**:
   * Captures higher-order interactions between non-linear flux ratios and distance indicators.
5. **Angular Object Size Estimation (`ObjectSizeAdder`)**:
   * Estimates angular spread and physical footprint using right ascension (RA) and declination (Dec).

---

## Model Architecture: Segmented Support Vector Machine
Standard Support Vector Classifiers (SVC) equipped with Radial Basis Function (RBF) kernels offer exceptional decision boundaries for complex physical domains, but suffer from super-linear computational scaling ($\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$) and struggle when global decision boundaries shift across distance regimes.

* **Segmented Architecture (`Segmentedclassifer`)**:
  * Segments the dataset along the spectroscopic `redshift` coordinate into discrete astronomical distance bins.
  * Trains independent, hyperparameter-tuned RBF SVC estimators for each redshift interval.
  * Dispatches unseen samples to their corresponding regional sub-model during inference, with a global estimator fallback.
* **Benefits**:
  * **Efficiency**: Sub-dividing the training data into smaller slices drastically reduces kernel matrix computations and training wall-clock time.
  * **Accuracy**: Allows the boundary between quasars (predominantly high redshift) and stars/galaxies (low-to-moderate redshift) to adapt locally without cross-regime interference.

---

## Experimental Results
* **Baseline Comparison**:
  * Baseline Logistic Regression & LDA/QDA: 0.93 – 0.94 accuracy.
  * Global SVC (unsegmented): 0.95 accuracy.
  * Hyperparameter-tuned Global RBF SVC ($C, \gamma$): ~0.96 accuracy.
* **Segmented SVC with Domain Features**:
  * **Accuracy**: **~0.97**
  * **Macro F1-Score**: **~0.97**
  * Substantial reduction in total training latency compared to monolithic SVC.

---

## Files
* [stellar_final.ipynb](stellar_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
