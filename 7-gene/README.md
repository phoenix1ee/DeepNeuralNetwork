# Lab 7: Multi-Branch Late-Fusion Neural Networks for Genomic Sequence Analysis

## Overview
This project focuses on sequence-level genomic classification across complex genetic data characterized by high-dimensional nucleotide patterns and ambiguous decision boundaries. The solution features an explicit **overlap-separation methodology**: isolating overlapping, difficult sequences and resolving them through **unsupervised Gaussian Mixture Model (GMM) clustering**, **k-mer frequency engineering (dinucleotides & trinucleotides)**, and a **multi-branch late-fusion neural network**.

* **Primary Notebook**: [7_Genomics_final.ipynb](7_Genomics_final.ipynb)
* **Task**: Genomic Sequence Classification
* **Target Variable**: Class Label `Y`
* **Primary Evaluation Metric**: Classification Accuracy / Cross-Validation Score
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `numpy`, `pandas`

---

## Genomic Feature Extraction & Overlap Decomposition

### 1. K-mer Frequency & Nucleotide Proportions
* **Nucleotide Proportions (`NucleotideCountProportion`)**: Extracts fundamental biological composition ratios: $A_{\text{ratio}}, C_{\text{ratio}}, G_{\text{ratio}}, T_{\text{ratio}}$, alongside aggregate GC-content ($\text{GC}_{\text{prop}} = G_{\text{ratio}} + C_{\text{ratio}}$).
* **High-Order K-mer Tokenization**: Enumerates all 16 dinucleotide pairs (`AA`, `AC`, ..., `TT`) and 64 trinucleotide triplets (`AAA` through `TTT`) using sliding window counters across each genetic sequence.

### 2. Overlap Region Decomposition
* Exploratory sequence analysis revealed that a substantial portion of the dataset is cleanly separable, while a specific "overlap portion" exhibits high class ambiguity and overlapping spectral densities.
* The pipeline segments the workflow to isolate and evaluate this difficult overlap subspace specifically.

---

## Architecture: Multi-Branch Late-Fusion Neural Network

```
          Raw K-mers + Proportions                 Selected Dinucleotide Pairs
                     │                                         │
                     │                                  GMM Clustering
                     │                              (Posterior Probabilities)
                     │                                         │
                     ▼                                         ▼
              Branch 1: Dense                           Branch 2: Dense
          (K-mer Representation)                      (GMM Latent Clusters)
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         ▼
                                Concatenation Layer
                                         │
                             Polynomial Feature Mixing
                                         │
                                Dense Fusion Layers
                                         │
                                Output Classification
```

1. **Unsupervised GMM Representation**:
   * Evaluates unsupervised clustering of Gaussian Mixture Models on a discriminative subset of dinucleotide pairs (`selection = ['AA', 'AT', 'CG', 'TT', 'TA', 'CC']`).
   * GMM is fitted strictly on training folds to prevent data leakage, generating posterior cluster probabilities.
2. **Multi-Branch Fusion (`build_NN_model_wide`)**:
   * **Primary Branch**: Processes high-dimensional k-mer counts and base proportions.
   * **Auxiliary Branch**: Ingests GMM posterior probabilities and polynomial interaction features.
   * **Late Fusion**: Concatenates both representations before final classification layers, combining global sequence statistics with localized cluster boundaries.

---

## Experimental Results
* **5-Seed Cross-Validation Protocol** (`seeds = [2, 12, 22, 32, 42]`, 80/20 train/val splits):
  * Baseline NN on raw features: ~0.885 CV
  * GMM Feature Augmentation + NN: **0.8935 CV**
  * Late-Fusion Multi-Branch NN with Polynomial Interactions: **0.8940 CV**

---

## Files
* [7_Genomics_final.ipynb](7_Genomics_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
