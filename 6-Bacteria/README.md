# Lab 6: Deep Neural Networks with Unsupervised GMM Representation Learning for Bacterial Classification

## Overview
This project classifies bacterial species from genomic DNA sequence signatures represented as 10-mer nucleotide frequencies (286 features corresponding to counts of 10-base DNA sub-sequences, e.g., `A0T4G2C4`). The solution combines **deep neural networks with input Batch Normalization and wide skip connections** with **unsupervised representation learning via Gaussian Mixture Models (GMM)**, achieving near-perfect classification performance across 5 bacterial classes.

* **Primary Notebook**: [6_Bacteria_v4.ipynb](6_Bacteria_v4.ipynb)
* **Task**: Multiclass Classification (5 Bacterial Species)
* **Input Features**: 286 genomic 10-mer frequency counts
* **Primary Evaluation Metrics**: Accuracy, Macro F1-Score
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `numpy`, `pandas`

---

## Unsupervised Representation Learning (GMM Feature Extraction)
Before training the deep classifier, we analyze the structure of the genomic feature space:
1. **Discriminative K-mer Selection**:
   * A targeted subset of highly discriminative k-mers (`A0T9G1C0`, `A1T4G4C1`, `A2T2G3C3`, `A3T2G0C5`, `A4T4G1C1`, `A5T4G0C1`, `A4T1G4C1`) is identified through exploratory variance and separation analysis.
2. **Log-Ratio & L1 Preprocessing**:
   * Features pass through logarithmic transformation ($\log(1 + x)$), $\ell_1$-normalization, and standard scaling.
3. **Gaussian Mixture Modeling**:
   * A GMM with 5 components and **tied covariance matrix** is fitted on the preprocessed training subset:
     $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma)$$
   * The posterior component probabilities $P(z = k \mid x)$ serve as dense unsupervised representations that capture latent biological sub-clusters and phylogenetic similarity.

---

## Neural Network Architecture
The classification engine employs a tailored multi-stage deep MLP:
* **Input Batch Normalization**: Normalizes raw k-mer count distributions directly at the network's input layer without dropping physical extremes.
* **Deep Layer Hierarchy**: Cascading Dense layers with ReLU activations:
  $$\text{Input (286)} \to \text{BN} \to 256 \to 256 \to 128 \to 64 \to 32 \to 10 \text{ (Softmax)}$$
* **Wide Skip Concatenation**: A wide feature bypass injects earlier high-capacity representations into later dense layers (after layer 128), preventing vanishing gradients and preserving fine-grained genomic signals.
* **Custom Keras Metrics**: Real-time evaluation tracking via a stateful `MulticlassF1Score` metric that computes running confusion matrices per epoch.

---

## Experimental Benchmarks & Progression

Across a rigorous 5-seed stratified cross-validation protocol (`seeds = [2, 12, 22, 32, 42]`):

| Architecture & Optimization | Epochs / BS | Mean Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| Baseline MLP (256, 128, 64, 32) + Input BN | 35 / 1024 | 0.9938 ± 0.0002 | 0.9856 ± 0.0006 |
| MLP (256, 196, 128, 32) + Input BN | 35 / 1024 | 0.9940 ± 0.0006 | 0.9859 ± 0.0017 |
| Deep MLP (256, 256, 128, 64, 32) | 35 / 1024 | 0.9945 ± 0.0012 | 0.9874 ± 0.0028 |
| **Wide-Skip MLP + GMM Posteriors (Best)** | **35 / 1024** | **0.9947 ± 0.0009** | **0.9877 ± 0.0020** |

---

## Files
* [6_Bacteria_v4.ipynb](6_Bacteria_v4.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
