# Lab 10: Cross-Resolution Multi-Scale Deep Sequence Ensemble for Cryptocurrency Forecasting

## Overview
This project presents an advanced financial time-series forecasting pipeline predicting **24,431 consecutive 1-minute Close prices (~17 days)** from 500,000 historical intervals (~347 days). Financial prices follow integrated non-stationary random walks ($I(1)$); direct long-horizon point autoregression collapses into flat lines or suffers catastrophic compounding error. This pipeline deploys a **Cross-Resolution Multi-Scale Ensemble** combining 1D-CNN + Bidirectional GRU neural architectures operating across complementary temporal resolutions ($K = 100$ macro and $K = 50$ intraday knots), continuous **PCHIP monotonic cubic splines**, and a **differentiable sequence Pearson correlation loss**.

* **Primary Notebook**: [10Crypto_final.ipynb](10Crypto_final.ipynb)
* **Task**: Multi-Step Sequence Time-Series Forecasting (24,431 continuous 1-minute steps)
* **Primary Evaluation Metric**: Sequence-Level Pearson Correlation Coefficient (PCC)
* **Dataset**: 500,000 1-minute OHLCV intervals (`tXY.csv`)
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `scipy`, `pandas`, `numpy`

---

## Core Engineering Innovations

### 1. Multi-Resolution Macro Resampling & Stationary Transformations
* Directly training recurrent networks over 24,431 steps induces severe vanishing/exploding gradients. We downsample raw 1-minute data into macro-bars:
  * **Macro-Scale ($K_1 = 100$ min)**: Compresses the 24,431-step test horizon into **245 macro knots**.
  * **Sub-Macro Scale ($K_2 = 50$ min)**: Compresses the horizon into **489 knots** (2× higher knot density).
* **Stationary Feature Set**: Log-returns ($\Delta \log P_t$), rolling normalized volatility, volume log-ratios ($\log(1 + V_t / \bar{V}_t)$), RSI, and fast/slow exponential momentum spans eliminate non-stationary drift.

### 2. Dual-Scale Neural Architectures (1D-CNN + BiGRU)

| Feature | Model 1: Macro ($K = 100$) | Model 2: Sub-Macro ($K = 50$) |
| :--- | :--- | :--- |
| **Lookback Window** | 200 bars (~13.9 days) | 400 bars (~13.9 days) |
| **Output Density** | 245 macro forecast knots | 489 forecast knots |
| **Neural Backbone** | `Conv1D(64)` + `BiGRU(64)` + `Dense(96)` | `Conv1D(64)` + `BiGRU(64)` + `Dense(128)` |
| **Specialization** | Ultra-smooth macro cycle orientation | Intraday inflection & turning point detection |

* **1D-CNN Layer**: Filters local micro-structure noise and extracts local temporal patterns.
* **Bidirectional GRU**: Captures long-range temporal dependencies across the 14-day lookback window.
* **Direct Multi-Horizon Head**: Fully-connected dense head projects hidden state directly to the entire forecast vector simultaneously.

### 3. Calibrated Joint Objective Function
To align training with the global trajectory shape while maintaining scale calibration:
$$\mathcal{L}(y_{\text{true}}, y_{\text{pred}}) = -\text{PCC}(y_{\text{true}}, y_{\text{pred}}) + 1.5 \cdot \text{MSE}(y_{\text{true}}, y_{\text{pred}})$$
The sequence Pearson correlation loss optimizes relative directional movement, while the MSE anchor stabilizes scale and prevents degenerate zero-variance solutions.

### 4. Continuous PCHIP Splines & Log-Space Geometric Ensembling
1. **PCHIP Cubic Splines**: Monotonic cubic spline interpolation (`scipy.interpolate.PchipInterpolator`) expands discrete knot predictions ($z^{(1)} \in \mathbb{R}^{245}$ and $z^{(2)} \in \mathbb{R}^{489}$) to continuous 24,431-minute vectors $\hat{z}_1(t)$ and $\hat{z}_2(t)$ without non-physical polynomial overshoot.
2. **Log-Trajectory Ensembling**:
   $$\hat{z}_{\text{ens}}(t) = 0.5 \cdot \hat{z}_1(t) + 0.5 \cdot \hat{z}_2(t)$$
3. **Boundary Anchorage**: Guarantees zero jump discontinuity at the boundary between past and future:
   $$\hat{z}_{\text{ens}}(t) \leftarrow \hat{z}_{\text{ens}}(t) - \hat{z}_{\text{ens}}(0)$$
4. **Exponential Price Reconstruction**: $\hat{P}_t = P_{\text{last}} \cdot \exp(\hat{z}_{\text{ens}}(t))$, guaranteeing strictly positive prices ($\hat{P}_t > 0$) as a geometric mean.

---

## 3-Fold Purged Walk-Forward Cross-Validation
Evaluated across three distinct 24,431-step sequential market regimes (~51 days total) with strict causal isolation:
* **Fold 1**: Consolidation $\to$ Breakout
* **Fold 2**: Continuation Breakout
* **Fold 3**: Mountain Rally & Retracement

### Empirical Validation Results:
* **Model 1 ($K = 100$) Mean CV**: 0.3532 ± 0.5656
* **Model 2 ($K = 50$) Mean CV**: 0.6247 ± 0.3099
* **v5 Cross-Resolution Ensemble**: **0.6783 ± 0.2089** (Fold 1: 0.8475, Fold 2: 0.4448, Fold 3: 0.7427)

> **Key Takeaway**: Blending macro ($K=100$) and intraday ($K=50$) models yields a massive diversification boost (+0.0536 PCC over the best single model and -32% reduction in variance), validating multi-resolution ensembling for financial time-series.

---

## Files
* [10Crypto_final.ipynb](10Crypto_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
