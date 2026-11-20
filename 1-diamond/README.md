# Lab 1: Segmented Linear Regression for Diamond Valuation

## Overview
This project tackles high-precision diamond price valuation using tabular gemological and geometric attributes. Rather than relying on a monolithic global linear model or black-box regressors, this lab designs a **Segmented Linear Regressor** that divides the complex, non-linear feature space into localized physical regimes, fitting tailored estimators per segment while guaranteeing fallback stability and physical boundaries.

* **Primary Notebook**: [diamond_v2.ipynb](diamond_v2.ipynb)
* **Task**: Regression (Continuous Price Prediction)
* **Target Variable**: `price` (USD)
* **Primary Evaluation Metric**: Mean Absolute Error (MAE)
* **Core Framework**: `scikit-learn`, `numpy`, `pandas`

---

## Dataset & Features
The dataset contains physical and optical attributes of diamonds:
* **Continuous Geometric Features**: `carat` (weight), `depth` (total depth percentage), `table` (width of top facet relative to widest point), and spatial bounding dimensions `x` (length in mm), `y` (width in mm), and `z` (depth in mm).
* **Categorical Quality Grades**:
  * `cut`: Quality grading (Fair, Good, Very Good, Premium, Ideal).
  * `color`: Diamond color scale from J (worst) to D (best).
  * `clarity`: Inclusions assessment (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF).

---

## Preprocessing & Feature Engineering
1. **Geometric Interaction Features (`FeatureCreator`)**:
   * `x_table`: Ratio of length to table width ($x / \text{table}$), capturing proportion anomalies and facet geometry.
   * `girdle_ratio`: Ratio of length to width ($x / y$), identifying stone symmetry deviations.
   * `log_table`: Log-transformed facet width ($\log(1 + \text{table})$) to stabilize variance.
2. **Missing & Zero Imputation Pipelines**:
   * Diamonds with zero or missing measurements in $x, y, z$ represent physical sensor errors or missing entries; these are handled via multi-stage `SimpleImputer` using median statistics.
   * Categorical missing values are imputed via `most_frequent` strategy.
3. **Outlier Filtering**:
   * Quantile filtering between 0.001 and 0.999 across continuous dimensions cleans unphysical measurement anomalies while retaining natural extremes.
4. **Polynomial & Categorical Encoding**:
   * `OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')` for ordinal/nominal quality grades.
   * Second-degree polynomial feature expansion (`PolynomialFeatures(degree=2, include_bias=False)`) followed by `StandardScaler` to capture non-linear carat-dimension interactions.

---

## Model Architecture: Segmented Linear Regressor
Standard linear regression fails to capture the steep super-linear price scaling of diamonds across varying sizes and cut proportions. The custom `SegmentedLinearRegressor` solves this via:
* **Quantile Binning**: Automatically splits the feature space along `x_table` into 24 distinct quantile bins.
* **Localized Estimators**: Fits an independent `LinearRegression` pipeline with degree-2 polynomial interactions within each bin that meets the minimum sample threshold (`min_samples_per_bin = 500`).
* **Global Fallback**: Samples falling into sparsely populated bins fall back to a robust global polynomial pipeline.
* **Physical Post-Processing**: Bounded clamping prevents negative price predictions ($\hat{y} \ge \$10.0$).

---

## Validation Strategy & Results
* **Cross-Validation**: 10-Fold Cross-Validation (`KFold(n_splits=10, shuffle=True, random_state=1)`).
* **Key Finding**: Segmenting by facet proportion ($x / \text{table}$) allows simple linear models with polynomial features to achieve high local fidelity without over-fitting extreme market segments.

---

## Files
* [diamond_v2.ipynb](diamond_v2.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
