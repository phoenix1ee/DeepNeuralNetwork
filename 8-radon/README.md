# Lab 8: Deep Neural Regression with Spatial-Environmental Feature Engineering for Indoor Radon Prediction

## Overview
This project predicts indoor radon gas concentrations (`Y`) from environmental, geological, spatial, and measurement error attributes. Radon prediction involves steep non-linear relationships, heavy-tailed measurement distributions, and spatial correlations. To handle these dynamics, this lab combines **custom domain-specific feature engineering** (log transforms, error indicators, inverse error weighting, and geographic coordinate modeling) with a **deep multilayer perceptron regressor** in TensorFlow/Keras.

* **Primary Notebook**: [8_Radon_final.ipynb](8_Radon_final.ipynb)
* **Task**: Regression (Continuous Environmental Radon Concentration)
* **Target Variable**: `Y` (Radon level)
* **Primary Evaluation Metric**: Mean Squared Error (MSE)
* **Core Framework**: `tensorflow.keras`, `scikit-learn`, `numpy`, `pandas`

---

## Domain Feature Engineering (`RadonFeatureEngineering`)
Geological and measurement characteristics exhibit significant skewness and heteroskedasticity:
1. **Measurement Error Decomposition**:
   * `log_pcterr`: Logarithmic transformation of measurement percentage error ($\log(1 + \text{pcterr})$).
   * `pcterr_zero`: Binary indicator capturing unperturbed readings where measurement error is exactly zero.
   * `inv_pcterr`: Inverse error weighting ($\frac{1}{\text{pcterr} + \epsilon}$), providing high-confidence weighting to precise measurements.
2. **Geological & Demographic Log Scaling**:
   * `log_Uppm`: Log-transformed county-level uranium concentration, reflecting the exponential decay and distribution of radioactive elements in soil.
   * `log_adjwt`: Log-transformed sample adjustment weight, normalizing sample representation.
3. **Spatial Coordinate Modeling**:
   * Direct utilization of spatial coordinates $(x, y)$ allows the neural network to approximate geographic continuous variation in regional soil permeability and bedrock geology.

---

## Model Architecture: Deep Neural Regressor
The continuous regression model is built as an end-to-end Keras Multilayer Perceptron:
* **Feature Processing**: Combined numerical and engineered pipelines pass through standard scaling (`StandardScaler`) to ensure well-conditioned gradient propagation.
* **Network Structure**:
  * Dense layers with non-linear activations (ReLU / ELU) paired with **Batch Normalization** to stabilize training dynamics under skewed geological inputs.
  * Dropout regularization to prevent the network from memorizing individual geographic cluster coordinates.
* **Loss & Optimization**:
  * Trained using Adam optimizer with Mean Squared Error (MSE) loss, with learning rate scheduling and early stopping.

---

## Validation Strategy
* Evaluated via 5-Fold Cross-Validation (`StratifiedKFold` / `KFold`), evaluating out-of-fold MSE to ensure spatial generalization across unseen residential dwellings.

---

## Files
* [8_Radon_final.ipynb](8_Radon_final.ipynb): Core reproducible research notebook
* [README.md](README.md): Lab technical documentation
