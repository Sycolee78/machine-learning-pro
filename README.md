# Tank Failure Pressure Prediction

This project predicts **Target Pressure (bar)** (tank failure pressure) from engineering and process features using supervised machine learning.

It includes:
- an end-to-end notebook workflow (data prep, training, evaluation, visualization)
- a reproducible Python script for benchmark evaluation

---

## Project objective
Given tank-related input features, train regression models that estimate the pressure at which the tank fails.

The target variable is:
- `Target Pressure (bar)`

---

## Repository structure
- `train.csv` — training dataset
- `prediction.csv` — saved predictions (generated output)
- `Buruvuru_Emmanuel_Machine_Learning_Assignment.ipynb` — main notebook
- `Buruvuru_Emmanuel_Machine_Learning_Assignment1.ipynb` — alternate notebook version
- `model_evaluation.py` — script-based model training and evaluation

---

## How the pipeline works

### 1) Load data
- Read `train.csv`
- Remove non-informative ID column when present

### 2) Preprocess
- Handle missing values
  - numeric: median imputation
  - categorical: most-frequent imputation
- Encode categorical variables (one-hot encoding)
- Scale features for models that need it (e.g., SVR)

### 3) Split data
- Train/test split: 80/20 (`random_state=42`)

### 4) Train models
Current baseline models include:
- Linear Regression
- Support Vector Regression (RBF)
- Random Forest Regressor
- Gradient Boosting Regressor

### 5) Evaluate
Metrics used:
- **R²** (goodness of fit)
- **MAE** (mean absolute error)
- **MAPE** (mean absolute percentage error)

The script also runs **5-fold cross-validation** for Random Forest.

---

## Quick start

### Requirements
- Python 3.9+

### Install dependencies
```bash
python3 -m pip install pandas scikit-learn
```

### Run evaluation
```bash
python3 model_evaluation.py
```

This prints model performance on:
- holdout test set (80/20 split)
- 5-fold CV summary for Random Forest

---

## Using the notebook
Open either notebook and run cells top-to-bottom.

```bash
jupyter notebook
```

In the notebook, you can:
- inspect preprocessing steps
- train and compare models interactively
- generate plots for feature relationships and model behavior

---

## Typical output
Example metrics from the script:
- LinearRegression: R²=0.4217, MAE=0.2115, MAPE=1.1475
- SVR_rbf: R²=0.8070, MAE=0.0949, MAPE=0.4796
- RandomForest: R²=0.8476, MAE=0.0709, MAPE=0.1936
- GradientBoosting: R²=0.6726, MAE=0.1256, MAPE=0.4756
- RandomForest 5-fold CV: R² mean=0.8805, MAE=0.0672, MAPE=0.1963

---

## Notes for users
- This is a regression project (continuous target), not classification.
- If you add new models, keep evaluation consistent (same split/CV and metrics).
- For deployment, save preprocessing + model as one pipeline object.
