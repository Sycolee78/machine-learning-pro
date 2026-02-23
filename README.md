# Predicting Tank Failure Pressure (Model Audit + Improvements)

## What I evaluated
I reviewed the notebook workflow and tested the dataset with a corrected validation setup.

Repository audited: `Sycolee78/machine-learning-pro`

---

## Key finding (important)
The current notebook has a **data leakage issue** in `split_data()`:

- `X_train` and `X_test` are both assigned from the same dataset
- `y_train` and `y_test` are also the same

That means the model is effectively being tested on training data, so reported metrics are over-optimistic and not trustworthy for real-world performance.

---

## Re-evaluation with proper split
I added a standalone script (`model_evaluation.py`) that uses:

- Proper 80/20 train-test split
- Median/most-frequent imputation
- One-hot encoding for categorical columns
- Scaling for SVR
- Holdout metrics + 5-fold cross-validation

### Holdout results (80/20)
- **LinearRegression**: R2=0.4217, MAE=0.2115, MAPE=1.1475
- **SVR (RBF)**: R2=0.8070, MAE=0.0949, MAPE=0.4796
- **RandomForest**: R2=0.8476, MAE=0.0709, MAPE=0.1936
- **GradientBoosting**: R2=0.6726, MAE=0.1256, MAPE=0.4756

### 5-fold CV (RandomForest)
- **R2 mean = 0.8805** (std 0.0191)
- **MAE = 0.0672**
- **MAPE = 0.1963**

### Conclusion
The project can perform well, but **only after fixing the validation approach**. With corrected evaluation, RandomForest is currently the strongest model among tested baselines.

---

## Additions made
- Added: `model_evaluation.py`
  - Reproducible evaluation pipeline
  - Correct split logic
  - Cross-validation summary

---

## How to run
```bash
python3 -m pip install pandas scikit-learn
python3 model_evaluation.py
```

---

## Recommended next improvements
1. Fix `split_data()` in notebook to use a real train-test split.
2. Remove `SMOTE` for regression target (SMOTE is mainly classification-focused; current logic is not ideal for continuous targets).
3. Add residual plots and error-by-feature diagnostics.
4. Try XGBoost/LightGBM/CatBoost with proper CV and early stopping.
5. Save best model + preprocessing pipeline (`joblib`) for deployment consistency.

---

## Notes
This update is intentionally focused on evaluation validity and reliability of model quality claims.