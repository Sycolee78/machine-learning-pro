import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


def build_preprocessors(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    pre_common = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    pre_scaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    return pre_common, pre_scaled


def main():
    data_path = "train.csv"
    target_col = "Target Pressure (bar)"

    df = pd.read_csv(data_path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    pre_common, pre_scaled = build_preprocessors(X)

    models = {
        "LinearRegression": (pre_common, LinearRegression()),
        "SVR_rbf": (pre_scaled, SVR(C=10, epsilon=0.1)),
        "RandomForest": (
            pre_common,
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        ),
        "GradientBoosting": (pre_common, GradientBoostingRegressor(random_state=42)),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Data shape: {df.shape}")
    print("\nHold-out evaluation (80/20 split):")
    for name, (pre, model) in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(
            f"{name:18s} | "
            f"R2={r2_score(y_test, pred):.4f} "
            f"MAE={mean_absolute_error(y_test, pred):.4f} "
            f"MAPE={mean_absolute_percentage_error(y_test, pred):.4f}"
        )

    print("\n5-fold CV (RandomForest):")
    best = Pipeline(
        [
            ("pre", pre_common),
            (
                "model",
                RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
            ),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        best,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        },
        n_jobs=-1,
    )

    print(
        "RandomForest CV | "
        f"R2_mean={scores['test_r2'].mean():.4f} "
        f"R2_std={scores['test_r2'].std():.4f} "
        f"MAE={-scores['test_mae'].mean():.4f} "
        f"MAPE={-scores['test_mape'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
