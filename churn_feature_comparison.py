"""
E-Commerce Customer Churn: Does Filtering Features by Univariate Correlation
Before Modeling Actually Help?
------------------------------------------------------------------------------
The original notebook (churn-portfolio.ipynb) hit 33% row-wise missingness
across all 20 columns, and fixed it by keeping only the 6 features most
correlated with churn individually, then dropping the remaining incomplete
rows (10.1% of data). This script tests whether that was the right call, by
comparing it against keeping all 18 usable features and imputing missing
values instead of dropping rows.

Data: E-Commerce-Dataset.xlsx, sheet "E Comm"
Source: https://github.com/anvarnarz/praktikum_datasets

Author: Dostonbek URINOV
https://github.com/DostonUr | https://www.linkedin.com/in/doston-urinov/
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_PATH = "E-Commerce-Dataset.xlsx"
ORIGINAL_FEATURES = ["Tenure", "Complain", "DaySinceLastOrder", "CashbackAmount",
                      "MaritalStatus", "Gender"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="E Comm")


def report_missingness(df: pd.DataFrame):
    full_missing = df[df.isnull().any(axis=1)].shape[0]
    print(f"Rows with ANY missing value across all 20 columns: "
          f"{full_missing} ({100*full_missing/len(df):.1f}%)")

    subset = df[["Churn"] + ORIGINAL_FEATURES]
    subset_missing = subset[subset.isnull().any(axis=1)].shape[0]
    print(f"Rows with a missing value in the 6 correlation-selected features only: "
          f"{subset_missing} ({100*subset_missing/len(subset):.1f}%)")


def evaluate(X, y, model, label, cv_folds=5):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model.fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)

    print(f"\n=== {label} ===")
    print(f"Rows used: {len(X)}")
    print(f"Test  Accuracy={accuracy_score(yte, pred):.3f}  F1={f1_score(yte, pred):.3f}  "
          f"Recall={recall_score(yte, pred):.3f}  Precision={precision_score(yte, pred):.3f}")

    cv_scores = cross_val_score(
        model, scaler.fit_transform(X), y,
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=0), scoring="f1"
    )
    print(f"5-fold CV F1: mean={cv_scores.mean():.3f} std={cv_scores.std():.3f} "
          f"range=[{cv_scores.min():.3f}, {cv_scores.max():.3f}]")
    return model


def original_approach(df: pd.DataFrame):
    data = df[["Churn"] + ORIGINAL_FEATURES].dropna()
    X = pd.get_dummies(data.drop("Churn", axis=1)).astype(float)
    y = data["Churn"]
    evaluate(X, y, RandomForestClassifier(n_estimators=9, random_state=0),
              "ORIGINAL: 6 correlation-selected features, dropna, 9-tree RF")


def improved_approach(df: pd.DataFrame):
    X = df.drop(["Churn", "CustomerID"], axis=1).copy()
    y = df["Churn"]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    X_encoded = pd.get_dummies(X, columns=cat_cols).astype(float)

    model = evaluate(
        X_encoded, y,
        RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced"),
        "IMPROVED: all 18 features, median-imputed, 300-tree RF, balanced"
    )

    # Refit on full data to report feature importance
    scaler = StandardScaler()
    model.fit(scaler.fit_transform(X_encoded), y)
    importance = pd.Series(model.feature_importances_, index=X_encoded.columns) \
        .sort_values(ascending=False)
    print("\nTop 10 features (full model):")
    print(importance.head(10).round(4).to_string())


def main():
    df = load_data()
    report_missingness(df)
    original_approach(df)
    improved_approach(df)


if __name__ == "__main__":
    main()
