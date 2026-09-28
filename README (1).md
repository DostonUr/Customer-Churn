# E-Commerce Customer Churn — Revisiting the Feature Selection

The original notebook (`churn-portfolio.ipynb`) hit 33% row-wise missingness across all 20
columns, and fixed it by keeping only the 6 features most correlated with churn individually,
then dropping the incomplete rows in that smaller set. This follow-up tests whether that was the
right call.

Full write-up: **[Read the article](https://urinovd.github.io/2027/02/06/customer-churn-feature-selection-mistake.html)**

**Note:** this project's README previously described the dataset as "telecom" — it's actually an
e-commerce dataset (online retail customer churn), corrected here.

## What's here

- `churn-portfolio.ipynb` — the original notebook
- `churn_feature_comparison.py` — the follow-up: compares the original 6-feature/dropna approach
  against using all 18 usable features with median imputation
- `E-Commerce-Dataset.xlsx` — the dataset ([source](https://github.com/anvarnarz/praktikum_datasets))

## The comparison

| Approach | Data used | Test F1 | Test Recall | Test Precision | 5-fold CV F1 (mean) |
|---|---|---|---|---|---|
| Original (6 features, dropna, 9-tree RF) | 89.9% of rows | 0.796 | 0.767 | 0.828 | 0.774 |
| **Full features (18), imputed, 300-tree RF, balanced** | **100% of rows** | **0.869** | **0.805** | **0.944** | **0.888** |

Filtering to 6 features by univariate correlation with the target, before modeling, discarded
13 features that turned out to carry real signal once given the chance — **WarehouseToHome,
OrderAmountHikeFromlastYear, NumberOfAddress,** and **SatisfactionScore** all rank in the
improved model's top 10 most important features, despite none of them making the original
correlation-based cut.

## Run it

```bash
pip install -r requirements.txt
python churn_feature_comparison.py
```

## Author

Dostonbek URINOV — [LinkedIn](https://www.linkedin.com/in/doston-urinov/) ·
[GitHub](https://github.com/DostonUr) · [Kaggle](https://www.kaggle.com/dostonur)
