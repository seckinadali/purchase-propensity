# Purchase Propensity Model

This project provides a model that predicts the probability that a user purchases from each product category in the next 5 days, given their browsing behavior over a month. Output is a calibrated probability score per (user, category)-pair, suitable for ranking users within a category for targeting, or for expected-value calculations.

**Dataset:** [REES46 eCommerce behavior data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — ~110M events across Oct–Nov 2019 (view / cart / purchase).

---

## Problem Setup

| | Window | Role |
|---|---|---|
| Features | Oct 1–31 2019 | Behavioral signal |
| Labels | Nov 1–5 2019 | Purchase targets |

Nov 1–5 is chosen as the stable period before Black Friday run-up (~22k purchases/day).

**Category scope:** Eight categories with ≥500 training positives are used: electronics, appliances, computers, furniture, apparel, auto, construction, kids. The remaining five categories are excluded from the scaffold but their Oct events still contribute to user-level features.

**Scaffold:** All (user, category)-pairs across the 8 categories, 80/20 user split — train 1.4M rows (3.22% positive), test 349K (3.24%).

**Features:** Six groups: recency (days since last view/cart/purchase per category), frequency (event counts at 7/14/30-day windows, category-specific and cross-category), engagement (active days, session count, category breadth), intent ratios (cart-to-view at 7d and 30d), price (avg viewed and carted price per category), and category base rate.

**Modeling:** 5-fold GroupKFold CV (grouped by user to prevent scaffold leakage), HP search on fold 0 only, a 20% val set held out before any fitting, and isotonic regression calibration fitted on val-set predictions.

---

## Results

| Model | OOF AUC | Val AUC | Val LogLoss |
|---|---|---|---|
| LightGBM (tuned) | 0.9456 | 0.9471 | 0.0725 |
| Logistic Regression | 0.9086 | 0.9095 | 0.0877 |
| Naive baseline (category rate) | 0.8699 | 0.8705 | 0.1048 |

**Key findings:**

- **Price segment is the top signal:** `avg_price_viewed_30d` leads feature importance by a notable margin. It likely captures where a user sits in the price range of a category, which is strongly predictive of whether they'll buy there.
- **Cross-category activity outranks category-specific:** `total_views_30d`, `total_purchases_30d`, and `session_count_30d` rank above their category-specific counterparts — overall engagement level generalizes across categories.
- **Category-specific cart counts are among the weakest features** despite representing explicit purchase intent, possibly because short-window cart behavior is noisy at this dataset's scale.
- **LightGBM is already well-calibrated in expectation** before any calibration step (val mean prob 0.0320 vs true rate 0.0318). Isotonic regression corrects small bin-level deviations visible in the reliability diagram.
- **Model is robust to hyperparameter changes:** AUC spread across 20 random search trials was under 0.001.

---

## Repo Structure

```
.
├── data/                      # gitignored
│   ├── 2019-Oct.csv           # download from data source
│   ├── 2019-Nov.csv           # download from data source
│   ├── events.parquet
│   ├── events_clean.parquet
│   ├── train_features.parquet
│   ├── test_features.parquet
│   └── test_predictions.parquet
├── notebooks/
│   ├── 01_eda.ipynb           # EDA & cleaning
│   ├── 02_features.ipynb      # Feature engineering
│   └── 03_modeling.ipynb      # Modeling, calibration, final predictions
├── src/                       # planned
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Stack

Python · Polars (EDA & feature engineering) · pandas · LightGBM · XGBoost · scikit-learn
