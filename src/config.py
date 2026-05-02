from pathlib import Path
from datetime import datetime

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / 'data'
FIGURES_DIR = ROOT_DIR / 'figures'
MODELS_DIR  = ROOT_DIR / 'models'
OUTPUT_DIR  = ROOT_DIR / 'output'

CUTOFF_DT  = datetime(2019, 11, 1)
TARGET_END = datetime(2019, 11, 6)
W14_START  = datetime(2019, 10, 18)
W7_START   = datetime(2019, 10, 25)

SEED                = 42
N_JOBS              = 1   # pair with deterministic=True, force_col_wise=True in train.py for full LGBm reproducibility
TRAIN_FRAC          = 0.8
VAL_FRAC            = 0.2
NON_PURCHASER_RATIO = 3
N_FOLDS             = 5
N_HP_TRIALS         = 20

KEPT_CATS = [
    'electronics', 'appliances', 'computers', 'furniture',
    'apparel', 'auto', 'construction', 'kids',
]

CAT_COLS = ['category_l1']
NUM_COLS = [
    'views_30d', 'carts_30d', 'purchases_30d',
    'views_14d', 'carts_14d', 'purchases_14d',
    'views_7d', 'carts_7d', 'purchases_7d',
    'days_since_view', 'days_since_cart', 'days_since_purchase',
    'total_views_30d', 'total_carts_30d', 'total_purchases_30d',
    'active_days_30d', 'category_breadth_30d', 'session_count_30d',
    'cat_purchase_rate', 'cart_view_ratio_30d', 'cart_view_ratio_7d',
    'brand_count_30d',
    'avg_price_viewed_30d', 'avg_price_carted_30d',
]
FEATURE_COLS = NUM_COLS + CAT_COLS
EVENT_COUNT_COLS = [
    'views_30d', 'carts_30d', 'purchases_30d',
    'views_14d', 'carts_14d', 'purchases_14d',
    'views_7d', 'carts_7d', 'purchases_7d',
]
