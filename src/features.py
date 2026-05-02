"""
features.py — builds train and test feature matrices from events_clean.parquet.

User pool: Nov 1-5 purchasers + Oct-active non-purchasers sampled at 3:1.
Train/test split: 80/20 by user.
Scaffold: (user, category) cross-join across KEPT_CATS.
Labels: 1 if user purchased in that category in Nov 1-5, else 0.
Features: event counts (7/14/30d), recency, user-level aggregates,
          category priors, brand count, price averages, cart-view ratios.
"""
import polars as pl
from config import (
    DATA_DIR, CUTOFF_DT, TARGET_END, W14_START, W7_START,
    SEED, TRAIN_FRAC, NON_PURCHASER_RATIO, KEPT_CATS, EVENT_COUNT_COLS,
)


# --- feature computation helpers -------------------------------------------

def user_cat_counts(lf: pl.LazyFrame, suffix: str, start=None) -> pl.DataFrame:
    if start is not None:
        lf = lf.filter(pl.col('event_time') >= start)
    return (
        lf
        .group_by(['user_id', 'category_l1'])
        .agg(
            (pl.col('event_type') == 'view').sum().alias(f'views_{suffix}'),
            (pl.col('event_type') == 'cart').sum().alias(f'carts_{suffix}'),
            (pl.col('event_type') == 'purchase').sum().alias(f'purchases_{suffix}'),
        )
        .collect()
    )


def user_cat_recency(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf
        .group_by(['user_id', 'category_l1'])
        .agg(
            pl.col('event_time').filter(pl.col('event_type') == 'view').max().alias('last_view'),
            pl.col('event_time').filter(pl.col('event_type') == 'cart').max().alias('last_cart'),
            pl.col('event_time').filter(pl.col('event_type') == 'purchase').max().alias('last_purchase'),
        )
        .with_columns(
            (pl.lit(CUTOFF_DT) - pl.col('last_view')).dt.total_days().alias('days_since_view'),
            (pl.lit(CUTOFF_DT) - pl.col('last_cart')).dt.total_days().alias('days_since_cart'),
            (pl.lit(CUTOFF_DT) - pl.col('last_purchase')).dt.total_days().alias('days_since_purchase'),
        )
        .drop(['last_view', 'last_cart', 'last_purchase'])
        .collect()
    )


def user_level_feats(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf
        .group_by('user_id')
        .agg(
            (pl.col('event_type') == 'view').sum().alias('total_views_30d'),
            (pl.col('event_type') == 'cart').sum().alias('total_carts_30d'),
            (pl.col('event_type') == 'purchase').sum().alias('total_purchases_30d'),
            pl.col('event_time').dt.date().n_unique().alias('active_days_30d'),
            pl.col('category_l1').n_unique().alias('category_breadth_30d'),
            pl.col('user_session').n_unique().alias('session_count_30d'),
        )
        .collect()
    )


def user_cat_brand_feats(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf
        .filter(pl.col('brand').is_not_null())
        .group_by(['user_id', 'category_l1'])
        .agg(pl.col('brand').n_unique().alias('brand_count_30d'))
        .collect()
    )


def user_cat_price_feats(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf
        .group_by(['user_id', 'category_l1'])
        .agg(
            pl.col('price').filter(pl.col('event_type') == 'view').mean().alias('avg_price_viewed_30d'),
            pl.col('price').filter(pl.col('event_type') == 'cart').mean().alias('avg_price_carted_30d'),
        )
        .collect()
    )


def assemble_features(
    labeled:        pl.DataFrame,
    user_cat_events: pl.DataFrame,
    recency_feats:  pl.DataFrame,
    user_feats:     pl.DataFrame,
    cat_priors:     pl.DataFrame,
    brand_feats:    pl.DataFrame,
    price_feats:    pl.DataFrame,
) -> pl.DataFrame:
    return (
        labeled
        .join(user_cat_events, on=['user_id', 'category_l1'], how='left')
        .join(recency_feats,   on=['user_id', 'category_l1'], how='left')
        .join(user_feats,      on='user_id',                  how='left')
        .join(cat_priors,      on='category_l1',              how='left')
        .join(brand_feats,     on=['user_id', 'category_l1'], how='left')
        .join(price_feats,     on=['user_id', 'category_l1'], how='left')
        .with_columns(
            (pl.col('carts_30d') / pl.col('views_30d').clip(lower_bound=1)).alias('cart_view_ratio_30d'),
            (pl.col('carts_7d')  / pl.col('views_7d').clip(lower_bound=1)).alias('cart_view_ratio_7d'),
        )
        .with_columns(
            [pl.col(c).fill_null(0) for c in EVENT_COUNT_COLS]
        )
    )


# --- main pipeline -----------------------------------------------------------

def main() -> None:
    lf = pl.scan_parquet(DATA_DIR / 'events_clean.parquet')

    cutoff_lit     = pl.lit(CUTOFF_DT)
    target_end_lit = pl.lit(TARGET_END)

    # Feature and label windows
    oct_lf = lf.filter(pl.col('event_time') < cutoff_lit)

    nov_purchases = (
        lf
        .filter(
            (pl.col('event_type') == 'purchase') &
            (pl.col('event_time') >= cutoff_lit) &
            (pl.col('event_time') <  target_end_lit)
        )
        .select(['user_id', 'category_l1'])
        .unique()
        .collect()
    )
    print(f'Nov 1-5 purchase (user, category) pairs: {len(nov_purchases):,}')

    # User pool: purchasers + sampled Oct-active non-purchasers
    purchaser_users     = nov_purchases.select('user_id').unique()
    oct_active_users    = oct_lf.select('user_id').unique().collect()
    non_purchaser_users = oct_active_users.join(purchaser_users, on='user_id', how='anti')

    n_non              = min(len(purchaser_users) * NON_PURCHASER_RATIO, len(non_purchaser_users))
    # sort before sample: Polars unique/group_by don't guarantee row order, so without
    # sorting the seed produces different splits depending on upstream execution order
    sampled_non        = non_purchaser_users.sort('user_id').sample(n=n_non, seed=SEED)
    all_users          = pl.concat([purchaser_users, sampled_non]).sort('user_id')
    train_users        = all_users.sample(fraction=TRAIN_FRAC, seed=SEED)
    test_users         = all_users.join(train_users, on='user_id', how='anti')

    print(f'Train users: {len(train_users):,} | Test users: {len(test_users):,}')

    # Scaffold: (user, category) cross-join, labels attached, filtered to KEPT_CATS
    categories = pl.DataFrame({'category_l1': KEPT_CATS})

    def build_and_label(users: pl.DataFrame) -> pl.DataFrame:
        scaffold = users.join(categories, how='cross')
        return (
            scaffold
            .join(nov_purchases.with_columns(pl.lit(1).alias('label')),
                  on=['user_id', 'category_l1'], how='left')
            .with_columns(pl.col('label').fill_null(0))
        )

    train_labeled = build_and_label(train_users)
    test_labeled  = build_and_label(test_users)

    for name, df in [('Train', train_labeled), ('Test', test_labeled)]:
        rate = df['label'].mean() * 100
        print(f'{name}: {len(df):,} rows | positive rate: {rate:.2f}%')

    # Category priors from train_labeled (includes val users — see note in 02_features.ipynb)
    cat_priors = (
        train_labeled
        .group_by('category_l1')
        .agg(pl.col('label').mean().alias('cat_purchase_rate'))
    )

    # Feature groups from Oct behavioral data
    print('Computing features...')
    counts_30d = user_cat_counts(oct_lf, '30d')
    counts_14d = user_cat_counts(oct_lf, '14d', start=W14_START)
    counts_7d  = user_cat_counts(oct_lf, '7d',  start=W7_START)
    user_cat_events = (
        counts_30d
        .join(counts_14d, on=['user_id', 'category_l1'], how='left')
        .join(counts_7d,  on=['user_id', 'category_l1'], how='left')
    )

    recency_feats = user_cat_recency(oct_lf)
    user_feats    = user_level_feats(oct_lf)
    brand_feats   = user_cat_brand_feats(oct_lf)
    price_feats   = user_cat_price_feats(oct_lf)

    args = (user_cat_events, recency_feats, user_feats, cat_priors, brand_feats, price_feats)
    train_features = assemble_features(train_labeled, *args)
    test_features  = assemble_features(test_labeled,  *args)

    train_features.write_parquet(DATA_DIR / 'train_features.parquet')
    test_features.write_parquet(DATA_DIR / 'test_features.parquet')
    print(f'Saved train_features.parquet {train_features.shape}')
    print(f'Saved test_features.parquet  {test_features.shape}')


if __name__ == '__main__':
    main()
