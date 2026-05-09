"""Unit tests for src/features.py.

Strategy: hand-build a tiny events frame, hand-compute expected outputs, and check
each helper. Catches off-by-one window errors, wrong join sides, and null-handling
bugs — the bug class that bit this project at the cold-start scaffold.
"""
from datetime import datetime

import polars as pl

import features as F
from config import CUTOFF_DT, EVENT_COUNT_COLS, W7_START, W14_START


def make_events() -> pl.DataFrame:
    """Three users with hand-known counts.

    u1: four electronics events in late Oct (3 views + 1 purchase, all within 7d window)
    u2: one apparel view on Oct 5 (inside 30d window, OUTSIDE 7d and 14d windows)
    u3: one Nov 2 view (no October activity at all → cold-start user)
    """
    return pl.DataFrame({
        'event_time': [
            datetime(2019, 10, 28),
            datetime(2019, 10, 29),
            datetime(2019, 10, 30),
            datetime(2019, 10, 30),
            datetime(2019, 10, 5),
            datetime(2019, 11, 2),
        ],
        'event_type':   ['view', 'view', 'view', 'purchase', 'view', 'view'],
        'user_id':      [1, 1, 1, 1, 2, 3],
        'category_l1':  ['electronics'] * 4 + ['apparel', 'apparel'],
        'brand':        ['acme'] * 4 + ['beta', 'gamma'],
        'price':        [100.0] * 4 + [20.0, 30.0],
        'user_session': ['s1', 's1', 's2', 's2', 's3', 's4'],
    })


def _oct_lf(events: pl.DataFrame) -> pl.LazyFrame:
    return events.filter(pl.col('event_time') < CUTOFF_DT).lazy()


def test_user_cat_counts_30d_includes_all_october_events():
    counts = F.user_cat_counts(_oct_lf(make_events()), '30d')
    u1 = counts.filter(
        (pl.col('user_id') == 1) & (pl.col('category_l1') == 'electronics')
    )
    assert u1['views_30d'][0] == 3
    assert u1['purchases_30d'][0] == 1


def test_user_cat_counts_7d_excludes_events_before_window_start():
    """u2's Oct 5 view sits inside the 30d window but outside the 7d (W7_START = Oct 25).

    This is the most load-bearing assertion in features.py: a one-character bug
    (>= → >, < → <=) here would silently shift every count by a day's worth of events.
    """
    counts_7d = F.user_cat_counts(_oct_lf(make_events()), '7d', start=W7_START)
    counts_14d = F.user_cat_counts(_oct_lf(make_events()), '14d', start=W14_START)

    assert counts_7d.filter(pl.col('user_id') == 2).height == 0
    assert counts_14d.filter(pl.col('user_id') == 2).height == 0


def test_user_cat_recency_measured_from_cutoff_not_from_max_event():
    """days_since_view = CUTOFF_DT - last_view, NOT max_event - last_view.

    The distinction matters when the cutoff is in the future relative to user data:
    the model expects "as of Nov 1" semantics, not "as of the user's most recent event."
    """
    rec = F.user_cat_recency(_oct_lf(make_events()))
    u1 = rec.filter((pl.col('user_id') == 1) & (pl.col('category_l1') == 'electronics'))
    # last view at Oct 30, cutoff Nov 1 → 2 days
    assert u1['days_since_view'][0] == 2


def test_assemble_features_fills_event_counts_for_cold_users():
    """Cold users (no rows in any feature frame) get 0 in EVENT_COUNT_COLS but
    null in user-level aggregates. This asymmetry is what the cold-start fix relies on
    end-to-end: the model sees zero-feature rows for cold users at training time, and
    serve.py preserves null for the same fields when a cold user calls the API.
    """
    events = make_events()
    oct_lf = _oct_lf(events)

    counts_30d = F.user_cat_counts(oct_lf, '30d')
    counts_14d = F.user_cat_counts(oct_lf, '14d', start=W14_START)
    counts_7d  = F.user_cat_counts(oct_lf, '7d', start=W7_START)
    user_cat_events = (
        counts_30d
        .join(counts_14d, on=['user_id', 'category_l1'], how='left')
        .join(counts_7d,  on=['user_id', 'category_l1'], how='left')
    )
    rec    = F.user_cat_recency(oct_lf)
    user_l = F.user_level_feats(oct_lf)
    brand  = F.user_cat_brand_feats(oct_lf)
    price  = F.user_cat_price_feats(oct_lf)

    # u3 has no October activity → fully cold
    labeled = pl.DataFrame({
        'user_id':     [3],
        'category_l1': ['kids'],
        'label':       [0],
    })

    out = F.assemble_features(labeled, user_cat_events, rec, user_l, brand, price)
    cold = out.row(0, named=True)

    for col in EVENT_COUNT_COLS:
        assert cold[col] == 0, f'{col} should fill to 0 for cold users, got {cold[col]!r}'

    # User-level aggregates remain null — this is the cold-start signal LightGBM uses.
    assert cold['total_views_30d'] is None
    assert cold['active_days_30d'] is None
    assert cold['avg_price_viewed_30d'] is None
