"""
clean.py — reads raw CSVs, applies cleaning steps, writes events_clean.parquet.

Cleaning steps applied in a single streaming pass:
  - Parse event_time, cast types
  - Drop rows with null category_code or user_session
  - Semi-join to users who have at least one view event
  - Deduplicate on (event_time, user_id, event_type, product_id, user_session)
  - Add category_l1 (first dot-separated level of category_code)
"""
import polars as pl
from config import DATA_DIR


def main() -> None:
    files = [str(DATA_DIR / '2019-Oct.csv'), str(DATA_DIR / '2019-Nov.csv')]
    out   = DATA_DIR / 'events_clean.parquet'

    schema_overrides = {
        'event_type': pl.Categorical,
        'product_id': pl.Int32,
        'user_id':    pl.Int32,
        'price':      pl.Float32,
    }

    def scan() -> pl.LazyFrame:
        return (
            pl.scan_csv(files, schema_overrides=schema_overrides)
            .with_columns(
                pl.col('event_time')
                  .str.replace(r' UTC$', '')
                  .str.to_datetime('%Y-%m-%d %H:%M:%S'),
                pl.col('category_code').replace('', None),
                pl.col('brand').replace('', None),
            )
        )

    print('Cleaning events... (first pass: collecting view users)')
    view_users = (
        scan()
        .filter(pl.col('event_type') == 'view')
        .select('user_id')
        .unique()
        .collect()
    )
    print(f'  {len(view_users):,} users with at least one view — second pass: filter, dedup, write')

    (
        scan()
        .filter(
            pl.col('category_code').is_not_null() &
            pl.col('user_session').is_not_null()
        )
        .join(view_users.lazy(), on='user_id', how='semi')
        .unique(subset=['event_time', 'user_id', 'event_type', 'product_id', 'user_session'])
        .with_columns(
            pl.col('category_code').str.split('.').list.get(0).alias('category_l1')
        )
        .sink_parquet(out)
    )
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
