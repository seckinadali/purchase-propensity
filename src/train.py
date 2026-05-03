"""
train.py — HP search, 5-fold GroupKFold CV, final model, figures, test predictions.

Outputs:
  models/lgb_model.pkl       final LightGBM model (trained on full train set)
  models/model_info.json     params, n_estimators, OOF/val metrics
  figures/feature_importance.png
  figures/top_k_recall.png
  figures/reliability_diagram.png
  figures/prediction_distribution.png
  data/test_predictions.parquet
"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, OUTPUT_DIR,
    SEED, N_JOBS, N_FOLDS, N_HP_TRIALS, VAL_FRAC,
    KEPT_CATS, FEATURE_COLS, NUM_COLS, CAT_COLS,
)


# --- helpers ----------------------------------------------------------------

def make_lgb_arrays(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].astype('category')
    return X


def make_val_split(train: pd.DataFrame):
    all_users = np.sort(train['user_id'].unique())
    rng       = np.random.default_rng(SEED)
    val_ids   = set(rng.choice(all_users, size=int(len(all_users) * VAL_FRAC), replace=False))
    pool = train[~train['user_id'].isin(val_ids)].copy()
    val  = train[ train['user_id'].isin(val_ids)].copy()
    return pool, val


def hp_search(X_tr: pd.DataFrame, y_tr, X_vl: pd.DataFrame, y_vl) -> dict:
    rng = np.random.default_rng(SEED)
    best_auc, best_params = -1.0, {}
    print(f'HP search: {N_HP_TRIALS} trials on fold 0')
    print('-' * 56)
    for i in range(N_HP_TRIALS):
        params = dict(
            num_leaves        = int(rng.integers(15, 128)),
            min_child_samples = int(rng.integers(20, 200)),
            learning_rate     = float(rng.uniform(0.02, 0.15)),
            subsample         = float(rng.uniform(0.6, 1.0)),
            colsample_bytree  = float(rng.uniform(0.6, 1.0)),
            reg_alpha         = float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            reg_lambda        = float(rng.choice([0.0, 0.1, 0.5, 1.0])),
        )
        m = lgb.LGBMClassifier(n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=N_JOBS,
                               deterministic=True, force_col_wise=True, **params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        auc = roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1])
        print(f'  [{i+1:2d}] AUC={auc:.4f}  leaves={params["num_leaves"]:3d}'
              f'  lr={params["learning_rate"]:.3f}  iter={m.best_iteration_}')
        if auc > best_auc:
            best_auc, best_params = auc, params
    print(f'Best AUC: {best_auc:.4f}  params: {best_params}\n')
    return best_params


def run_cv(X_pool: pd.DataFrame, y_pool, groups, best_params: dict, gkf: GroupKFold):
    lgb_params = dict(n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=N_JOBS,
                      deterministic=True, force_col_wise=True, **best_params)
    oof              = np.zeros(len(X_pool))
    best_iterations  = []
    feat_importances = np.zeros(len(FEATURE_COLS))
    fold0_val_idx    = None

    print('LightGBM GroupKFold CV')
    print('-' * 56)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_pool, y_pool, groups)):
        if fold == 0:
            fold0_val_idx = val_idx
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_pool.iloc[tr_idx], y_pool.iloc[tr_idx],
              eval_set=[(X_pool.iloc[val_idx], y_pool.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = m.predict_proba(X_pool.iloc[val_idx])[:, 1]
        best_iterations.append(m.best_iteration_)
        feat_importances += m.feature_importances_
        auc = roc_auc_score(y_pool.iloc[val_idx], oof[val_idx])
        print(f'  Fold {fold}: AUC={auc:.4f}  best_iter={m.best_iteration_}')

    feat_importances /= N_FOLDS
    eval_mask                = np.ones(len(X_pool), dtype=bool)
    eval_mask[fold0_val_idx] = False
    oof_auc    = roc_auc_score(y_pool[eval_mask], oof[eval_mask])
    oof_ll     = log_loss(y_pool[eval_mask], oof[eval_mask])
    mean_iters = int(np.round(np.mean(best_iterations)))

    print(f'\nOOF AUC (folds 1-4): {oof_auc:.4f}  LogLoss: {oof_ll:.4f}')
    print(f'Best iterations: {best_iterations}  → mean: {mean_iters}\n')
    return mean_iters, feat_importances, oof_auc, oof_ll


def top_k_recall(y_true, y_score, fracs):
    order    = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order]
    total    = y_sorted.sum()
    if total == 0:
        return np.zeros(len(fracs))
    return np.array([y_sorted[:max(1, int(len(y_sorted) * f))].sum() / total for f in fracs])


# --- figure savers ----------------------------------------------------------

def save_feature_importance(feat_importances: np.ndarray) -> None:
    imp_df = (
        pd.DataFrame({'feature': FEATURE_COLS, 'importance': feat_importances})
        .sort_values('importance', ascending=False)
    )
    top20 = imp_df.head(20).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top20['feature'], top20['importance'], color='tab:blue')
    ax.set_xlabel('Mean split importance (across folds)')
    ax.set_title('Top 20 Feature Importances — LightGBM')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150)
    plt.close(fig)
    print('Saved figures/feature_importance.png')


def save_top_k_recall(val_df: pd.DataFrame, val_preds: np.ndarray, y_val: np.ndarray) -> None:
    fracs = np.linspace(0.01, 1.0, 100)
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in KEPT_CATS:
        mask   = (val_df['category_l1'] == cat).values
        recall = top_k_recall(y_val[mask], val_preds[mask], fracs)
        ax.plot(fracs * 100, recall * 100, label=cat)
    overall = top_k_recall(y_val, val_preds, fracs)
    ax.plot(fracs * 100, overall * 100, 'k-', linewidth=2, label='Overall')
    ax.plot([0, 100], [0, 100], 'gray', linestyle=':', linewidth=1, label='Random')
    for x in [20, 40, 60, 80]:
        ax.axvline(x, color='lightgray', linestyle='--', linewidth=0.8, zorder=0)
    ax.set_xlabel('% of users targeted (ranked by propensity score)')
    ax.set_ylabel('% of buyers captured')
    ax.set_title('Top-K Recall Curve — LightGBM (val set)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'top_k_recall.png', dpi=150)
    plt.close(fig)
    print('Saved figures/top_k_recall.png')


def save_reliability_diagram(y_val: np.ndarray, val_preds: np.ndarray) -> None:
    frac_pos, mean_pred = calibration_curve(y_val, val_preds, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, 's-', color='tab:blue', label='Model')
    ax.plot([0, 0.3], [0, 0.3], 'k--', lw=1, label='Perfect calibration')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction of positives')
    ax.set_title('Reliability Diagram — LightGBM (val set)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'reliability_diagram.png', dpi=150)
    plt.close(fig)
    print('Saved figures/reliability_diagram.png')


def save_prediction_distribution(test: pd.DataFrame, test_preds: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(test_preds, bins=50, edgecolor='white', color='tab:blue')
    axes[0].set_xlabel('Predicted probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Test prediction distribution')
    cat_means = (
        pd.DataFrame({'category': test['category_l1'], 'prob': test_preds})
        .groupby('category')['prob'].mean()
        .sort_values()
    )
    axes[1].barh(cat_means.index, cat_means.values, color='tab:blue')
    axes[1].set_xlabel('Mean predicted probability')
    axes[1].set_title('Mean prediction by category (test)')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'prediction_distribution.png', dpi=150)
    plt.close(fig)
    print('Saved figures/prediction_distribution.png')


# --- main -------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load features
    train = pd.read_parquet(DATA_DIR / 'train_features.parquet')
    test  = pd.read_parquet(DATA_DIR / 'test_features.parquet')
    print(f'Train: {len(train):,} rows | positive rate: {train["label"].mean():.2%}')
    print(f'Test:  {len(test):,} rows  | positive rate: {test["label"].mean():.2%}\n')

    # Val split — held out before any fitting
    train_pool, val_df = make_val_split(train)
    y_val        = val_df['label'].values
    val_pos_rate = val_df['label'].mean()
    print(f'train_pool: {len(train_pool):,} rows | val_df: {len(val_df):,} rows\n')

    X_pool   = make_lgb_arrays(train_pool)
    y_pool   = train_pool['label']
    groups   = train_pool['user_id']
    X_val    = make_lgb_arrays(val_df)
    gkf      = GroupKFold(n_splits=N_FOLDS)

    # HP search on fold 0 only
    _tr0, _vl0 = next(iter(gkf.split(X_pool, y_pool, groups)))
    best_params = hp_search(X_pool.iloc[_tr0], y_pool.iloc[_tr0],
                            X_pool.iloc[_vl0], y_pool.iloc[_vl0])

    # Full CV (fold 0 excluded from reported OOF — it was the HP tuning fold)
    mean_iters, feat_importances, oof_auc, oof_ll = run_cv(
        X_pool, y_pool, groups, best_params, gkf
    )

    # Retrain on train_pool → val evaluation
    print('Retraining on train_pool for val evaluation...')
    lgb_params = dict(n_estimators=mean_iters, random_state=SEED, verbose=-1, n_jobs=N_JOBS,
                      deterministic=True, force_col_wise=True, **best_params)
    val_model  = lgb.LGBMClassifier(**lgb_params)
    val_model.fit(X_pool, y_pool)
    val_preds = val_model.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_preds)
    val_ll    = log_loss(y_val, val_preds)
    print(f'Val AUC: {val_auc:.4f}  LogLoss: {val_ll:.4f}'
          f'  MeanProb: {val_preds.mean():.4f}  (true rate: {val_pos_rate:.4f})\n')

    # Final model on full train (pool + val)
    print('Training final model on full train set...')
    X_full     = make_lgb_arrays(train)
    y_full     = train['label']
    full_model = lgb.LGBMClassifier(**lgb_params)
    full_model.fit(X_full, y_full)

    X_test     = make_lgb_arrays(test)
    test_preds = full_model.predict_proba(X_test)[:, 1]
    print(f'Test mean prob: {test_preds.mean():.4f}  (train pos rate: {y_full.mean():.4f})\n')

    # Figures
    print('Saving figures...')
    save_feature_importance(feat_importances)
    save_top_k_recall(val_df, val_preds, y_val)
    save_reliability_diagram(y_val, val_preds)
    save_prediction_distribution(test, test_preds)

    # Model + metadata
    with open(MODELS_DIR / 'lgb_model.pkl', 'wb') as f:
        pickle.dump(full_model, f)
    model_info = {
        'final_model_n_estimators': mean_iters,       # mean CV best_iter; used for final model
        'params':                   best_params,
        'oof_auc':                  round(oof_auc, 4), # GroupKFold on train_pool (folds 1-4)
        'oof_logloss':              round(oof_ll, 4),
        'val_model_auc':            round(val_auc, 4), # model trained on train_pool, evaluated on val
        'val_model_logloss':        round(val_ll, 4),
    }
    with open(MODELS_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print('Saved models/lgb_model.pkl and models/model_info.json')

    # Test predictions
    output = test[['user_id', 'category_l1']].copy()
    output['propensity_score'] = test_preds
    output = output.sort_values(['category_l1', 'propensity_score'], ascending=[True, False])
    output.to_parquet(DATA_DIR / 'test_predictions.parquet', index=False)
    print(f'Saved data/test_predictions.parquet  shape: {output.shape}')

    output.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
    print(f'Saved output/test_predictions.csv')

    sample = output.groupby('category_l1', observed=True).head(5)
    sample.to_csv(OUTPUT_DIR / 'test_predictions_sample.csv', index=False)
    print(f'Saved output/test_predictions_sample.csv  ({len(sample)} rows, top 5 per category)')


if __name__ == '__main__':
    main()
