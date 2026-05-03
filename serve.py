"""
serve.py — FastAPI scoring endpoint for the purchase propensity model.

Usage:
  uvicorn serve:app --reload
  uvicorn serve:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health       — liveness + model-load status
  POST /score        — score a single (user, category) feature row
  POST /score/batch  — score multiple rows; response sorted by score descending
"""
import json
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from config import FEATURE_COLS, KEPT_CATS

MODEL_PATH      = Path(__file__).resolve().parent / 'models' / 'lgb_model.pkl'
MODEL_INFO_PATH = Path(__file__).resolve().parent / 'models' / 'model_info.json'

_model: object = None
_cat_purchase_rates: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _cat_purchase_rates
    for path in (MODEL_PATH, MODEL_INFO_PATH):
        if not path.exists():
            raise RuntimeError(f'{path.name} not found at {path}. Run the training pipeline first.')
    with open(MODEL_PATH, 'rb') as f:
        _model = pickle.load(f)
    with open(MODEL_INFO_PATH) as f:
        _cat_purchase_rates = json.load(f)['cat_purchase_rates']
    yield


app = FastAPI(
    title='Purchase Propensity API',
    description=(
        'Returns the probability a user purchases from a product category '
        'in the next 5 days, given their October browsing behavior.'
    ),
    lifespan=lifespan,
)

Category = Literal[
    'electronics', 'appliances', 'computers', 'furniture',
    'apparel', 'auto', 'construction', 'kids',
]


class ScoreRequest(BaseModel):
    record_id: Optional[str] = None  # echoed back in response for caller correlation

    # ── categorical ──────────────────────────────────────────────────────────
    category_l1: Category

    # ── event counts (0 when no events observed in the window) ───────────────
    views_30d:     int = 0
    carts_30d:     int = 0
    purchases_30d: int = 0
    views_14d:     int = 0
    carts_14d:     int = 0
    purchases_14d: int = 0
    views_7d:      int = 0
    carts_7d:      int = 0
    purchases_7d:  int = 0

    # ── recency (None = event type never observed for this user × category) ──
    days_since_view:     Optional[float] = None
    days_since_cart:     Optional[float] = None
    days_since_purchase: Optional[float] = None

    # ── user-level cross-category aggregates ─────────────────────────────────
    total_views_30d:      int = 0
    total_carts_30d:      int = 0
    total_purchases_30d:  int = 0
    active_days_30d:      int = 0
    category_breadth_30d: int = 0
    session_count_30d:    int = 0

    # ── brand and price (None = no data for this user × category) ────────────
    brand_count_30d:      Optional[int]   = None
    avg_price_viewed_30d: Optional[float] = None
    avg_price_carted_30d: Optional[float] = None


class ScoreResponse(BaseModel):
    record_id:        Optional[str]
    category_l1:      str
    propensity_score: float


def _to_feature_frame(records: list[ScoreRequest]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = r.model_dump(exclude={'record_id'})
        row['cat_purchase_rate'] = _cat_purchase_rates[r.category_l1]
        rows.append(row)
    X = pd.DataFrame(rows)
    # derived from raw counts — same formula as features.py
    X['cart_view_ratio_30d'] = X['carts_30d'] / X['views_30d'].clip(lower=1)
    X['cart_view_ratio_7d']  = X['carts_7d']  / X['views_7d'].clip(lower=1)
    X = X[FEATURE_COLS]
    X['category_l1'] = pd.Categorical(X['category_l1'], categories=KEPT_CATS)
    return X


def _predict(records: list[ScoreRequest]) -> np.ndarray:
    return _model.predict_proba(_to_feature_frame(records))[:, 1]


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': _model is not None}


@app.post('/score', response_model=ScoreResponse)
def score(request: ScoreRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return ScoreResponse(
        record_id=request.record_id,
        category_l1=request.category_l1,
        propensity_score=round(float(_predict([request])[0]), 6),
    )


@app.post('/score/batch', response_model=list[ScoreResponse])
def score_batch(requests: list[ScoreRequest]):
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if not requests:
        return []
    scores = _predict(requests)
    results = [
        ScoreResponse(
            record_id=r.record_id,
            category_l1=r.category_l1,
            propensity_score=round(float(s), 6),
        )
        for r, s in zip(requests, scores)
    ]
    return sorted(results, key=lambda x: x.propensity_score, reverse=True)
