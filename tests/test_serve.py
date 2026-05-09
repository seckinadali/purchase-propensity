"""Smoke tests for the FastAPI serving layer.

Contract tests run anywhere; behaviour tests need the real model. The whole module
is skipped if the trained model isn't on disk (e.g. in CI without training artifacts).
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from serve import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lgb_model.pkl'

if not MODEL_PATH.exists():
    pytest.skip(
        'trained model not on disk; run the training pipeline before testing serve.py',
        allow_module_level=True,
    )


WARM_PAYLOAD = {
    'record_id': 'u_warm',
    'category_l1': 'electronics',
    'views_30d': 20, 'carts_30d': 3, 'purchases_30d': 1,
    'views_14d': 10, 'carts_14d': 2, 'purchases_14d': 0,
    'views_7d': 4,   'carts_7d': 1,  'purchases_7d': 0,
    'days_since_view': 1, 'days_since_cart': 3, 'days_since_purchase': 25,
    'total_views_30d': 55, 'total_carts_30d': 6, 'total_purchases_30d': 2,
    'active_days_30d': 18, 'category_breadth_30d': 4, 'session_count_30d': 12,
    'brand_count_30d': 5, 'avg_price_viewed_30d': 349.0, 'avg_price_carted_30d': 299.0,
}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


# --- contract --------------------------------------------------------------

def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model_loaded': True}


def test_score_returns_expected_schema(client):
    r = client.post('/score', json=WARM_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {'record_id', 'category_l1', 'propensity_score'}
    assert body['record_id'] == 'u_warm'
    assert body['category_l1'] == 'electronics'
    assert isinstance(body['propensity_score'], float)


def test_score_rejects_unknown_category(client):
    """Pydantic Literal[...] guards the category set; everything else is a 422."""
    r = client.post('/score', json={'category_l1': 'spaceship'})
    assert r.status_code == 422


def test_score_batch_empty_list_returns_empty_list(client):
    r = client.post('/score/batch', json=[])
    assert r.status_code == 200
    assert r.json() == []


def test_score_batch_response_sorted_descending(client):
    r = client.post('/score/batch', json=[
        {'record_id': 'a', 'category_l1': 'kids'},
        {'record_id': 'b', 'category_l1': 'electronics'},
        {'record_id': 'c', 'category_l1': 'apparel'},
    ])
    assert r.status_code == 200
    scores = [row['propensity_score'] for row in r.json()]
    assert scores == sorted(scores, reverse=True)


# --- behaviour (requires the trained model) --------------------------------

def test_score_in_unit_interval(client):
    r = client.post('/score', json=WARM_PAYLOAD)
    score = r.json()['propensity_score']
    assert 0.0 <= score <= 1.0


def test_cold_start_scores_track_category_priors(client):
    """A cold user (only category_l1 specified) should land near each category's prior.

    Electronics has the highest prior (~0.185); kids the lowest (~0.003). Same caller,
    very different scores — proof the model uses the category prior for cold rows.
    """
    elec = client.post('/score', json={'category_l1': 'electronics'}).json()
    kids = client.post('/score', json={'category_l1': 'kids'}).json()

    assert elec['propensity_score'] > kids['propensity_score']
    assert elec['propensity_score'] > 0.05   # near 0.185 prior
    assert kids['propensity_score'] < 0.05   # near 0.003 prior


def test_warm_user_scores_higher_than_cold_for_same_category(client):
    """Holding category constant, the warm fully-featured user should outscore the cold one."""
    warm = client.post('/score', json=WARM_PAYLOAD).json()
    cold = client.post('/score', json={'category_l1': 'electronics'}).json()
    assert warm['propensity_score'] > cold['propensity_score']
