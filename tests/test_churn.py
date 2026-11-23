import numpy as np
import pytest

from churn.data import make_subscribers
from churn.evaluate import decile_table
from churn.explain import global_importance, reason_codes, shap_values
from churn.features import build_features
from churn.train import make_split, train_xgb


@pytest.fixture(scope="module")
def fitted():
    s = make_split(n=12_000)
    return s, train_xgb(s)


def test_features_have_no_leakage_columns():
    X = build_features(make_subscribers(500))
    assert "churned" not in X.columns and "subscriber_id" not in X.columns
    assert X.select_dtypes(exclude="number").empty


def test_decile_table_shape_and_lift():
    rng = np.random.default_rng(0)
    p = rng.random(10_000)
    y = (rng.random(10_000) < p).astype(int)
    t = decile_table(y, p)
    assert len(t) == 10 and t["customers"].sum() == 10_000
    assert t.loc[1, "lift"] > 1.5 > t.loc[10, "lift"]
    assert t["cum_share_of_churners"].iloc[-1] == 1.0


def test_model_beats_random(fitted):
    s, m = fitted
    t = decile_table(s.y_test, m.predict_proba(s.X_test)[:, 1])
    assert t.loc[1, "lift"] > 2.5


def test_shap_ranks_real_drivers_above_noise(fitted):
    s, m = fitted
    X = s.X_test.iloc[:1500]
    imp = global_importance(shap_values(m, X), X)
    for real in ("contract", "usage_trend", "support_calls_90d", "port_out_inquiry"):
        assert imp[real] > 5 * imp["favorite_color_survey"]


def test_shap_values_add_up_to_prediction(fitted):
    s, m = fitted
    X = s.X_test.iloc[:50]
    import shap
    ex = shap.TreeExplainer(m)
    margin = ex.expected_value + ex.shap_values(X).sum(axis=1)
    gap = m.predict(X, output_margin=True) - margin
    # per-row contributions are exact; some shap/xgboost version pairs disagree on the base
    # score by a small constant, so check the gap is the same for every row and small
    assert np.ptp(gap) < 1e-3 and abs(gap.mean()) < 0.05


def test_reason_codes_are_readable(fitted):
    s, m = fitted
    X = s.X_test.iloc[:200]
    codes = reason_codes(shap_values(m, X), X)
    assert len(codes) == 200 and all(len(c) <= 3 for c in codes)
    assert all(isinstance(r, str) for c in codes for r in c)
