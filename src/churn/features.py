"""Model features: one-hot categoricals, booleans as 0/1, plus two derived features."""
from __future__ import annotations

import pandas as pd

CATEGORICAL = ["contract", "plan", "region", "favorite_color_survey"]
DROP = ["subscriber_id", "churned"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    X["autopay"] = X["autopay"].astype(int)
    X["port_out_inquiry"] = X["port_out_inquiry"].astype(int)
    X["bill_per_line"] = (X["monthly_bill"] / X["lines"]).round(2)
    X["is_new_customer"] = (X["tenure_months"] <= 6).astype(int)
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)
    return X
