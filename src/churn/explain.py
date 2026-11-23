"""SHAP explanations: which drivers matter overall, and why each customer is at risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

REASONS = {
    "contract_month-to-month": "on a month-to-month contract",
    "tenure_months": "short time as a customer",
    "is_new_customer": "joined in the last 6 months",
    "support_calls_90d": "several support calls recently",
    "dropped_call_rate": "high dropped-call rate",
    "usage_trend": "data usage is falling",
    "device_age_months": "device is getting old",
    "monthly_bill": "high monthly bill",
    "bill_per_line": "high bill per line",
    "autopay": "not on autopay",
    "lines": "single-line account",
    "port_out_inquiry": "asked about porting out",
}


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    return shap.TreeExplainer(model).shap_values(X)


def global_importance(sv: np.ndarray, X: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature, one-hot columns folded back into their source column."""
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    groups = imp.index.str.replace(r"_(month-to-month|1-year|2-year|Basic|Unlimited.*|Premium|Northeast|Southeast|Midwest|West|red|blue|green|none)$", "", regex=True)
    return imp.groupby(groups).sum().sort_values(ascending=False)


def reason_codes(sv: np.ndarray, X: pd.DataFrame, top: int = 3) -> list[list[str]]:
    """Plain-English top reasons pushing each customer's risk up."""
    out = []
    for row in sv:
        order = np.argsort(row)[::-1]
        reasons = [REASONS[X.columns[i]] for i in order if row[i] > 0 and X.columns[i] in REASONS][:top]
        out.append(reasons)
    return out
