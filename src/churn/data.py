"""Synthetic mobile subscriber base with realistic churn drivers.

Churn is generated from a hidden logistic model so we know which drivers are real:
month-to-month contracts, short tenure, repeated support calls, dropped calls,
data usage falling off, an old device, high bills, and asking about porting out.
Autopay and family plans make customers stickier. Some columns are pure noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_subscribers(n: int = 50_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "1-year", "2-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.clip(rng.gamma(2.0, 14, n), 1, 96).round()
    lines = rng.choice([1, 2, 3, 4, 5], n, p=[0.45, 0.25, 0.15, 0.1, 0.05])
    plan = rng.choice(["Basic", "Unlimited", "Unlimited Plus", "Premium"], n, p=[0.3, 0.35, 0.2, 0.15])
    base_bill = {"Basic": 40, "Unlimited": 65, "Unlimited Plus": 80, "Premium": 95}
    monthly_bill = np.array([base_bill[p] for p in plan]) * lines * rng.uniform(0.85, 1.25, n)
    support_calls_90d = rng.poisson(0.8, n)
    dropped_call_rate = np.clip(rng.gamma(1.5, 1.2, n), 0, 15)  # % of calls dropped
    data_gb_m1 = rng.gamma(3, 6, n)
    usage_trend = rng.normal(0, 0.15, n)                         # change in data usage, last 3 months
    device_age_months = np.clip(rng.gamma(2.5, 9, n), 0, 72).round()
    autopay = rng.random(n) < 0.6
    port_out_inquiry = rng.random(n) < 0.04
    region = rng.choice(["Northeast", "Southeast", "Midwest", "West"], n)
    favorite_color_survey = rng.choice(["red", "blue", "green", "none"], n)  # noise

    logit = (
        -2.6
        + 1.3 * (contract == "month-to-month") - 0.6 * (contract == "2-year")
        - 0.025 * tenure
        + 0.45 * np.minimum(support_calls_90d, 6)
        + 0.18 * dropped_call_rate
        - 4.0 * usage_trend
        + 0.02 * device_age_months
        + 0.004 * (monthly_bill - 100)
        - 0.5 * autopay
        - 0.25 * (lines - 1)
        + 2.2 * port_out_inquiry
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "subscriber_id": np.arange(1, n + 1),
        "contract": contract, "tenure_months": tenure, "lines": lines, "plan": plan,
        "monthly_bill": monthly_bill.round(2), "support_calls_90d": support_calls_90d,
        "dropped_call_rate": dropped_call_rate.round(2), "data_gb_m1": data_gb_m1.round(1),
        "usage_trend": usage_trend.round(3), "device_age_months": device_age_months,
        "autopay": autopay, "port_out_inquiry": port_out_inquiry, "region": region,
        "favorite_color_survey": favorite_color_survey, "churned": churned.astype(int),
    })
