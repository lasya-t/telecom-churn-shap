"""Metrics a retention team cares about, not just AUC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def decile_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Sort customers by risk, split into 10 equal groups, show churn rate and lift per group."""
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = np.arange(len(df)) * 10 // len(df) + 1
    base = df["y"].mean()
    t = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    t["churn_rate"] = t["churners"] / t["customers"]
    t["lift"] = t["churn_rate"] / base
    t["cum_share_of_churners"] = t["churners"].cumsum() / df["y"].sum()
    return t.round(3)


def summary(y, p) -> dict:
    t = decile_table(y, p)
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "top-10% lift": float(t.loc[1, "lift"]),
        "churners caught in top 20%": float(t.loc[2, "cum_share_of_churners"]),
    }
