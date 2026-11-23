"""Train a logistic baseline and an XGBoost model on the same split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_subscribers
from .features import build_features


@dataclass
class Split:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


def make_split(n=50_000, seed=0) -> Split:
    df = make_subscribers(n, seed)
    X, y = build_features(df), df["churned"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
    return Split(Xtr, Xva, Xte, ytr, yva, yte)


def train_baseline(s: Split):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(s.X_train, s.y_train)


def train_xgb(s: Split, seed=0) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=1.0, eval_metric="aucpr", early_stopping_rounds=100,
        random_state=seed, n_jobs=-1,
    )
    model.fit(s.X_train, s.y_train, eval_set=[(s.X_valid, s.y_valid)], verbose=False)
    return model
