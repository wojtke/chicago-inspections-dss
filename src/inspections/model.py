"""Prediction module: probability that a routine inspection fails."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, NUMERIC, build_features, canvass_outcomes

MODELS = Path("models")
MODEL_PATH = MODELS / "risk_model.pkl"
METRICS_PATH = MODELS / "metrics.json"

# 2010-2014 only serves as history for the features.
SPLITS = {"train": ("2015-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2099-12-31")}


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = build_features(df, canvass_outcomes(df))
    rows["split"] = None
    for name, (lo, hi) in SPLITS.items():
        rows.loc[rows["date"].between(lo, hi), "split"] = name
    return rows.dropna(subset=["split"])


def gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, min_samples_leaf=200,
                                          categorical_features="from_dtype", random_state=0)


def logistic() -> object:
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=1000))


def evaluate(y: np.ndarray, p: np.ndarray, top: float = 0.1) -> dict:
    """AUC, plus how often the top 10% by predicted risk actually failed."""
    k = max(1, int(len(p) * top))
    top_idx = np.argsort(-p)[:k]
    return {"auc": float(roc_auc_score(y, p)), "fail_rate_top10": float(y[top_idx].mean()),
            "fail_rate_all": float(y.mean()), "n": int(len(y))}


def train(df: pd.DataFrame) -> dict:
    rows = training_rows(df)
    tr = rows[rows["split"] == "train"]
    models = {"gradient_boosting": gradient_boosting(), "logistic": logistic()}
    metrics = {}
    for name, model in models.items():
        model.fit(tr[FEATURES], tr["fail"])
        metrics[name] = {s: evaluate(part["fail"].to_numpy(), model.predict_proba(part[FEATURES])[:, 1])
                         for s, part in rows.groupby("split") if s != "train"}
    MODELS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(models["gradient_boosting"], fh)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"canvass inspections: train {len(tr):,}, " + ", ".join(
        f"{s} {n:,}" for s, n in rows["split"].value_counts().items() if s != "train"))
    print(f"{'model':<18} {'split':<6} {'AUC':>6} {'fail rate top 10%':>18} {'fail rate all':>14}")
    for name, by_split in metrics.items():
        for s, m in by_split.items():
            print(f"{name:<18} {s:<6} {m['auc']:6.3f} {m['fail_rate_top10']:18.1%} {m['fail_rate_all']:14.1%}")
    return metrics


def load_model() -> HistGradientBoostingClassifier:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found, run `insp train` first")
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
