"""Prediction module: probability that a routine inspection fails."""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import fairness
from .city import CityConfig
from .data import load_socio
from .features import CATEGORICAL, build_features, feature_list, routine_outcomes

# The earliest years only serve as history for the features.
SPLITS = {"train": ("2015-01-01", "2023-12-31"), "valid": ("2024-01-01", "2024-12-31"),
          "test": ("2025-01-01", "2099-12-31")}
# Keep coordinates only if they buy more than this much validation AUC: location can stand in
# for a neighbourhood's income, so we drop it unless it clearly earns its place.
LOCATION_MIN_GAIN = 0.01


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrated:
    """A fitted classifier whose probabilities are remapped by Platt scaling.

    The decision module weighs the chance of failing against other criteria, so a score of 0.3
    should mean roughly 30% of such places fail. The map (a logistic regression on the model's
    log-odds) is fitted on the validation year, which the classifier never trained on. Platt
    scaling is smooth, so unlike isotonic regression it never claims 0% or 100%.
    """

    def __init__(self, model, platt: LogisticRegression, features: list[str]):
        self.model, self.platt, self.features = model, platt, features

    def predict_proba(self, X) -> np.ndarray:
        return self.platt.predict_proba(_logit(self.model.predict_proba(X[self.features])[:, 1]))


def paths(city: CityConfig) -> dict:
    d = city.models_dir
    return {"model": d / "risk_model.pkl", "metrics": d / "metrics.json", "importance": d / "importance.json"}


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = build_features(df, routine_outcomes(df))
    rows["split"] = None
    for name, (lo, hi) in SPLITS.items():
        rows.loc[rows["date"].between(lo, hi), "split"] = name
    return rows.dropna(subset=["split"])


def gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, min_samples_leaf=200,
                                          categorical_features="from_dtype", random_state=0)


def logistic(features: list[str]) -> object:
    numeric = [f for f in features if f not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=1000))


def evaluate(y: np.ndarray, p: np.ndarray, top: float = 0.1) -> dict:
    """AUC and Brier score, plus how often the top 10% by predicted risk actually failed."""
    k = max(1, int(len(p) * top))
    top_idx = np.argsort(-p)[:k]
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "fail_rate_top10": float(y[top_idx].mean()), "fail_rate_all": float(y.mean()), "n": int(len(y))}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Mean predicted vs observed fail rate in equal-size bins of predicted risk."""
    order = np.argsort(p)
    return [{"predicted": float(p[i].mean()), "observed": float(y[i].mean()), "n": int(len(i))}
            for i in np.array_split(order, bins)]


def _variant(tr, va, te, location: bool, socio: pd.DataFrame) -> dict:
    feats = feature_list(location)
    gb = gradient_boosting().fit(tr[feats], tr["fail"])
    lr = logistic(feats).fit(tr[feats], tr["fail"])
    platt = LogisticRegression().fit(_logit(gb.predict_proba(va[feats])[:, 1]), va["fail"])
    final = Calibrated(gb, platt, feats)
    p_te = final.predict_proba(te)[:, 1]
    audit = fairness.audit_model(fairness.with_quintile(te, socio), p_te)
    return {
        "model": final, "features": feats,
        "metrics": {
            "gradient_boosting": {s: evaluate(d["fail"].to_numpy(), gb.predict_proba(d[feats])[:, 1])
                                  for s, d in (("valid", va), ("test", te))},
            "logistic": {s: evaluate(d["fail"].to_numpy(), lr.predict_proba(d[feats])[:, 1])
                         for s, d in (("valid", va), ("test", te))},
            "calibrated": {"test": evaluate(te["fail"].to_numpy(), p_te)},
            "reliability": reliability(te["fail"].to_numpy(), p_te),
        },
        "fairness": audit.to_dict("records"), "fairness_summary": fairness.summary(audit),
    }


def train(df: pd.DataFrame, city: CityConfig) -> dict:
    rows = training_rows(df)
    tr, va, te = (rows[rows["split"] == s] for s in ("train", "valid", "test"))
    socio = load_socio(city)
    variants = {"with_location": _variant(tr, va, te, True, socio),
                "without_location": _variant(tr, va, te, False, socio)}
    gain = (variants["with_location"]["metrics"]["gradient_boosting"]["valid"]["auc"]
            - variants["without_location"]["metrics"]["gradient_boosting"]["valid"]["auc"])
    chosen = "with_location" if gain > LOCATION_MIN_GAIN else "without_location"
    final = variants[chosen]["model"]

    sample = te.sample(min(len(te), 15000), random_state=0)
    imp = permutation_importance(final.model, sample[final.features], sample["fail"], scoring="roc_auc",
                                 n_repeats=3, random_state=0)
    importance = sorted(({"feature": f, "auc_drop": float(m), "std": float(s)}
                         for f, m, s in zip(final.features, imp.importances_mean, imp.importances_std)),
                        key=lambda r: -r["auc_drop"])

    p = paths(city)
    city.models_dir.mkdir(parents=True, exist_ok=True)
    with open(p["model"], "wb") as fh:
        pickle.dump(final, fh)
    metrics = {"chosen": chosen, "location_auc_gain": gain, "counts": {"train": len(tr), "valid": len(va), "test": len(te)},
               **{k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in variants.items()}}
    p["metrics"].write_text(json.dumps(metrics, indent=2, default=float))
    p["importance"].write_text(json.dumps(importance, indent=2))

    print(f"routine inspections: train {len(tr):,}, valid {len(va):,}, test {len(te):,}")
    for name, v in variants.items():
        m = v["metrics"]
        print(f"[{name}] GB AUC valid {m['gradient_boosting']['valid']['auc']:.3f} test {m['gradient_boosting']['test']['auc']:.3f}"
              f" | LR test {m['logistic']['test']['auc']:.3f} | calibrated Brier {m['calibrated']['test']['brier']:.3f}"
              f" | fairness {v['fairness_summary']}")
    print(f"location adds {gain:+.3f} validation AUC -> using {chosen}")
    a = pd.DataFrame(variants[chosen]["fairness"]).round(3)
    print(a[["income_quintile", "n", "fail_rate", "mean_predicted", "auc", "share_of_fails", "share_of_flagged",
             "burden_ratio", "false_alarm_rate"]].to_string(index=False))
    print("top features:", ", ".join(f"{r['feature']} {r['auc_drop']:.3f}" for r in importance[:6]))
    return metrics


def load_model(city: CityConfig) -> Calibrated:
    path = paths(city)["model"]
    if not path.exists():
        raise SystemExit(f"{path} not found, run `insp train` first")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_metrics(city: CityConfig) -> dict:
    p = paths(city)["metrics"]
    return json.loads(p.read_text()) if p.exists() else {}


def load_importance(city: CityConfig) -> list[dict]:
    p = paths(city)["importance"]
    return json.loads(p.read_text()) if p.exists() else []
