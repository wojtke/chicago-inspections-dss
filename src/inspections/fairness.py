"""Does the system treat poorer neighbourhoods differently?

Two ways a ranking like ours can hurt a poorer area:
- over-inspection: its places are flagged far more often than they actually fail
  (`burden_ratio` = share of flagged places / share of failing places, 1 = proportional);
- under-protection: its failing places are found later than elsewhere (`days_sooner` per group).
We also check the model itself per group: is it as accurate (AUC) and as honest (predicted vs
observed fail rate) for poor areas as for rich ones? Groups are income quintiles of the
neighbourhoods, 1 = poorest.
"""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def with_quintile(rows: pd.DataFrame, socio: pd.DataFrame) -> pd.DataFrame:
    q = socio.set_index("area")["income_quintile"]
    out = rows.copy()
    out["income_quintile"] = out["area"].map(q)
    return out


def audit_model(rows: pd.DataFrame, p: np.ndarray, top: float = 0.1) -> pd.DataFrame:
    """rows need fail and income_quintile; p is the predicted chance of failing."""
    d = rows[["fail", "income_quintile"]].copy()
    d["p"] = p
    d["flagged"] = d["p"] >= np.quantile(d["p"], 1 - top)
    d = d.dropna(subset=["income_quintile"])
    out = []
    for q, g in d.groupby("income_quintile"):
        out.append({
            "income_quintile": int(q), "n": len(g),
            "fail_rate": g["fail"].mean(), "mean_predicted": g["p"].mean(),
            "auc": roc_auc_score(g["fail"], g["p"]) if g["fail"].nunique() == 2 else np.nan,
            "share_of_fails": g["fail"].sum() / d["fail"].sum(),
            "share_of_flagged": g["flagged"].sum() / d["flagged"].sum(),
            "false_alarm_rate": 1 - g.loc[g["flagged"], "fail"].mean() if g["flagged"].any() else np.nan,
        })
    a = pd.DataFrame(out)
    a["calibration_gap"] = a["mean_predicted"] - a["fail_rate"]
    a["burden_ratio"] = a["share_of_flagged"] / a["share_of_fails"]
    return a


def audit_plan(rows: pd.DataFrame, assigned_day: np.ndarray, days: int) -> pd.DataFrame:
    """Per income quintile: share of early visits vs share of failures, and days sooner for fails.

    rows need fail, actual_day and income_quintile; assigned_day comes from a policy.
    """
    d = rows[["fail", "actual_day", "income_quintile"]].copy()
    d["day"] = assigned_day
    d["early"] = d["day"] < days / 2
    d = d.dropna(subset=["income_quintile"])
    out = []
    for q, g in d.groupby("income_quintile"):
        f = g[g["fail"] == 1]
        out.append({
            "income_quintile": int(q),
            "share_of_early_visits": g["early"].sum() / max(1, d["early"].sum()),
            "share_of_fails": g["fail"].sum() / max(1, d["fail"].sum()),
            "days_sooner": (f["actual_day"] - f["day"]).mean() if len(f) else np.nan,
        })
    a = pd.DataFrame(out)
    a["burden_ratio"] = a["share_of_early_visits"] / a["share_of_fails"]
    return a


def balanced_order(priority: np.ndarray, p_fail: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Visit order that shares out early visits in proportion to each group's *expected* failures.

    At every point of the list, each group (income quintile) has had about as many visits as
    its share of the predicted failures (sum of p_fail) says; inside a group places go by
    priority. The model is calibrated within each quintile, so expected failures are a fair
    stand-in for actual ones, which nobody knows before the visit.
    """
    group = np.where(pd.isna(group), -1, group).astype(int)
    labels = np.unique(group)
    share = np.array([p_fail[group == g].sum() for g in labels]) / p_fail.sum()
    queues = [deque(np.flatnonzero(group == g)[np.argsort(-priority[group == g], kind="stable")]) for g in labels]
    taken = np.zeros(len(labels))
    order = []
    for k in range(len(priority)):
        deficit = share * (k + 1) - taken
        deficit[[len(q) == 0 for q in queues]] = -np.inf
        g = int(np.argmax(deficit))
        order.append(queues[g].popleft())
        taken[g] += 1
    return np.array(order, dtype=int)


def summary(audit: pd.DataFrame) -> dict:
    """The few numbers to watch: the spread of burden ratios and of calibration gaps."""
    s = {"burden_min": float(audit["burden_ratio"].min()), "burden_max": float(audit["burden_ratio"].max())}
    if "calibration_gap" in audit:
        s["max_abs_calibration_gap"] = float(audit["calibration_gap"].abs().max())
    if "days_sooner" in audit:
        s["days_sooner_min"] = float(audit["days_sooner"].min())
        s["days_sooner_max"] = float(audit["days_sooner"].max())
    return s
