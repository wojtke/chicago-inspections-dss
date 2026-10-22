"""Decision module: which places to inspect on which day, and a replay to score any order.

The replay follows the city's own 2014 evaluation. Take the routine inspections actually done
in a window, keep the number done each day, and change only the order. Then count how much
sooner the failing places would have been reached.
"""
import numpy as np
import pandas as pd

from .features import FEATURES, build_features, canvass_outcomes


def pool(df: pd.DataFrame, start: str | pd.Timestamp, days: int = 60) -> pd.DataFrame:
    """Routine inspections done in [start, start + days), first visit per licence, scored as of `start`."""
    start = pd.Timestamp(start)
    rows = canvass_outcomes(df)
    rows = rows[(rows["date"] >= start) & (rows["date"] < start + pd.Timedelta(days=days))]
    rows = rows.sort_values("date").drop_duplicates("license")
    rows = build_features(df, rows, as_of=start)
    rows["actual_day"] = (rows["date"] - start).dt.days
    return rows.reset_index(drop=True)


def score(model, rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["p_fail"] = model.predict_proba(rows[FEATURES])[:, 1]
    return rows


def capacity(rows: pd.DataFrame, days: int) -> np.ndarray:
    """How many inspections were actually done on each day of the window."""
    return np.bincount(rows["actual_day"], minlength=days)


def assign(order: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Give each row a day: rows are taken in `order` and fill each day's capacity in turn."""
    slots = np.repeat(np.arange(len(cap)), cap)
    day = np.empty(len(order), dtype=int)
    day[order] = slots[: len(order)]
    return day


def schedules(rows: pd.DataFrame, days: int, n_random: int = 20, seed: int = 0) -> dict[str, list[np.ndarray]]:
    """Assigned day per row for each policy (random has several draws)."""
    cap = capacity(rows, days)
    rng = np.random.default_rng(seed)
    return {
        "actual": [rows["actual_day"].to_numpy()],
        "random": [assign(rng.permutation(len(rows)), cap) for _ in range(n_random)],
        "risk_first": [assign(np.argsort(-rows["p_fail"].to_numpy(), kind="stable"), cap)],
    }


def backtest(rows: pd.DataFrame, days: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary per policy and the cumulative number of failures found by each day."""
    fail = rows["fail"].to_numpy() == 1
    actual = rows["actual_day"].to_numpy()
    summary, curves = [], {}
    for policy, draws in schedules(rows, days).items():
        sooner = np.mean([(actual[fail] - d[fail]).mean() for d in draws])
        first_half = np.mean([(d[fail] < days / 2).mean() for d in draws])
        curves[policy] = np.mean([np.cumsum(np.bincount(d[fail], minlength=days)) for d in draws], axis=0)
        summary.append({"policy": policy, "days_sooner": sooner, "fails_in_first_half": first_half})
    summary = pd.DataFrame(summary).set_index("policy")
    summary.attrs = {"inspections": len(rows), "fails": int(fail.sum())}
    return summary, pd.DataFrame(curves).rename_axis("day")


def daily_list(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    """The top-k places by predicted risk: the list for the first day of the window."""
    return rows.sort_values("p_fail", ascending=False).head(k)
