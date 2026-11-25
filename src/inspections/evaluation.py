"""Evaluation: replay past windows to score any visiting policy, the way the city tested its own model.

Take the routine inspections actually done in a 60-day window. Keep each day's total workload,
let the mandatory visits of that day (complaints, licences, re-inspections) take their share
first, and give what is left to routine visits. Then change only the order in which the routine
places are visited, and count how much sooner the failing ones are reached. A scenario
multiplies the workload (e.g. 0.8 = a fifth fewer inspectors); places a policy never reaches
count as reached the day after the window ends.
"""
import numpy as np
import pandas as pd

from . import fairness
from .city import CityConfig
from .data import load, load_socio
from .features import build_features, routine_outcomes
from .model import load_model
from .preferences import Preferences, preferences, priority
from .routing import RoutingConfig, route_km, solve, vehicle_caps, xy_km

WINDOWS = ["2025-01-06", "2025-04-01", "2025-09-01", "2026-03-02", "2026-07-01"]
SCENARIOS = [1.0, 0.9, 0.8, 0.7]
POLICIES = ["actual", "random", "risk", "ahp", "ahp_fair", "system"]
POLICIES_PL = {"actual": "Kolejność miasta", "random": "Losowa", "risk": "Tylko ryzyko",
               "ahp": "Ryzyko + preferencje (AHP)", "ahp_fair": "AHP + równoważenie dzielnic",
               "system": "Pełny system (AHP + równoważenie + trasy)"}
DAYS = 60


SNAPSHOT = ["name", "facility_type", "facility_group", "risk", "address", "lat", "lon", "area"]
KM_DRAWS = 5  # random order: km are averaged over this many draws (routing every draw is slow)


def _window(df: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """First routine visit with an outcome per place in [start, start + days)."""
    rows = routine_outcomes(df)
    rows = rows[(rows["date"] >= start) & (rows["date"] < start + pd.Timedelta(days=days))]
    return rows.sort_values("date", kind="stable").drop_duplicates("place_id")


def pool(df: pd.DataFrame, start: str | pd.Timestamp, days: int = DAYS) -> pd.DataFrame:
    """The routine visits of the window that a planner could have reordered on day `start`.

    Only places already known before `start` qualify, and they are described as they were then
    (risk category, type, location from their latest earlier record); features are as of `start`.
    Visits to places first seen inside the window stay where they were (see daily_load).
    """
    start = pd.Timestamp(start)
    rows = _window(df, start, days)
    known = df[df["date"] < start].sort_values("date", kind="stable").drop_duplicates("place_id", keep="last")
    known = known.set_index("place_id")[SNAPSHOT]
    rows = rows[rows["place_id"].isin(known.index)].drop(columns=SNAPSHOT)
    rows = rows.join(known, on="place_id").dropna(subset=["lat", "lon"])
    rows = build_features(df, rows, as_of=start)
    rows["actual_day"] = (rows["date"] - start).dt.days
    return rows.reset_index(drop=True)


def score(model, rows: pd.DataFrame, prefs: Preferences | None = None) -> pd.DataFrame:
    rows = rows.copy()
    rows["p_fail"] = model.predict_proba(rows)[:, 1]
    return priority(rows, prefs)


def daily_load(df: pd.DataFrame, rows: pd.DataFrame, start: str | pd.Timestamp, days: int = DAYS) -> pd.DataFrame:
    """Per day of the window: visits that cannot move (mandatory ones, and routine visits to places
    nobody knew at the start), routine visits in the pool, and their total."""
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=days)
    m = df[(df["kind"] == "mandatory") & (df["date"] >= start) & (df["date"] < end)]
    w = _window(df, start, days)
    new = w[~w["place_id"].isin(rows["place_id"])]
    fixed = np.bincount((m["date"] - start).dt.days, minlength=days)[:days] + \
        np.bincount((new["date"] - start).dt.days, minlength=days)[:days]
    out = pd.DataFrame({"fixed": fixed, "routine": np.bincount(rows["actual_day"], minlength=days)[:days]})
    out["total"] = out["fixed"] + out["routine"]
    return out


def routine_capacity(load: pd.DataFrame, multiplier: float = 1.0) -> np.ndarray:
    """Visits that cannot move go first; routine visits get what is left of the (scaled) day."""
    return np.maximum(0, np.rint(load["total"] * multiplier).astype(int) - load["fixed"]).to_numpy()


def assign(order: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Rows taken in `order` fill each day's capacity in turn; rows left over get day = len(cap)."""
    slots = np.repeat(np.arange(len(cap)), cap)
    day = np.full(len(order), len(cap), dtype=int)
    k = min(len(order), len(slots))
    day[order[:k]] = slots[:k]
    return day


def route_days(rows: pd.DataFrame, cap: np.ndarray, cfg: RoutingConfig, xy: np.ndarray,
               order: np.ndarray) -> np.ndarray:
    """Day by day, the router picks and orders each day's routine visits (prize = AHP priority)
    from the next candidates in `order` (the balanced ranking)."""
    day = np.full(len(rows), len(cap), dtype=int)
    prio = rows["priority"].to_numpy()
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    left = np.ones(len(rows), dtype=bool)
    for d, c in enumerate(cap):
        if c == 0 or not left.any():
            continue
        cand = np.flatnonzero(left)
        cand = cand[np.argsort(rank[cand], kind="stable")][: max(cfg.candidates, c)]
        routes = solve(xy[cand], vehicle_caps(int(min(c, len(cand))), cfg.visits_per_inspector), cfg, prize=prio[cand])
        visited = cand[[i for r in routes for i in r]]
        day[visited] = d
        left[visited] = False
    return day


def schedules(rows: pd.DataFrame, cap: np.ndarray, cfg: RoutingConfig, xy: np.ndarray,
              n_random: int = 20, seed: int = 0, route: bool = True) -> dict[str, list[np.ndarray]]:
    """Assigned day per row for each policy (random has several draws)."""
    rng = np.random.default_rng(seed)
    balanced = fairness.balanced_order(rows["priority"].to_numpy(), rows["p_fail"].to_numpy(),
                                       rows["income_quintile"].to_numpy())
    out = {
        "actual": [assign(np.argsort(rows["actual_day"].to_numpy(), kind="stable"), cap)],
        "random": [assign(rng.permutation(len(rows)), cap) for _ in range(n_random)],
        "risk": [assign(np.argsort(-rows["p_fail"].to_numpy(), kind="stable"), cap)],
        "ahp": [assign(np.argsort(-rows["priority"].to_numpy(), kind="stable"), cap)],
        "ahp_fair": [assign(balanced, cap)],
    }
    if route:
        out["system"] = [route_days(rows, cap, cfg, xy, balanced)]
    return out


def km_per_visit(day: np.ndarray, cap: np.ndarray, cfg: RoutingConfig, xy: np.ndarray) -> float:
    """Drive the same quick router over each day's routine list, whatever policy chose it.

    A like-for-like yardstick for comparing orders, not real mileage: mandatory stops are left
    out, the day goes to the fewest inspectors, and the 8-hour limit is not applied.
    """
    km, visits = 0.0, 0
    for d in range(len(cap)):
        idx = np.flatnonzero(day == d)
        if len(idx) < 2:
            continue
        routes = solve(xy[idx], vehicle_caps(len(idx), cfg.visits_per_inspector), cfg, optimise=False)
        km += route_km(xy[idx], routes)
        visits += len(idx)
    return km / max(visits, 1)


def metrics(rows: pd.DataFrame, draws: list[np.ndarray], days: int) -> dict:
    fail = rows["fail"].to_numpy() == 1
    high = fail & rows["facility_group"].isin(["children", "healthcare"]).to_numpy()
    actual = rows["actual_day"].to_numpy()
    return {
        "days_sooner": float(np.mean([(actual[fail] - d[fail]).mean() for d in draws])),
        "fails_first_half": float(np.mean([(d[fail] < days / 2).mean() for d in draws])),
        "fails_reached": float(np.mean([(d[fail] < days).mean() for d in draws])),
        "high_impact_days_sooner": float(np.mean([(actual[high] - d[high]).mean() for d in draws])) if high.any() else np.nan,
    }


def evaluate_window(df: pd.DataFrame, model, city: CityConfig, start: str, multiplier: float,
                    prefs: Preferences, socio: pd.DataFrame, cfg: RoutingConfig, days: int = DAYS) -> dict:
    rows = fairness.with_quintile(score(model, pool(df, start, days), prefs), socio)
    load_ = daily_load(df, rows, start, days)
    cap = routine_capacity(load_, multiplier)
    xy = xy_km(rows["lon"], rows["lat"], city.center_lat)
    draws = schedules(rows, cap, cfg, xy)
    summary, curves, areas, quint = [], [], [], []
    fail = rows["fail"].to_numpy() == 1
    for policy, ds in draws.items():
        m = metrics(rows, ds, days)
        m["km_per_visit"] = float(np.mean([km_per_visit(d, cap, cfg, xy) for d in ds[:KM_DRAWS]]))
        summary.append({"window": start, "multiplier": multiplier, "policy": policy,
                        "inspections": len(rows), "fails": int(fail.sum()), "capacity": int(cap.sum()), **m})
        curve = np.mean([np.cumsum(np.bincount(np.minimum(d[fail], days), minlength=days + 1)[:days]) for d in ds], axis=0)
        curves += [{"window": start, "multiplier": multiplier, "policy": policy, "day": i, "found": v}
                   for i, v in enumerate(curve)]
        sooner = np.mean([rows["actual_day"].to_numpy() - d for d in ds], axis=0)
        per_area = pd.DataFrame({"area": rows["area"], "sooner": sooner})[fail]
        for a, g in per_area.groupby("area"):
            areas.append({"window": start, "multiplier": multiplier, "policy": policy, "area": int(a),
                          "fails": len(g), "days_sooner": float(g["sooner"].mean())})
        q = pd.concat([fairness.audit_plan(rows, d, days) for d in ds]).groupby("income_quintile", as_index=False).mean()
        quint += [{"window": start, "multiplier": multiplier, "policy": policy, **r} for r in q.to_dict("records")]
    return {"summary": summary, "curves": curves, "areas": areas, "quintiles": quint}


def run(city: CityConfig, quick: bool = False, prefs: Preferences | None = None,
        cfg: RoutingConfig | None = None) -> dict[str, pd.DataFrame]:
    df, model, socio = load(city), load_model(city), load_socio(city)
    prefs, cfg = prefs or preferences(), cfg or RoutingConfig()
    windows, scenarios = (WINDOWS[2:3], [1.0]) if quick else (WINDOWS, SCENARIOS)
    parts = {"summary": [], "curves": [], "areas": [], "quintiles": []}
    for start in windows:
        for mult in scenarios:
            r = evaluate_window(df, model, city, start, mult, prefs, socio, cfg)
            for k in parts:
                parts[k] += r[k]
            s = pd.DataFrame(r["summary"]).set_index("policy")
            print(f"{start} x{mult}: " + " | ".join(
                f"{p} {v['days_sooner']:+.1f}d {v['fails_first_half']:.0%} {v['km_per_visit']:.1f}km"
                for p, v in s.iterrows()), flush=True)
    out = {k: pd.DataFrame(v) for k, v in parts.items()}
    city.results_dir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(city.results_dir / f"{k}.parquet", index=False)
    print(f"results -> {city.results_dir}")
    return out


def load_results(city: CityConfig) -> dict[str, pd.DataFrame]:
    d = city.results_dir
    return {k: pd.read_parquet(d / f"{k}.parquet") for k in ("summary", "curves", "areas", "quintiles")
            if (d / f"{k}.parquet").exists()}
