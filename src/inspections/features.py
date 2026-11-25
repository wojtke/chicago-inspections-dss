"""Features for the risk model, built only from what was known before a given date.

Every feature of a row uses the place's inspections strictly before that row's `as_of` date
(same-day inspections are excluded), so nothing from the visit being predicted or later can
leak in. Works on the canonical schema, so on any city.
"""
import numpy as np
import pandas as pd

from .city import GROUPS

HISTORY = ["risk", "days_since_last", "days_since_first", "n_past", "n_past_fails", "past_fail_rate",
           "last_failed", "last_n_violations", "n_past_complaints", "month"]
LOCATION = ["lat", "lon"]
CATEGORICAL = ["facility_group"]
# Plain-Polish names of the features, for the report and the app.
FEATURES_PL = {"past_fail_rate": "odsetek wcześniejszych niezaliczeń", "last_n_violations": "naruszenia przy ostatniej kontroli",
               "days_since_last": "dni od ostatniej kontroli", "days_since_first": "staż lokalu", "month": "miesiąc",
               "facility_group": "grupa lokalu", "risk": "kategoria ryzyka miasta", "n_past": "liczba kontroli",
               "n_past_fails": "liczba niezaliczeń", "last_failed": "ostatnia kontrola niezaliczona",
               "n_past_complaints": "kontrole po skargach", "lat": "szer. geogr.", "lon": "dł. geogr."}


def feature_list(location: bool = True) -> list[str]:
    """The model's inputs; `location=False` drops coordinates (see the fairness audit)."""
    return HISTORY + (LOCATION if location else []) + CATEGORICAL


def routine_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Routine visits with a real pass/fail outcome: the visits the planner can reorder."""
    rows = df[(df["kind"] == "routine") & df["outcome"].notna()].copy()
    rows["fail"] = (rows["outcome"] == "fail").astype(int)
    return rows


def _history(events: pd.DataFrame) -> pd.DataFrame:
    """Running totals per place, one row per inspection, including that inspection."""
    e = events.sort_values(["place_id", "date", "inspection_id"])[
        ["place_id", "date", "outcome", "complaint", "closed", "n_violations"]].copy()
    g = e.groupby("place_id")
    is_fail = (e["outcome"] == "fail").astype(int)
    e["n_past"] = g.cumcount() + 1
    e["n_past_fails"] = is_fail.groupby(e["place_id"]).cumsum()
    e["n_outcomes"] = e["outcome"].notna().astype(int).groupby(e["place_id"]).cumsum()
    e["n_past_complaints"] = e["complaint"].astype(int).groupby(e["place_id"]).cumsum()
    e["first_date"] = g["date"].transform("min")
    e["last_failed"] = is_fail
    e["last_closed"] = e["closed"].astype(int)
    e["last_n_violations"] = e["n_violations"]
    e = e.rename(columns={"date": "last_date"})
    return e[["place_id", "last_date", "first_date", "n_past", "n_past_fails", "n_outcomes",
              "n_past_complaints", "last_failed", "last_closed", "last_n_violations"]]


def build_features(events: pd.DataFrame, targets: pd.DataFrame,
                   as_of: pd.Series | pd.Timestamp | None = None) -> pd.DataFrame:
    """Add model features to `targets`.

    events  all inspections (any kind), the history to learn from
    targets rows to score: must have place_id, risk, facility_group, lat, lon
    as_of   the date each target is scored on; defaults to the target's own inspection date
    """
    t = targets.copy()
    t["as_of"] = t["date"] if as_of is None else as_of
    # One time unit on both sides: merge_asof refuses to compare e.g. datetime64[s] with [us].
    t["as_of"] = pd.to_datetime(t["as_of"]).astype("datetime64[ns]")
    t["_order"] = np.arange(len(t))
    # Stable sort keeps same-day inspections in id order, so merge_asof picks the day's last running total.
    hist = _history(events).sort_values("last_date", kind="stable")
    hist["last_date"] = hist["last_date"].astype("datetime64[ns]")
    hist["first_date"] = hist["first_date"].astype("datetime64[ns]")
    # For each target, take the place's running totals at its last inspection strictly before as_of.
    m = pd.merge_asof(t.sort_values("as_of"), hist, left_on="as_of", right_on="last_date",
                      by="place_id", allow_exact_matches=False, direction="backward")
    m = m.sort_values("_order").set_index(t.index)

    out = t.drop(columns="_order")
    out["days_since_last"] = (m["as_of"] - m["last_date"]).dt.days
    out["days_since_first"] = (m["as_of"] - m["first_date"]).dt.days
    for col in ["n_past", "n_past_fails", "n_past_complaints"]:
        out[col] = m[col].fillna(0)
    out["past_fail_rate"] = m["n_past_fails"] / m["n_outcomes"].where(m["n_outcomes"] > 0)
    out["last_failed"] = m["last_failed"]
    out["last_closed"] = m["last_closed"]
    out["last_n_violations"] = m["last_n_violations"]
    out["month"] = out["as_of"].dt.month
    out["facility_group"] = pd.Categorical(out["facility_group"], categories=GROUPS)
    return out
