"""Features for the risk model, built only from what was known before a given date.

Every feature of a row uses the establishment's inspections strictly before that row's
`as_of` date (same-day inspections are excluded), so nothing from the visit being predicted
or later can leak in.
"""
import numpy as np
import pandas as pd

OUTCOMES = ["Pass", "Pass w/ Conditions", "Fail"]
TOP_FACILITIES = ["Restaurant", "Grocery Store", "School", "Children's Services Facility", "Bakery", "Daycare (2 - 6 Years)"]

NUMERIC = ["risk", "days_since_last", "days_since_first", "n_past", "n_past_fails", "past_fail_rate",
           "last_failed", "last_n_violations", "n_past_complaints", "lat", "lon", "month"]
CATEGORICAL = ["facility"]
FEATURES = NUMERIC + CATEGORICAL


def canvass_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Routine (canvass) inspections with a real pass/fail outcome: the visits the city can reorder."""
    rows = df[(df["inspection_type"] == "Canvass") & df["result"].isin(OUTCOMES)].copy()
    rows["fail"] = (rows["result"] == "Fail").astype(int)
    return rows


def _history(events: pd.DataFrame) -> pd.DataFrame:
    """Running totals per licence, one row per inspection, including that inspection."""
    e = events.sort_values(["license", "date", "inspection_id"])[
        ["license", "date", "inspection_type", "result", "n_violations"]].copy()
    g = e.groupby("license")
    is_fail = (e["result"] == "Fail").astype(int)
    e["n_past"] = g.cumcount() + 1
    e["n_past_fails"] = is_fail.groupby(e["license"]).cumsum()
    e["n_outcomes"] = e["result"].isin(OUTCOMES).astype(int).groupby(e["license"]).cumsum()
    is_complaint = e["inspection_type"].str.contains("Complaint|Food Poisoning", regex=True).astype(int)
    e["n_past_complaints"] = is_complaint.groupby(e["license"]).cumsum()
    e["first_date"] = g["date"].transform("min")
    e["last_failed"] = is_fail
    e["last_n_violations"] = e["n_violations"]
    e = e.rename(columns={"date": "last_date"})
    return e[["license", "last_date", "first_date", "n_past", "n_past_fails", "n_outcomes",
              "n_past_complaints", "last_failed", "last_n_violations"]]


def build_features(events: pd.DataFrame, targets: pd.DataFrame, as_of: pd.Series | pd.Timestamp | None = None) -> pd.DataFrame:
    """Add model features to `targets`.

    events  all inspections (any type), the history to learn from
    targets rows to score: must have license, risk, facility_type, lat, lon
    as_of   the date each target is scored on; defaults to the target's own inspection date
    """
    t = targets.copy()
    t["as_of"] = t["date"] if as_of is None else as_of
    # One time unit on both sides: merge_asof refuses to compare e.g. datetime64[s] with [us].
    t["as_of"] = pd.to_datetime(t["as_of"]).astype("datetime64[ns]")
    t["_order"] = np.arange(len(t))
    hist = _history(events).sort_values("last_date")
    hist["last_date"] = hist["last_date"].astype("datetime64[ns]")
    hist["first_date"] = hist["first_date"].astype("datetime64[ns]")
    # For each target, take the licence's running totals at its last inspection strictly before as_of.
    m = pd.merge_asof(t.sort_values("as_of"), hist, left_on="as_of", right_on="last_date",
                      by="license", allow_exact_matches=False, direction="backward")
    m = m.sort_values("_order").set_index(t.index)

    out = t.drop(columns="_order")
    out["days_since_last"] = (m["as_of"] - m["last_date"]).dt.days
    out["days_since_first"] = (m["as_of"] - m["first_date"]).dt.days
    for col in ["n_past", "n_past_fails", "n_past_complaints"]:
        out[col] = m[col].fillna(0)
    out["past_fail_rate"] = m["n_past_fails"] / m["n_outcomes"].where(m["n_outcomes"] > 0)
    out["last_failed"] = m["last_failed"]
    out["last_n_violations"] = m["last_n_violations"]
    out["month"] = out["as_of"].dt.month
    out["facility"] = pd.Categorical(
        out["facility_type"].where(out["facility_type"].isin(TOP_FACILITIES), "Other"),
        categories=TOP_FACILITIES + ["Other"])
    return out


def reasons(row: pd.Series) -> str:
    """A short plain-English explanation of why a place is high on the list."""
    r = []
    if row.get("n_past", 0) == 0:
        r.append("no inspection history")
    if row.get("last_failed") == 1:
        r.append("failed last inspection")
    if row.get("n_past_fails", 0) >= 2:
        r.append(f"{int(row['n_past_fails'])} past fails")
    if pd.notna(row.get("days_since_last")) and row["days_since_last"] > 365:
        r.append(f"{int(row['days_since_last'])} days since last visit")
    if row.get("n_past_complaints", 0) >= 1:
        r.append(f"{int(row['n_past_complaints'])} complaint visit(s)")
    if row.get("risk") == 1:
        r.append("high-risk category")
    return ", ".join(r[:3])
