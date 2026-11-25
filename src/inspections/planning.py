"""Operations: the plan for one working day.

Unlike the replay, live planning starts from every place that is *due* for a routine visit,
not only from the places the city happened to visit. A place is due when its last completed
inspection is older than the frequency for its risk category and it was still open then (an
attempt with no entry does not restart the clock). The day's mandatory visits (complaints,
licences, re-inspections) are taken from the data as the queue that came in that day, plus
requests earlier approved plans did not get done and re-inspections after a failed visit; they
go into the routes first, routine places fill the rest. Visits the inspectors recorded a result
for are added to the history (add_field_visits), so later plans and the model's features take
them into account.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fairness
from .city import CityConfig
from .features import build_features
from .preferences import Preferences, explain, preferences, priority
from .routing import RoutingConfig, leg_km, solve, xy_km

ACTIVE_YEARS = 3          # a place unseen for longer is treated as gone
MANDATORY_PRIZE = 1000.0  # in priority units: the router drops a request only when it cannot fit at all
PINNED_PRIZE = 10.0       # a pinned place beats any unpinned one but never a request
CARRY_DAYS = 7            # requests an approved plan did not get done wait this long for a later one
REINSPECT_DAYS = 7        # a failed visit from a plan is re-inspected this many days later
FIELD_TYPE = "Field visit (DSS)"            # a routine visit from an approved plan
FIELD_REQUEST_TYPE = "Request visit (DSS)"  # a request handled on a later day than it came in
STOP_COLUMNS = ["inspector", "seq", "eta", "kind", "place_id", "place_ids", "name", "address", "facility_type",
                "facility_group", "p_fail", "priority", "why", "lat", "lon", "km_from_prev"]
DAY_START_MIN = 8 * 60


@dataclass
class DayPlan:
    day: pd.Timestamp
    routes: pd.DataFrame      # one row per stop: inspector, seq, kind, place, eta ...
    due: pd.DataFrame         # every due routine place with its priority (the manager's long list)
    queue: pd.DataFrame       # the day's mandatory requests
    unassigned: pd.DataFrame  # mandatory requests that did not fit
    km: float


def latest_places(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """One row per place seen in the last few years: its latest known name, type, address, risk."""
    past = df[(df["date"] < day) & (df["date"] >= day - pd.DateOffset(years=ACTIVE_YEARS))]
    cols = ["place_id", "name", "facility_type", "facility_group", "risk", "address", "lat", "lon", "area"]
    return past.sort_values("date").drop_duplicates("place_id", keep="last")[cols].reset_index(drop=True)


def add_field_visits(df: pd.DataFrame, visits: pd.DataFrame | None) -> pd.DataFrame:
    """Visits the inspectors recorded a result for, as rows of the inspection history.

    visits: place_id, day, outcome (pass / conditional / fail / closed / no_entry) and kind
    (rutynowa / zgłoszenie; routine when missing). Pass / conditional / fail feed the place's
    features like any city inspection; "closed" marks the place as gone; "no entry" is kept but
    does not restart the clock. A visit the city's data already has (the same place and day, e.g.
    a request the city handled that day) is not added twice: the recorded result replaces the
    city's on that row. Every row a result came from is marked `field`; `reinspect` marks failures
    the city's data does not already follow up (it holds its own re-inspections after its failures).
    """
    if visits is None or visits.empty:
        return df
    out = df.assign(field=False, reinspect=False) if "field" not in df else df.copy()
    last = df.sort_values("date", kind="stable").drop_duplicates("place_id", keep="last").set_index("place_id")
    v = visits[visits["place_id"].isin(last.index)].copy()
    v["date"] = pd.to_datetime(v["day"]).astype(df["date"].dtype)
    v = v.drop_duplicates(["place_id", "date"], keep="last").reset_index(drop=True)
    if v.empty:
        return out
    v["kept"] = v["outcome"].where(v["outcome"].isin(["pass", "conditional", "fail"]))
    v["text"] = v["outcome"].str.replace("_", " ").str.title()
    rows_key = pd.MultiIndex.from_frame(out[["place_id", "date"]])
    v_key = pd.MultiIndex.from_frame(v[["place_id", "date"]])
    same = rows_key.isin(v_key)
    if same.any():
        r = v.set_index(["place_id", "date"]).reindex(rows_key[same])
        out.loc[same, "reinspect"] = ((r["outcome"] == "fail").to_numpy() & (out.loc[same, "outcome"] != "fail").to_numpy())
        out.loc[same, "outcome"] = r["kept"].to_numpy()
        out.loc[same, "closed"] = (r["outcome"] == "closed").to_numpy()
        out.loc[same, "result"] = r["text"].to_numpy()
        out.loc[same, "field"] = True
    v = v[~v_key.isin(rows_key)].reset_index(drop=True)
    if v.empty:
        return out
    routine = v.get("kind", pd.Series("rutynowa", index=v.index)).fillna("rutynowa").eq("rutynowa").to_numpy()
    rows = last.loc[v["place_id"]].reset_index()
    rows = rows.assign(inspection_id=-(np.arange(len(v)) + 1), date=v["date"].to_numpy(),
                       inspection_type=np.where(routine, FIELD_TYPE, FIELD_REQUEST_TYPE),
                       kind=np.where(routine, "routine", "other"), complaint=False, outcome=v["kept"].to_numpy(),
                       closed=(v["outcome"] == "closed").to_numpy(), n_violations=np.nan,
                       result=v["text"].to_numpy(), field=True, reinspect=(v["outcome"] == "fail").to_numpy())
    return pd.concat([out, rows[out.columns]], ignore_index=True)


def due_places(df: pd.DataFrame, city: CityConfig, day: str | pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day)
    places = latest_places(df, day).dropna(subset=["lat", "lon"])
    places["date"] = day
    f = build_features(df, places, as_of=day)
    f = f[f["last_closed"] != 1]
    # The clock runs from the last completed inspection: a visit with an outcome, a closure, or a
    # visit from an approved plan. "No entry" and similar attempts do not restart it.
    done = df["outcome"].notna() | df["closed"].astype(bool)
    if "field" in df:
        done |= df["field"].fillna(False).astype(bool) & (df["result"] != "No Entry")
    last_done = df[done & (df["date"] < day)].groupby("place_id")["date"].max()
    days_since_done = (day - f["place_id"].map(last_done)).dt.days
    limit = f["risk"].map(city.due_days).fillna(max(city.due_days.values()))
    f["due_since_days"] = days_since_done.fillna(10 * max(city.due_days.values())) - limit
    return f[f["due_since_days"] >= 0].reset_index(drop=True)


REQUEST_LABELS = [  # (regex on the published inspection type, Polish label), first match wins
    (r"Food Poisoning", "podejrzenie zatrucia"), (r"Complaint.*Re-Inspection", "ponowna kontrola po skardze"),
    (r"License.*Re-Inspection", "ponowna kontrola licencyjna"), (r"Re-Inspection", "ponowna kontrola"),
    (r"Complaint", "skarga"), (r"License", "kontrola licencyjna"),
]


def request_label(inspection_type: str) -> str:
    for pattern, label in REQUEST_LABELS:
        if re.search(pattern, inspection_type, re.IGNORECASE):
            return label
    return inspection_type.lower()


def requests_of(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The mandatory requests that came in on `day`, one row per request."""
    q = df[(df["kind"] == "mandatory") & (df["date"] == day)].dropna(subset=["lat", "lon"]).copy()
    q["request"] = q["inspection_type"].map(request_label)
    q["requested"] = day
    return q


def _stop_key(q: pd.DataFrame) -> pd.Series:
    """Requests with the same name and address are one stop; rows without either stand alone."""
    name = q["name"].fillna("").str.upper().str.strip()
    addr = q["address"].fillna("").str.upper().str.strip()
    return (name + "|" + addr).where((name != "") | (addr != ""), "id:" + q["place_id"].astype(str))


def pending_requests(df: pd.DataFrame, day: str | pd.Timestamp, served: dict) -> pd.DataFrame:
    """Requests from the last CARRY_DAYS days that approved plans did not get done.

    served: {"YYYY-MM-DD": place_ids visited that day} for every day with an approved plan; a
    request counts as done once its visit has a result other than "no entry" (Store.served).
    A day without an approved plan changes nothing: the system did not plan it.
    """
    day = pd.Timestamp(day)
    pending = requests_of(df, day).iloc[0:0]
    for d in sorted(pd.Timestamp(x) for x in served):
        if not day - pd.Timedelta(days=CARRY_DAYS) <= d < day:
            continue
        q = pd.concat([pending, requests_of(df, d)], ignore_index=True)
        key = _stop_key(q)
        done = set(key[q["place_id"].isin(served[d.date().isoformat()])])
        pending = q[~key.isin(done)]
    return pending


def followups(df: pd.DataFrame, day: str | pd.Timestamp) -> pd.DataFrame:
    """Re-inspections owed after a failure recorded in the app that the city's data does not
    follow up itself (see add_field_visits): due REINSPECT_DAYS after the failure and pending until
    the place has been visited again. Shaped like requests_of."""
    day = pd.Timestamp(day)
    if "reinspect" not in df:
        return requests_of(df, day).iloc[0:0]
    past = df[df["date"] < day]
    fails = past[past["reinspect"].fillna(False).astype(bool)]
    fails = fails.sort_values("date").drop_duplicates("place_id", keep="last")
    later = past[past["outcome"].notna() | past["closed"].astype(bool)].groupby("place_id")["date"].max()
    owed = fails[(fails["date"] <= day - pd.Timedelta(days=REINSPECT_DAYS))
                 & (fails["place_id"].map(later) <= fails["date"])].copy()
    owed["request"] = "ponowna kontrola po niezaliczeniu " + owed["date"].dt.strftime("%d.%m")
    owed["requested"] = day
    return owed.reset_index(drop=True)


def _combine(labels: pd.Series) -> str:
    """'kontrola licencyjna ×3' for repeats, 'skarga + kontrola licencyjna' for a mix."""
    counts = labels.value_counts(sort=False)
    return " + ".join(lab if n == 1 else f"{lab} ×{n}" for lab, n in counts.items())


def mandatory_queue(df: pd.DataFrame, day: str | pd.Timestamp, carried: pd.DataFrame | None = None) -> pd.DataFrame:
    """The day's requests (and any carried over), one stop per establishment: several licences or
    a complaint and a licence check at one address are one visit. `place_ids` lists every licence
    the stop covers, so none of them is planned again as a routine visit."""
    day = pd.Timestamp(day)
    q = requests_of(df, day)
    q = pd.concat([x for x in (carried, followups(df, day), q) if x is not None and len(x)] or [q], ignore_index=True)
    if q.empty:
        return q.assign(place_ids=pd.Series(dtype=object))
    q = q.reset_index(drop=True)
    key = _stop_key(q)
    late = q["requested"] < day
    q.loc[late, "request"] = q.loc[late, "request"] + " · od " + q.loc[late, "requested"].dt.strftime("%d.%m")
    groups = q.groupby(key)
    q["place_ids"] = key.map(groups["place_id"].agg(lambda s: sorted({int(x) for x in s})))
    q["request"] = key.map(groups["request"].agg(_combine))
    return q[~key.duplicated()].reset_index(drop=True)


def _etas(xy: np.ndarray, route: list[int], cfg: RoutingConfig) -> list[int]:
    t, out = DAY_START_MIN, []
    for k, i in enumerate(route):
        if k:
            t += int(round(leg_km(xy, route[k - 1], i) / cfg.speed_kmh * 60))
        out.append(t)
        t += cfg.service_min
    return out


def score_due(df: pd.DataFrame, model, city: CityConfig, day: pd.Timestamp, prefs: Preferences,
              socio: pd.DataFrame | None = None, balance: bool = True) -> pd.DataFrame:
    """Every due place with its chance of failing, AHP priority, reason and rank on the long list.

    With `balance`, the rank shares early visits between income quintiles in proportion to their
    expected failures (see fairness.balanced_order); otherwise it is plain priority order.
    """
    due = due_places(df, city, day)
    due["p_fail"] = model.predict_proba(due)[:, 1] if len(due) else np.array([])
    due = priority(due, prefs)
    due["why"] = due.apply(explain, axis=1) if len(due) else pd.Series(dtype=str)
    if socio is not None and len(socio):
        due = fairness.with_quintile(due, socio)
    else:
        due["income_quintile"] = np.nan
    if balance and len(due) and due["income_quintile"].notna().any():
        order = fairness.balanced_order(due["priority"].to_numpy(), due["p_fail"].to_numpy(),
                                        due["income_quintile"].to_numpy())
    else:
        order = np.argsort(-due["priority"].to_numpy(), kind="stable")
    due["rank"] = 0
    due.loc[due.index[order], "rank"] = np.arange(1, len(due) + 1)
    return due.sort_values("rank").reset_index(drop=True)


def plan_day(df: pd.DataFrame, model, city: CityConfig, day: str | pd.Timestamp, inspectors: int | list[int],
             prefs: Preferences | None = None, cfg: RoutingConfig | None = None,
             removed: set | None = None, pinned: set | None = None,
             socio: pd.DataFrame | None = None, balance: bool = True, due: pd.DataFrame | None = None,
             carried: pd.DataFrame | None = None) -> DayPlan:
    """Routes for the inspectors: the mandatory queue first, then the most important due places.

    `inspectors` is a head count (routes numbered 1..n) or the team's inspector numbers, which
    then label the routes. Pass a precomputed `due` (from score_due) to skip re-scoring when only
    the team or the manager's edits change; `carried` are requests left over from earlier plans
    (pending_requests).
    """
    team = list(range(1, inspectors + 1)) if isinstance(inspectors, int) else [int(i) for i in inspectors]
    day, prefs, cfg = pd.Timestamp(day), prefs or preferences(), cfg or RoutingConfig()
    removed, pinned = set(removed or ()), set(pinned or ())
    if due is None:
        due = score_due(df, model, city, day, prefs, socio, balance)
    queue = mandatory_queue(df, day, carried)
    covered = {pid for ids in queue["place_ids"] for pid in ids}

    routine = due[~due["place_id"].isin(removed) & ~due["place_id"].isin(covered)]
    cand = pd.concat([routine[routine["place_id"].isin(pinned)],
                      routine[~routine["place_id"].isin(pinned)].head(cfg.candidates)])
    nodes = pd.concat([
        queue.assign(kind_pl="zgłoszenie", prize=MANDATORY_PRIZE, why=queue["request"]),
        cand.assign(kind_pl="rutynowa", prize=np.where(cand["place_id"].isin(pinned), PINNED_PRIZE, cand["priority"])),
    ], ignore_index=True)
    xy = xy_km(nodes["lon"], nodes["lat"], city.center_lat)
    routes = solve(xy, [cfg.visits_per_inspector] * len(team), cfg, prize=nodes["prize"].to_numpy())

    stops, km = [], 0.0
    for v, r in zip(team, routes):
        for seq, (i, eta) in enumerate(zip(r, _etas(xy, r, cfg)), start=1):
            leg = 0.0 if seq == 1 else leg_km(xy, r[seq - 2], i)
            km += leg
            n = nodes.iloc[i]
            ids = n["place_ids"] if isinstance(n.get("place_ids"), list) else [int(n["place_id"])]
            stops.append({"inspector": v, "seq": seq, "eta": f"{eta // 60:02d}:{eta % 60:02d}", "kind": n["kind_pl"],
                          "place_id": int(n["place_id"]), "place_ids": ids, "name": n["name"], "address": n["address"],
                          "facility_type": n["facility_type"], "facility_group": n["facility_group"],
                          "p_fail": n.get("p_fail", np.nan), "priority": n.get("priority", np.nan),
                          "why": n["why"], "lat": n["lat"], "lon": n["lon"], "km_from_prev": round(leg, 1)})
    visited = {s["place_id"] for s in stops}
    unassigned = queue[~queue["place_id"].isin(visited)]
    return DayPlan(day, pd.DataFrame(stops, columns=STOP_COLUMNS), due, queue, unassigned, km)


def last_full_day(df: pd.DataFrame) -> pd.Timestamp:
    """The newest day the city has published in full (the last day of an export is often partial)."""
    n = df.groupby("date").size()
    recent = n[n.index >= n.index.max() - pd.Timedelta(days=60)]
    return recent[recent >= 0.5 * recent.median()].index.max()


def plan_params(team: list[int], removed=(), pinned=(), balance: bool = True, prefs_key: str = "[null, null]") -> dict:
    """The inputs a plan was made from, saved with an approval so the app can tell when they change."""
    return {"team": [int(x) for x in team], "removed": sorted(int(x) for x in removed),
            "pinned": sorted(int(x) for x in pinned), "balance": bool(balance), "prefs": prefs_key}


def place_profile(df: pd.DataFrame, model, city: CityConfig, day: str | pd.Timestamp, place_id: int) -> dict | None:
    """Everything the place view shows about one place as of `day`: its latest details, the
    model's chance of failing, the history counts behind it and when its next routine visit is due."""
    day = pd.Timestamp(day)
    h = df[(df["place_id"] == place_id) & (df["date"] < day)]
    if h.empty:
        return None
    place = h.sort_values("date", kind="stable").iloc[[-1]].assign(date=day)
    feats = build_features(df[df["place_id"] == place_id], place, as_of=day)
    f = feats.iloc[0]
    done = h["outcome"].notna() | h["closed"].astype(bool)
    if "field" in h:
        done |= h["field"].fillna(False).astype(bool) & (h["result"] != "No Entry")
    last_done = h.loc[done, "date"].max()
    every = city.due_days.get(f["risk"], max(city.due_days.values())) if pd.notna(f["risk"]) else max(city.due_days.values())
    due_in = None if pd.isna(last_done) else int((last_done + pd.Timedelta(days=every) - day).days)
    p_fail = float(model.predict_proba(feats)[:, 1][0])
    return {**f.to_dict(), "p_fail": p_fail, "last_done": last_done, "every_days": int(every), "due_in_days": due_in}


OUTCOME_PL = {"pass": "zaliczona", "conditional": "z zastrzeżeniami", "fail": "niezaliczona"}
RESULT_PL = {"no entry": "brak dostępu", "not ready": "lokal nieprzygotowany", "out of business": "działalność zakończona",
             "business not located": "nie znaleziono lokalu", "closed": "działalność zakończona"}


def history(df: pd.DataFrame, place_id: int, before: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """A place's past inspections, newest first, in plain Polish: what the inspector reads before going in."""
    h = df[df["place_id"] == place_id]
    if before is not None:
        h = h[h["date"] < pd.Timestamp(before)]
    h = h.sort_values("date", ascending=False)
    kind = np.where(h["inspection_type"] == FIELD_TYPE, "rutynowa (z planu)",
                    np.where(h["inspection_type"] == FIELD_REQUEST_TYPE, "zgłoszenie (z planu)",
                    np.where(h["kind"] == "routine", "rutynowa",
                             np.where(h["kind"] == "mandatory", h["inspection_type"].map(request_label), "inna"))))
    other = h["result"].str.lower().map(lambda r: RESULT_PL.get(r, r))
    result = np.where(h["closed"], "działalność zakończona", h["outcome"].map(OUTCOME_PL).fillna(other))
    return pd.DataFrame({"date": h["date"].to_numpy(), "inspection_type": kind, "result": result,
                         "n_violations": h["n_violations"].to_numpy(), "outcome": h["outcome"].to_numpy(),
                         "inspection_id": h["inspection_id"].to_numpy()})


def print_plan(city: CityConfig, date: str, out: str | None = None, inspectors: int = 13) -> None:
    from .data import load, load_socio
    from .model import load_model
    df = load(city)
    p = plan_day(df, load_model(city), city, date, inspectors, socio=load_socio(city))
    print(f"{city.name}, {date}: {len(p.due):,} places due, {len(p.queue)} mandatory requests, "
          f"{len(p.routes)} stops for {inspectors} inspectors, {p.km:.0f} km, {len(p.unassigned)} requests unassigned")
    print(p.routes[["inspector", "seq", "eta", "kind", "name", "address", "why"]].to_string(index=False))
    if out:
        p.routes.to_csv(out, index=False)
