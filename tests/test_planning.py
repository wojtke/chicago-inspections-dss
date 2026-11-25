"""Live planning: when a place is due, the request queue, field visits, follow-ups and the replay pool."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from inspections import evaluation, planning
from inspections.cities.chicago import CHICAGO
from inspections.routing import RoutingConfig, solve
from inspections.store import Store

DAY = pd.Timestamp("2026-10-20")


def history(rows) -> pd.DataFrame:
    """Canonical inspections from (id, place, date, kind, inspection_type, outcome, name, address)."""
    df = pd.DataFrame(rows, columns=["inspection_id", "place_id", "date", "kind", "inspection_type", "outcome",
                                     "name", "address"])
    df["date"] = pd.to_datetime(df["date"])
    df["closed"] = False
    df["complaint"] = df["inspection_type"].str.contains("Complaint")
    df["n_violations"] = 0
    df["result"] = df["outcome"].fillna("No Entry").str.title()
    df["facility_type"], df["facility_group"], df["risk"], df["area"] = "Restaurant", "restaurant", 1.0, 1
    df["lat"] = 41.9 + df["place_id"] * 0.001
    df["lon"] = -87.7
    return df


def test_a_no_entry_attempt_does_not_restart_the_clock():
    df = history([(1, 10, "2025-09-01", "routine", "Canvass", "pass", "A", "1 Main"),
                  (2, 10, "2026-10-10", "routine", "Canvass", None, "A", "1 Main"),   # no entry, 10 days ago
                  (3, 20, "2026-09-01", "routine", "Canvass", "pass", "B", "2 Main")])  # done 49 days ago
    due = planning.due_places(df, CHICAGO, DAY)
    assert due["place_id"].tolist() == [10]  # risk 1 = every 182 days; 10 was last inspected 414 days ago


def test_a_recorded_visit_counts_and_its_result_feeds_the_features():
    df = history([(1, 10, "2025-09-01", "routine", "Canvass", "pass", "A", "1 Main")])
    visits = pd.DataFrame({"place_id": [10], "day": ["2026-10-19"], "outcome": ["fail"]})
    df2 = planning.add_field_visits(df, visits)
    assert planning.due_places(df2, CHICAGO, DAY).empty  # visited yesterday, so not due today
    f = planning.build_features(df2, df2.iloc[[0]].assign(date=DAY), as_of=DAY)
    assert f["last_failed"].iloc[0] == 1 and f["n_past"].iloc[0] == 2
    h = planning.history(df2, 10, before=DAY)
    assert h["inspection_type"].iloc[0] == "rutynowa (z planu)" and h["result"].iloc[0] == "niezaliczona"


def test_requests_at_one_address_are_one_stop_that_covers_every_licence():
    df = history([(1, 10, "2026-10-20", "mandatory", "License", None, "Cafe", "1 Main"),
                  (2, 11, "2026-10-20", "mandatory", "License", None, "Cafe", "1 Main"),
                  (3, 12, "2026-10-20", "mandatory", "Complaint", None, "Cafe", "1 Main"),
                  (4, 20, "2026-10-20", "mandatory", "Complaint", None, "Bar", "2 Main")])
    q = planning.mandatory_queue(df, DAY)
    assert len(q) == 2
    cafe = q[q["name"] == "Cafe"].iloc[0]
    assert cafe["place_ids"] == [10, 11, 12]
    assert cafe["request"] == "kontrola licencyjna ×2 + skarga"


def test_requests_an_approved_plan_left_out_come_back_the_next_day():
    df = history([(1, 10, "2026-10-19", "mandatory", "Complaint", None, "A", "1 Main"),
                  (2, 20, "2026-10-19", "mandatory", "Complaint", None, "B", "2 Main"),
                  (3, 30, "2026-10-20", "mandatory", "License", None, "C", "3 Main")])
    carried = planning.pending_requests(df, DAY, {"2026-10-19": {10}})
    assert carried["place_id"].tolist() == [20]
    q = planning.mandatory_queue(df, DAY, carried)
    assert sorted(q["place_id"]) == [20, 30]
    assert q.loc[q["place_id"] == 20, "request"].iloc[0] == "skarga · od 19.10"
    assert planning.pending_requests(df, DAY, {}).empty  # no approved plan: nothing to carry


def test_a_failed_visit_from_a_plan_brings_a_reinspection_until_the_place_is_seen_again():
    df = history([(1, 10, "2025-09-01", "routine", "Canvass", "pass", "A", "1 Main"),
                  (2, 20, "2025-09-01", "routine", "Canvass", "pass", "B", "2 Main")])
    fail = pd.DataFrame({"place_id": [10, 20], "day": ["2026-10-12", "2026-10-16"], "outcome": ["fail", "fail"]})
    df2 = planning.add_field_visits(df, fail)
    owed = planning.followups(df2, DAY)
    assert owed["place_id"].tolist() == [10]  # 20 failed only 4 days ago
    assert owed["request"].iloc[0] == "ponowna kontrola po niezaliczeniu 12.10"
    assert 10 in set(planning.mandatory_queue(df2, DAY)["place_id"])
    seen = planning.add_field_visits(df2, pd.DataFrame({"place_id": [10], "day": ["2026-10-19"], "outcome": ["pass"],
                                                        "kind": ["zgłoszenie"]}))
    assert planning.followups(seen, DAY).empty
    tried = planning.add_field_visits(df2, pd.DataFrame({"place_id": [10], "day": ["2026-10-19"], "outcome": ["no_entry"]}))
    assert planning.followups(tried, DAY)["place_id"].tolist() == [10]  # no entry does not settle it


def test_a_result_for_a_visit_the_city_data_has_replaces_it_and_can_bring_a_reinspection():
    df = history([(1, 10, "2026-10-12", "mandatory", "Complaint", "pass", "A", "1 Main")])
    out = planning.add_field_visits(df, pd.DataFrame({"place_id": [10], "day": ["2026-10-12"], "outcome": ["fail"],
                                                      "kind": ["zgłoszenie"]}))
    assert len(out) == 1 and out["outcome"].iloc[0] == "fail" and out["field"].iloc[0]
    assert df["outcome"].iloc[0] == "pass"  # the input (the app's shared cache) is left alone
    assert planning.followups(out, DAY)["place_id"].tolist() == [10]
    city_fail = history([(1, 10, "2026-10-12", "mandatory", "Complaint", "fail", "A", "1 Main")])
    agreed = planning.add_field_visits(city_fail, pd.DataFrame({"place_id": [10], "day": ["2026-10-12"],
                                                                "outcome": ["fail"]}))
    assert planning.followups(agreed, DAY).empty  # the city's data has its own re-inspection for this one


def test_a_request_beats_pinned_places_when_not_everything_fits():
    xy = np.array([[0, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [3, 3]], dtype=float)
    prize = np.r_[np.full(6, planning.PINNED_PRIZE), planning.MANDATORY_PRIZE]
    routes = solve(xy, [6], RoutingConfig(time_limit_s=1.0), prize=prize)
    assert 6 in routes[0] and len(routes[0]) == 6


def test_routes_fit_in_the_working_day():
    xy = np.array([[0, 0], [20, 0], [40, 0], [60, 0]], dtype=float)  # 20 km apart = 60 min at 20 km/h
    cfg = RoutingConfig(time_limit_s=1.0, day_min=200)            # room for 2 visits and one drive
    routes = solve(xy, [4], cfg, prize=np.ones(4))
    assert len(routes[0]) == 2


def test_replay_pool_holds_only_places_known_at_the_start_as_they_were_then():
    df = history([(1, 10, "2024-10-01", "routine", "Canvass", "pass", "A", "1 Main"),
                  (2, 10, "2025-01-10", "routine", "Canvass", "fail", "A", "1 Main"),
                  (3, 20, "2025-01-12", "routine", "Canvass", "pass", "New", "9 Main")])  # first seen in the window
    df.loc[1, "risk"] = 3.0  # re-rated at the visit being replayed: the planner could not know that
    rows = evaluation.pool(df, "2025-01-06")
    assert rows["place_id"].tolist() == [10] and rows["risk"].iloc[0] == 1.0
    load = evaluation.daily_load(df, rows, "2025-01-06")
    assert load["fixed"].sum() == 1 and load["routine"].sum() == 1  # the new place keeps its day


def test_store_upgrades_an_old_database_and_counts_only_recorded_visits(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:  # the schema before plans had params and users had an active flag
        db.execute("CREATE TABLE users (username TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL, "
                   "inspector_no INTEGER)")
        db.execute("CREATE TABLE plans (day TEXT PRIMARY KEY, approved_by TEXT, approved_at TEXT, inspectors INTEGER, "
                   "stops TEXT)")
        stops = [{"inspector": 1, "seq": 1, "eta": "08:00", "kind": "rutynowa", "place_id": 10, "name": "A", "address": "1"},
                 {"inspector": 1, "seq": 2, "eta": "09:00", "kind": "zgłoszenie", "place_id": 11, "name": "B", "address": "2"},
                 {"inspector": 1, "seq": 3, "eta": "10:00", "kind": "zgłoszenie", "place_id": 12, "name": "C", "address": "3"}]
        db.execute("INSERT INTO plans VALUES ('2026-10-19', 'kierownik', '2026-10-18T10:00:00', 1, ?)", (json.dumps(stops),))
    db.close()
    s = Store(CHICAGO, path=path)
    s.seed_demo(inspectors=2)
    assert s.user("inspektor1")["role"] == "inspector" and s.users()["active"].all()
    assert s.plan("2026-10-19")[1]["params"] is None
    s.record("2026-10-19", 10, "inspektor1", "conditional")
    s.record("2026-10-19", 12, "inspektor1", "no_entry")
    v = s.field_visits(before="2026-10-20")
    assert sorted(v["place_id"]) == [10, 12] and set(v["kind"]) == {"rutynowa", "zgłoszenie"}  # 11 has no result yet
    assert s.served("2026-10-20", 7) == {"2026-10-19": {10}}  # no entry is not done; 11 and 12 stay pending
    assert s.visits("2026-10-19", "2026-10-19")["outcome"].isna().sum() == 1


def test_a_visit_to_an_address_with_several_licences_settles_all_of_them(tmp_path):
    s = Store(CHICAGO, path=tmp_path / "group.sqlite")
    stops = pd.DataFrame([{"inspector": 1, "seq": 1, "eta": "08:00", "kind": "zgłoszenie", "place_id": 10,
                           "place_ids": [10, 11], "name": "Cafe", "address": "1 Main"}])
    s.approve_plan("2026-10-19", stops, "kierownik", 1)
    s.record("2026-10-19", 10, "inspektor1", "pass")
    assert sorted(s.field_visits(before="2026-10-20")["place_id"]) == [10, 11]
    assert s.served("2026-10-20", 7) == {"2026-10-19": {10, 11}}
    with pytest.raises(ValueError):  # a day with results keeps its plan
        s.approve_plan("2026-10-19", stops, "kierownik", 1)


def test_absent_and_inactive_inspectors_are_not_available(tmp_path):
    s = Store(CHICAGO, path=tmp_path / "team.sqlite")
    s.seed_demo(inspectors=3)
    s.add_absence("inspektor2", "2026-10-19", "2026-10-21", "urlop")
    s.update_user("inspektor3", active=False)
    assert s.available("2026-10-20")["inspector_no"].tolist() == [1]
    assert s.available("2026-10-22")["inspector_no"].tolist() == [1, 2]
    assert s.add_inspector("Nowa Osoba") == "inspektor4"


def test_routes_carry_the_inspectors_numbers():
    df = history([(i, 10 + i, "2024-01-01", "routine", "Canvass", "pass", f"P{i}", f"{i} Main") for i in range(1, 9)])
    p = planning.plan_day(df, _Flat(), CHICAGO, DAY, [4, 9], cfg=RoutingConfig(time_limit_s=0.5))
    assert set(p.routes["inspector"]) == {4, 9}
    assert all(ids == [pid] for ids, pid in zip(p.routes["place_ids"], p.routes["place_id"]))


class _Flat:
    """A stand-in model: every place has the same chance of failing."""
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.7), np.full(len(X), 0.3)])
