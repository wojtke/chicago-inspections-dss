import numpy as np
import pandas as pd

from inspections import fairness
from inspections.evaluation import assign, metrics, routine_capacity
from inspections.routing import RoutingConfig, solve, vehicle_caps


def test_assign_respects_capacity_and_marks_the_rest_unreached():
    day = assign(np.array([4, 3, 2, 1, 0]), np.array([2, 0, 1]))
    assert day.tolist() == [3, 3, 2, 0, 0]  # rows 1 and 0 did not fit: day = len(cap)


def test_mandatory_visits_take_capacity_first():
    load = pd.DataFrame({"fixed": [5, 5, 5], "routine": [5, 3, 0]})
    load["total"] = load["fixed"] + load["routine"]
    assert routine_capacity(load, 1.0).tolist() == [5, 3, 0]   # x1 = what really happened
    assert routine_capacity(load, 0.8).tolist() == [3, 1, 0]   # a smaller team cuts routine visits only


def test_unreached_failures_are_penalised():
    rows = pd.DataFrame({"actual_day": [0, 1], "fail": [1, 1], "facility_group": ["restaurant", "children"]})
    m = metrics(rows, [np.array([0, 3])], days=3)  # the second failure is never reached in a 3-day window
    assert m["days_sooner"] == -1.0 and m["fails_reached"] == 0.5


def test_vehicle_caps_split_evenly():
    assert vehicle_caps(13, 6) == [5, 4, 4]
    assert sum(vehicle_caps(31, 6)) == 31 and vehicle_caps(0, 6) == []


def test_routes_respect_caps_visit_each_place_once_and_prefer_prizes():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(12, 2))
    prize = np.r_[np.full(6, 1.0), np.full(6, 0.01)]
    cfg = RoutingConfig(time_limit_s=1.0)
    routes = solve(xy, [3, 3], cfg, prize=prize)
    visited = [i for r in routes for i in r]
    assert all(len(r) <= 3 for r in routes)
    assert len(visited) == len(set(visited)) == 6
    assert set(visited) == set(range(6))  # the valuable places, not the cheap ones


def test_routes_visit_everything_when_no_prizes():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    routes = solve(xy, [2, 2], RoutingConfig(), optimise=False)
    assert sorted(i for r in routes for i in r) == [0, 1, 2, 3]


def test_balanced_order_follows_expected_failures():
    # Group 1 holds half the expected failures but its places rank lower: it still gets every other slot.
    priority = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.01])
    p_fail = np.array([0.1] * 8)
    group = np.array([2, 2, 2, 2, 1, 1, 1, 1])
    order = fairness.balanced_order(priority, p_fail, group)
    first_four = group[order[:4]]
    assert sorted(first_four.tolist()) == [1, 1, 2, 2]


def test_burden_ratio_is_one_when_flags_follow_failures():
    rows = pd.DataFrame({"fail": [1, 0, 1, 0] * 25, "income_quintile": [1, 1, 2, 2] * 25})
    p = np.where(rows["fail"] == 1, 0.9, 0.1) + np.random.default_rng(0).uniform(0, 0.01, len(rows))
    a = fairness.audit_model(rows, p, top=0.5)
    assert np.allclose(a["burden_ratio"], 1.0)
