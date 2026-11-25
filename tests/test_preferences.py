import numpy as np
import pandas as pd

from inspections.preferences import ahp_weights, matrix, preferences, priority


def test_consistent_matrix_gives_exact_weights_and_zero_cr():
    w, cr = ahp_weights(matrix(["a", "b", "c"], {("a", "b"): 2, ("a", "c"): 4, ("b", "c"): 2}))
    assert np.allclose(w, [4 / 7, 2 / 7, 1 / 7])
    assert cr < 1e-9


def test_contradictory_judgments_are_flagged():
    # a > b > c > a, each strongly: a cycle no weights can satisfy.
    _, cr = ahp_weights(matrix(["a", "b", "c"], {("a", "b"): 9, ("b", "c"): 9, ("a", "c"): 1 / 9}))
    assert cr > 0.1


def test_default_preferences_are_consistent_and_put_children_first():
    p = preferences()
    assert p.consistent
    assert abs(sum(p.criteria.values()) - 1) < 1e-9
    assert np.isclose(p.groups["children"], max(p.groups.values())) and np.isclose(p.groups["healthcare"], 1.0)
    assert p.groups["other"] < p.groups["restaurant"]


def test_priority_rewards_vulnerable_places_at_equal_risk():
    rows = pd.DataFrame({"p_fail": [0.3, 0.3, 0.1], "facility_group": ["children", "other", "children"],
                         "days_since_last": [400, 400, 400]})
    out = priority(rows)
    assert out.loc[0, "priority"] > out.loc[1, "priority"]
    assert out.loc[0, "priority"] > out.loc[2, "priority"]
