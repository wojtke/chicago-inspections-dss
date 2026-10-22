import pandas as pd

from inspections.features import build_features, canvass_outcomes


def test_features_use_only_earlier_inspections(events):
    f = build_features(events, canvass_outcomes(events)).set_index("inspection_id")
    # First ever visit: no history at all.
    assert f.loc[1, "n_past"] == 0 and pd.isna(f.loc[1, "days_since_last"])
    # Visit 4 sees visits 1-3 only: one fail out of three outcomes, one complaint.
    assert f.loc[4, "n_past"] == 3
    assert f.loc[4, "n_past_fails"] == 1
    assert abs(f.loc[4, "past_fail_rate"] - 1 / 3) < 1e-9
    assert f.loc[4, "n_past_complaints"] == 1
    assert f.loc[4, "last_failed"] == 0 and f.loc[4, "last_n_violations"] == 3
    assert f.loc[4, "days_since_last"] == (pd.Timestamp("2021-06-01") - pd.Timestamp("2021-03-01")).days


def test_same_day_inspections_do_not_leak(events):
    f = build_features(events, canvass_outcomes(events)).set_index("inspection_id")
    # The complaint fail on the same day as visit 5 must not count as history.
    assert f.loc[5, "n_past"] == 0 and f.loc[5, "n_past_fails"] == 0


def test_snapshot_date_hides_later_visits(events):
    target = canvass_outcomes(events).query("inspection_id == 4")
    f = build_features(events, target, as_of=pd.Timestamp("2020-06-01"))
    assert f["n_past"].item() == 2  # only the two visits in January 2020
