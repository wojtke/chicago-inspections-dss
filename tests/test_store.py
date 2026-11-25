import pandas as pd

from inspections.cities.chicago import CHICAGO
from inspections.store import Store


def test_users_plan_and_results_round_trip(tmp_path):
    s = Store(CHICAGO, path=tmp_path / "t.sqlite")
    s.seed_demo(inspectors=2)
    assert s.user("kierownik")["role"] == "manager"
    assert s.user("inspektor2")["inspector_no"] == 2
    assert s.user("nobody") is None
    stops = pd.DataFrame({"inspector": [1, 2], "seq": [1, 1], "place_id": [10, 20], "name": ["A", "B"]})
    s.approve_plan("2026-10-20", stops, "kierownik", 2)
    got, meta = s.plan("2026-10-20")
    assert got["place_id"].tolist() == [10, 20] and meta["approved_by"] == "kierownik"
    s.record("2026-10-20", 20, "inspektor2", "fail", "brudna chłodnia")
    assert s.results("2026-10-20")["outcome"].tolist() == ["fail"]
    s.set_setting("ahp_criteria", [["risk", "overdue", 3]], "kierownik")
    assert s.setting("ahp_criteria") == [["risk", "overdue", 3]]
