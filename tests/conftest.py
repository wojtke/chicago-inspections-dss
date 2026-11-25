import pandas as pd
import pytest


@pytest.fixture
def events() -> pd.DataFrame:
    """Two places with a short, hand-checkable inspection history, in the canonical schema."""
    rows = [
        # inspection_id, place_id, date, kind, complaint, outcome, n_violations
        (1, 10, "2020-01-10", "routine", False, "fail", 5),
        (2, 10, "2020-01-20", "mandatory", False, "pass", 1),
        (3, 10, "2021-03-01", "mandatory", True, "conditional", 3),
        (4, 10, "2021-06-01", "routine", False, "fail", 7),
        (5, 20, "2021-06-01", "routine", False, "pass", 0),
        (6, 20, "2021-06-01", "mandatory", True, "fail", 4),  # same day as id 5
    ]
    df = pd.DataFrame(rows, columns=["inspection_id", "place_id", "date", "kind", "complaint", "outcome", "n_violations"])
    df["date"] = pd.to_datetime(df["date"])
    df["closed"] = False
    df["facility_type"], df["facility_group"], df["risk"], df["lat"], df["lon"] = "Restaurant", "restaurant", 1.0, 41.9, -87.7
    return df
