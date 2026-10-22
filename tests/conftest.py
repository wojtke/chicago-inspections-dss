import pandas as pd
import pytest


@pytest.fixture
def events() -> pd.DataFrame:
    """Two establishments with a short, hand-checkable inspection history."""
    rows = [
        # inspection_id, license, date, inspection_type, result, n_violations
        (1, 10, "2020-01-10", "Canvass", "Fail", 5),
        (2, 10, "2020-01-20", "Canvass Re-Inspection", "Pass", 1),
        (3, 10, "2021-03-01", "Complaint", "Pass w/ Conditions", 3),
        (4, 10, "2021-06-01", "Canvass", "Fail", 7),
        (5, 20, "2021-06-01", "Canvass", "Pass", 0),
        (6, 20, "2021-06-01", "Complaint", "Fail", 4),  # same day as id 5
    ]
    df = pd.DataFrame(rows, columns=["inspection_id", "license", "date", "inspection_type", "result", "n_violations"])
    df["date"] = pd.to_datetime(df["date"])
    df["facility_type"], df["risk"], df["lat"], df["lon"] = "Restaurant", 1.0, 41.9, -87.7
    return df
