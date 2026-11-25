"""The pipeline runs on any city that provides a CityConfig: here a made-up one."""
import numpy as np
import pandas as pd

from inspections.city import CityConfig
from inspections.data import clean, facility_group, tag_areas
from inspections.features import feature_list, routine_outcomes, build_features

TESTVILLE = CityConfig(
    slug="testville", name="Testville", inspections_url="",
    columns={"ID": "inspection_id", "Permit": "place_id", "Business": "name", "Category": "facility_type",
             "Tier": "risk", "Street": "address", "Y": "lat", "X": "lon", "When": "date",
             "Reason": "inspection_type", "Grade": "result"},
    date_format="%Y-%m-%d",
    result_outcome={"A": "pass", "B": "conditional", "C": "fail"},
    closed_results=("Closed",),
    routine_types=r"^Routine$", mandatory_types=r"Complaint|Follow-up", complaint_types=r"Complaint",
    risk_pattern=r"Tier (\d)",
    facility_groups=(("children", r"school"), ("restaurant", r"restaurant")),
    center_lat=50.0,
)


def raw_testville() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": [1, 2, 3, 4], "Permit": ["7", "7", "8", "8"], "Business": ["Bistro", "Bistro", "Primary", "Primary"],
        "Category": ["restaurant", "restaurant", "elementary school", "elementary school"],
        "Tier": ["Tier 1", "Tier 1", "Tier 2", "Tier 2"], "Street": ["1 Main", "1 Main", "2 Elm", "2 Elm"],
        "Y": ["50.0", "50.0", "50.1", "50.1"], "X": ["10.0", "10.0", "10.1", "10.1"],
        "When": ["2024-01-05", "2024-02-01", "2024-01-10", "2024-06-01"],
        "Reason": ["Routine", "Follow-up", "Routine", "Complaint"], "Grade": ["C", "A", "A", "Closed"],
    })


def test_testville_maps_onto_the_canonical_schema():
    df = clean(raw_testville(), TESTVILLE)
    assert df["kind"].tolist() == ["routine", "routine", "mandatory", "mandatory"]
    assert df["outcome"].tolist()[:3] == ["fail", "pass", "pass"] and pd.isna(df["outcome"].iloc[3])
    assert df["closed"].tolist() == [False, False, False, True]
    assert df["facility_group"].tolist() == ["restaurant", "children", "restaurant", "children"]
    assert df["risk"].tolist() == [1, 2, 1, 2]


def test_testville_features_build():
    df = clean(raw_testville(), TESTVILLE)
    f = build_features(df, routine_outcomes(df))
    assert set(feature_list(location=False)) <= set(f.columns)


def test_chicago_quirks():
    from inspections.cities.chicago import CHICAGO
    assert facility_group(pd.Series(["Children's Services Facility", "Golden Diner", "Tavern", "Grocery Store"]),
                          CHICAGO).tolist() == ["children", "healthcare", "other", "retail"]


def test_tag_areas_point_in_polygon():
    import shapely
    sq = {"area": 3, "name": "Square", "geometry": shapely.box(0, 0, 1, 1)}
    assert tag_areas(np.array([0.5, 2.0, np.nan]), np.array([0.5, 2.0, 0.5]), [sq]).tolist() == [3, 0, 0]
