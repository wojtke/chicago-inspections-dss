"""Chicago: Food Inspections on the city's open-data portal (Socrata dataset 4ijn-s7e5)."""
from ..city import CityConfig

CHICAGO = CityConfig(
    slug="chicago",
    name="Chicago",
    inspections_url="https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD",
    columns={
        "Inspection ID": "inspection_id", "License #": "place_id", "DBA Name": "name",
        "Facility Type": "facility_type", "Risk": "risk", "Address": "address",
        "Latitude": "lat", "Longitude": "lon", "Inspection Date": "date",
        "Inspection Type": "inspection_type", "Results": "result",
    },
    date_format="%m/%d/%Y",
    result_outcome={"Pass": "pass", "Pass w/ Conditions": "conditional", "Fail": "fail"},
    closed_results=("Out of Business", "Business Not Located"),
    routine_types=r"^Canvass$",
    # Complaints, food poisoning, licence checks and every re-inspection after a failure.
    mandatory_types=r"Complaint|Food Poisoning|License|Re-Inspection",
    complaint_types=r"Complaint|Food Poisoning",
    not_a_visit=("Non-Inspection",),
    risk_pattern=r"Risk (\d)",                 # "Risk 1 (High)" -> 1
    violations_column="Violations",
    violations_sep="|",
    detail_columns={"AKA Name": "aka", "Zip": "zip", "Violations": "violations"},
    facility_groups=(
        ("children", r"school|daycare|day care|children|preschool|pre school|charter|youth|kids"),
        ("healthcare", r"hospital|long term care|nursing|assisted living|senior|golden diner|adult day|rehab|hospice"),
        ("restaurant", r"restaurant|catering|banquet|cafe|coffee|diner|kitchen"),
        ("retail", r"grocery|bakery|wholesale|convenience|store|market|deli|butcher|meat|candy|ice cream|liquor"),
    ),
    areas_url="https://data.cityofchicago.org/resource/igwz-8jzy.geojson?$limit=100",
    area_id_field="area_numbe",
    area_name_field="community",
    socio_url="https://data.cityofchicago.org/resource/kn9c-c2s2.json?$limit=100",
    socio_area_field="ca",
    socio_income_field="per_capita_income_",
    socio_hardship_field="hardship_index",
    socio_year="2008–2012",
    # Inspection frequency by risk category (to check against the city's food code).
    due_days={1: 182, 2: 365, 3: 730},
    center_lat=41.84,
    center_lon=-87.68,
)
