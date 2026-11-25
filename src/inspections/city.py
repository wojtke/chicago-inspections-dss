"""What the system needs to know about a city. Everything city-specific lives in a CityConfig.

The rest of the code only sees the canonical schema below, so plugging in another city means
writing one config (see `cities/chicago.py`) that says where its open data is and how its
columns and codes map onto these names.

Canonical inspection columns
    inspection_id, place_id, name, facility_type, risk (1 high .. 3 low), address, lat, lon,
    date, inspection_type, result (as published), n_violations
Optional details, kept in a separate file for the place view (`data.load_details`)
    aka (other name), zip, violations (the inspector's notes as published)
Derived by `data.clean`
    kind       routine | mandatory | other     (routine = the visits the planner may reorder)
    complaint  True for visits triggered by a complaint or suspected food poisoning
    outcome    pass | conditional | fail | None (None = no outcome, e.g. place closed, no entry)
    closed     True when the visit found the place out of business
    facility_group  children | healthcare | restaurant | retail | other
    area       neighbourhood id (0 = outside every area)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# Vulnerability groups the decision module weighs; every city maps its facility types onto them.
GROUPS = ["children", "healthcare", "restaurant", "retail", "other"]
OUTCOMES = ["pass", "conditional", "fail"]


@dataclass(frozen=True)
class CityConfig:
    slug: str
    name: str
    # Inspections export (CSV) and how to read it.
    inspections_url: str
    columns: dict[str, str]                 # source column -> canonical name
    date_format: str
    result_outcome: dict[str, str]          # published result -> pass | conditional | fail
    closed_results: tuple[str, ...]         # published results meaning the place is gone
    routine_types: str                      # regex on inspection_type
    mandatory_types: str                    # regex on inspection_type
    complaint_types: str                    # regex on inspection_type
    not_a_visit: tuple[str, ...] = ()
    risk_pattern: str = r"(\d)"             # regex whose first group is the 1-3 risk category
    violations_column: str | None = None    # source column with all violations in one field
    violations_sep: str = "|"
    detail_columns: dict[str, str] = field(default_factory=dict)  # source column -> aka | zip | violations,
                                                                  # shown in the place view only
    facility_groups: tuple[tuple[str, str], ...] = ()  # (group, regex on facility type), first match wins
    # Neighbourhoods (GeoJSON) and their socioeconomic indicators (JSON rows).
    areas_url: str = ""
    area_id_field: str = ""
    area_name_field: str = ""
    socio_url: str = ""
    socio_area_field: str = ""
    socio_income_field: str = ""            # per-capita income
    socio_hardship_field: str = ""          # composite deprivation index, higher = poorer
    socio_year: str = ""
    # Planning rules.
    due_days: dict[int, int] = field(default_factory=lambda: {1: 182, 2: 365, 3: 730})
    center_lat: float = 0.0                 # for the local km projection and the maps
    center_lon: float = 0.0

    @property
    def data_dir(self) -> Path:
        return home() / "data" / self.slug

    @property
    def models_dir(self) -> Path:
        return home() / "models" / self.slug

    @property
    def results_dir(self) -> Path:
        return home() / "results" / self.slug


def home() -> Path:
    return Path(os.environ.get("INSP_HOME", "."))


def get_city(slug: str | None = None) -> CityConfig:
    from .cities import CITIES
    slug = slug or os.environ.get("INSP_CITY", "chicago")
    if slug not in CITIES:
        raise SystemExit(f"unknown city {slug!r}; known: {', '.join(CITIES)}")
    return CITIES[slug]
