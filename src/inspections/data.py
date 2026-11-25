"""Download a city's inspection export, its neighbourhoods and their socioeconomic indicators,
and turn them into one clean parquet file in the canonical schema (see `city.py`)."""
import json
import re
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import shapely

from .city import CityConfig


def _urlopen(url: str):
    """urlopen with certifi's CA bundle: some Python builds ship without system certificates."""
    return urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))


def _fetch(url: str, path: Path, force: bool = False) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading {url}")
    with _urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.rename(path)
    return path


def raw_path(city: CityConfig) -> Path:
    return city.data_dir / "raw" / "inspections.csv"


def clean_path(city: CityConfig) -> Path:
    return city.data_dir / "inspections.parquet"


def details_path(city: CityConfig) -> Path:
    return city.data_dir / "details.parquet"


def load_areas(city: CityConfig) -> list[dict]:
    """[{area, name, geometry}] for each neighbourhood, geometry as a shapely (Multi)Polygon."""
    if not city.areas_url:
        return []
    gj = json.loads(_fetch(city.areas_url, city.data_dir / "raw" / "areas.geojson").read_text())
    areas = [{"area": int(float(f["properties"][city.area_id_field])),
              "name": str(f["properties"][city.area_name_field]).title(),
              "geometry": shapely.geometry.shape(f["geometry"])} for f in gj["features"]]
    return sorted(areas, key=lambda a: a["area"])


def load_socio(city: CityConfig) -> pd.DataFrame:
    """area, income (per capita), hardship (higher = poorer) and income_quintile (1 = poorest)."""
    if not city.socio_url:
        return pd.DataFrame(columns=["area", "income", "hardship", "income_quintile"])
    rows = json.loads(_fetch(city.socio_url, city.data_dir / "raw" / "socio.json").read_text())
    s = pd.DataFrame(rows)
    s = s[pd.to_numeric(s[city.socio_area_field], errors="coerce").notna()]  # drops the city total
    out = pd.DataFrame({
        "area": s[city.socio_area_field].astype(float).astype(int),
        "income": pd.to_numeric(s[city.socio_income_field], errors="coerce"),
        "hardship": pd.to_numeric(s[city.socio_hardship_field], errors="coerce"),
    })
    out["income_quintile"] = pd.qcut(out["income"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return out.reset_index(drop=True)


def tag_areas(lon: np.ndarray, lat: np.ndarray, areas: list[dict]) -> np.ndarray:
    """Neighbourhood id of each point; 0 where a point is missing or outside every area."""
    out = np.zeros(len(lon), dtype=int)
    ok = ~(np.isnan(lon) | np.isnan(lat))
    for a in areas:
        shapely.prepare(a["geometry"])
        inside = np.zeros(len(lon), dtype=bool)
        inside[ok] = shapely.contains_xy(a["geometry"], lon[ok], lat[ok])
        out[inside & (out == 0)] = a["area"]
    return out


def _title(s: pd.Series) -> pd.Series:
    """Title case without str.title()'s "Children'S" quirk."""
    return s.fillna("Unknown").astype(str).str.strip().str.title().str.replace(r"'S\b", "'s", regex=True)


def facility_group(facility_type: pd.Series, city: CityConfig) -> pd.Series:
    """Map a city's free-text facility types onto the five vulnerability groups."""
    t = facility_type.fillna("").str.lower()
    out = pd.Series("other", index=t.index)
    for group, pattern in reversed(city.facility_groups):  # earlier patterns win
        out[t.str.contains(pattern, regex=True)] = group
    return out


def clean(raw: pd.DataFrame, city: CityConfig, areas: list[dict] | None = None) -> pd.DataFrame:
    """Rename to the canonical schema, parse types, derive kind/outcome/group/area."""
    df = raw[list(city.columns)].rename(columns=city.columns)
    df["date"] = pd.to_datetime(df["date"], format=city.date_format)
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["place_id"] = pd.to_numeric(df["place_id"], errors="coerce")
    df = df[df["place_id"].fillna(0) > 0].copy()
    df["place_id"] = df["place_id"].astype("int64")
    df["risk"] = pd.to_numeric(df["risk"].astype(str).str.extract(city.risk_pattern)[0], errors="coerce")
    df["facility_type"] = _title(df["facility_type"])
    df["facility_group"] = facility_group(df["facility_type"], city)
    df["inspection_type"] = _title(df["inspection_type"])
    df["result"] = df["result"].fillna("Unknown").astype(str).str.strip()
    t = df["inspection_type"]
    visit = ~t.isin(city.not_a_visit)
    df["kind"] = np.select([t.str.contains(city.routine_types, flags=re.I) & visit,
                            t.str.contains(city.mandatory_types, flags=re.I) & visit],
                           ["routine", "mandatory"], "other")
    df["complaint"] = t.str.contains(city.complaint_types, flags=re.I) & visit
    df["outcome"] = df["result"].map(city.result_outcome)
    df["closed"] = df["result"].isin(city.closed_results)
    if city.violations_column and city.violations_column in raw:
        v = raw.loc[df.index, city.violations_column].fillna("")
        df["n_violations"] = v.map(lambda s: s.count(city.violations_sep) + 1 if s else 0)
    else:
        df["n_violations"] = np.nan
    df["area"] = tag_areas(df["lon"].to_numpy(float), df["lat"].to_numpy(float), areas) if areas else 0
    return df.sort_values(["date", "inspection_id"]).reset_index(drop=True)


def build_details(city: CityConfig, raw: pd.DataFrame | None = None) -> None:
    """Per-inspection details for the place view (other name, ZIP, violation notes). They are big
    and only needed one place at a time, so they live in their own file, sorted by place."""
    if not city.detail_columns:
        return
    ids = {src: dst for src, dst in city.columns.items() if dst in ("inspection_id", "place_id")}
    cols = ids | city.detail_columns
    if raw is None:
        raw = pd.read_csv(raw_path(city), dtype=str, usecols=list(cols))
    d = raw[list(cols)].rename(columns=cols)
    d["place_id"] = pd.to_numeric(d["place_id"], errors="coerce")
    d = d.dropna(subset=["place_id"]).astype({"place_id": "int64"})
    d.sort_values("place_id").to_parquet(details_path(city), index=False, row_group_size=20_000, compression="zstd")


def load_details(city: CityConfig, place_id: int) -> pd.DataFrame:
    """The details rows of one place (empty if the city has none)."""
    path = details_path(city)
    if not path.exists():
        return pd.DataFrame(columns=["inspection_id", "place_id", *city.detail_columns.values()])
    return pd.read_parquet(path, filters=[("place_id", "==", int(place_id))])


def violations(text, sep: str = "|") -> list[dict]:
    """Split a published violations field into [{no, title, comment}] in sentence case."""
    out = []
    if not isinstance(text, str):  # missing (NaN / None): no violations recorded
        return out
    for item in text.split(sep):
        item = item.strip()
        if not item:
            continue
        head, _, comment = item.partition(" - Comments:")
        no, _, title = head.partition(". ")
        if not no.strip().isdigit():
            no, title = "", head
        out.append({"no": no.strip(), "title": _sentence(title), "comment": _sentence(comment)})
    return out


def _sentence(text: str) -> str:
    """'OBSERVED NO SOAP. INSTRUCTED TO…' -> 'Observed no soap. Instructed to…'"""
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.strip().lower())


def build(city: CityConfig, force_download: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(_fetch(city.inspections_url, raw_path(city), force_download), dtype=str)
    df = clean(raw, city, load_areas(city))
    load_socio(city)  # cache it next to the data
    df.to_parquet(clean_path(city), index=False)
    build_details(city, raw)
    print(f"{city.name}: {len(raw):,} rows in the export, {len(df):,} kept -> {clean_path(city)}")
    print(f"dates {df['date'].min():%d.%m.%Y} - {df['date'].max():%d.%m.%Y}, {df['place_id'].nunique():,} places")
    print(f"in a neighbourhood: {(df['area'] > 0).mean():.1%}; kinds: {df['kind'].value_counts(normalize=True).round(3).to_dict()}")
    return df


def load(city: CityConfig) -> pd.DataFrame:
    path = clean_path(city)
    if not path.exists():
        raise SystemExit(f"{path} not found, run `insp data` first")
    return pd.read_parquet(path)
