"""Download the Chicago Food Inspections export and turn it into a clean parquet file."""
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"
DATA = Path("data")
RAW = DATA / "raw" / "food_inspections.csv"
CLEAN = DATA / "inspections.parquet"

COLUMNS = {
    "Inspection ID": "inspection_id",
    "License #": "license",
    "DBA Name": "name",
    "Facility Type": "facility_type",
    "Risk": "risk",
    "Address": "address",
    "Zip": "zip",
    "Latitude": "lat",
    "Longitude": "lon",
    "Inspection Date": "date",
    "Inspection Type": "inspection_type",
    "Results": "result",
    "Violations": "violations",
}


def download(force: bool = False) -> Path:
    """Fetch the full CSV export (about 330 MB, no account needed) unless it is already there."""
    if RAW.exists() and not force:
        print(f"{RAW} already downloaded")
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW.with_suffix(".part")
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.rename(RAW)
    return RAW


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse types and drop rows that cannot be tied to an establishment."""
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["license"] = pd.to_numeric(df["license"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[df["license"].fillna(0) > 0].copy()
    df["license"] = df["license"].astype("int64")
    # "Risk 1 (High)" -> 1, "Risk 3 (Low)" -> 3; "All" and blanks become missing.
    df["risk"] = pd.to_numeric(df["risk"].str.extract(r"Risk (\d)")[0], errors="coerce")
    df["facility_type"] = df["facility_type"].fillna("Unknown").str.strip().str.title()
    df["inspection_type"] = df["inspection_type"].fillna("Unknown").str.strip().str.title()
    df["result"] = df["result"].fillna("Unknown").str.strip()
    # Violations are one free-text field, items separated by " | ".
    df["n_violations"] = df["violations"].fillna("").map(lambda s: s.count("|") + 1 if s else 0)
    df = df.drop(columns="violations")
    return df.sort_values(["date", "inspection_id"]).reset_index(drop=True)


def build(force_download: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(download(force_download), dtype=str)
    df = clean(raw)
    df.to_parquet(CLEAN, index=False)
    print(f"{len(raw):,} rows in the export, {len(df):,} kept -> {CLEAN}")
    print(f"dates {df['date'].min():%d.%m.%Y} - {df['date'].max():%d.%m.%Y}, {df['license'].nunique():,} licences")
    return df


def load() -> pd.DataFrame:
    if not CLEAN.exists():
        raise SystemExit(f"{CLEAN} not found, run `insp data` first")
    return pd.read_parquet(CLEAN)
