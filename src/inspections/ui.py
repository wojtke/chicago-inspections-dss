"""Pieces shared by the two Streamlit apps: the product (app.py) and the technical report (report.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pydeck.types import String

from . import evaluation
from .city import get_city
from .data import load, load_areas, load_socio
from .model import load_model

CITY = get_city()
ASSETS = Path(__file__).resolve().parents[2] / "assets"

NAVY, ORANGE, MUTED, INK, RED, DONE = "#1F4E79", "#C9731E", "#C3CCD6", "#1B2430", "#B42318", "#98A2B3"
OUTCOMES_PL = {"pass": "zaliczona", "conditional": "z zastrzeżeniami", "fail": "niezaliczona",
               "no_entry": "brak dostępu", "closed": "działalność zakończona"}
OUTCOME_COLOR = {"pass": "green", "conditional": "orange", "fail": "red", "closed": "gray", "no_entry": "gray"}
QUINTILE_PL = {1: "1 (najbiedniejsze)", 2: "2", 3: "3", 4: "4", 5: "5 (najbogatsze)"}
_WEEKDAYS = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
_MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
           "października", "listopada", "grudnia"]

CSS = """
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 600;}
[data-testid="stMetricLabel"] p {color: #667085; font-weight: 500;}
h2 {letter-spacing: -0.01em;}
</style>
"""


def setup(title: str, logo: str) -> None:
    st.set_page_config(page_title=title, page_icon=str(ASSETS / "icon.svg"), layout="wide")
    st.logo(str(ASSETS / logo), size="large")
    st.html(CSS)


def long_date(day) -> str:
    d = pd.Timestamp(day)
    return f"{_WEEKDAYS[d.weekday()]}, {d.day} {_MONTHS[d.month - 1]} {d.year}"


def header(title: str, subtitle: str | None = None) -> None:
    st.header(title, anchor=False)
    if subtitle:
        st.caption(subtitle)


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------- data, cached ----------------
@st.cache_data(show_spinner="Wczytuję dane…")
def data() -> pd.DataFrame:
    return load(CITY)


@st.cache_resource
def model():
    return load_model(CITY)


@st.cache_data
def socio() -> pd.DataFrame:
    return load_socio(CITY)


@st.cache_data
def eval_results() -> dict:
    return evaluation.load_results(CITY)


@st.cache_data
def area_names() -> dict[int, str]:
    return {a["area"]: a["name"] for a in load_areas(CITY)}


@st.cache_data
def areas_geojson() -> dict:
    s = socio().set_index("area")
    feats = []
    for a in load_areas(CITY):
        props = {"area": a["area"], "name": a["name"]}
        if a["area"] in s.index:
            props.update(income=float(s.loc[a["area"], "income"]), quintile=int(s.loc[a["area"], "income_quintile"]))
        feats.append({"type": "Feature", "properties": props, "geometry": a["geometry"].__geo_interface__})
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------- maps ------------------------
def rgb(hex_color: str, alpha: int = 255) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def pick_place(key: str) -> None:
    """on_select callback of a map: remember the clicked stop; the app opens its card once."""
    objects = st.session_state[key].selection.get("objects", {})
    for layer in ("stops_sel", "stops"):
        if objects.get(layer):
            st.session_state.open_place = int(objects[layer][0]["place_id"])
            return


def route_map(routes: pd.DataFrame, highlight: int | None = None, height: int = 560, names: dict | None = None,
              results: dict | None = None, key: str = "map") -> None:
    """Routes on a map. All routes in one colour; with `highlight`, that inspector's route stands
    out with numbered stops and the others fade. Orange stops are requests, navy ones routine;
    once a result is recorded a stop turns grey, or red for a failed inspection.
    Clicking a stop opens its place card (see pick_place)."""
    if routes.empty:
        st.info("Brak wizyt do pokazania na mapie.")
        return
    names, results = names or {}, results or {}
    pts = routes.sort_values(["inspector", "seq"]).copy()
    on = pts["inspector"].eq(highlight) if highlight else pd.Series(True, index=pts.index)

    def colour(kind, place_id):
        r = results.get(int(place_id))
        return RED if r == "fail" else DONE if r else ORANGE if kind == "zgłoszenie" else NAVY
    pts["fill"] = [rgb(colour(k, p), 255 if o else 70) for k, p, o in zip(pts["kind"], pts["place_id"], on)]
    who = pts["inspector"].map(lambda i: names.get(int(i), f"Inspektor {i}"))
    pts["label"] = who + " · wizyta " + pts["seq"].astype(str) + " · ok. " + pts["eta"]
    pts["num"] = pts["seq"].astype(str)
    pts = pts[["place_id", "name", "address", "lon", "lat", "fill", "label", "num"]]
    paths = pd.DataFrame([{"path": g[["lon", "lat"]].values.tolist(), "inspector": i,
                           "name": names.get(int(i), f"Inspektor {i}"), "address": "", "label": "trasa"}
                          for i, g in routes.sort_values(["inspector", "seq"]).groupby("inspector")])
    chosen = paths["inspector"].eq(highlight) if highlight else pd.Series(True, index=paths.index)
    big = highlight is not None
    dot = dict(get_position=["lon", "lat"], get_fill_color="fill", get_line_color=[255, 255, 255], stroked=True,
               get_radius=1, pickable=True)
    layers = [
        pdk.Layer("PathLayer", paths[~chosen], id="paths", get_path="path", get_color=rgb(MUTED, 200),
                  width_min_pixels=2),
        pdk.Layer("PathLayer", paths[chosen], id="paths_sel", get_path="path", get_color=rgb(NAVY, 230),
                  width_min_pixels=4 if big else 2.5),
    ]
    if big:
        sel = pts[on.to_numpy()]
        layers += [
            pdk.Layer("ScatterplotLayer", pts[~on.to_numpy()], id="stops", line_width_min_pixels=1,
                      radius_min_pixels=4, radius_max_pixels=4, **dot),
            pdk.Layer("ScatterplotLayer", sel, id="stops_sel", line_width_min_pixels=2, radius_min_pixels=11,
                      radius_max_pixels=11, **dot),
            pdk.Layer("TextLayer", sel, id="nums", get_position=["lon", "lat"], get_text="num", get_size=13,
                      get_color=[255, 255, 255, 255], get_text_anchor=String("middle"),
                      get_alignment_baseline=String("center"), font_weight=700),
        ]
        focus = sel
    else:
        layers.append(pdk.Layer("ScatterplotLayer", pts, id="stops", line_width_min_pixels=1.5, radius_min_pixels=5,
                                radius_max_pixels=5, **dot))
        focus = pts
    span = max(focus["lat"].max() - focus["lat"].min(), (focus["lon"].max() - focus["lon"].min()) * 0.75, 0.01)
    zoom = float(np.clip(np.log2(0.9 / span) + 7.6, 9.5, 14))
    view = pdk.ViewState(latitude=float(focus["lat"].mean()), longitude=float(focus["lon"].mean()), zoom=zoom)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=None,
                             tooltip={"text": "{name}\n{address}\n{label}"}), height=height, key=key,
                    on_select=lambda: pick_place(key), selection_mode="single-object")
    legend = ":orange[●] wizyta obowiązkowa  :blue[●] kontrola rutynowa"
    if results:
        legend += "  :gray[●] wynik wpisany  :red[●] niezaliczona"
    st.caption(legend + " · kliknij punkt, żeby otworzyć kartę lokalu")


def area_map(values: pd.Series, title: str, palette=("#EEF2F7", NAVY), height: int = 460) -> None:
    """Neighbourhoods coloured by a value per area id (light = low, dark = high)."""
    gj = json.loads(json.dumps(areas_geojson()))
    v = values.dropna()
    lo, hi = (v.min(), v.max()) if len(v) else (0, 1)
    a, b = np.array(rgb(palette[0])[:3]), np.array(rgb(palette[1])[:3])
    for f in gj["features"]:
        x = values.get(f["properties"]["area"], np.nan)
        t = 0.0 if pd.isna(x) or hi == lo else (x - lo) / (hi - lo)
        f["properties"]["fill"] = [int(c) for c in a + (b - a) * t] + [200 if pd.notna(x) else 40]
        f["properties"]["value"] = "brak danych" if pd.isna(x) else f"{x:.1f}"
    layer = pdk.Layer("GeoJsonLayer", gj, get_fill_color="properties.fill", get_line_color=[255, 255, 255],
                      line_width_min_pixels=1, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=CITY.center_lat, longitude=CITY.center_lon, zoom=9.6),
                             map_style=None, tooltip={"text": "{name}\n" + title + ": {value}"}), height=height)
