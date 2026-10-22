"""Streamlit app: a daily inspection list, a replay of past inspections and the model's quality."""
import datetime as dt

import pandas as pd
import streamlit as st

from inspections import data, model, schedule
from inspections.features import reasons

POLICIES = {"actual": "Order the city used", "random": "Random order", "risk_first": "Highest risk first"}

st.set_page_config(page_title="Chicago inspections DSS", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    return data.load()


@st.cache_resource
def risk_model():
    return model.load_model()


@st.cache_data
def scored_pool(start: pd.Timestamp, days: int) -> pd.DataFrame:
    rows = schedule.score(risk_model(), schedule.pool(load(), start, days))
    rows["why"] = rows.apply(reasons, axis=1)
    return rows


@st.cache_data
def replay(start: pd.Timestamp, days: int):
    return schedule.backtest(scored_pool(start, days), days)


df = load()
first_test = pd.Timestamp(model.SPLITS["test"][0]).date()
last_day = df["date"].max().date()

st.title("Which restaurants should Chicago inspect first?")
st.caption("**Prediction:** how likely a routine inspection is to fail, from the place's past inspections. "
           "**Decision:** the order of visits, given how many inspections fit in a day. "
           "Everything replays real inspections from the City of Chicago open-data portal; the model never saw these dates.")

with st.sidebar:
    st.header("Planning")
    days = st.slider("Planning window (days)", 14, 90, 60, step=7)
    start = st.date_input("Planning date", value=dt.date(2025, 9, 1), min_value=first_test,
                          max_value=last_day - dt.timedelta(days=days))
    st.caption(f"The model was trained on 2015–2023. Dates from {first_test:%d.%m.%Y} are unseen.")

start = pd.Timestamp(start)
rows = scored_pool(start, days)
plan_tab, replay_tab, model_tab = st.tabs(["Daily plan", "Backtest", "Model"])

with plan_tab:
    per_day = len(rows) / max(1, (schedule.capacity(rows, days) > 0).sum())
    st.write(f"**{len(rows):,} places** are due for a routine inspection in the {days} days from "
             f"{start:%d.%m.%Y} (the places the city actually visited). They are ranked by predicted risk as of that morning.")
    k = st.slider("Inspections that fit in a day", 5, 120, int(round(per_day)),
                  help=f"The city did about {per_day:.0f} routine inspections per working day in this window.")
    top = schedule.daily_list(rows, k)

    st.subheader(f"Today's list: {k} places")
    shown = top.assign(pct=top["p_fail"] * 100)
    st.dataframe(
        shown[["name", "address", "facility_type", "risk", "pct", "why"]],
        hide_index=True, use_container_width=True,
        column_config={
            "name": "Name", "address": "Address", "facility_type": "Type",
            "risk": st.column_config.NumberColumn("Risk cat.", help="The city's category: 1 = high, 3 = low"),
            "pct": st.column_config.ProgressColumn("Chance of failing", format="%.0f%%", min_value=0, max_value=100),
            "why": "Why it is on the list",
        })
    st.caption(f"In hindsight, **{top['fail'].mean():.0%}** of these places failed when they were inspected, "
               f"against **{rows['fail'].mean():.0%}** of all places due in the window.")

    st.subheader("Map")
    pts = rows.dropna(subset=["lat", "lon"]).copy()
    on_list = pts.index.isin(top.index)
    pts["color"] = ["#d1453b" if t else "#8a96a366" for t in on_list]
    pts["size"] = [120 if t else 50 for t in on_list]
    st.map(pts.sort_values("p_fail"), latitude="lat", longitude="lon", color="color", size="size", zoom=10, height=480)
    st.caption("Red: today's list. Grey: other places due in the window.")

with replay_tab:
    summary, curves = replay(start, days)
    st.write(f"Replay of **{summary.attrs['inspections']:,} routine inspections** from {start:%d.%m.%Y}, "
             f"**{summary.attrs['fails']:,}** of which failed. Each policy keeps the number of inspections done on "
             "each day and changes only the order, as in the city's own 2014 evaluation.")
    st.caption("Big number: share of the failing places found in the first half of the window. "
               "Below it: how many days sooner, on average, a failing place is reached than in the city's real order.")
    cols = st.columns(3)
    for col, (key, label) in zip(cols, POLICIES.items()):
        s = summary.loc[key]
        col.metric(label, f"{s['fails_in_first_half']:.0%}", f"{s['days_sooner']:+.1f} days sooner",
                   delta_color="normal" if key != "actual" else "off")
    st.line_chart(curves.rename(columns=POLICIES), color=["#4c78a8", "#9aa5b1", "#d1453b"],
                  x_label="Day of the window", y_label="Failing places found so far")
    st.caption("Limits: we can only replay places that were actually inspected, so we never learn what we would have found "
               "elsewhere. The random line averages 20 shuffles.")

with model_tab:
    metrics = model.load_metrics()
    table = pd.DataFrame([
        {"Model": name.replace("_", " "), "Period": {"valid": "2024", "test": "2025–2026"}[split], "AUC": m["auc"],
         "Fail rate in top 10%": 100 * m["fail_rate_top10"], "Fail rate overall": 100 * m["fail_rate_all"],
         "Inspections": m["n"]}
        for name, by_split in metrics.items() for split, m in sorted(by_split.items(), key=lambda kv: kv[0] != "valid")])
    st.dataframe(table, hide_index=True, column_config={
        "AUC": st.column_config.NumberColumn(format="%.3f"),
        "Fail rate in top 10%": st.column_config.NumberColumn(format="%.1f%%"),
        "Fail rate overall": st.column_config.NumberColumn(format="%.1f%%"),
        "Inspections": st.column_config.NumberColumn(format="localized")})
    st.caption("AUC is the chance that a random failing place gets a higher score than a random passing one "
               "(0.5 = guessing, 1 = perfect). The app uses the gradient boosting model.")
