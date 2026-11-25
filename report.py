"""Raport techniczny: how well the system works, kept apart from the product.

Run:  uv run streamlit run report.py --server.port 8502
Reads what `insp train` and `insp evaluate` saved: model quality, the replay of 2025-26 against
the city's own order, team-size scenarios and the fairness audit by neighbourhood income.
"""
import altair as alt
import pandas as pd
import streamlit as st

from inspections import evaluation
from inspections.features import FEATURES_PL
from inspections.model import load_importance, load_metrics
from inspections.ui import CITY, QUINTILE_PL, area_map, eval_results, header, setup, socio

setup("Raport techniczny", "logo-report.svg")
POLICY_ORDER = [evaluation.POLICIES_PL[p] for p in evaluation.POLICIES]


def legend(columns: int = 1) -> alt.Legend:
    """Legend to the right for full-width charts (columns=1), below for half-width ones."""
    if columns == 1:
        return alt.Legend(orient="right", title=None, labelLimit=230)
    return alt.Legend(orient="bottom", title=None, columns=columns, labelLimit=320, columnPadding=32)


def summary() -> pd.DataFrame | None:
    r = eval_results()
    if "summary" not in r:
        st.info("Brak wyników oceny. Uruchom `uv run insp evaluate`.", icon=":material/info:")
        return None
    return r["summary"]


def overview() -> None:
    header("Ocena na danych historycznych",
           "Pięć okien po 60 dni z lat 2025–2026, których model nie widział. Pełny zespół.")
    s = summary()
    if s is None:
        return
    with st.expander("Jak liczymy", icon=":material/info:"):
        st.markdown(
            "- Każdego dnia zgłoszenia zabierają swoją część czasu zespołu, a kontrole rutynowe dostają resztę, "
            "tyle samo co w rzeczywistości. Zmienia się tylko **kolejność** wizyt.\n"
            "- **Dni wcześniej**: o ile wcześniej lokale, które nie przeszły kontroli, zostałyby odwiedzone niż w "
            "kolejności, której faktycznie użyło miasto.\n"
            "- Wynik znamy tylko dla lokali, które miasto skontrolowało, więc porównujemy kolejności tych samych "
            "wizyt. Losowa kolejność to średnia z 20 losowań.")
    full = s[s["multiplier"] == 1.0]
    agg = full.groupby("policy")[["days_sooner", "high_impact_days_sooner", "fails_first_half", "km_per_visit"]].mean()
    agg = agg.reindex(evaluation.POLICIES)
    sysrow, base = agg.loc["system"], agg.loc["actual"]
    k = st.columns(3)
    k[0].metric("Niezaliczone lokale wykryte", f"{sysrow['days_sooner']:+.1f} dnia", border=True,
                help="Pełny system względem kolejności miasta, średnio.")
    k[1].metric("Szkoły, żłobki, opieka", f"{sysrow['high_impact_days_sooner']:+.1f} dnia", border=True)
    k[2].metric("Dojazd na wizytę", f"{sysrow['km_per_visit']:.1f} km", border=True,
                help=f"Kolejność miasta: {base['km_per_visit']:.1f} km na wizytę.")
    agg.index = agg.index.map(evaluation.POLICIES_PL)
    agg["fails_first_half"] *= 100
    st.dataframe(agg.reset_index(names="Polityka"), hide_index=True, width="stretch",
                 column_config={"days_sooner": st.column_config.NumberColumn("Dni wcześniej, wszystkie", format="%+.1f"),
                                "high_impact_days_sooner": st.column_config.NumberColumn("Dni wcześniej, szkoły i opieka", format="%+.1f"),
                                "fails_first_half": st.column_config.NumberColumn("Wykryte w 1. połowie okna", format="%.0f%%"),
                                "km_per_visit": st.column_config.NumberColumn("km na wizytę", format="%.1f")})
    c = eval_results()["curves"]
    st.subheader("Wykrywanie w czasie", anchor=False)
    w = st.selectbox("Okno", evaluation.WINDOWS, index=2, format_func=lambda x: f"60 dni od {pd.Timestamp(x):%d.%m.%Y}")
    cw = c[(c["window"] == w) & (c["multiplier"] == 1.0)].copy()
    cw["Polityka"] = cw["policy"].map(evaluation.POLICIES_PL)
    st.altair_chart(alt.Chart(cw).mark_line(strokeWidth=2).encode(
        x=alt.X("day:Q", title="Dzień okna"), y=alt.Y("found:Q", title="Wykryte niezaliczone lokale"),
        color=alt.Color("Polityka:N", sort=POLICY_ORDER, legend=legend()),
        tooltip=["Polityka", "day", "found"]).properties(height=380), width="stretch")


def scenarios() -> None:
    header("Wielkość zespołu", "Co się dzieje, gdy w terenie jest mniej inspektorów. Średnia z pięciu okien po 60 dni.")
    s = summary()
    if s is None:
        return
    st.caption("Zgłoszenia zawsze idą pierwsze, więc mniejszy zespół uderza w kontrole rutynowe. Lokale nieodwiedzone w "
               "oknie liczą się jako odwiedzone dzień po jego końcu.")
    agg = s.groupby(["multiplier", "policy"], as_index=False)[["days_sooner", "fails_reached", "km_per_visit", "capacity"]].mean()
    agg["Polityka"] = agg["policy"].map(evaluation.POLICIES_PL)
    sel = [p for p in evaluation.POLICIES if p != "random"]
    base = alt.Chart(agg[agg["policy"].isin(sel)]).encode(
        x=alt.X("multiplier:Q", title="Wielkość zespołu", scale=alt.Scale(reverse=True), axis=alt.Axis(format="%", tickCount=4)),
        color=alt.Color("Polityka:N", sort=[evaluation.POLICIES_PL[p] for p in sel],
                        legend=legend()))
    st.altair_chart(base.mark_line(point=True).encode(
        y=alt.Y("fails_reached:Q", title="Odwiedzone niezaliczone lokale", axis=alt.Axis(format="%")))
        .properties(height=280), width="stretch")
    st.altair_chart(base.mark_line(point=True).encode(y=alt.Y("days_sooner:Q", title="Dni wcześniej niż miasto"))
                    .properties(height=280), width="stretch")
    cap = agg[agg["policy"] == "actual"][["multiplier", "capacity"]].copy()
    full = cap.loc[cap["multiplier"] == 1.0, "capacity"].iloc[0]
    cap["Zespół"] = (cap["multiplier"] * 100).round().astype(int).astype(str) + "%"
    cap["Kontrole rutynowe w 60 dni"] = cap["capacity"].round().astype(int)
    cap["Część kontroli rutynowych"] = (100 * cap["capacity"] / full).round().astype(int).astype(str) + "%"
    st.dataframe(cap.sort_values("multiplier", ascending=False)[["Zespół", "Kontrole rutynowe w 60 dni",
                                                                 "Część kontroli rutynowych"]], hide_index=True)
    st.caption("Wizyty obowiązkowe zostają, więc mniejszy zespół traci nieproporcjonalnie dużo kontroli rutynowych. "
               "Opóźnienie lokali nieodwiedzonych w oknie jest liczone do dnia po jego końcu, więc to dolne oszacowanie.")


def fairness() -> None:
    header("Sprawiedliwość wobec dzielnic",
           f"Dzielnice podzielone na piątki według dochodu na mieszkańca (dane miasta, {CITY.socio_year}).")
    st.markdown("Sprawdzamy trzy rzeczy: czy model jest w biedniejszych dzielnicach tak samo trafny, czy ich lokale nie są "
                "wskazywane częściej, niż faktycznie nie przechodzą kontroli (**nadmierne kontrolowanie**), i czy ich "
                "niezaliczone lokale nie są wykrywane później (**niedostateczna ochrona**).")
    met = load_metrics(CITY)
    if met:
        st.subheader("Model", anchor=False)
        st.caption(f"Współrzędne lokalu mogą pośrednio kodować zamożność dzielnicy. Dają tylko "
                   f"{met['location_auc_gain']:+.3f} AUC na walidacji, a nierówności rosną, więc model ich nie używa.")
        rows = []
        for v in ("with_location", "without_location"):
            for r in met[v]["fairness"]:
                rows.append({"Wariant": "ze współrzędnymi" if v == "with_location" else "bez współrzędnych (używany)",
                             "Kwintyl": QUINTILE_PL[r["income_quintile"]], "Wskaźnik obciążenia": r["burden_ratio"],
                             "AUC": r["auc"], "Odsetek fałszywych alarmów": r["false_alarm_rate"],
                             "Przewidziane − faktyczne": r["calibration_gap"]})
        t = pd.DataFrame(rows)
        bars = alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Kwintyl:N", sort=list(QUINTILE_PL.values()), title=None, axis=alt.Axis(labelLimit=200)), yOffset="Wariant:N",
            x=alt.X("Wskaźnik obciążenia:Q", title="Udział wskazań w top 10% / udział niezaliczeń (1 = proporcjonalnie)"),
            color=alt.Color("Wariant:N", scale=alt.Scale(range=["#9AA6B4", "#1F4E79"]),
                            legend=legend(2)))
        rule = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4], color="#667085").encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=300, padding={"left": 24}), width="stretch")
        st.dataframe(t[t["Wariant"].str.startswith("bez")].drop(columns="Wariant"), hide_index=True, width="stretch",
                     column_config={"Wskaźnik obciążenia": st.column_config.NumberColumn(format="%.2f"),
                                    "AUC": st.column_config.NumberColumn(format="%.3f"),
                                    "Odsetek fałszywych alarmów": st.column_config.NumberColumn(format="%.2f"),
                                    "Przewidziane − faktyczne": st.column_config.NumberColumn(format="%+.3f")})
    r = eval_results()
    if "quintiles" in r:
        st.subheader("Plan, odtworzenie 2025–2026", anchor=False)
        q = r["quintiles"][r["quintiles"]["multiplier"] == 1.0]
        agg = q.groupby(["policy", "income_quintile"], as_index=False)[["burden_ratio", "days_sooner"]].mean()
        agg = agg[agg["policy"].isin(["actual", "ahp", "ahp_fair", "system"])]
        agg["Polityka"] = agg["policy"].map(evaluation.POLICIES_PL)
        agg["Kwintyl"] = agg["income_quintile"].map(QUINTILE_PL)
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(agg).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Kwintyl:N", sort=list(QUINTILE_PL.values()), title=None, axis=alt.Axis(labelLimit=200)), yOffset="Polityka:N",
            x=alt.X("burden_ratio:Q", title="Obciążenie wczesnymi wizytami (1 = proporcjonalnie)"),
            color=alt.Color("Polityka:N", legend=legend(2))).properties(height=360, padding={"left": 24}), width="stretch")
        c2.altair_chart(alt.Chart(agg[agg["policy"] != "actual"]).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Kwintyl:N", sort=list(QUINTILE_PL.values()), title=None, axis=alt.Axis(labelLimit=200)), yOffset="Polityka:N",
            x=alt.X("days_sooner:Q", title="Niezaliczone lokale wykryte dni wcześniej"),
            color=alt.Color("Polityka:N", legend=legend(2))).properties(height=360, padding={"left": 24}), width="stretch")
    st.subheader("Dochód na mieszkańca", anchor=False)
    area_map(socio().set_index("area")["income"] / 1000, "dochód (tys. USD)")


def model_quality() -> None:
    header("Model ryzyka", "Szansa, że kontrola rutynowa lokalu zakończy się niezaliczeniem.")
    met = load_metrics(CITY)
    if not met:
        st.info("Brak modelu. Uruchom `uv run insp train`.", icon=":material/info:")
        return
    v = met[met["chosen"]]["metrics"]
    test = v["calibrated"]["test"]
    k = st.columns(3)
    k[0].metric("AUC, 2025–26", f"{test['auc']:.3f}", border=True,
                help="Szansa, że losowy lokal, który nie przeszedł kontroli, dostanie wyższą ocenę niż losowy, który "
                     "przeszedł (0,5 = zgadywanie, 1 = ideał).")
    k[1].metric("Niezaliczenia w top 10%", f"{test['fail_rate_top10']:.0%}", f"ogółem {test['fail_rate_all']:.0%}",
                delta_color="off", border=True)
    k[2].metric("Brier", f"{test['brier']:.3f}", border=True, help="Średni kwadrat błędu prawdopodobieństwa; mniej = lepiej.")
    rows = [{"Model": "Gradient boosting", "Okres": "2024 (walidacja)", **v["gradient_boosting"]["valid"]},
            {"Model": "Gradient boosting", "Okres": "2025–26 (test)", **v["gradient_boosting"]["test"]},
            {"Model": "Regresja logistyczna", "Okres": "2025–26 (test)", **v["logistic"]["test"]},
            {"Model": "Gradient boosting + kalibracja", "Okres": "2025–26 (test)", **v["calibrated"]["test"]}]
    mt = pd.DataFrame(rows)
    mt[["fail_rate_top10", "fail_rate_all"]] *= 100
    st.dataframe(mt, hide_index=True, width="stretch",
                 column_config={"auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                                "brier": st.column_config.NumberColumn("Brier", format="%.3f"),
                                "fail_rate_top10": st.column_config.NumberColumn("Niezaliczenia w top 10%", format="%.1f%%"),
                                "fail_rate_all": st.column_config.NumberColumn("Niezaliczenia ogółem", format="%.1f%%"),
                                "n": st.column_config.NumberColumn("Kontrole", format="localized")})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Kalibracja, 2025–26", anchor=False)
        rel = pd.DataFrame(v["reliability"])
        pts = alt.Chart(rel).mark_line(point=True, color="#1F4E79").encode(
            x=alt.X("predicted:Q", title="Przewidziana szansa", axis=alt.Axis(format="%")),
            y=alt.Y("observed:Q", title="Faktyczny odsetek niezaliczeń", axis=alt.Axis(format="%")))
        diag = alt.Chart(pd.DataFrame({"x": [0.05, 0.5], "y": [0.05, 0.5]})).mark_line(
            strokeDash=[4, 4], color="#9AA6B4").encode(x="x", y="y")
        st.altair_chart((diag + pts).properties(height=320), width="stretch")
    with c2:
        st.subheader("Co model bierze pod uwagę", anchor=False)
        imp = pd.DataFrame(load_importance(CITY))
        imp["Cecha"] = imp["feature"].map(FEATURES_PL).fillna(imp["feature"])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#1F4E79", cornerRadiusEnd=3).encode(
            x=alt.X("auc_drop:Q", title="Spadek AUC po wymieszaniu cechy"), y=alt.Y("Cecha:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
            .properties(height=320), width="stretch")
    st.caption("Trening 2015–2023, kalibracja Platta na 2024, test 2025–26. Model jest umiarkowanie trafny, dlatego plan "
               "łączy ryzyko z innymi kryteriami, a decyzję podejmuje kierownik.")


with st.sidebar:
    st.caption(f"{CITY.name} · wyniki `insp train` i `insp evaluate`. Produkt dla kierownika i inspektorów: `app.py`.")
st.navigation([st.Page(overview, title="Ocena", icon=":material/insights:", url_path="ocena", default=True),
               st.Page(scenarios, title="Wielkość zespołu", icon=":material/groups:", url_path="zespol"),
               st.Page(fairness, title="Sprawiedliwość", icon=":material/balance:", url_path="sprawiedliwosc"),
               st.Page(model_quality, title="Model ryzyka", icon=":material/model_training:", url_path="model")],
              position="top").run()
