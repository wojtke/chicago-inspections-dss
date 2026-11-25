"""Inspekcje sanitarne: the product. The manager plans and approves the day, inspectors carry it out.

Run:  uv run insp demo                 # optional: a lived-in database (team, plans, results)
      uv run streamlit run app.py
No passwords: switch between the manager and any inspector in the sidebar. Clicking a place on a
map or in a list opens its card. The analysis behind the system (model quality, replay of
2025-26, fairness audit) is a separate app: report.py.
"""
import datetime as dt
import json

import altair as alt
import pandas as pd
import streamlit as st

from inspections import planning
from inspections.city import GROUPS
from inspections.data import load_details, violations
from inspections.model import SPLITS
from inspections.pl import count, every, facility_pl, title_case
from inspections.preferences import CRITERIA, CRITERIA_PL, DEFAULT_CRITERIA, DEFAULT_GROUPS, GROUPS_PL, preferences
from inspections.routing import RoutingConfig
from inspections.store import ABSENCE_REASONS, Store
from inspections.ui import (
    CITY,
    NAVY,
    OUTCOME_COLOR,
    OUTCOMES_PL,
    QUINTILE_PL,
    area_names,
    data,
    fmt_int,
    header,
    long_date,
    model,
    route_map,
    setup,
    socio,
)

setup("Inspekcje sanitarne", "logo.svg")
STORE = Store(CITY)
if STORE.users().empty:
    STORE.seed_demo()  # a fresh database gets the demo team; `insp demo` adds plans and results
SERVICE_MIN = RoutingConfig().service_min
FIRST_DAY = pd.Timestamp(SPLITS["test"][0]).date()  # earlier days are in the model's training data
RISK_PL = {1: "wysokie", 2: "średnie", 3: "niskie"}


# ---------------------------------------------------------------- planning, cached ------------
def _judgments(key: str, default: dict) -> dict:
    raw = STORE.setting(key)
    return {(a, b): v for a, b, v in raw} if raw else dict(default)


def current_prefs():
    return preferences(_judgments("ahp_criteria", DEFAULT_CRITERIA), _judgments("ahp_groups", DEFAULT_GROUPS))


def prefs_key() -> str:
    return json.dumps([STORE.setting("ahp_criteria"), STORE.setting("ahp_groups")])


@st.cache_resource(max_entries=4, show_spinner=False)
def history_until(day: str, version: str) -> pd.DataFrame:
    """The city's inspections plus the visits inspectors recorded before `day`. `version` changes
    whenever approvals or results before `day` do, which refreshes the cache. Shared: never mutate."""
    return planning.add_field_visits(data(), STORE.field_visits(before=day))


@st.cache_data(show_spinner="Liczę priorytety lokali…")
def due_list(day: str, prefs_key: str, balance: bool, version: str) -> pd.DataFrame:
    return planning.score_due(history_until(day, version), model(), CITY, pd.Timestamp(day), current_prefs(), socio(),
                              balance)


@st.cache_data(show_spinner=False)
def day_requests(day: str, version: str):
    """(carried, queue): requests earlier plans did not get done, and the day's whole request queue."""
    df = history_until(day, version)
    carried = planning.pending_requests(df, day, STORE.served(day, planning.CARRY_DAYS))
    return carried, planning.mandatory_queue(df, day, carried)


@st.cache_data(show_spinner="Układam trasy…")
def day_plan(day: str, inspectors: int, removed: tuple, pinned: tuple, prefs_key: str, balance: bool, version: str):
    """Routes numbered 1..inspectors; the page maps them onto the chosen people, so changing who
    goes out does not re-solve."""
    carried, _ = day_requests(day, version)
    return planning.plan_day(history_until(day, version), model(), CITY, day, inspectors, current_prefs(),
                             RoutingConfig(), set(removed), set(pinned), due=due_list(day, prefs_key, balance, version),
                             carried=carried)


@st.cache_data(show_spinner=False)
def last_full_day() -> dt.date:
    return planning.last_full_day(data()).date()


@st.cache_data(show_spinner=False, max_entries=64)
def place_details(place_id: int) -> pd.DataFrame:
    return load_details(CITY, place_id)


def team_names() -> dict[int, str]:
    u = STORE.users()
    u = u[u["role"] == "inspector"]
    return dict(zip(u["inspector_no"].astype(int), u["display_name"]))


def user_names() -> pd.Series:
    return STORE.users().set_index("username")["display_name"]


def end_time(eta: str) -> str:
    h, m = map(int, eta.split(":"))
    t = h * 60 + m + SERVICE_MIN
    return f"{t // 60:02d}:{t % 60:02d}"


def kind_badge(kind: str, where=st) -> None:
    where.badge("Wizyta obowiązkowa" if kind == "zgłoszenie" else "Kontrola rutynowa",
                color="orange" if kind == "zgłoszenie" else "blue")


OUTCOME_ICON = {"pass": "check_circle", "conditional": "error", "fail": "cancel", "no_entry": "lock", "closed": "block"}


def outcome_badge(outcome: str, where=st) -> None:
    where.badge(OUTCOMES_PL[outcome].capitalize(), icon=f":material/{OUTCOME_ICON[outcome]}:", color=OUTCOME_COLOR[outcome])


def open_card_on_select(key: str, place_ids: list):
    """on_select callback of a table: remember the clicked row's place; main() opens its card once."""
    def callback():
        rows = st.session_state[key].selection.rows
        if rows:
            st.session_state.open_place = int(place_ids[rows[0]])
    return callback


def card_button(place_id: int, key: str, where=st) -> None:
    if where.button("Karta lokalu", key=key, icon=":material/storefront:"):
        st.session_state.open_place = int(place_id)


def nav_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}"


def set_day(day: str) -> None:
    st.session_state.day_input = pd.Timestamp(day).date()


# ---------------------------------------------------------------- place card -----------------
def place_card(place_id: int) -> None:
    day = st.session_state.day
    df = history_until(day, STORE.version(before=day))
    prof = planning.place_profile(df, model(), CITY, day, place_id)
    if prof is None:
        st.toast("Brak danych o tym lokalu przed wybranym dniem.", icon=":material/info:")
        return
    st.dialog(title_case(prof["name"]), width="large")(_card)(place_id, prof, day)


def _card(pid: int, prof: dict, day: str) -> None:
    hist = planning.history(history_until(day, STORE.version(before=day)), pid, before=day)
    det = place_details(pid).merge(hist[["inspection_id", "date"]].astype({"inspection_id": str}),
                                   on="inspection_id").sort_values("date")  # known before the day, oldest first
    now = st.session_state.get("plan_now", {})
    now = now if now.get("day") == day else {}
    routes, queue, due = now.get("routes"), now.get("queue"), now.get("due")
    if routes is None:
        approved = STORE.plan(day)
        routes = approved[0] if approved else None

    def latest(col):
        return det[col].dropna().iloc[-1] if col in det and det[col].notna().any() else ""
    aka, zipc = latest("aka"), latest("zip")
    line = f"**{title_case(prof['address'])}**{', ' + zipc if zipc else ''} · {facility_pl(prof['facility_type'])} · licencja {pid}"
    if aka and aka.strip().upper() != str(prof["name"]).strip().upper():
        line += f" · znany też jako {title_case(aka)}"
    st.markdown(line)

    b = st.container(horizontal=True)
    b.badge("Klienci: " + GROUPS_PL[str(prof["facility_group"])], color="violet")
    if pd.notna(prof["risk"]):
        r = int(prof["risk"])
        b.badge(f"Kategoria ryzyka miasta: {r} ({RISK_PL.get(r, '')}), kontrola {every(prof['every_days'])}", color="gray")
    if prof.get("last_closed") == 1:
        b.badge("Działalność zakończona", color="gray", icon=":material/block:")
    elif prof["due_in_days"] is None:
        b.badge("Bez zakończonej kontroli", color="orange", icon=":material/schedule:")
    elif prof["due_in_days"] < 0:
        b.badge(f"Kontrola zaległa o {count(-prof['due_in_days'], 'dzień', 'dni', 'dni')}", color="red",
                icon=":material/schedule:")
    else:
        b.badge(f"Następna kontrola za {count(prof['due_in_days'], 'dzień', 'dni', 'dni')}", color="green",
                icon=":material/event_available:")

    _card_today(pid, day, routes, queue, due)

    area, socio_rows = int(prof.get("area") or 0), socio().set_index("area")
    k = st.columns(4)
    k[0].metric("Szansa niezaliczenia", f"{prof['p_fail']:.0%}", border=True,
                help="Ocena modelu na podstawie historii lokalu. To co innego niż kategoria ryzyka miasta, która "
                     "opisuje rodzaj działalności.")
    last = prof["last_done"]
    k[1].metric("Ostatnia kontrola", "brak" if pd.isna(last) else f"{last:%d.%m.%Y}", border=True,
                help="Ostatnia zakończona kontrola (z wynikiem). Próby bez dostępu się nie liczą.")
    fails = int(prof["n_past_fails"])
    k[2].metric("Kontrole w historii", int(prof["n_past"]), border=True,
                help=f"W tym {count(fails, 'niezaliczona', 'niezaliczone', 'niezaliczonych')}.")
    q = int(socio_rows.loc[area, "income_quintile"]) if area in socio_rows.index else None
    k[3].metric("Dzielnica", area_names().get(area, "poza granicami"), border=True,
                help=None if q is None else f"Piątka dochodu: {QUINTILE_PL[q]}")

    _card_history(pid, day, hist, det)
    st.link_button("Nawiguj (Mapy Google)", nav_url(prof["lat"], prof["lon"]), icon=":material/directions:")


def _card_today(pid: int, day: str, routes, queue, due) -> None:
    """Where the place stands in this day's plan, and the manager's actions on it."""
    names = team_names()
    in_route = pd.DataFrame()
    if routes is not None and len(routes):
        ids = routes["place_ids"] if "place_ids" in routes else routes["place_id"].map(lambda p: [p])
        in_route = routes[ids.map(lambda x: pid in x)]
    res = STORE.results(day)
    res = res[res["place_id"] == (int(in_route["place_id"].iloc[0]) if len(in_route) else pid)]
    row = due[due["place_id"] == pid] if due is not None else pd.DataFrame()
    with st.container(border=True):
        if len(in_route):
            s = in_route.iloc[0]
            st.markdown(f":material/route: **W planie na ten dzień:** {names.get(int(s['inspector']), '?')}, wizyta "
                        f"{s['seq']}, ok. {s['eta']}")
            st.caption(f"{'Wizyta obowiązkowa' if s['kind'] == 'zgłoszenie' else 'Kontrola rutynowa'}: {s['why']}")
            if len(res):
                outcome_badge(res["outcome"].iloc[0])
                if res["note"].iloc[0]:
                    st.caption(f"Uwagi inspektora: {res['note'].iloc[0]}")
        elif queue is not None and len(queue) and queue["place_ids"].map(lambda ids: pid in ids).any():
            req = queue[queue["place_ids"].map(lambda ids: pid in ids)].iloc[0]
            st.markdown(f":material/priority_high: **Wizyta obowiązkowa poza planem:** {req['request']}")
        elif len(row):
            st.markdown(f":material/format_list_numbered: **Zaległy lokal, pozycja {int(row['rank'].iloc[0])} w kolejce.**")
        else:
            st.markdown(":material/event_available: W tym dniu nie czeka na kontrolę.")
        if len(row):
            r = row.iloc[0]
            parts = " + ".join(f"{CRITERIA_PL[c].lower()} {r[f'w_{c}']:.2f}" for c in CRITERIA)
            st.caption(f"Priorytet {r['priority']:.2f} (na 1) = {parts}.")
        _card_actions(pid, day, routes, row)


def _card_actions(pid: int, day: str, routes, due_row) -> None:
    """Pin or skip a due place in the draft plan (manager only)."""
    s = st.session_state
    now = s.get("plan_now", {})
    if s.user["role"] != "manager" or not now.get("draft") or now.get("day") != day or not len(due_row):
        return
    removed = s.setdefault("removed", {}).setdefault(day, set())
    pinned = s.setdefault("pinned", {}).setdefault(day, set())
    planned = routes is not None and pid in set(routes["place_id"])
    a = b = st.container(horizontal=True)
    if pid in removed:
        if a.button("Przywróć", icon=":material/undo:"):
            removed.discard(pid)
            st.rerun()
        return
    if pid in pinned:
        if a.button("Odepnij", icon=":material/keep_off:"):
            pinned.discard(pid)
            st.rerun()
    elif not planned:
        if a.button("Dodaj do planu", type="primary", icon=":material/add_location:",
                    help="Lokal dostaje pierwszeństwo przed innymi kontrolami rutynowymi; ustępuje tylko wizytom "
                         "obowiązkowym."):
            pinned.add(pid)
            st.rerun()
    if b.button("Pomiń dziś", icon=":material/event_busy:"):
        removed.add(pid)
        pinned.discard(pid)
        st.rerun()


def _card_history(pid: int, day: str, h: pd.DataFrame, det: pd.DataFrame) -> None:
    if h.empty:
        return
    notes = STORE.results()
    notes = notes[notes["place_id"] == pid].assign(date=lambda x: pd.to_datetime(x["day"]))
    notes["kto"] = notes["username"].map(user_names())
    h = h.merge(notes[["date", "note", "kto"]], on="date", how="left").fillna({"note": "", "kto": ""})
    h["n_violations"] = h["n_violations"].map(lambda x: "" if pd.isna(x) else str(int(x)))
    st.markdown("**Historia kontroli**")
    ev = st.dataframe(h[["date", "inspection_type", "result", "n_violations", "kto", "note"]], hide_index=True,
                      width="stretch", height=min(36 * len(h) + 38, 260), on_select="rerun",
                      selection_mode="single-row", key=f"hist_{pid}_{day}",
                      column_config={"date": st.column_config.DateColumn("Data", format="DD.MM.YYYY"),
                                     "inspection_type": "Rodzaj", "result": "Wynik",
                                     "n_violations": st.column_config.TextColumn("Naruszenia", width="small"),
                                     "kto": "Inspektor", "note": "Uwagi"})
    rows = ev.selection.rows
    with_v = h[pd.to_numeric(h["n_violations"], errors="coerce").fillna(0) > 0]
    pick = h.iloc[rows[0]] if rows else (with_v.iloc[0] if len(with_v) else None)
    if pick is None or "violations" not in det:
        return
    text = det.loc[det["inspection_id"] == str(pick["inspection_id"]), "violations"]
    items = violations(text.iloc[0], CITY.violations_sep) if len(text) else []
    if not items:
        st.caption("Ta kontrola nie ma opisanych naruszeń. Kliknij inną w tabeli.")
        return
    st.markdown(f"**Naruszenia z kontroli {pick['date']:%d.%m.%Y}** · {count(len(items), 'pozycja', 'pozycje', 'pozycji')}")
    st.caption("Opis z protokołu miasta (po angielsku). Kliknij inną kontrolę w tabeli, żeby zobaczyć jej naruszenia.")
    with st.container(height=min(70 * len(items) + 20, 280), border=False):
        for v in items:
            st.markdown(f"**{v['no']}. {v['title']}**" if v["no"] else f"**{v['title']}**")
            if v["comment"]:
                st.caption(v["comment"][:400] + ("…" if len(v["comment"]) > 400 else ""))


# ---------------------------------------------------------------- manager: day plan -----------
def balance_check(routes: pd.DataFrame, due: pd.DataFrame) -> None:
    """Today's routine visits per income fifth against what its expected failures would justify.
    A fifth is flagged when its count falls outside a rough binomial range (about two standard deviations)."""
    due = due.dropna(subset=["income_quintile"])
    routine = routes[routes["kind"] == "rutynowa"]
    exp_share = due.groupby("income_quintile")["p_fail"].sum() / due["p_fail"].sum()
    got = routine.merge(due[["place_id", "income_quintile"]], on="place_id").groupby("income_quintile").size()
    got = got.reindex(exp_share.index, fill_value=0)
    exp = exp_share * got.sum()
    off = [QUINTILE_PL[int(q)] for q in exp.index if abs(got[q] - exp[q]) > 2 * exp[q] ** 0.5 + 1]
    with st.popover("Rozkład wg dochodu", icon=":material/balance:" if not off else ":material/warning:",
                    help="W normie" if not off else "Sprawdź: " + ", ".join(off), width="stretch"):
        st.markdown("**Kontrole rutynowe według zamożności dzielnic**")
        t = pd.DataFrame({"Kwintyl": [QUINTILE_PL[int(i)] for i in exp.index] * 2, "Wizyty": list(exp) + list(got),
                          "Seria": ["proporcjonalnie do ryzyka"] * len(exp) + ["w planie"] * len(got)})
        st.altair_chart(alt.Chart(t).mark_bar(cornerRadiusEnd=3, height=10).encode(
            y=alt.Y("Kwintyl:N", sort=list(QUINTILE_PL.values()), title=None),
            yOffset=alt.YOffset("Seria:N", sort="descending"),
            x=alt.X("Wizyty:Q", title=None), tooltip=["Kwintyl", "Seria", alt.Tooltip("Wizyty:Q", format=".1f")],
            color=alt.Color("Seria:N", scale=alt.Scale(range=["#9AA6B4", NAVY]), legend=alt.Legend(orient="bottom", title=None)))
            .properties(height=230, width=380))
        if off:
            st.warning("Odbiega od proporcji: " + ", ".join(off) + ". Rozważ dodanie do planu lokali z tych dzielnic "
                       "(Kolejka lokali → karta lokalu).", icon=":material/warning:")
        st.caption(f"Dzielnice podzielone na piątki według dochodu na mieszkańca (dane miasta, {CITY.socio_year}). To "
                   "zgrubna kontrola: każda piątka powinna dostać mniej więcej tyle kontroli, ile uzasadnia ryzyko jej lokali.")


def start_editing(day: str, approved_meta: dict, options: list[int]) -> None:
    """Edit an approved plan: a draft that starts from the inputs it was approved with."""
    s, p = st.session_state, approved_meta.get("params") or {}
    s.editing.add(day)
    s.removed[day], s.pinned[day] = set(p.get("removed", [])), set(p.get("pinned", []))
    s.inputs[day] = {"team": [n for n in p.get("team", options) if n in options], "balance": p.get("balance", True)}


def drop_draft(day: str) -> None:
    s = st.session_state
    s.editing.discard(day)
    for d in (s.removed, s.pinned, s.inputs):
        d.pop(day, None)


def manager_plan() -> None:
    s, day = st.session_state, st.session_state.day
    for k, v in (("removed", {}), ("pinned", {}), ("editing", set()), ("inputs", {})):
        s.setdefault(k, v)
    names, version = team_names(), STORE.version(before=day)
    approved = STORE.plan(day)
    results = STORE.results(day)
    if day in s.editing and len(results):
        drop_draft(day)
        st.toast("Inspektorzy zaczęli wpisywać wyniki, więc edycję planu anulowano.", icon=":material/lock:")
    draft = approved is None or day in s.editing
    options = STORE.available(day)["inspector_no"].astype(int).tolist()
    absent = STORE.absences(day)

    top, actions = st.columns([3, 2], vertical_alignment="bottom")
    with top:
        header("Plan dnia", f"{long_date(day)} · {CITY.name}")
        status = st.container()
    bar = st.columns([1.6, 1.5, 1.7, 1.2], vertical_alignment="bottom")

    rem = pin = ()
    if draft:
        # The draft's inputs live in s.inputs (widget state is dropped when the page is left); the
        # widgets are seeded from it and write back, with no default values to conflict with.
        inp = s.inputs.setdefault(day, {"team": options, "balance": True})
        for key, value in ((f"team_{day}", [n for n in inp["team"] if n in options]), (f"bal_{day}", inp["balance"])):
            if key not in s:  # first run on this page: seed the widget; otherwise it holds the user's change
                s[key] = value
        with bar[0].popover(f"W terenie: {count(len(s[f'team_{day}']), 'inspektor', 'inspektorów', 'inspektorów')}",
                            icon=":material/groups:", width="stretch"):
            chosen = inp["team"] = st.multiselect("Kto dziś wychodzi w teren", options, format_func=names.get,
                                                  key=f"team_{day}")
            if len(absent):
                st.caption("Nieobecni: " + ", ".join(f"{user_names().get(u, u)} ({r})"
                                                     for u, r in zip(absent["username"], absent["reason"])))
            st.page_link(s.pages["team"], label="Zespół i nieobecności", icon=":material/badge:")
        balance = inp["balance"] = bar[1].toggle("Równoważenie dzielnic", key=f"bal_{day}",
                                help="Dzieli kontrole rutynowe między biedniejsze i bogatsze dzielnice proporcjonalnie "
                                     "do oczekiwanej liczby niezaliczeń.")
        if not chosen:
            st.warning("Wybierz inspektorów, którzy wychodzą dziś w teren.", icon=":material/groups:")
            return
        rem, pin = tuple(sorted(s.removed.get(day, ()))), tuple(sorted(s.pinned.get(day, ())))
        if approved and approved[1]["params"] == planning.plan_params(chosen, rem, pin, balance, prefs_key()):
            routes = approved[0]  # nothing changed yet: keep the approved routes rather than re-solve
            due = due_list(day, prefs_key(), balance, version)
            _, queue = day_requests(day, version)
        else:
            plan = day_plan(day, len(chosen), rem, pin, prefs_key(), balance, version)
            routes = plan.routes.assign(inspector=plan.routes["inspector"].map(lambda i: chosen[i - 1]))
            due, queue = plan.due, plan.queue
    else:
        routes = approved[0]
        balance = (approved[1]["params"] or {}).get("balance", True)
        n_team = routes["inspector"].nunique()
        bar[0].button(f"W terenie: {count(n_team, 'inspektor', 'inspektorów', 'inspektorów')}", icon=":material/groups:",
                      disabled=True, width="stretch")
        bar[1].toggle("Równoważenie dzielnic", value=balance, disabled=True)
        due = due_list(day, prefs_key(), balance, version)
        _, queue = day_requests(day, version)
    s.plan_now = {"day": day, "routes": routes, "queue": queue, "due": due, "draft": draft}
    done = dict(zip(results["place_id"].astype(int), results["outcome"])) if approved else {}

    with bar[3]:
        balance_check(routes, due)
    with status:
        if approved and not draft:
            meta = approved[1]
            st.badge(f"Zatwierdzony · {user_names().get(meta['approved_by'], meta['approved_by'])} · "
                     f"{pd.Timestamp(meta['approved_at']):%d.%m %H:%M}", icon=":material/check_circle:", color="green")
        elif approved:
            st.badge("Edycja zatwierdzonego planu · inspektorzy widzą poprzednią wersję", icon=":material/edit:",
                     color="orange")
        else:
            st.badge("Szkic · niezatwierdzony", icon=":material/draft:", color="gray")
        later = [d for d in STORE.plan_days() if d > day]
        if draft and later:
            st.caption(f"Plan na {pd.Timestamp(min(later)):%d.%m} jest już zatwierdzony i nie uwzględnia tego dnia.")
    with actions:
        a, b = st.columns(2)
        if draft:
            if a.button("Zatwierdź plan", type="primary", icon=":material/check:", width="stretch",
                        help="Inspektorzy zobaczą swoje trasy w widoku „Moje wizyty”."):
                try:
                    STORE.approve_plan(day, routes, s.user["username"], len(chosen),
                                       planning.plan_params(chosen, rem, pin, balance, prefs_key()))
                    st.toast("Plan zatwierdzony. Inspektorzy widzą swoje trasy.", icon=":material/check_circle:")
                except ValueError:
                    st.toast("Inspektorzy wpisali już wyniki na ten dzień, więc planu nie można zmienić.",
                             icon=":material/lock:")
                drop_draft(day)
                st.rerun()
            if approved and b.button("Odrzuć zmiany", icon=":material/close:", width="stretch"):
                drop_draft(day)
                st.rerun()
        elif a.button("Edytuj plan", icon=":material/edit:", width="stretch", disabled=len(results) > 0,
                      help="Inspektorzy wpisali już wyniki na ten dzień, więc plan jest zablokowany." if len(results)
                      else "Zmień zespół, przypnij lub pomiń lokale i zatwierdź nową wersję."):
            start_editing(day, approved[1], options)
            st.rerun()
        if not (draft and approved):
            out = routes.assign(inspektor=routes["inspector"].map(names)).drop(columns=["place_ids"], errors="ignore")
            b.download_button("Eksport CSV", out.to_csv(index=False).encode("utf-8"), file_name=f"plan_{day}.csv",
                              mime="text/csv", icon=":material/download:", width="stretch")

    routine = routes[routes["kind"] == "rutynowa"]
    n_req = len(routes) - len(routine)
    k = st.columns(5)
    k[0].metric("Wizyty obowiązkowe", f"{n_req} z {len(queue)}", border=True,
                help="Skargi, kontrole licencyjne i ponowne kontrole na ten dzień, także zaległe z poprzednich dni "
                     "i ponowne kontrole po niezaliczeniu. Mają pierwszeństwo.")
    k[1].metric("Kontrole rutynowe", len(routine), border=True, help="Pozostały czas zespołu, najważniejsze zaległe lokale.")
    k[2].metric("Dojazd łącznie", f"ok. {routes['km_from_prev'].sum():.0f} km", border=True,
                help="Odległość w linii ulic między kolejnymi wizytami, bez dojazdu do pierwszej i powrotu z ostatniej.")
    k[3].metric("Zaległe lokale", fmt_int(len(due)), border=True,
                help="Ostatnia zakończona kontrola starsza niż wymaga kategoria ryzyka lokalu.")
    k[4].metric("Wpisane wyniki", f"{sum(p in done for p in routes['place_id'])} z {len(routes)}" if approved else "–",
                border=True, help="Wyniki wizyt z zatwierdzonego planu wpisane przez inspektorów.")
    if draft and n_req < len(queue):
        left = len(queue) - n_req
        st.warning(f"{count(left, 'wizyta obowiązkowa', 'wizyty obowiązkowe', 'wizyt obowiązkowych')} poza planem: "
                   f"nie mieszczą się w dniu. Dodaj inspektorów albo zatwierdź plan bez nich: wrócą w kolejnych dniach "
                   f"(do {planning.CARRY_DAYS} dni).", icon=":material/warning:")
    missed = [p for p in pin if p not in set(routes["place_id"])]
    if draft and missed:
        miss = due[due["place_id"].isin(missed)]["name"].map(title_case)
        st.warning("Nie zmieściły się mimo przypięcia: " + ", ".join(miss) + ".", icon=":material/warning:")

    t_routes, t_queue, t_requests = st.tabs(["Trasy", f"Kolejka lokali ({fmt_int(len(due))})",
                                             f"Wizyty obowiązkowe ({len(queue)})"])
    with t_routes:
        routes_tab(routes, names, done, day)
    with t_queue:
        queue_tab(due, routes, day, draft)
    with t_requests:
        requests_tab(queue, routes, names, done, day)


def routes_tab(r: pd.DataFrame, names: dict, done: dict, day: str) -> None:
    r = r.assign(req=r["kind"].eq("zgłoszenie"), wynik=r["place_id"].isin(done))
    per = (r.groupby("inspector").agg(wizyty=("seq", "size"), obowiazkowe=("req", "sum"), start=("eta", "min"),
                                      koniec=("eta", "max"), km=("km_from_prev", "sum"), wyniki=("wynik", "sum"))
           .reset_index())
    per["koniec"] = per["koniec"].map(end_time)
    per["kto"] = per["inspector"].map(names)
    per["postep"] = per["wyniki"] / per["wizyty"]
    info = per.set_index("inspector")
    left, right = st.columns([3, 2], gap="medium")
    with right:
        sel = st.selectbox("Trasa", [0] + per["inspector"].tolist(), key=f"route_sel_{day}",
                           format_func=lambda i: "Wszyscy inspektorzy" if i == 0 else
                           f"{names.get(i, i)} · {count(info.loc[i, 'wizyty'], 'wizyta', 'wizyty', 'wizyt')} · "
                           f"ok. {info.loc[i, 'km']:.0f} km")
        if sel == 0:
            cols = ["kto", "wizyty", "obowiazkowe", "start", "koniec", "km"] + (["postep"] if done else [])
            st.dataframe(per[cols], hide_index=True, width="stretch", height=470,
                         column_config={"kto": "Inspektor", "wizyty": "Wizyty", "obowiazkowe": "Obowiązkowe",
                                        "start": "Start", "koniec": "Koniec wizyt",
                                        "km": st.column_config.NumberColumn("km", format="%.0f"),
                                        "postep": st.column_config.ProgressColumn("Wyniki", format="percent",
                                                                                  min_value=0, max_value=1)})
        else:
            with st.container(height=470, border=False):
                for _, s in r[r["inspector"] == sel].sort_values("seq").iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{s['seq']}. {title_case(s['name'])}**  \n:gray[ok. {s['eta']} · "
                                    f"{title_case(s['address'])}]")
                        row = st.container(horizontal=True, vertical_alignment="center")
                        kind_badge(s["kind"], row)
                        if int(s["place_id"]) in done:
                            outcome_badge(done[int(s["place_id"])], row)
                        st.caption(str(s["why"]))
                        card_button(s["place_id"], f"rc_{day}_{s['place_id']}")
    with left:
        route_map(r, highlight=sel or None, height=520, names=names, results=done, key=f"map_{day}_{sel}")


def queue_tab(due: pd.DataFrame, routes: pd.DataFrame, day: str, draft: bool) -> None:
    s = st.session_state
    rem, pin = s.removed.get(day, set()), s.pinned.get(day, set())
    search = st.text_input("Szukaj lokalu", placeholder="Szukaj po nazwie lub adresie", key=f"qs_{day}",
                           label_visibility="collapsed", icon=":material/search:")
    view = due
    if search:
        hit = (due["name"].str.contains(search, case=False, regex=False)
               | due["address"].str.contains(search, case=False, regex=False))
        view = due[hit]
    top = view.head(200).copy()
    planned = set(routes["place_id"])
    top["status"] = ["w planie" + (" (przypięty)" if p in pin else "") if p in planned else
                     "przypięty, nie zmieścił się" if p in pin else "pominięty dziś" if p in rem else ""
                     for p in top["place_id"]]
    top["ryzyko"] = (top["p_fail"] * 100).round()
    top["klienci"] = top["facility_group"].astype(str).map(GROUPS_PL)
    top["lokal"], top["adres"] = top["name"].map(title_case), top["address"].map(title_case)
    key = f"q_{day}_{search}"
    st.dataframe(top[["rank", "status", "lokal", "adres", "klienci", "ryzyko", "priority", "why"]], hide_index=True,
                 width="stretch", height=520, key=key,
                 on_select=open_card_on_select(key, top["place_id"].tolist()), selection_mode="single-row",
                 column_config={"rank": st.column_config.NumberColumn("Poz.", width="small"), "lokal": "Lokal",
                                "adres": "Adres", "klienci": "Klienci",
                                "ryzyko": st.column_config.ProgressColumn("Szansa niezaliczenia", format="%d%%",
                                                                          min_value=0, max_value=100),
                                "priority": st.column_config.ProgressColumn("Priorytet", format="%.2f", min_value=0,
                                                                            max_value=1),
                                "why": st.column_config.TextColumn("Uzasadnienie", width="large"), "status": "Status"})
    shown = f"Pokazano {len(top)} z {fmt_int(len(view))}" + (" pasujących" if search else "") + ". "
    st.caption(shown + ("Kliknij wiersz, żeby otworzyć kartę lokalu, dodać go do planu albo pominąć dziś."
                        if draft else "Kliknij wiersz, żeby otworzyć kartę lokalu. Zmiany w planie: „Edytuj plan”."))
    if draft and (rem or pin):
        if st.button(f"Cofnij zmiany (przypięte: {len(pin)}, pominięte: {len(rem)})", icon=":material/undo:"):
            s.removed.pop(day, None)
            s.pinned.pop(day, None)
            st.rerun()


def requests_tab(queue: pd.DataFrame, routes: pd.DataFrame, names: dict, done: dict, day: str) -> None:
    if queue.empty:
        st.info("Na ten dzień nie ma wizyt obowiązkowych.", icon=":material/inbox:")
        return
    q = queue.merge(routes[["place_id", "inspector", "eta"]], on="place_id", how="left")
    q["przydzial"] = [f"{names.get(int(i), i)}, ok. {e}" if pd.notna(i) else "poza planem"
                      for i, e in zip(q["inspector"], q["eta"])]
    q["wynik"] = q["place_id"].map(lambda p: OUTCOMES_PL.get(done.get(int(p)), ""))
    q["lokal"], q["adres"] = q["name"].map(title_case), q["address"].map(title_case)
    q["rodzaj"] = q["facility_type"].map(facility_pl)
    key = f"req_{day}"
    st.dataframe(q[["request", "lokal", "adres", "rodzaj", "przydzial", "wynik"]], hide_index=True,
                 width="stretch", key=key, on_select=open_card_on_select(key, q["place_id"].tolist()),
                 selection_mode="single-row",
                 height=min(35 * len(q) + 38, 600),
                 column_config={"request": st.column_config.TextColumn("Powód", width="large"), "lokal": "Lokal",
                                "adres": "Adres", "rodzaj": "Rodzaj lokalu", "przydzial": "Przydział", "wynik": "Wynik"})
    st.caption("Kliknij wiersz, żeby otworzyć kartę lokalu.")


# ---------------------------------------------------------------- manager: team ---------------
def manager_team() -> None:
    day = st.session_state.day
    d = pd.Timestamp(day)
    header("Zespół", f"Inspektorzy, ich dostępność i wizyty z ostatnich 30 dni · stan na {d:%d.%m.%Y}")
    users = STORE.users()
    team = users[users["role"] == "inspector"].copy()
    absent = STORE.absences(day).drop_duplicates("username").set_index("username")
    approved = STORE.plan(day)
    today = approved[0]["inspector"].value_counts() if approved else pd.Series(dtype=int)
    v = STORE.visits((d - pd.Timedelta(days=30)).date().isoformat(), day)
    v["missing"] = v["outcome"].isna() & (v["day"] < day)
    per = v.groupby("inspector").agg(visits=("place_id", "size"), missing=("missing", "sum"))

    def status(u):
        if not u["active"]:
            return "nieaktywny"
        if u["username"] in absent.index:
            a = absent.loc[u["username"]]
            return f"{a['reason']} do {pd.Timestamp(a['end']):%d.%m}"
        n = int(today.get(u["inspector_no"], 0))
        return f"w terenie · {count(n, 'wizyta', 'wizyty', 'wizyt')}" if n else "dostępny"
    team["status"] = [status(u) for _, u in team.iterrows()]
    team["visits"] = team["inspector_no"].map(per["visits"]).fillna(0).astype(int)
    team["missing"] = team["inspector_no"].map(per["missing"]).fillna(0).astype(int)

    active = team[team["active"]]
    k = st.columns(3)
    k[0].metric("Aktywni inspektorzy", len(active), border=True)
    k[1].metric(f"Dostępni {d:%d.%m}", len(STORE.available(day)), border=True)
    k[2].metric("Wizyty bez wyniku", int(team["missing"].sum()), border=True,
                help="Wizyty z zatwierdzonych planów z poprzednich dni, dla których nie wpisano wyniku. Lokal nie liczy "
                     "się jako skontrolowany, dopóki wynik nie zostanie wpisany.")

    with st.form("team_form", border=False):
        ed = st.data_editor(team[["inspector_no", "display_name", "active", "status", "visits", "missing"]],
                            hide_index=True, width="stretch", key="team_editor", height=35 * len(team) + 38,
                            disabled=["inspector_no", "status", "visits", "missing"],
                            column_config={"inspector_no": st.column_config.NumberColumn("Nr", width="small"),
                                           "display_name": st.column_config.TextColumn("Imię i nazwisko", required=True),
                                           "active": st.column_config.CheckboxColumn("Aktywny", width="small"),
                                           "status": f"W dniu {d:%d.%m}",
                                           "visits": st.column_config.NumberColumn("Wizyty (30 dni)"),
                                           "missing": st.column_config.NumberColumn("Bez wyniku")})
        if st.form_submit_button("Zapisz zmiany", icon=":material/save:"):
            changed = 0
            for (_, old), (_, new) in zip(team.iterrows(), ed.iterrows()):
                name = str(new["display_name"] or "").strip()
                if name and (name != old["display_name"] or bool(new["active"]) != bool(old["active"])):
                    STORE.update_user(old["username"], name, bool(new["active"]))
                    changed += 1
            st.toast(f"Zapisano zmiany: {changed}." if changed else "Bez zmian.", icon=":material/check_circle:")
            st.rerun()
    st.caption("Inspektorów się nie usuwa: nieaktywni zostają w historii planów i wyników, ale nie trafiają do nowych planów.")

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("**Nieobecności**")
        with st.form("absence", border=False, clear_on_submit=True):
            who = st.selectbox("Inspektor", active["username"].tolist(),
                               format_func=active.set_index("username")["display_name"].get)
            dates = st.date_input("Od – do", value=(d.date(), d.date()), format="DD.MM.YYYY")
            reason = st.selectbox("Powód", ABSENCE_REASONS)
            if st.form_submit_button("Dodaj nieobecność", icon=":material/event_busy:"):
                dates = dates if isinstance(dates, (tuple, list)) else (dates,)
                STORE.add_absence(who, dates[0].isoformat(), dates[-1].isoformat(), reason)
                st.rerun()
        a = STORE.absences()
        a = a[a["end"] >= (d - pd.Timedelta(days=14)).date().isoformat()]
        names = users.set_index("username")["display_name"]
        for _, r in a.iterrows():
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(f"{names.get(r['username'], r['username'])} · {r['reason']}  \n"
                        f":gray[{pd.Timestamp(r['start']):%d.%m} – {pd.Timestamp(r['end']):%d.%m.%Y}]")
            if c2.button("Usuń", key=f"abs_{r['id']}", type="tertiary", icon=":material/delete:"):
                STORE.remove_absence(r["id"])
                st.rerun()
        if a.empty:
            st.caption("Brak bieżących i planowanych nieobecności.")
    with right, st.container(border=True):
        st.markdown("**Nowy inspektor**")
        with st.form("new_inspector", border=False, clear_on_submit=True):
            name = st.text_input("Imię i nazwisko")
            if st.form_submit_button("Dodaj do zespołu", icon=":material/person_add:") and name.strip():
                STORE.add_inspector(name)
                st.toast(f"Dodano: {name.strip()}.", icon=":material/check_circle:")
                st.rerun()


# ---------------------------------------------------------------- manager: priorities ---------
def saaty_slider(a: str, b: str, value: float, labels: dict, key: str) -> float:
    """Pairwise comparison on Saaty's 1-9 scale as one slider: left = b more important, right = a."""
    def fmt(s):
        if s == 0:
            return "równie ważne"
        return f"{labels[a] if s > 0 else labels[b]} {abs(s) + 1}× ważniejsze"
    start = int(round(value - 1)) if value >= 1 else -int(round(1 / value - 1))
    s = st.select_slider(f"{labels[a]}  ↔  {labels[b]}", options=list(range(-8, 9)), value=start, format_func=fmt, key=key)
    return float(s + 1) if s >= 0 else 1.0 / (1 - s)


def weights_chart(weights: dict, labels: dict, order: list) -> None:
    t = pd.DataFrame({"Kryterium": [labels[x] for x in order], "Waga": [weights[x] for x in order]})
    bars = alt.Chart(t).mark_bar(color=NAVY, cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("Kryterium:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Waga:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])))
    text = bars.mark_text(align="left", dx=4, color="#1B2430").encode(text=alt.Text("Waga:Q", format=".0%"))
    st.altair_chart((bars + text).properties(height=34 * len(order) + 20), width="stretch")


def cr_badge(cr: float) -> None:
    ok = cr < 0.1
    st.badge(f"Spójność ocen CR = {cr:.2f}" + ("" if ok else " · popraw porównania"),
             icon=":material/check:" if ok else ":material/error:", color="green" if ok else "red")


def manager_prefs() -> None:
    header("Priorytety", "Jak system wybiera lokale do kontroli rutynowych. Porównaj kryteria parami w skali Saaty’ego (1–9).")
    crit, groups = _judgments("ahp_criteria", DEFAULT_CRITERIA), _judgments("ahp_groups", DEFAULT_GROUPS)
    with st.container(border=True):
        st.subheader("Kryteria", anchor=False)
        l1, r1 = st.columns([3, 2], gap="large")
        with l1:
            new_c = {(a, b): saaty_slider(a, b, crit[(a, b)], CRITERIA_PL, f"c_{a}_{b}") for (a, b) in DEFAULT_CRITERIA}
    with st.container(border=True):
        st.subheader("Kto korzysta z lokalu", anchor=False)
        st.caption("Jak bardzo niezaliczona kontrola zagraża klientom danej grupy lokali.")
        l2, r2 = st.columns([3, 2], gap="large")
        with l2:
            new_g = {(a, b): saaty_slider(a, b, groups[(a, b)], GROUPS_PL, f"g_{a}_{b}") for (a, b) in DEFAULT_GROUPS}
    p = preferences(new_c, new_g)
    with r1:
        st.markdown("**Wagi kryteriów**")
        weights_chart(p.criteria, CRITERIA_PL, CRITERIA)
        cr_badge(p.cr_criteria)
    with r2:
        st.markdown("**Wrażliwość grup** (najwyższa = 100%)")
        weights_chart({g: p.groups[g] / max(p.groups.values()) for g in GROUPS}, GROUPS_PL, GROUPS)
        cr_badge(p.cr_groups)
    a, b, _ = st.columns([1, 1, 3])
    user = st.session_state.user["username"]
    if a.button("Zapisz", type="primary", icon=":material/save:", disabled=not p.consistent, width="stretch"):
        STORE.set_setting("ahp_criteria", [[x, y, v] for (x, y), v in new_c.items()], user)
        STORE.set_setting("ahp_groups", [[x, y, v] for (x, y), v in new_g.items()], user)
        st.toast("Zapisano. Nowe i edytowane plany uwzględnią nowe wagi; zatwierdzone zostają bez zmian.",
                 icon=":material/check_circle:")
    if b.button("Przywróć domyślne", icon=":material/restart_alt:", width="stretch"):
        STORE.set_setting("ahp_criteria", [[x, y, v] for (x, y), v in DEFAULT_CRITERIA.items()], user)
        STORE.set_setting("ahp_groups", [[x, y, v] for (x, y), v in DEFAULT_GROUPS.items()], user)
        for k in list(st.session_state):
            if k.startswith(("c_", "g_")):
                del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------- manager: field results ------
def manager_results() -> None:
    day = st.session_state.day
    header("Wyniki kontroli", "Wyniki wpisane przez inspektorów w terenie.")
    scope = st.segmented_control("Zakres", ["Wybrany dzień", "Ostatnie 7 dni", "Wszystkie dni"], default="Wybrany dzień",
                                 required=True, label_visibility="collapsed")
    since = {"Wybrany dzień": day, "Ostatnie 7 dni": (pd.Timestamp(day) - pd.Timedelta(days=6)).date().isoformat(),
             "Wszystkie dni": "0000"}[scope]
    v = STORE.visits(since, day)
    if v.empty:
        st.info("W tym okresie nie ma zatwierdzonych planów.", icon=":material/info:")
        return
    r = v[v["outcome"].notna()].copy()
    counts = {"pass": "Zaliczone", "conditional": "Z zastrzeżeniami", "fail": "Niezaliczone", "no_entry": "Brak dostępu",
              "closed": "Zlikwidowane"}
    k = st.columns(len(counts) + 1)
    for col, (key, label) in zip(k, counts.items()):
        col.metric(label, int((r["outcome"] == key).sum()), border=True)
    k[-1].metric("Bez wyniku", int(v["outcome"].isna().sum()), border=True,
                 help="Wizyty z zatwierdzonych planów bez wpisanego wyniku (w bieżącym dniu: jeszcze przed inspektorami).")
    if r.empty:
        st.info("Inspektorzy nie wpisali jeszcze wyników w tym okresie.", icon=":material/info:")
        return
    r["kto"] = r["username"].map(user_names())
    r["Wynik"] = r["outcome"].map(OUTCOMES_PL)
    r["rodzaj"] = r["kind"].map({"zgłoszenie": "obowiązkowa", "rutynowa": "rutynowa"})
    r["lokal"], r["adres"] = r["name"].map(title_case), r["address"].map(title_case)
    r["day"], r["recorded_at"] = pd.to_datetime(r["day"]), pd.to_datetime(r["recorded_at"])
    r = r.sort_values(["day", "recorded_at"], ascending=False).reset_index(drop=True)
    key = f"res_{scope}_{day}"
    st.dataframe(r[["day", "kto", "lokal", "adres", "rodzaj", "Wynik", "note", "recorded_at"]], hide_index=True,
                 width="stretch", key=key, height=min(35 * len(r) + 38, 600), on_select=open_card_on_select(key, r["place_id"].tolist()),
                 selection_mode="single-row",
                 column_config={"day": st.column_config.DateColumn("Dzień", format="DD.MM.YYYY"), "kto": "Inspektor",
                                "lokal": "Lokal", "adres": "Adres", "rodzaj": "Wizyta", "note": "Uwagi",
                                "recorded_at": st.column_config.DatetimeColumn("Wpisano", format="DD.MM HH:mm")})
    st.caption(f"Kliknij wiersz, żeby otworzyć kartę lokalu. Wyniki trafiają do historii lokali: od nich liczy się termin "
               f"następnej kontroli i cechy modelu. Niezaliczona wizyta wraca po {planning.REINSPECT_DAYS} dniach jako "
               "wizyta obowiązkowa (ponowna kontrola). Ponowne trenowanie modelu to osobny, okresowy krok.")


# ---------------------------------------------------------------- inspector -------------------
def record_form(day: str, pid: int, user: dict, current: pd.Series | None = None) -> None:
    key = f"{day}_{user['username']}_{pid}"
    with st.form(f"r_{key}", border=False):
        o = st.segmented_control("Wynik", list(OUTCOMES_PL), format_func=lambda k: OUTCOMES_PL[k].capitalize(),
                                 default=None if current is None else current["outcome"], key=f"o_{key}")
        note = st.text_input("Uwagi", value="" if current is None else (current["note"] or ""),
                             placeholder="np. brak termometru w chłodni", key=f"n_{key}")
        if st.form_submit_button("Zapisz wynik", type="primary", icon=":material/save:", width="stretch"):
            if o is None:
                st.error("Wybierz wynik kontroli.")
                return
            at = dt.datetime.combine(pd.Timestamp(day).date(), dt.datetime.now().time())  # the simulated working day
            STORE.record(day, pid, user["username"], o, note, at=at)
            st.rerun()


def stop_header(s: pd.Series) -> None:
    st.markdown(f"**{s['seq']}. {title_case(s['name'])}**  \n{title_case(s['address'])} · {facility_pl(s['facility_type'])}")
    row = st.container(horizontal=True, vertical_alignment="center")
    kind_badge(s["kind"], row)
    row.caption(f"ok. {s['eta']}")
    st.caption(str(s["why"]))


def inspector_route() -> None:
    user, day = st.session_state.user, st.session_state.day
    if user["username"] is None:
        header("Moje wizyty")
        st.info("W zespole nie ma aktywnych inspektorów.", icon=":material/group_off:")
        return
    header("Moje wizyty", f"{long_date(day)} · {user['display_name']}")
    approved = STORE.plan(day)
    if not approved:
        st.info("Plan na ten dzień nie jest jeszcze zatwierdzony.", icon=":material/schedule:")
        days = STORE.plan_days()[:6]
        if days:
            row = st.container(horizontal=True, vertical_alignment="center")
            row.caption("Zatwierdzone dni:")
            for d in days:
                row.button(f"{pd.Timestamp(d):%d.%m}", key=f"go_{d}", on_click=set_day, args=(d,))
        return
    stops, meta = approved
    mine = stops[stops["inspector"] == user["inspector_no"]].sort_values("seq")
    if mine.empty:
        away = STORE.absences(day)
        away = away[away["username"] == user["username"]]
        st.info(f"Nieobecność: {away['reason'].iloc[0]}." if len(away) else "Nie masz dziś przydzielonych wizyt.",
                icon=":material/event_available:")
        return
    st.session_state.plan_now = {"day": day, "routes": stops, "draft": False}
    res = STORE.results(day).set_index("place_id")
    res = res[res.index.isin(mine["place_id"])]
    st.progress(min(1.0, len(res) / len(mine)),
                text=f"Wykonane {len(res)} z {len(mine)} · ok. {mine['eta'].iloc[0]}–{end_time(mine['eta'].iloc[-1])} · "
                     f"ok. {mine['km_from_prev'].sum():.0f} km dojazdu")
    left, right = st.columns([1.1, 1], gap="medium")
    with right:
        route_map(mine, highlight=int(user["inspector_no"]), height=600, names=team_names(),
                  results=res["outcome"].to_dict(), key=f"imap_{day}_{user['username']}")
    with left:
        todo, done = mine[~mine["place_id"].isin(res.index)], mine[mine["place_id"].isin(res.index)]
        if todo.empty:
            st.success("Wszystkie dzisiejsze wizyty mają wpisany wynik.", icon=":material/task_alt:")
        for i, (_, s) in enumerate(todo.iterrows()):
            pid = int(s["place_id"])
            with st.container(border=True):
                if i == 0:
                    st.caption(":blue[**NASTĘPNA WIZYTA**]")
                stop_header(s)
                row = st.container(horizontal=True)
                row.link_button("Nawiguj", nav_url(s["lat"], s["lon"]), icon=":material/directions:")
                card_button(pid, f"ic_{day}_{pid}", row)
                if i == 0:
                    record_form(day, pid, user)
                else:
                    with row.popover("Wpisz wynik", icon=":material/edit_note:"):
                        record_form(day, pid, user)
        if len(done):
            with st.expander(f"Wykonane ({len(done)})", expanded=todo.empty):
                for _, s in done.iterrows():
                    pid = int(s["place_id"])
                    cur = res.loc[pid]
                    with st.container(border=True):
                        a, b = st.columns([4, 1.6], vertical_alignment="center")
                        a.markdown(f"**{s['seq']}. {title_case(s['name'])}**  \n:gray[{title_case(s['address'])}]")
                        outcome_badge(cur["outcome"], b)
                        if cur["note"]:
                            st.caption(cur["note"])
                        row = st.container(horizontal=True)
                        card_button(pid, f"dc_{day}_{pid}", row)
                        with row.popover("Zmień wynik", icon=":material/edit:"):
                            record_form(day, pid, user, cur)
    st.caption(f"Plan zatwierdził(a) {user_names().get(meta['approved_by'], meta['approved_by'])}, "
               f"{pd.Timestamp(meta['approved_at']):%d.%m.%Y %H:%M}. Godziny są orientacyjne.")


# ---------------------------------------------------------------- shell -----------------------
def switcher(team: pd.DataFrame) -> dict:
    """One click between the manager and any inspector; no passwords in the local prototype."""
    day = st.session_state.day_input.isoformat()
    view = st.segmented_control("Widok", ["Kierownik", "Inspektor"], default="Kierownik", required=True,
                                key="view", width="stretch")
    if view == "Kierownik":
        boss = team[team["role"] == "manager"].iloc[0].to_dict()
        st.caption(f"{boss['display_name']} · kierownik zespołu")
        return boss
    inspectors = team[(team["role"] == "inspector") & team["active"]].set_index("username")
    if inspectors.empty:
        st.caption("Brak aktywnych inspektorów. Dodaj ich na stronie Zespół (widok kierownika).")
        return {"username": None, "role": "inspector", "display_name": "", "inspector_no": None}
    approved = STORE.plan(day)
    counts = approved[0]["inspector"].value_counts() if approved else pd.Series(dtype=int)

    def label(u):
        n = int(counts.get(inspectors.loc[u, "inspector_no"], 0))
        return f"{inspectors.loc[u, 'display_name']} · " + (count(n, "wizyta", "wizyty", "wizyt") if n else "brak wizyt")
    names = list(inspectors.index)
    who = st.session_state.get("who")
    who = st.selectbox("Inspektor", names, format_func=label, index=names.index(who) if who in names else 0)
    st.session_state.who = who  # survives switching to the manager and back
    return {"username": who, **inspectors.loc[who].to_dict()}


def main() -> None:
    team = STORE.users()
    last = last_full_day()
    st.session_state.setdefault("day_input", last)
    with st.sidebar:
        user = st.session_state.user = switcher(team)
        st.date_input("Dzień pracy", key="day_input", min_value=FIRST_DAY, max_value=last, format="DD.MM.YYYY",
                      help="System działa na danych historycznych miasta: dzień pokazuje zgłoszenia, które wtedy "
                           "wpłynęły, i lokale, które czekały na kontrolę.")
        st.session_state.day = st.session_state.day_input.isoformat()
        st.caption(f"{CITY.name} · dane miasta do {last:%d.%m.%Y}. Zespół, plany i wyniki z ostatnich dni to dane "
                   "demonstracyjne.")
    if user["role"] == "manager":
        pages = {"plan": st.Page(manager_plan, title="Plan dnia", icon=":material/map:", url_path="plan", default=True),
                 "team": st.Page(manager_team, title="Zespół", icon=":material/badge:", url_path="zespol"),
                 "results": st.Page(manager_results, title="Wyniki kontroli", icon=":material/fact_check:",
                                    url_path="wyniki"),
                 "prefs": st.Page(manager_prefs, title="Priorytety", icon=":material/tune:", url_path="priorytety")}
    else:
        pages = {"route": st.Page(inspector_route, title="Moje wizyty", icon=":material/route:", url_path="wizyty",
                                  default=True)}
    st.session_state.pages = pages
    st.navigation(list(pages.values()), position="top").run()
    pid = st.session_state.pop("open_place", None)
    if pid:
        place_card(pid)


main()
