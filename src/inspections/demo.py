"""A lived-in demo state for the app: the team, a few absences, approved plans for the last working
days and the results the inspectors recorded. The newest day is the morning of a working day: its
plan was approved the afternoon before and nobody has recorded a result yet.

    uv run insp demo        # rebuilds var/<city>.sqlite from scratch

Each day is planned exactly as the app would plan it (the earlier days' visits and results are
already in the history), so the app opens every seeded day as approved. Outcomes are invented
but plausible: a request gets the result the city recorded for it, a routine visit the system
chose (which the city did not necessarily make) draws its result from the model's chance of
failing, and the notes come from a short list of typical findings.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import planning
from .city import CityConfig
from .data import load, load_socio
from .model import load_model
from .preferences import preferences
from .routing import RoutingConfig
from .store import Store

NOTES = {
    "pass": ["", "", "", "Bez uwag.", "Temperatury w chłodniach w normie.", "Dokumentacja i szkolenia personelu aktualne."],
    "conditional": ["Brak termometru w chłodni, do uzupełnienia.", "Brak mydła i ręczników przy umywalce personelu.",
                    "Pojemniki z żywnością bez etykiet z datą.", "Uszczelka chłodni do wymiany w ciągu 14 dni.",
                    "Zapisy temperatur prowadzone niesystematycznie.", "Żywność przechowywana na podłodze w magazynie."],
    "fail": ["Ślady gryzoni w magazynie suchym. Konieczna ponowna kontrola.",
             "Chłodnia 9°C, produkty wycofane i zniszczone na miejscu.", "Brak ciepłej wody przy umywalce i zmywaku.",
             "Karaluchy pod zmywakiem. Lokal zobowiązany do dezynsekcji.",
             "Brak certyfikowanego kierownika ds. bezpieczeństwa żywności na zmianie.",
             "Surowe mięso nad gotowymi potrawami w chłodni, ryzyko skażenia."],
    "no_entry": ["Lokal zamknięty w godzinach otwarcia, kartka na drzwiach.", "Brak osoby upoważnionej, nie wpuszczono."],
    "closed": ["Lokal zlikwidowany, witryna pusta.", "W lokalu działa już inna firma."],
}
# (inspector number, reason, first and last working day of the window it covers, as offsets from the end)
ABSENCES = [(5, "urlop", -6, 4), (12, "szkolenie", -2, -2), (9, "zwolnienie lekarskie", 0, 1)]


def working_days(df: pd.DataFrame, end: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """The last `n` weekdays up to `end` on which the city inspected as usual."""
    per_day = df.groupby("date").size()
    usual = per_day[(per_day.index <= end) & (per_day.index.weekday < 5) & (per_day >= 0.5 * per_day.median())]
    return list(usual.index[-n:])


def city_outcome(df: pd.DataFrame, place_id: int, day: pd.Timestamp) -> str | None:
    """What the city recorded for this request on this day: pass / conditional / fail / closed /
    no_entry. None for a request handled on a later day than the city handled it (carried over
    or a re-inspection the system added): its result is drawn like a routine visit's."""
    rows = df[(df["place_id"] == place_id) & (df["date"] == day) & (df["kind"] == "mandatory")]
    if rows.empty:
        return None
    last = rows.iloc[-1]
    if last["closed"]:
        return "closed"
    return last["outcome"] if pd.notna(last["outcome"]) else "no_entry"


def drawn_outcome(p_fail: float, rng: np.random.Generator) -> str:
    u = rng.random()
    if u < 0.03:
        return "no_entry"
    if u < 0.04:
        return "closed"
    if rng.random() < p_fail:
        return "fail"
    return "conditional" if rng.random() < 0.2 else "pass"


def seed(city: CityConfig, days: int = 8, seed: int = 7) -> None:
    df, model, socio = load(city), load_model(city), load_socio(city)
    store = Store(city)
    store.reset()
    store.seed_demo()
    end = planning.last_full_day(df)
    window = working_days(df, end, days)
    for no, reason, a, b in ABSENCES:
        store.add_absence(f"inspektor{no}", (end + pd.Timedelta(days=a)).date().isoformat(),
                          (end + pd.Timedelta(days=b)).date().isoformat(), reason)
    rng = np.random.default_rng(seed)
    prefs, cfg = preferences(), RoutingConfig()
    for d in window:
        day = d.date().isoformat()
        hist = planning.add_field_visits(df, store.field_visits(before=day))
        carried = planning.pending_requests(hist, day, store.served(day, planning.CARRY_DAYS))
        team = store.available(day)
        plan = planning.plan_day(hist, model, city, day, team["inspector_no"].tolist(), prefs, cfg, socio=socio,
                                 carried=carried)
        approved = datetime.combine((d - pd.Timedelta(days=1)).date(), datetime.min.time()) + timedelta(
            hours=15, minutes=int(rng.integers(20, 55)))
        store.approve_plan(day, plan.routes, "kierownik", len(team), planning.plan_params(team["inspector_no"]),
                           at=approved)
        who = team.set_index("inspector_no")["username"]
        recorded = 0
        for s in plan.routes.itertuples(index=False):
            if d == end:
                break  # the newest day has not started yet
            outcome = city_outcome(df, s.place_id, d) if s.kind == "zgłoszenie" else None
            outcome = outcome or drawn_outcome(float(s.p_fail) if pd.notna(s.p_fail) else 0.2, rng)
            h, m = map(int, s.eta.split(":"))
            at = datetime.combine(d.date(), datetime.min.time()) + timedelta(
                hours=h, minutes=m + cfg.service_min + int(rng.integers(0, 25)))
            store.record(day, s.place_id, who[s.inspector], outcome, str(rng.choice(NOTES[outcome])), at=at)
            recorded += 1
        print(f"{day}: {len(team)} inspectors, {len(plan.routes)} visits ({(plan.routes['kind'] == 'zgłoszenie').sum()} "
              f"requests), {recorded} results")
    print(f"demo state in {store.path}; open the app on {end:%d.%m.%Y}")
