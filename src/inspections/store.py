"""Local state of the running system in SQLite: the team, absences, approved day plans and results.

This is what makes the prototype behave like a product: the manager keeps the team list,
approves a plan for the inspectors who are in that day, each inspector sees their own route,
and the results they record flow back into the history the next plans are made from
(field_visits -> planning.add_field_visits). There are no passwords: the prototype runs
locally and the app switches between the team's accounts in one click. Everything lives in
var/<city>.sqlite (not committed).
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from .city import CityConfig, home

# A fictional team for the demo: the manager and 13 inspectors.
DEMO_MANAGER = "Patricia Morales"
DEMO_TEAM = ["Michael Brennan", "Aisha Johnson", "Daniel Kowalczyk", "Maria Hernández", "James O'Connor",
             "Tanya Williams", "Kevin Nguyen", "Laura Sánchez", "Robert Jackson", "Emily Novak", "Marcus Davis",
             "Sofia Ramírez", "Anthony Russo"]
DEMO_INSPECTORS = len(DEMO_TEAM)
ABSENCE_REASONS = ["urlop", "zwolnienie lekarskie", "szkolenie"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('manager', 'inspector')),
    inspector_no INTEGER, active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS absences (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
    reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans (
    day TEXT PRIMARY KEY, approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, inspectors INTEGER NOT NULL,
    stops TEXT NOT NULL, params TEXT);
CREATE TABLE IF NOT EXISTS results (
    day TEXT NOT NULL, place_id INTEGER NOT NULL, username TEXT NOT NULL, outcome TEXT NOT NULL
    CHECK (outcome IN ('pass', 'conditional', 'fail', 'closed', 'no_entry')),
    note TEXT, recorded_at TEXT NOT NULL, PRIMARY KEY (day, place_id));
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_by TEXT, updated_at TEXT);
"""


class Store:
    def __init__(self, city: CityConfig, path=None):
        self.path = path or home() / "var" / f"{city.slug}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)
            if "params" not in [c[1] for c in db.execute("PRAGMA table_info(plans)")]:
                db.execute("ALTER TABLE plans ADD COLUMN params TEXT")
            if "active" not in [c[1] for c in db.execute("PRAGMA table_info(users)")]:
                db.execute("ALTER TABLE users ADD COLUMN active INTEGER NOT NULL DEFAULT 1")

    @contextmanager
    def _db(self):
        """A connection that commits on success, rolls back on error and is always closed."""
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    # team ----------------------------------------------------------------------------------
    def add_user(self, username: str, display_name: str, role: str, inspector_no: int | None = None,
                 active: bool = True):
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?)",
                       (username, display_name, role, inspector_no, int(active)))

    def add_inspector(self, display_name: str) -> str:
        """A new inspector with the next free number; returns the username."""
        with self._db() as db:
            no = (db.execute("SELECT max(inspector_no) FROM users").fetchone()[0] or 0) + 1
        self.add_user(f"inspektor{no}", display_name.strip(), "inspector", no)
        return f"inspektor{no}"

    def update_user(self, username: str, display_name: str | None = None, active: bool | None = None) -> None:
        with self._db() as db:
            if display_name is not None:
                db.execute("UPDATE users SET display_name = ? WHERE username = ?", (display_name.strip(), username))
            if active is not None:
                db.execute("UPDATE users SET active = ? WHERE username = ?", (int(active), username))

    def user(self, username: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def users(self) -> pd.DataFrame:
        with self._db() as db:
            u = pd.read_sql("SELECT username, display_name, role, inspector_no, active FROM users "
                            "ORDER BY role, inspector_no", db)
        u["active"] = u["active"].astype(bool)
        return u

    def seed_demo(self, inspectors: int = DEMO_INSPECTORS) -> None:
        """The demo team: a manager and N named inspectors."""
        self.add_user("kierownik", DEMO_MANAGER, "manager")
        for i in range(1, inspectors + 1):
            name = DEMO_TEAM[i - 1] if i <= len(DEMO_TEAM) else f"Inspektor {i}"
            self.add_user(f"inspektor{i}", name, "inspector", i)

    def reset(self) -> None:
        """Empty every table (the demo seed starts from here)."""
        with self._db() as db:
            for t in ("users", "absences", "plans", "results", "settings"):
                db.execute(f"DELETE FROM {t}")

    def add_absence(self, username: str, start: str, end: str, reason: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO absences (username, start, end, reason) VALUES (?, ?, ?, ?)",
                       (username, str(start), str(end), reason))

    def remove_absence(self, absence_id: int) -> None:
        with self._db() as db:
            db.execute("DELETE FROM absences WHERE id = ?", (int(absence_id),))

    def absences(self, day: str | None = None) -> pd.DataFrame:
        """All absences, or those covering `day`."""
        with self._db() as db:
            if day:
                return pd.read_sql("SELECT * FROM absences WHERE start <= ? AND end >= ? ORDER BY start", db,
                                   params=(day, day))
            return pd.read_sql("SELECT * FROM absences ORDER BY start DESC", db)

    def available(self, day: str) -> pd.DataFrame:
        """Active inspectors not absent on `day`, in team order: the people a plan can use."""
        u = self.users()
        u = u[(u["role"] == "inspector") & u["active"]]
        return u[~u["username"].isin(self.absences(day)["username"])].reset_index(drop=True)

    # plans -----------------------------------------------------------------------------------
    def approve_plan(self, day: str, stops: pd.DataFrame, by: str, inspectors: int, params: dict | None = None,
                     at: datetime | None = None) -> None:
        """Save the approved routes with `params`, the inputs they were made from (the app restores
        them when the plan is edited). Once a result is recorded for the day its plan is final."""
        with self._db() as db:
            if db.execute("SELECT count(*) FROM results WHERE day = ?", (day,)).fetchone()[0]:
                raise ValueError(f"results are already recorded for {day}; its plan can no longer change")
            db.execute("INSERT OR REPLACE INTO plans (day, approved_by, approved_at, inspectors, stops, params) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (day, by, (at or datetime.now()).isoformat(timespec="seconds"), inspectors,
                        stops.to_json(orient="records", force_ascii=False), json.dumps(params)))

    def plan(self, day: str) -> tuple[pd.DataFrame, dict] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM plans WHERE day = ?", (day,)).fetchone()
        if not row:
            return None
        meta = {k: row[k] for k in ("approved_by", "approved_at", "inspectors")}
        meta["params"] = json.loads(row["params"]) if row["params"] else None
        return pd.DataFrame(json.loads(row["stops"])), meta

    def plan_days(self) -> list[str]:
        with self._db() as db:
            return [r["day"] for r in db.execute("SELECT day FROM plans ORDER BY day DESC")]

    def _recorded(self, since: str, before: str) -> pd.DataFrame:
        """Results recorded for days in [since, before), one row per licence the stop covered
        (a visit to an address with several licences settles all of them), with the stop's kind."""
        with self._db() as db:
            plans = db.execute("SELECT day, stops FROM plans WHERE day >= ? AND day < ?", (since, before)).fetchall()
            res = pd.read_sql("SELECT day, place_id, outcome FROM results WHERE day >= ? AND day < ?", db,
                              params=(since, before))
        stops = pd.DataFrame([{"day": r["day"], "place_id": int(s["place_id"]), "kind": s["kind"],
                               "place_ids": s.get("place_ids") or [int(s["place_id"])]}
                              for r in plans for s in json.loads(r["stops"])],
                             columns=["day", "place_id", "kind", "place_ids"]).drop_duplicates(["day", "place_id"])
        res = res.merge(stops, on=["day", "place_id"], how="left")
        res["place_ids"] = [ids if isinstance(ids, list) else [p] for ids, p in zip(res["place_ids"], res["place_id"])]
        out = res.drop(columns="place_id").explode("place_ids").rename(columns={"place_ids": "place_id"})
        return out.astype({"place_id": "int64"}).drop_duplicates(["day", "place_id"])[["day", "place_id", "outcome", "kind"]]

    def served(self, before: str, days: int) -> dict[str, set[int]]:
        """{day: places visited} for each day with an approved plan in the `days` days before
        `before`; a visit counts once its result is recorded and is not "no entry"."""
        since = (pd.Timestamp(before) - pd.Timedelta(days=days)).date().isoformat()
        with self._db() as db:
            planned = [r["day"] for r in db.execute("SELECT day FROM plans WHERE day >= ? AND day < ?", (since, before))]
        r = self._recorded(since, before)
        r = r[r["outcome"] != "no_entry"]
        out = {d: set() for d in planned}
        for d, p in zip(r["day"], r["place_id"]):
            if d in out:
                out[d].add(int(p))
        return out

    def field_visits(self, before: str) -> pd.DataFrame:
        """Every result recorded for days before `before`, per licence, with the kind of stop it was."""
        return self._recorded("0000", before).reset_index(drop=True)

    def visits(self, since: str, until: str) -> pd.DataFrame:
        """Every stop of the plans approved for days in [since, until], with its result if recorded."""
        with self._db() as db:
            plans = db.execute("SELECT day, stops FROM plans WHERE day >= ? AND day <= ?", (since, until)).fetchall()
            res = pd.read_sql("SELECT day, place_id, username, outcome, note, recorded_at FROM results "
                              "WHERE day >= ? AND day <= ?", db, params=(since, until))
        rows = [{"day": r["day"], **{k: s[k] for k in ("inspector", "seq", "eta", "kind", "place_id", "name", "address")}}
                for r in plans for s in json.loads(r["stops"])]
        v = pd.DataFrame(rows, columns=["day", "inspector", "seq", "eta", "kind", "place_id", "name", "address"])
        return v.merge(res, on=["day", "place_id"], how="left").sort_values(["day", "inspector", "seq"])

    def version(self, before: str) -> str:
        """Changes whenever anything later plans depend on changes: approvals and results before `before`."""
        with self._db() as db:
            rows = db.execute("SELECT day, approved_at, stops FROM plans WHERE day < ? ORDER BY day", (before,)).fetchall()
            rows += db.execute("SELECT day, place_id, outcome FROM results WHERE day < ? ORDER BY day, place_id",
                               (before,)).fetchall()
        return hashlib.md5(repr([tuple(r) for r in rows]).encode()).hexdigest()

    # settings (e.g. the manager's AHP judgments) ------------------------------------------------
    def setting(self, key: str, default=None):
        with self._db() as db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value, by: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?, ?, ?)",
                       (key, json.dumps(value), by, datetime.now().isoformat(timespec="seconds")))

    # results (the feedback loop) ---------------------------------------------------------------
    def record(self, day: str, place_id: int, username: str, outcome: str, note: str = "",
               at: datetime | None = None) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?)",
                       (day, int(place_id), username, outcome, note,
                        (at or datetime.now()).isoformat(timespec="seconds")))

    def results(self, day: str | None = None) -> pd.DataFrame:
        with self._db() as db:
            if day:
                return pd.read_sql("SELECT * FROM results WHERE day = ?", db, params=(day,))
            return pd.read_sql("SELECT * FROM results ORDER BY recorded_at DESC", db)
