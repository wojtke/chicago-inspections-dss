"""Decision-maker preferences: AHP weights for the criteria and for the vulnerability groups.

The prediction module only says how likely a place is to fail. What the inspection service
cares about also depends on who eats there and how long nobody has looked. The supervisor
states that as pairwise comparisons on Saaty's 1-9 scale; AHP turns them into weights and
checks they are consistent (consistency ratio below 0.1).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .city import GROUPS
from .pl import count

# Saaty's random consistency index for n x n matrices.
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

CRITERIA = ["risk", "vulnerability", "overdue"]
CRITERIA_PL = {"risk": "Ryzyko niezaliczenia", "vulnerability": "Wrażliwość klientów",
               "overdue": "Czas od ostatniej kontroli"}
GROUPS_PL = {"children": "dzieci (szkoły, żłobki)", "healthcare": "szpitale i opieka",
             "restaurant": "restauracje", "retail": "sklepy", "other": "inne (bary, food trucki)"}

# judgments[(a, b)] = how much more important a is than b; a value below 1 means b matters more.
DEFAULT_CRITERIA = {("risk", "vulnerability"): 2, ("risk", "overdue"): 4, ("vulnerability", "overdue"): 2}
DEFAULT_GROUPS = {
    ("children", "healthcare"): 1, ("children", "restaurant"): 3, ("children", "retail"): 5, ("children", "other"): 5,
    ("healthcare", "restaurant"): 3, ("healthcare", "retail"): 5, ("healthcare", "other"): 5,
    ("restaurant", "retail"): 2, ("restaurant", "other"): 3,
    ("retail", "other"): 1,
}
# Days since the last visit stop counting as more urgent after two years.
OVERDUE_CAP = 730


def matrix(names: list[str], judgments: dict[tuple[str, str], float]) -> np.ndarray:
    """Reciprocal pairwise-comparison matrix from the upper-triangle judgments."""
    idx = {n: i for i, n in enumerate(names)}
    m = np.ones((len(names), len(names)))
    for (a, b), v in judgments.items():
        m[idx[a], idx[b]] = v
        m[idx[b], idx[a]] = 1 / v
    return m


def ahp_weights(m: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal-eigenvector weights (summing to 1) and the consistency ratio."""
    n = len(m)
    vals, vecs = np.linalg.eig(m)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (vals[k].real - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RI[n] if RI.get(n) else 0.0
    return w, float(max(cr, 0.0))


@dataclass
class Preferences:
    criteria: dict[str, float]   # weight of each criterion, sums to 1
    groups: dict[str, float]     # vulnerability score of each group, the most vulnerable = 1
    cr_criteria: float
    cr_groups: float

    @property
    def consistent(self) -> bool:
        return self.cr_criteria < 0.1 and self.cr_groups < 0.1


def preferences(criteria: dict | None = None, groups: dict | None = None) -> Preferences:
    wc, crc = ahp_weights(matrix(CRITERIA, criteria or DEFAULT_CRITERIA))
    wg, crg = ahp_weights(matrix(GROUPS, groups or DEFAULT_GROUPS))
    return Preferences(dict(zip(CRITERIA, wc)), dict(zip(GROUPS, wg / wg.max())), crc, crg)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)


def priority(rows: pd.DataFrame, prefs: Preferences | None = None) -> pd.DataFrame:
    """Add each criterion (scaled to 0-1 within the pool), its weighted contribution and `priority`.

    rows need p_fail, facility_group and days_since_last.
    """
    prefs = prefs or preferences()
    out = rows.copy()
    raw = {
        "risk": out["p_fail"],
        "vulnerability": out["facility_group"].map(prefs.groups).fillna(prefs.groups["other"]),
        # A place never inspected before counts as fully overdue.
        "overdue": out["days_since_last"].fillna(OVERDUE_CAP).clip(upper=OVERDUE_CAP) / OVERDUE_CAP,
    }
    out["priority"] = 0.0
    for c in CRITERIA:
        out[f"c_{c}"] = _minmax(raw[c].astype(float))
        out[f"w_{c}"] = prefs.criteria[c] * out[f"c_{c}"]
        out["priority"] += out[f"w_{c}"]
    return out


def explain(row: pd.Series) -> str:
    """Why a place is high on the list, in Polish: its strongest criteria first, then facts."""
    parts = []
    for c in sorted(CRITERIA, key=lambda c: -row.get(f"w_{c}", 0)):
        if row.get(f"w_{c}", 0) <= 0.05:
            continue
        if c == "risk":
            parts.append(f"szansa niezaliczenia {row['p_fail']:.0%}")
        elif c == "vulnerability" and row["facility_group"] in ("children", "healthcare"):
            parts.append(GROUPS_PL[row["facility_group"]])
        elif c == "overdue":
            d = row.get("days_since_last")
            parts.append("brak wcześniejszych kontroli" if pd.isna(d) else f"{int(d)} dni od ostatniej kontroli")
    if row.get("last_failed") == 1:
        parts.append("nie przeszedł ostatniej kontroli")
    if row.get("n_past_complaints", 0) >= 2:
        parts.append(count(row["n_past_complaints"], "kontrola", "kontrole", "kontroli") + " po skargach")
    return ", ".join(parts[:3])
