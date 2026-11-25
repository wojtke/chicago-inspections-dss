"""Write what the LaTeX report needs: CSVs for the pgfplots charts and numbers.tex for the text.

Run after `insp train` and `insp evaluate`:  uv run python docs/tech-report/export_data.py
Everything is read from what those commands saved (plus the cleaned data for a few counts), so
every number in the report comes from the code; none is typed by hand.
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from inspections import evaluation, planning
from inspections.city import get_city
from inspections.data import load
from inspections.features import FEATURES_PL
from inspections.model import SPLITS, load_importance, load_metrics
from inspections.preferences import DEFAULT_CRITERIA, preferences

OUT = Path(__file__).resolve().parent / "data"
NAMES = {"actual": "Kolejność faktyczna", "random": "Losowa", "risk": "Tylko ryzyko", "ahp": "Ryzyko + AHP",
         "ahp_fair": "AHP + równoważenie", "system": "Pełny system (+ trasy)"}


# The report says "wyniki negatywne"; the app's shorter labels say "niezaliczenia".
REPORT_LABELS = {"past_fail_rate": "odsetek wyników negatywnych", "n_past_fails": "liczba wyników negatywnych",
                 "last_failed": "ostatni wynik negatywny"}


def rnd(x: float, digits: int = 1) -> float:
    """Round half up, the way pgfplots prints, so the text and the tables agree."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def pl(x: float, digits: int = 1) -> str:
    """Polish decimal comma, thin space between thousands."""
    return f"{rnd(x, digits):,.{digits}f}".replace(",", "\\,").replace(".", ",")


def main() -> None:
    city = get_city()
    OUT.mkdir(exist_ok=True)
    n: dict[str, str] = {}  # macro name -> value, written to numbers.tex

    # Data.
    df = load(city)
    recent = df[df["date"] >= SPLITS["test"][0]]
    n.update(NInspections=pl(len(df), 0), NPlaces=pl(df["place_id"].nunique(), 0),
             FirstYear=str(df["date"].min().year), DataUntil=f"{df['date'].max():%d.%m.%Y}",
             MandatoryShare=pl(100 * (recent["kind"] == "mandatory").mean(), 0),
             MedianDaily=pl(recent.groupby("date").size().median(), 0),
             RoutinePerDay=pl(recent[recent["kind"] == "routine"].groupby("date").size().median(), 0))
    daily = df.groupby("date").size()
    daily = daily[daily.index >= daily.index.max() - pd.Timedelta(days=60)]
    ref = daily[daily >= 0.5 * daily.median()].index.max()  # the last day published in full
    n.update(RefDay=f"{ref:%d.%m.%Y}", DueRef=pl(len(planning.due_places(df, city, ref)), 0))

    # Model.
    met = load_metrics(city)
    chosen = met[met["chosen"]]
    v = chosen["metrics"]
    for split in ("train", "valid", "test"):
        n[f"N{split.title()}"] = pl(met["counts"][split], 0)
    for key, tag in (("gradient_boosting", "GB"), ("logistic", "LR")):
        for split, s in (("valid", "Valid"), ("test", "Test")):
            m = v[key][split]
            n[f"Auc{s}{tag}"], n[f"Brier{s}{tag}"] = pl(m["auc"], 3), pl(m["brier"], 3)
            n[f"Top{s}{tag}"] = pl(100 * m["fail_rate_top10"], 1)
    n["AllValid"] = pl(100 * v["gradient_boosting"]["valid"]["fail_rate_all"], 1)
    n["AllTest"] = pl(100 * v["gradient_boosting"]["test"]["fail_rate_all"], 1)
    n["BrierTestCal"] = pl(v["calibrated"]["test"]["brier"], 3)
    n["TopTestCal"] = pl(100 * v["calibrated"]["test"]["fail_rate_top10"], 0)
    n["AllTestCal"] = pl(100 * v["calibrated"]["test"]["fail_rate_all"], 0)
    top = v["reliability"][-1]
    n["TopDecilePred"], n["TopDecileObs"] = pl(100 * top["predicted"], 1), pl(100 * top["observed"], 1)
    fair = pd.DataFrame(chosen["fairness"])
    loc = pd.DataFrame(met["with_location"]["fairness"])
    n.update(LocGain=pl(met["location_auc_gain"], 3),
             LocTestAuc=pl(met["with_location"]["metrics"]["calibrated"]["test"]["auc"], 3),
             LocBurdenMax=pl(loc["burden_ratio"].max(), 2),
             LocBurdenMaxQ=str(int(loc.loc[loc["burden_ratio"].idxmax(), "income_quintile"])),
             BurdenModelMin=pl(fair["burden_ratio"].min(), 2), BurdenModelMax=pl(fair["burden_ratio"].max(), 2),
             CalGapMax=pl(100 * fair["calibration_gap"].abs().max(), 1),
             FalseAlarmMin=pl(100 * fair["false_alarm_rate"].min(), 0),
             FalseAlarmMax=pl(100 * fair["false_alarm_rate"].max(), 0),
             AucQMin=pl(fair["auc"].min(), 2), AucQMax=pl(fair["auc"].max(), 2),
             BurdenModelMaxQ=str(int(fair.loc[fair["burden_ratio"].idxmax(), "income_quintile"])),
             NFeatures=str(len(chosen["features"])))
    rel = pd.DataFrame(v["reliability"])[["predicted", "observed"]] * 100
    rel.round(2).to_csv(OUT / "reliability.csv", index=False)
    imp = pd.DataFrame(load_importance(city)).head(8)
    imp["label"] = imp["feature"].map({**FEATURES_PL, **REPORT_LABELS}).fillna(imp["feature"])
    imp["auc_drop"] *= 1000  # in thousandths of AUC
    imp[::-1][["label", "auc_drop"]].round(2).to_csv(OUT / "importance.csv", index=False)
    n["NShownFeatures"] = str(len(imp))
    burden_model = pd.DataFrame({k: {f["income_quintile"]: f["burden_ratio"] for f in met[k]["fairness"]}
                                 for k in ("with_location", "without_location")})
    burden_model.round(3).to_csv(OUT / "burden_model.csv", index_label="quintile")

    # Preferences: the defaults the replay used.
    p = preferences()
    n.update(WRisk=pl(p.criteria["risk"], 2), WVuln=pl(p.criteria["vulnerability"], 2),
             WOverdue=pl(p.criteria["overdue"], 2), CRCrit=pl(p.cr_criteria, 2), CRGroups=pl(p.cr_groups, 2),
             GChildren=pl(p.groups["children"], 2), GHealth=pl(p.groups["healthcare"], 2),
             GRest=pl(p.groups["restaurant"], 2), GRetail=pl(p.groups["retail"], 2), GOther=pl(p.groups["other"], 2),
             AhpRV=f"{DEFAULT_CRITERIA[('risk', 'vulnerability')]:g}", AhpRO=f"{DEFAULT_CRITERIA[('risk', 'overdue')]:g}",
             AhpVO=f"{DEFAULT_CRITERIA[('vulnerability', 'overdue')]:g}")

    # Replay, full team, mean over the five windows.
    r = evaluation.load_results(city)
    s, c, q = r["summary"], r["curves"], r["quintiles"]
    full = s[s["multiplier"] == 1.0]
    pol = (full.groupby("policy")[["days_sooner", "high_impact_days_sooner", "fails_first_half", "km_per_visit"]]
           .mean().reindex(evaluation.POLICIES))
    pol["fails_first_half"] *= 100
    pol.insert(0, "name", [NAMES[k] for k in pol.index])
    shown = pol.copy()  # rounded exactly as the text rounds, so table and text agree
    for col in ("days_sooner", "high_impact_days_sooner", "km_per_visit"):
        shown[col] = shown[col].map(rnd)
    shown["fails_first_half"] = shown["fails_first_half"].map(lambda x: rnd(x, 0))
    shown.to_csv(OUT / "policies.csv", index_label="policy")
    sys_w = full[full["policy"] == "system"]["days_sooner"]
    n.update(DaysSys=pl(pol.loc["system", "days_sooner"]), DaysHiSys=pl(pol.loc["system", "high_impact_days_sooner"]),
             DaysRisk=pl(pol.loc["risk", "days_sooner"]), DaysHiRisk=pl(pol.loc["risk", "high_impact_days_sooner"]),
             DaysAhp=pl(pol.loc["ahp", "days_sooner"]), DaysHiAhp=pl(pol.loc["ahp", "high_impact_days_sooner"]),
             DaysRandom=pl(pol.loc["random", "days_sooner"]),
             AhpLoss=pl(rnd(pol.loc["risk", "days_sooner"]) - rnd(pol.loc["ahp", "days_sooner"])),
             KmSys=pl(pol.loc["system", "km_per_visit"]), KmAhp=pl(pol.loc["ahp", "km_per_visit"]),
             KmFair=pl(pol.loc["ahp_fair", "km_per_visit"]),
             KmCity=pl(pol.loc["actual", "km_per_visit"]), SysDaysMin=pl(sys_w.min()), SysDaysMax=pl(sys_w.max()),
             HalfCity=pl(pol.loc["actual", "fails_first_half"], 0), HalfSys=pl(pol.loc["system", "fails_first_half"], 0),
             HalfRisk=pl(pol.loc["risk", "fails_first_half"], 0))
    fails = full.set_index(["window", "policy"])["fails"]
    cw = c[c["multiplier"] == 1.0].copy()
    cw["share"] = 100 * cw["found"] / fails.reindex(pd.MultiIndex.from_frame(cw[["window", "policy"]])).values
    curves = cw.groupby(["day", "policy"])["share"].mean().unstack()[evaluation.POLICIES]
    curves.round(2).to_csv(OUT / "curves.csv", index_label="day")

    # Team size.
    t = s.groupby(["multiplier", "policy"])[["fails_reached", "days_sooner", "capacity"]].mean()
    mults = t.index.levels[0]
    team = pd.DataFrame({"team": np.rint(100 * mults).astype(int)}, index=mults)
    for k in evaluation.POLICIES:
        team[f"{k}_reached"] = 100 * t.xs(k, level="policy")["fails_reached"]
        team[f"{k}_days"] = t.xs(k, level="policy")["days_sooner"]
    team["capacity"] = t.xs("actual", level="policy")["capacity"]
    team["capacity_share"] = 100 * team["capacity"] / team.loc[1.0, "capacity"]
    team.sort_values("team").round(2).to_csv(OUT / "team.csv", index=False)
    e = team.loc[0.8]
    n.update(CapFull=pl(team.loc[1.0, "capacity"], 0), CapEighty=pl(e["capacity"], 0),
             CapShareEighty=pl(e["capacity_share"], 0), ReachCity=pl(e["actual_reached"], 0),
             ReachSys=pl(e["system_reached"], 0), LateCity=pl(-e["actual_days"]), LateSys=pl(-e["system_days"]))

    # Plan fairness by income quintile, full team.
    qq = q[q["multiplier"] == 1.0].groupby(["income_quintile", "policy"])[["burden_ratio", "days_sooner"]].mean()
    burden = qq["burden_ratio"].unstack()[evaluation.POLICIES]
    burden.round(3).to_csv(OUT / "burden_plan.csv", index_label="quintile")
    days_q = qq["days_sooner"].unstack()[evaluation.POLICIES]
    days_q.round(2).to_csv(OUT / "days_by_quintile.csv", index_label="quintile")
    for k, tag in (("actual", "City"), ("ahp", "Ahp"), ("ahp_fair", "Fair"), ("system", "Sys")):
        n[f"Burden{tag}Min"], n[f"Burden{tag}Max"] = pl(burden[k].min(), 2), pl(burden[k].max(), 2)
    n.update(BurdenSysPoorest=pl(burden.loc[1, "system"], 2),
             QDaysSysMin=pl(days_q["system"].min()), QDaysSysMax=pl(days_q["system"].max()))

    tex = ["% Generated by export_data.py; do not edit by hand."]
    tex += [f"\\newcommand{{\\{k}}}{{{val}}}" for k, val in n.items()]
    (OUT / "numbers.tex").write_text("\n".join(tex) + "\n")
    print(pol.round(1).to_string())
    print(team.round(1).to_string())
    print("\n".join(f"{k} = {val}" for k, val in n.items()))
    print(f"data -> {OUT}")


if __name__ == "__main__":
    main()
