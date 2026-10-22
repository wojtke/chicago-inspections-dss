import numpy as np
import pandas as pd

from inspections.schedule import assign, backtest


def toy_window() -> pd.DataFrame:
    """Ten visits over five days, two per day; the two failing places were visited last."""
    return pd.DataFrame({
        "actual_day": np.repeat(np.arange(5), 2),
        "fail": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        "p_fail": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0.9, 0.8],
    })


def test_assign_respects_daily_capacity():
    cap = np.array([3, 0, 2])
    day = assign(np.array([4, 3, 2, 1, 0]), cap)
    assert np.bincount(day, minlength=3).tolist() == [3, 0, 2]
    assert day.tolist() == [2, 2, 0, 0, 0]


def test_backtest_policies():
    summary, curves = backtest(toy_window(), days=5)
    assert summary.loc["actual", "days_sooner"] == 0
    # Risk-first moves both failures from day 4 to day 0.
    assert summary.loc["risk_first", "days_sooner"] == 4
    assert summary.loc["risk_first", "days_sooner"] >= summary.loc["random", "days_sooner"]
    assert summary.loc["risk_first", "fails_in_first_half"] == 1
    assert curves["risk_first"].iloc[-1] == curves["actual"].iloc[-1] == 2
