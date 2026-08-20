"""Unit tests for the feature engineering logic (the part most prone to bugs)."""
import numpy as np
import pandas as pd
from train import build_features


def _toy_df():
    # one product, 40 consecutive days, known sales
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    return pd.DataFrame({
        "id": ["ITEM_A"] * 40,
        "date": dates,
        "wday": [(i % 7) + 1 for i in range(40)],
        "sales": list(range(40)),          # 0,1,2,...,39
        "sell_price": [5.0] * 40,
        "event_type_1": ["No Event"] * 40,
        "event_type_2": ["No Event"] * 40,
    })


def test_lag_7_is_seven_days_back():
    df = build_features(_toy_df())
    # sales on day index 10 is 10; lag_7 should be sales 7 rows earlier = 3
    row = df.iloc[10]
    assert row["lag_7"] == 3


def test_rolling_mean_excludes_current_day():
    """CRITICAL leakage test: rolling window must NOT include today's sales."""
    df = build_features(_toy_df())
    # rolling_mean_7 at row 10 must be mean of sales days 3..9 = mean(3,4,5,6,7,8,9)=6.0
    # If it wrongly included day 10 (value 10), it would be mean(4..10)=7.0
    assert df.iloc[10]["rolling_mean_7"] == 6.0


def test_no_cross_series_leakage():
    """Lags must not bleed from one product into another."""
    a = _toy_df()
    b = _toy_df()
    b["id"] = "ITEM_B"
    b["sales"] = [100 + i for i in range(40)]
    df = build_features(pd.concat([a, b], ignore_index=True))
    # first 7 rows of each series have no lag_7 -> NaN, not a value from the other series
    first_b = df[df["id"] == "ITEM_B"].iloc[0]
    assert np.isnan(first_b["lag_7"])


def test_is_weekend_flag():
    df = build_features(_toy_df())
    # wday 1 (Sat) and 7 (Sun) are weekend
    assert df[df["wday"] == 1]["is_weekend"].iloc[0] == 1
    assert df[df["wday"] == 3]["is_weekend"].iloc[0] == 0
