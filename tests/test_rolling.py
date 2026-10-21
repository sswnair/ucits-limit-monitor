"""Rolling VaR forecasts and exception days."""

import numpy as np
import pandas as pd
import pytest

from ucits_monitor.var import exception_hits, historical_var, parametric_var, rolling_var_forecasts


def returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(rng.normal(0, 0.01, n), index=index, name="return")


def test_shape_and_warm_up_period():
    r = returns()
    f = rolling_var_forecasts(r, window=250)
    assert f.index.equals(r.index)
    assert f.name == "var_historical"
    assert f.iloc[:250].isna().all()
    assert f.iloc[250:].notna().all()


@pytest.mark.parametrize("t", [250, 275, 299])
def test_each_forecast_uses_the_window_before_it(t):
    r = returns()
    f = rolling_var_forecasts(r, window=250)
    assert f.iloc[t] == pytest.approx(historical_var(r.iloc[t - 250 : t], 0.99))


def test_no_look_ahead():
    r = returns(260)
    r.iloc[250:253] = -0.20  # three crash days
    f = rolling_var_forecasts(r, window=250)
    assert f.iloc[250] < 0.05  # the crash is not in its own forecast
    assert f.iloc[253] == pytest.approx(0.20)  # three crashes in the window: the third-worst is a crash


def test_parametric_method():
    r = returns()
    f = rolling_var_forecasts(r, window=250, method="parametric")
    assert f.name == "var_parametric"
    assert f.iloc[255] == pytest.approx(parametric_var(r.iloc[5:255], 0.99))


def test_shorter_window():
    r = returns(60)
    f = rolling_var_forecasts(r, window=20)
    assert f.iloc[:20].isna().all()
    assert f.iloc[20] == pytest.approx(historical_var(r.iloc[0:20], 0.99))


def test_series_shorter_than_the_window_is_all_nan():
    f = rolling_var_forecasts(returns(100), window=250)
    assert len(f) == 100
    assert f.isna().all()


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        rolling_var_forecasts(returns(), method="monte carlo")


def test_hits():
    index = pd.bdate_range("2025-01-01", periods=4)
    r = pd.Series([-0.03, -0.01, 0.02, -0.05], index=index)
    f = pd.Series([np.nan, 0.02, 0.02, 0.04], index=index)
    hits = exception_hits(r, f)
    assert list(hits) == [0, 0, 1]
    assert list(hits.index) == list(index[1:])
    assert hits.name == "hit"
    assert pd.api.types.is_integer_dtype(hits)


def test_a_loss_equal_to_var_is_not_a_hit():
    index = pd.bdate_range("2025-01-01", periods=1)
    assert list(exception_hits(pd.Series([-0.02], index=index), pd.Series([0.02], index=index))) == [0]


def test_hits_line_up_by_date_not_position():
    index = pd.bdate_range("2025-01-01", periods=4)
    r = pd.Series([-0.03, -0.01, 0.02, -0.05], index=index)
    f = pd.Series([0.04, 0.02], index=[index[3], index[1]])  # different order, two dates only
    hits = exception_hits(r, f)
    assert len(hits) == 2
    assert hits.loc[index[1]] == 0
    assert hits.loc[index[3]] == 1
