"""Historical and parametric VaR, and scaling to 20 days."""

import math

import numpy as np
import pandas as pd
import pytest

from ucits_monitor.var import historical_var, parametric_var, scale_to_horizon

Z99 = 2.3263478740408408  # NormalDist().inv_cdf(0.99)


def test_third_worst_day_out_of_250():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 250)
    returns[[10, 50, 100, 200]] = [-0.08, -0.07, -0.06, -0.05]
    # ceil(250 x 1%) = 3, so VaR is the third-worst loss
    assert historical_var(returns, 0.99) == pytest.approx(0.06)


def test_one_percent_of_100_days_is_one_day():
    returns = np.full(100, 0.001)
    returns[3], returns[60] = -0.05, -0.04
    # 100 x 1% is exactly 1 day: the worst loss, not the second worst
    assert historical_var(returns, 0.99) == pytest.approx(0.05)


def test_nans_are_ignored():
    returns = [np.nan, -0.04, 0.01, np.nan, -0.02, 0.03]
    assert historical_var(returns, 0.99) == pytest.approx(0.04)


def test_never_negative():
    assert historical_var([0.01, 0.02, 0.005], 0.99) == 0.0


def test_other_confidence_levels():
    returns = np.linspace(-0.05, 0.05, 101)
    # ceil(101 x 5%) = 6, so the sixth-worst: -0.045
    assert historical_var(returns, 0.95) == pytest.approx(0.045)


def test_lists_arrays_and_series_agree():
    values = [-0.03, 0.01, -0.02, 0.02, -0.01]
    expected = historical_var(values, 0.99)
    assert historical_var(np.array(values), 0.99) == expected
    assert historical_var(pd.Series(values), 0.99) == expected


@pytest.mark.parametrize("returns", [[], [np.nan, np.nan]])
def test_no_data_raises(returns):
    with pytest.raises(ValueError):
        historical_var(returns, 0.99)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.2])
def test_bad_confidence_raises(confidence):
    with pytest.raises(ValueError):
        historical_var([0.01, -0.01], confidence)


def test_parametric_known_value():
    returns = [0.01, -0.01, 0.02, -0.02]
    expected = Z99 * math.sqrt(0.001 / 3)  # sample standard deviation, ddof=1
    assert parametric_var(returns, 0.99) == pytest.approx(expected)


def test_parametric_ignores_the_mean():
    returns = np.array([0.01, -0.01, 0.02, -0.02])
    assert parametric_var(returns + 0.003, 0.99) == pytest.approx(parametric_var(returns, 0.99))


def test_parametric_ignores_nans():
    assert parametric_var([0.01, np.nan, -0.01], 0.99) == pytest.approx(parametric_var([0.01, -0.01], 0.99))


@pytest.mark.parametrize("returns", [[0.01], [np.nan, 0.01], []])
def test_parametric_needs_two_returns(returns):
    with pytest.raises(ValueError):
        parametric_var(returns, 0.99)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_parametric_bad_confidence_raises(confidence):
    with pytest.raises(ValueError):
        parametric_var([0.01, -0.01, 0.02], confidence)


def test_square_root_of_time():
    assert scale_to_horizon(0.02, 20) == pytest.approx(0.02 * math.sqrt(20))
    assert scale_to_horizon(0.025) == pytest.approx(0.1118033989)  # UCITS default: 20 days
    assert scale_to_horizon(0.02, 1) == pytest.approx(0.02)


@pytest.mark.parametrize("days", [0, -5])
def test_horizon_must_be_at_least_one_day(days):
    with pytest.raises(ValueError):
        scale_to_horizon(0.02, days)
