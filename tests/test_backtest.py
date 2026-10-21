"""Kupiec, Christoffersen and conditional coverage tests, and the Basel zones."""

import math

import numpy as np
import pandas as pd
import pytest

from ucits_monitor.backtest import (
    basel_zone,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    notification_required,
)


def test_kupiec_zero_exceptions_rejects_the_model():
    lr, p = kupiec_pof(0, 250, 0.99)
    assert lr == pytest.approx(-500 * math.log(0.99))  # 5.0252
    assert p == pytest.approx(0.02497, abs=1e-4)


def test_kupiec_about_the_expected_number_passes():
    lr, p = kupiec_pof(3, 250, 0.99)
    assert p > 0.5


def test_kupiec_ten_exceptions_rejects_the_model():
    lr, p = kupiec_pof(10, 250, 0.99)
    assert p < 0.001


def test_kupiec_matches_the_formula():
    x, t, q = 5, 250, 0.01
    ll_null = (t - x) * math.log(1 - q) + x * math.log(q)
    ll_alt = (t - x) * math.log(1 - x / t) + x * math.log(x / t)
    expected = -2 * (ll_null - ll_alt)
    lr, p = kupiec_pof(x, t, 0.99)
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(math.erfc(math.sqrt(expected / 2)))


def test_kupiec_at_another_confidence_level():
    lr, p = kupiec_pof(12, 250, 0.95)  # 12.5 expected
    assert p > 0.8


def test_kupiec_every_day_an_exception():
    lr, p = kupiec_pof(250, 250, 0.99)
    assert lr > 0
    assert p < 1e-10


@pytest.mark.parametrize("x, t", [(-1, 250), (251, 250), (0, 0)])
def test_kupiec_bad_inputs(x, t):
    with pytest.raises(ValueError):
        kupiec_pof(x, t, 0.99)


def test_christoffersen_no_clustering_gives_zero():
    # pi01 = pi11 = 1/3: an exception is no more likely after an exception
    lr, p = christoffersen_independence([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_christoffersen_matches_the_formula():
    hits = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  # n00=5, n01=1, n10=1, n11=2
    pi01, pi11, pi = 1 / 6, 2 / 3, 3 / 9
    ll0 = 6 * math.log(1 - pi) + 3 * math.log(pi)
    ll1 = 5 * math.log(1 - pi01) + math.log(pi01) + math.log(1 - pi11) + 2 * math.log(pi11)
    expected = -2 * (ll0 - ll1)
    lr, p = christoffersen_independence(hits)
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(math.erfc(math.sqrt(expected / 2)))


@pytest.mark.parametrize("hits", [[0] * 250, [1] + [0] * 249])
def test_christoffersen_nothing_to_test(hits):
    assert christoffersen_independence(hits) == (0.0, 1.0)


def test_christoffersen_back_to_back_exceptions_are_flagged():
    lr, p = christoffersen_independence([0] * 100 + [1, 1, 1] + [0] * 100)
    assert lr > 0
    assert p < 0.01


def test_christoffersen_spread_out_exceptions_pass():
    hits = [0] * 250
    for day in (30, 130, 230):
        hits[day] = 1
    lr, p = christoffersen_independence(hits)
    assert p > 0.5


def test_christoffersen_accepts_series_and_arrays():
    hits = [0] * 100 + [1, 1, 1] + [0] * 100
    expected = christoffersen_independence(hits)
    assert christoffersen_independence(pd.Series(hits)) == pytest.approx(expected)
    assert christoffersen_independence(np.array(hits)) == pytest.approx(expected)


def test_conditional_coverage_adds_the_two_tests():
    hits = [0] * 100 + [1, 1, 1] + [0] * 147
    lr, p = conditional_coverage(hits, 0.99)
    expected = kupiec_pof(3, 250, 0.99)[0] + christoffersen_independence(hits)[0]
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(math.exp(-expected / 2))


@pytest.mark.parametrize("x, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (25, "red")])
def test_basel_zones(x, zone):
    assert basel_zone(x) == zone


def test_notification_above_four():
    assert notification_required(4) is False
    assert notification_required(5) is True
    assert notification_required(0) is False
    assert notification_required(3, threshold=2) is True
