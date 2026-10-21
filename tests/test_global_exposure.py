"""Absolute and relative VaR limits."""

import pytest

from ucits_monitor.exposure import check_absolute_var, check_relative_var


def test_absolute_var_within_limit():
    f = check_absolute_var(0.12)
    assert f.rule == "absolute_var"
    assert f.subject == "fund"
    assert f.value == pytest.approx(0.12)
    assert f.limit == 0.20
    assert f.status == "ok"


@pytest.mark.parametrize("var, status", [(0.185, "warning"), (0.20, "warning"), (0.23, "breach")])
def test_absolute_var_statuses(var, status):
    assert check_absolute_var(var).status == status


def test_warn_ratio_is_passed_through():
    assert check_absolute_var(0.15, warn_ratio=0.7).status == "warning"


def test_relative_var_is_a_ratio():
    f = check_relative_var(0.124, 0.091)
    assert f.rule == "relative_var"
    assert f.subject == "fund vs reference portfolio"
    assert f.value == pytest.approx(0.124 / 0.091)
    assert f.limit == 2.0
    assert f.status == "ok"
    assert f.note == "fund 12.40% vs reference 9.10%"


def test_relative_var_breach():
    assert check_relative_var(0.25, 0.12).status == "breach"


def test_relative_var_warning():
    assert check_relative_var(0.19, 0.10).status == "warning"


@pytest.mark.parametrize("reference", [0.0, -0.01])
def test_reference_var_must_be_positive(reference):
    with pytest.raises(ValueError):
        check_relative_var(0.10, reference)
