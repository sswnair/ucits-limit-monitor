"""The single-issuer limit, including the 35% exception for index trackers."""

import pandas as pd
import pytest

from ucits_monitor.limits import check_single_issuer
from ucits_monitor.models import INDEX_TRACKER, INDEX_TRACKER_EXCEPTIONAL, STANDARD


def exposures(mapping, sort=True):
    s = pd.Series(mapping, dtype=float, name="weight")
    return s.sort_values(ascending=False) if sort else s


def by_subject(findings):
    return {f.subject: f for f in findings}


def test_standard_regime():
    out = by_subject(check_single_issuer(exposures({"A": 0.12, "B": 0.09, "C": 0.04}), STANDARD))
    assert out["A"].status == "breach"
    assert out["B"].status == "warning"
    assert out["C"].status == "ok"
    assert all(f.limit == 0.10 for f in out.values())
    assert all(f.rule == "single_issuer" for f in out.values())


def test_values_are_carried_through():
    out = by_subject(check_single_issuer(exposures({"A": 0.12, "B": 0.09, "C": 0.04}), STANDARD))
    assert out["A"].value == pytest.approx(0.12)
    assert out["C"].value == pytest.approx(0.04)
    assert len(out) == 3


def test_index_regime_raises_the_limit_to_20_percent():
    out = by_subject(check_single_issuer(exposures({"A": 0.21, "B": 0.19, "C": 0.12}), INDEX_TRACKER))
    assert out["A"].status == "breach"
    assert out["B"].status == "warning"
    assert out["C"].status == "ok"
    assert all(f.limit == 0.20 for f in out.values())


def test_35_percent_exception_goes_to_the_largest_issuer_only():
    out = by_subject(check_single_issuer(exposures({"A": 0.30, "B": 0.22, "C": 0.10}), INDEX_TRACKER_EXCEPTIONAL))
    assert out["A"].limit == 0.35
    assert out["A"].status == "ok"
    assert "35%" in out["A"].note
    assert out["B"].limit == 0.20
    assert out["B"].status == "breach"  # only one issuer gets the exception
    assert out["C"].limit == 0.20
    assert out["B"].note == "" and out["C"].note == ""


def test_exception_is_not_used_when_nobody_needs_it():
    out = by_subject(check_single_issuer(exposures({"A": 0.15, "B": 0.05}), INDEX_TRACKER_EXCEPTIONAL))
    assert out["A"].limit == 0.20
    assert out["A"].note == ""


def test_the_exception_has_a_ceiling_too():
    out = by_subject(check_single_issuer(exposures({"A": 0.40, "B": 0.10}), INDEX_TRACKER_EXCEPTIONAL))
    assert out["A"].limit == 0.35
    assert out["A"].status == "breach"


def test_input_order_does_not_matter():
    out = by_subject(check_single_issuer(exposures({"C": 0.10, "A": 0.30, "B": 0.22}, sort=False), INDEX_TRACKER_EXCEPTIONAL))
    assert out["A"].limit == 0.35
    assert out["B"].limit == 0.20


def test_empty_input():
    assert check_single_issuer(pd.Series(dtype=float), STANDARD) == []
