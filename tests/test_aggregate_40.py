"""The 40% part of the 5/10/40 rule."""

import pandas as pd
import pytest

from ucits_monitor.limits import check_aggregate_40
from ucits_monitor.models import INDEX_TRACKER, INDEX_TRACKER_EXCEPTIONAL, STANDARD


def exposures(values):
    names = [chr(ord("A") + i) for i in range(len(values))]
    return pd.Series(values, index=names, dtype=float, name="weight").sort_values(ascending=False)


def test_issuers_above_5_percent_add_up_to_a_breach():
    out = check_aggregate_40(exposures([0.09, 0.08, 0.08, 0.08, 0.08, 0.03]), STANDARD)
    assert len(out) == 1
    f = out[0]
    assert f.rule == "aggregate_5_10_40"
    assert f.subject == "issuers above 5%"
    assert f.value == pytest.approx(0.41)
    assert f.limit == 0.40
    assert f.status == "breach"
    assert f.note == "5 issuers above 5%"


def test_exactly_5_percent_is_not_counted():
    f = check_aggregate_40(exposures([0.09, 0.08, 0.08, 0.08, 0.05, 0.03]), STANDARD)[0]
    assert f.value == pytest.approx(0.33)
    assert f.status == "ok"
    assert f.note == "4 issuers above 5%"


def test_a_hair_above_5_percent_is_still_5_percent():
    f = check_aggregate_40(exposures([0.09, 0.08, 0.08, 0.08, 0.05000000001]), STANDARD)[0]
    assert f.value == pytest.approx(0.33)


def test_close_to_the_cap_is_a_warning():
    f = check_aggregate_40(exposures([0.10, 0.10, 0.10, 0.09, 0.04]), STANDARD)[0]
    assert f.value == pytest.approx(0.39)
    assert f.status == "warning"


def test_nothing_above_5_percent():
    f = check_aggregate_40(exposures([0.05, 0.04, 0.03]), STANDARD)[0]
    assert f.value == 0.0
    assert f.status == "ok"
    assert f.note == "0 issuers above 5%"


@pytest.mark.parametrize("regime", [INDEX_TRACKER, INDEX_TRACKER_EXCEPTIONAL])
def test_index_trackers_have_no_40_percent_rule(regime):
    assert check_aggregate_40(exposures([0.19, 0.18, 0.17]), regime) == []
