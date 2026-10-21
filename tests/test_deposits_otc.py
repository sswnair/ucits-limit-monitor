"""Deposit and OTC counterparty limits."""

import pandas as pd
import pytest

from ucits_monitor.limits import check_deposits, check_otc_counterparties
from ucits_monitor.models import INDEX_TRACKER, STANDARD


def otc(rows):
    return pd.DataFrame(rows, columns=["counterparty", "exposure", "is_credit_institution"])


def by_subject(findings):
    return {f.subject: f for f in findings}


def test_deposits_against_20_percent():
    out = by_subject(check_deposits(pd.Series({"A": 0.18, "B": 0.21, "C": 0.05}), STANDARD))
    assert out["A"].status == "warning"
    assert out["B"].status == "breach"
    assert out["C"].status == "ok"
    assert all(f.limit == 0.20 and f.rule == "deposits" for f in out.values())


def test_deposit_limit_also_applies_to_index_trackers():
    out = check_deposits(pd.Series({"B": 0.21}), INDEX_TRACKER)
    assert out[0].status == "breach"


def test_no_deposits():
    assert check_deposits(pd.Series(dtype=float), STANDARD) == []


def test_otc_limit_depends_on_the_counterparty():
    out = by_subject(check_otc_counterparties(otc([("BANK", 0.09, True), ("FUND", 0.06, False)]), STANDARD))
    assert out["BANK"].limit == 0.10
    assert out["BANK"].status == "warning"
    assert out["FUND"].limit == 0.05
    assert out["FUND"].status == "breach"
    assert all(f.rule == "otc_counterparty" for f in out.values())


def test_trades_with_one_counterparty_are_added_first():
    out = check_otc_counterparties(otc([("BANK", 0.04, True), ("BANK", 0.05, True)]), STANDARD)
    assert len(out) == 1
    assert out[0].value == pytest.approx(0.09)
    assert out[0].status == "warning"


def test_small_exposure_to_a_non_bank_is_fine():
    out = check_otc_counterparties(otc([("FUND", 0.04, False)]), STANDARD)
    assert out[0].status == "ok"


def test_no_otc_trades():
    assert check_otc_counterparties(otc([]), STANDARD) == []
