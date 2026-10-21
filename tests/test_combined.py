"""Securities, deposits and OTC exposure to one body, taken together."""

import pandas as pd
import pytest

from ucits_monitor.limits import check_combined
from ucits_monitor.models import INDEX_TRACKER, STANDARD


def otc(rows):
    return pd.DataFrame(rows, columns=["counterparty", "exposure", "is_credit_institution"])


def test_securities_deposits_and_otc_add_up():
    out = check_combined(
        pd.Series({"XYZ BANK": 0.08, "APPLE INC": 0.09}),
        pd.Series({"XYZ BANK": 0.07}),
        otc([("XYZ BANK", 0.06, True)]),
        STANDARD,
    )
    assert [f.subject for f in out] == ["XYZ BANK"]  # Apple appears in one source only
    f = out[0]
    assert f.rule == "combined"
    assert f.value == pytest.approx(0.21)
    assert f.limit == 0.20
    assert f.status == "breach"
    assert f.note == "securities + deposits + otc"


def test_each_piece_within_its_limit_but_together_over():
    out = check_combined(
        pd.Series({"APPLE INC": 0.09}),
        pd.Series({"CORRIB BANK AG": 0.185}),
        otc([("CORRIB BANK AG", 0.02, True)]),
        STANDARD,
    )
    assert len(out) == 1
    assert out[0].value == pytest.approx(0.205)
    assert out[0].status == "breach"
    assert out[0].note == "deposits + otc"


def test_otc_trades_are_summed_before_combining():
    out = check_combined(
        pd.Series(dtype=float),
        pd.Series({"BANK": 0.10}),
        otc([("BANK", 0.03, True), ("BANK", 0.03, True)]),
        STANDARD,
    )
    assert out[0].value == pytest.approx(0.16)
    assert out[0].status == "ok"


def test_sorted_by_body_name():
    out = check_combined(
        pd.Series({"ZETA BANK": 0.05, "ALPHA BANK": 0.05}),
        pd.Series({"ZETA BANK": 0.01, "ALPHA BANK": 0.01}),
        otc([]),
        STANDARD,
    )
    assert [f.subject for f in out] == ["ALPHA BANK", "ZETA BANK"]
    assert all(f.note == "securities + deposits" for f in out)


def test_no_overlap_means_no_findings():
    out = check_combined(pd.Series({"APPLE INC": 0.09}), pd.Series(dtype=float), otc([]), STANDARD)
    assert out == []


def test_not_run_for_index_trackers():
    out = check_combined(pd.Series({"XYZ BANK": 0.15}), pd.Series({"XYZ BANK": 0.10}), otc([]), INDEX_TRACKER)
    assert out == []
