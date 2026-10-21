"""Adding up holdings by issuer."""

import pandas as pd
import pytest

from ucits_monitor.issuers import aggregate_by_issuer


def holdings(rows):
    return pd.DataFrame(rows, columns=["issuer", "asset_class", "weight"])


def test_share_classes_of_one_issuer_are_added():
    out = aggregate_by_issuer(
        holdings(
            [
                ("ALPHABET INC", "Equity", 0.03),
                ("ALPHABET INC", "Equity", 0.025),
                ("APPLE INC", "Equity", 0.05),
            ]
        )
    )
    assert out["ALPHABET INC"] == pytest.approx(0.055)
    assert out["APPLE INC"] == pytest.approx(0.05)
    assert len(out) == 2


def test_cash_futures_and_fx_do_not_count_but_bonds_do():
    out = aggregate_by_issuer(
        holdings(
            [
                ("APPLE INC", "Equity", 0.05),
                ("USD CASH", "Cash", 0.02),
                ("NASDAQ 100 E-MINI DEC 26", "Futures", 0.01),
                ("EUR USD", "FX", 0.001),
                ("REPUBLIC OF FRANCE", "Fixed Income", 0.04),
            ]
        )
    )
    assert set(out.index) == {"APPLE INC", "REPUBLIC OF FRANCE"}


def test_asset_class_ignores_case_and_spaces():
    out = aggregate_by_issuer(
        holdings(
            [
                ("A", "Equity", 0.01),
                ("B", "equity", 0.02),
                ("C", " EQUITY ", 0.03),
                ("D", "fixed income", 0.04),
                ("E", "Cash ", 0.05),
            ]
        )
    )
    assert set(out.index) == {"A", "B", "C", "D"}


def test_sorted_largest_first_and_named_weight():
    out = aggregate_by_issuer(holdings([("SMALL", "Equity", 0.01), ("BIG", "Equity", 0.09), ("MID", "Equity", 0.04)]))
    assert list(out.index) == ["BIG", "MID", "SMALL"]
    assert out.name == "weight"


def test_extra_columns_are_ignored():
    frame = holdings([("APPLE INC", "Equity", 0.05)]).assign(name="APPLE INC", ticker="AAPL", sector="IT")
    out = aggregate_by_issuer(frame)
    assert out["APPLE INC"] == pytest.approx(0.05)


def test_empty_input_gives_an_empty_series():
    out = aggregate_by_issuer(holdings([]))
    assert isinstance(out, pd.Series)
    assert out.empty


def test_nothing_investable_gives_an_empty_series():
    out = aggregate_by_issuer(holdings([("USD CASH", "Cash", 1.0)]))
    assert out.empty
