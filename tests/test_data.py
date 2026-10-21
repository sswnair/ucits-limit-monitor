"""Parsing holdings, prices and counterparties; issuer names; the shared models."""

import datetime as dt
import json

import pandas as pd
import pytest

from ucits_monitor.data import (
    SAMPLE_COUNTERPARTIES,
    SAMPLE_HOLDINGS,
    Fund,
    holdings_url,
    load_funds,
    parse_holdings,
    parse_prices,
    read_counterparties,
    read_holdings_file,
    simple_returns,
)
from ucits_monitor.issuers import issuer_key
from ucits_monitor.models import REGIMES, Finding

ISHARES_STYLE = (
    "\ufeffExample Equity UCITS ETF\n"
    'Fund Holdings as of,"16/Oct/2026"\n'
    'Inception Date,"26/Jan/2010"\n'
    "\xa0\n"
    "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Market Currency\n"
    '"NVDA","NVIDIA CORP","Information Technology","Equity","1,234,567.00","9.10","1,234,567.00","6,768.00","182.40","United States","NASDAQ","USD"\n'
    '"GOOGL","ALPHABET INC CLASS A","Communication","Equity","500,000.00","2.60","500,000.00","1,990.00","251.20","United States","NASDAQ","USD"\n'
    '"GOOG","ALPHABET INC CLASS C","Communication","Equity","480,000.00","2.50","480,000.00","1,905.00","252.00","United States","NASDAQ","USD"\n'
    '"USD","USD CASH","Cash and/or Derivatives","Cash","30,000.00","0.30","30,000.00","30,000.00","100.00","United States","-","USD"\n'
    "\xa0\n"
    '"Holdings are subject to change."\n'
)


def test_parse_ishares_style_csv():
    as_of, frame = parse_holdings(ISHARES_STYLE)
    assert as_of == dt.date(2026, 10, 16)
    assert len(frame) == 4
    assert list(frame["weight"]) == pytest.approx([0.091, 0.026, 0.025, 0.003])
    assert list(frame["issuer"]) == ["NVIDIA CORP", "ALPHABET INC", "ALPHABET INC", "USD CASH"]
    assert list(frame["asset_class"]) == ["Equity", "Equity", "Equity", "Cash"]
    assert frame.loc[0, "market_value"] == pytest.approx(1_234_567.0)


SWISS_SITE_STYLE = (
    'Fund Holdings as of,"16/Oct/2026"\n'
    " \n"
    "Ticker,Name,Type,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Market Currency\n"
    '"NVDA","NVIDIA","EQUITY","Information Technology","Equity","2\u2019502\u2019248\u2019448.63","8.16","2\u2019502\u2019248\u2019448.63","13\u2019500\u2019000.00","185.35","United States","NASDAQ","USD"\n'
    '"GOOGL","ALPHABET CLASS A","EQUITY","Communication","Equity","969\u2019737\u2019384.52","3.16","969\u2019737\u2019384.52","3\u2019860\u2019000.00","251.20","United States","NASDAQ","USD"\n'
    '"GOOG","ALPHABET CLASS C","EQUITY","Communication","Equity","902\u2019731\u2019814.95","2.94","902\u2019731\u2019814.95","3\u2019580\u2019000.00","252.00","United States","NASDAQ","USD"\n'
    '"EUR","EUR CASH","CASH","Cash and/or Derivatives","Cash","92\u2019994.35","0.00","92\u2019994.35","82\u2019640.00","112.53","European Union","-","EUR"\n'
)


def test_parse_swiss_site_csv():
    as_of, frame = parse_holdings(SWISS_SITE_STYLE)
    assert as_of == dt.date(2026, 10, 16)
    assert list(frame["weight"]) == pytest.approx([0.0816, 0.0316, 0.0294, 0.0])
    assert frame.loc[0, "market_value"] == pytest.approx(2_502_248_448.63)
    assert list(frame["issuer"]) == ["NVIDIA", "ALPHABET INC", "ALPHABET INC", "EUR CASH"]


def test_parse_us_style_date():
    as_of, _ = parse_holdings(ISHARES_STYLE.replace('"16/Oct/2026"', '"Oct 16, 2026"'))
    assert as_of == dt.date(2026, 10, 16)


def test_no_table_raises():
    with pytest.raises(ValueError):
        parse_holdings("just some text\nwith no holdings table\n")


def test_no_weight_column_raises():
    with pytest.raises(ValueError):
        parse_holdings("Ticker,Name,Asset Class\nAAPL,APPLE INC,Equity\n")


def test_sample_holdings_file():
    holdings = read_holdings_file(SAMPLE_HOLDINGS)
    assert holdings.as_of == dt.date(2026, 10, 16)
    assert len(holdings.frame) == 102
    assert holdings.frame["weight"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name, ticker, key",
    [
        ("ALPHABET INC CLASS A", None, "ALPHABET INC"),
        ("Alphabet Inc. Class C", None, "ALPHABET INC"),
        ("META PLATFORMS INC CLASS A", None, "META PLATFORMS INC"),
        ("BERKSHIRE HATHAWAY INC CLASS B", None, "BERKSHIRE HATHAWAY INC"),
        ("FOX CORP CL A", None, "FOX CORP"),
        ("NESTLE SA REG", None, "NESTLE SA"),
        ("AT&T INC", None, "AT&T INC"),
        ("Some Odd Name", "GOOGL", "ALPHABET INC"),
        ("  apple   inc  ", None, "APPLE INC"),
    ],
)
def test_issuer_key(name, ticker, key):
    assert issuer_key(name, ticker) == key


def test_parse_prices_sorts_by_date():
    text = "Date,Open,High,Low,Close,Volume\n2026-10-15,1,1,1,101.5,10\n2026-10-14,1,1,1,100.0,10\n"
    prices = parse_prices(text)
    assert list(prices) == [100.0, 101.5]
    assert prices.index[0] == pd.Timestamp("2026-10-14")
    assert prices.name == "close"


def test_parse_prices_needs_a_close_column():
    with pytest.raises(ValueError):
        parse_prices("Date,Price\n2026-10-15,100\n")


def test_simple_returns():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2026-01-05", periods=3))
    assert list(simple_returns(prices)) == pytest.approx([0.10, -0.10])


def test_sample_counterparties():
    cp = read_counterparties(SAMPLE_COUNTERPARTIES)
    assert cp.deposits.to_dict() == pytest.approx({"CORRIB BANK AG": 0.185, "LIFFEY BANK PLC": 0.03})
    assert len(cp.otc) == 4
    assert cp.otc["is_credit_institution"].dtype == bool
    northwind = cp.otc.loc[cp.otc["counterparty"] == "NORTHWIND MARKETS LTD"]
    assert not northwind["is_credit_institution"].iloc[0]


def test_fund_config_is_valid():
    funds = load_funds()
    assert funds
    for ticker, fund in funds.items():
        assert ticker == fund.ticker.upper()
        assert fund.regime in REGIMES
        assert fund.product_id and fund.slug and fund.price_symbol


def test_unknown_config_keys_are_ignored(tmp_path):
    path = tmp_path / "funds.json"
    entry = dict(ticker="test", name="Test", product_id="1", slug="test", regime="standard", price_symbol="t.uk", comment="x")
    path.write_text(json.dumps({"funds": [entry]}))
    assert load_funds(path)["TEST"].name == "Test"


def test_holdings_url():
    fund = Fund("CNDX", "Example", "253741", "example-slug", "index_tracker", "cndx.uk", file_ticker="CSNDX")
    url = holdings_url(fund)
    assert url.startswith("https://www.ishares.com/ch/individual/en/products/253741/example-slug/")
    assert "fileType=csv" in url and "fileName=CSNDX_holdings" in url


def test_finding_rejects_unknown_status():
    with pytest.raises(ValueError):
        Finding("single_issuer", "A", 0.05, 0.10, "fine")


def test_finding_headroom_and_utilization():
    f = Finding("single_issuer", "A", 0.08, 0.10, "ok")
    assert f.headroom == pytest.approx(0.02)
    assert f.utilization == pytest.approx(0.8)
