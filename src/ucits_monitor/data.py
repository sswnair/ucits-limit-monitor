"""Downloading and parsing public data.

* Fund holdings: the daily holdings file iShares publishes for each UCITS ETF.
* Price history: daily closes from Stooq.
* Counterparties: deposits and OTC derivative exposures, from a CSV you supply.
  Public holdings files do not show which bank holds the cash or who the
  derivative counterparties are, so these come from an illustrative file.

Downloads are cached in data/cache, so a second run the same day works offline.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import pathlib
from dataclasses import dataclass

import pandas as pd

from .issuers import issuer_key

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "funds.json"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

# Synthetic files for trying the tool offline. Not real funds.
SAMPLE_HOLDINGS = SAMPLES_DIR / "SAMPLE_holdings.csv"
SAMPLE_PRICES = SAMPLES_DIR / "SAMPLE_prices.csv"
SAMPLE_REFERENCE = SAMPLES_DIR / "SAMPLE_reference.csv"
SAMPLE_COUNTERPARTIES = SAMPLES_DIR / "counterparties_example.csv"

ISHARES_HOLDINGS_URL = (
    "https://www.ishares.com/{site}/products/{product_id}/{slug}/"
    "{ajax}.ajax?fileType=csv&fileName={ticker}_holdings&dataType=fund"
)
# The "Download Holdings" link on iShares' Swiss site. Other country sites use
# other ajax numbers; config/funds.json can override both.
ISHARES_DEFAULT_SITE = "ch/individual/en"
ISHARES_DEFAULT_AJAX = "1495092304805"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

USER_AGENT = "ucits-limit-monitor/0.1 (portfolio project)"

# ---------------------------------------------------------------------------
# Fund configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fund:
    ticker: str
    name: str
    product_id: str
    slug: str
    regime: str
    price_symbol: str
    reference_symbol: str | None = None
    reference_name: str | None = None
    site: str = ISHARES_DEFAULT_SITE
    holdings_ajax: str = ISHARES_DEFAULT_AJAX
    file_ticker: str | None = None  # the ticker iShares uses in the file name, if different
    notes: str = ""


def load_funds(path: pathlib.Path = CONFIG_PATH) -> dict[str, Fund]:
    """Read config/funds.json. Keys the Fund class does not know are ignored."""
    raw = json.loads(pathlib.Path(path).read_text())
    known = {f.name for f in dataclasses.fields(Fund)}
    funds = {}
    for entry in raw["funds"]:
        fund = Fund(**{k: v for k, v in entry.items() if k in known})
        funds[fund.ticker.upper()] = fund
    return funds


# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------

_COLUMN_MAP = {
    "ticker": "ticker",
    "issuer ticker": "ticker",
    "name": "name",
    "sector": "sector",
    "asset class": "asset_class",
    "market value": "market_value",
    "weight (%)": "weight_pct",
    "notional value": "notional_value",
    "shares": "quantity",
    "nominal": "quantity",
    "price": "price",
    "location": "location",
    "location of risk": "location",
    "exchange": "exchange",
    "currency": "currency",
    "market currency": "market_currency",
}

_DATE_FORMATS = ("%d/%b/%Y", "%b %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y")


@dataclass(frozen=True)
class Holdings:
    fund: str
    as_of: dt.date | None
    frame: pd.DataFrame


def _to_number(value: str) -> float:
    # Thousands separators vary by site: 1,234.56 or 1’234.56 (Swiss)
    text = str(value).strip()
    for sep in (",", "\u2019", "'", "\xa0", "\u2009", "%"):
        text = text.replace(sep, "")
    if text in {"", "-", "--", "N/A", "n/a"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_date(text: str) -> dt.date | None:
    text = text.strip().strip('"').strip().replace("Sept", "Sep")
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_holdings(text: str) -> tuple[dt.date | None, pd.DataFrame]:
    """Parse an iShares holdings CSV into (as_of_date, frame).

    The frame has one row per holding with canonical columns, including
    weight as a fraction of NAV and an issuer key for aggregation.
    """
    text = text.lstrip("\ufeff")
    lines = text.splitlines()

    as_of = None
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if as_of is None and "holdings as of" in low:
            parts = next(csv.reader([line]))
            if len(parts) > 1:
                as_of = _parse_date(parts[1])
        if low.startswith("ticker") or ("weight (%)" in low and "name" in low):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the holdings table header (a row starting 'Ticker').")

    rows = []
    reader = csv.reader(io.StringIO("\n".join(lines[header_idx:])))
    header = [h.strip() for h in next(reader)]
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            break  # the table ends at the first blank line
        if len(row) < len(header) - 1:
            break  # disclaimer text after the table
        rows.append(row[: len(header)] + [""] * (len(header) - len(row)))

    frame = pd.DataFrame(rows, columns=header)
    frame.columns = [_COLUMN_MAP.get(c.strip().lower(), c.strip().lower().replace(" ", "_")) for c in frame.columns]

    for col in ("market_value", "weight_pct", "notional_value", "quantity", "price"):
        if col in frame.columns:
            frame[col] = frame[col].map(_to_number)
    if "weight_pct" not in frame.columns:
        raise ValueError("Holdings file has no 'Weight (%)' column.")

    frame["weight"] = frame["weight_pct"] / 100.0
    frame["asset_class"] = frame.get("asset_class", pd.Series("Equity", index=frame.index)).astype(str).str.strip()
    frame["issuer"] = [issuer_key(n, t) for n, t in zip(frame["name"], frame.get("ticker", [None] * len(frame)))]
    frame = frame.dropna(subset=["weight"]).reset_index(drop=True)
    return as_of, frame


def _http_get(url: str) -> str:
    import requests  # imported here so the parsers work without network libraries

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def holdings_url(fund: Fund) -> str:
    return ISHARES_HOLDINGS_URL.format(
        site=fund.site,
        product_id=fund.product_id,
        slug=fund.slug,
        ajax=fund.holdings_ajax,
        ticker=fund.file_ticker or fund.ticker,
    )


def load_holdings(fund: Fund, use_cache: bool = True) -> Holdings:
    """Download the latest holdings file (or read today's cached copy) and parse it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{fund.ticker}_holdings_{dt.date.today():%Y-%m-%d}.csv"
    if use_cache and cache.exists():
        file_date, frame = parse_holdings(cache.read_text(encoding="utf-8"))
    else:
        text = _http_get(holdings_url(fund))
        file_date, frame = parse_holdings(text)  # fails before a bad download is cached
        cache.write_text(text, encoding="utf-8")
    return Holdings(fund=fund.ticker, as_of=file_date, frame=frame)


def read_holdings_file(path: pathlib.Path | str, fund: str = "SAMPLE") -> Holdings:
    """Parse a holdings CSV already on disk."""
    as_of, frame = parse_holdings(pathlib.Path(path).read_text(encoding="utf-8-sig"))
    return Holdings(fund=fund, as_of=as_of, frame=frame)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def parse_prices(text: str) -> pd.Series:
    """Parse a Date/Close CSV (Stooq's format) into a Series of closes by date."""
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        first_line = text.strip().splitlines()[0][:80] if text.strip() else "an empty file"
        raise ValueError(f"Expected a CSV with Date and Close columns, got: {first_line}")
    series = pd.Series(
        pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["date"]),
        name="close",
    )
    return series.dropna().sort_index()


def load_prices(symbol: str, use_cache: bool = True) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"prices_{symbol.replace('^', 'idx_')}_{dt.date.today():%Y%m%d}.csv"
    if use_cache and cache.exists():
        return parse_prices(cache.read_text())
    text = _http_get(STOOQ_URL.format(symbol=symbol))
    prices = parse_prices(text)  # fails before a bad download is cached
    cache.write_text(text)
    return prices


def read_prices_file(path: pathlib.Path | str) -> pd.Series:
    return parse_prices(pathlib.Path(path).read_text())


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns, r_t = P_t / P_(t-1) - 1."""
    return prices.pct_change().dropna().rename("return")


# ---------------------------------------------------------------------------
# Counterparties (deposits and OTC derivatives)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterparties:
    deposits: pd.Series  # index: body, values: fraction of NAV
    otc: pd.DataFrame  # columns: counterparty, exposure, is_credit_institution


def read_counterparties(path: pathlib.Path | str) -> Counterparties:
    """Read deposits and OTC exposures from a CSV with columns
    type (deposit|otc), body, exposure_pct, is_credit_institution (yes|no).
    Lines starting with # are comments."""
    frame = pd.read_csv(path, comment="#")
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame["body"] = [issuer_key(b) for b in frame["body"]]
    frame["exposure"] = frame["exposure_pct"].astype(float) / 100.0
    kind = frame["type"].str.strip().str.lower()

    dep = frame.loc[kind == "deposit"].groupby("body")["exposure"].sum().rename("exposure")
    otc = frame.loc[kind == "otc", ["body", "exposure", "is_credit_institution"]].rename(columns={"body": "counterparty"})
    otc["is_credit_institution"] = otc["is_credit_institution"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})
    return Counterparties(deposits=dep, otc=otc.reset_index(drop=True))
