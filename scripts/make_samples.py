"""Write the synthetic sample files in data/samples.

The samples let the tool run offline and give the tests fixed inputs. Every
number in them is made up. Run from the project root:

    python scripts/make_samples.py

The files are committed, so this only needs to run after a change to the samples.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"

AS_OF = pd.Timestamp("2026-10-16")
FUND_NAV = 1_000_000_000.0  # USD, only used to turn weights into market values
SEED = 131

# Ticker, name, sector, weight (%), price. The weights are chosen so the fund
# fails the 5/10/40 rule as a standard UCITS but passes as an index tracker.
NAMED_HOLDINGS = [
    ("NVDA", "NVIDIA CORP", "Information Technology", 9.10, 182.40),
    ("MSFT", "MICROSOFT CORP", "Information Technology", 8.60, 512.30),
    ("AAPL", "APPLE INC", "Information Technology", 8.30, 247.10),
    ("AMZN", "AMAZON COM INC", "Consumer Discretionary", 5.60, 221.80),
    ("AVGO", "BROADCOM INC", "Information Technology", 5.20, 342.50),
    ("META", "META PLATFORMS INC CLASS A", "Communication", 3.60, 717.40),
    ("TSLA", "TESLA INC", "Consumer Discretionary", 3.10, 433.70),
    ("GOOGL", "ALPHABET INC CLASS A", "Communication", 2.60, 251.20),
    ("GOOG", "ALPHABET INC CLASS C", "Communication", 2.50, 252.00),
    ("COST", "COSTCO WHOLESALE CORP", "Consumer Staples", 2.40, 935.20),
    ("NFLX", "NETFLIX INC", "Communication", 2.20, 1210.00),
]
FILLER_COUNT = 89
FILLER_SECTORS = [
    "Information Technology",
    "Health Care",
    "Consumer Discretionary",
    "Communication",
    "Industrials",
    "Consumer Staples",
    "Utilities",
]
CASH_WEIGHT = 0.30
FUTURES_WEIGHT = 0.10

HEADER = [
    "Ticker", "Name", "Sector", "Asset Class", "Market Value", "Weight (%)",
    "Notional Value", "Shares", "Price", "Location", "Exchange", "Market Currency",
]


def _money(x: float) -> str:
    return f"{x:,.2f}"


def filler_weights(total: float, count: int) -> list[float]:
    """Decreasing weights in hundredths of a percent that add up to `total` exactly."""
    raw = np.geomspace(1.0, 0.12, count)
    cents = np.floor(raw / raw.sum() * total * 100).astype(int)
    cents[: int(round(total * 100)) - int(cents.sum())] += 1  # hand out the rounding
    return [c / 100 for c in cents]


def write_holdings(path: pathlib.Path) -> None:
    named_total = sum(w for *_, w, _ in NAMED_HOLDINGS)
    filler_total = round(100 - named_total - CASH_WEIGHT - FUTURES_WEIGHT, 2)
    rows = []

    def add(ticker, name, sector, asset_class, weight, price, exchange, notional=None):
        mv = FUND_NAV * weight / 100
        shares = mv / price if price else 0.0
        rows.append([
            ticker, name, sector, asset_class, _money(mv), f"{weight:.2f}",
            _money(notional if notional is not None else mv), _money(shares), _money(price),
            "United States", exchange, "USD",
        ])

    for ticker, name, sector, weight, price in NAMED_HOLDINGS:
        add(ticker, name, sector, "Equity", weight, price, "NASDAQ")
    for i, weight in enumerate(filler_weights(filler_total, FILLER_COUNT), start=1):
        sector = FILLER_SECTORS[i % len(FILLER_SECTORS)]
        add(f"S{i:03d}", f"SAMPLE HOLDING {i:03d} INC", sector, "Equity", weight, 50.0 + i, "NASDAQ")
    add("USD", "USD CASH", "Cash and/or Derivatives", "Cash", CASH_WEIGHT, 100.0, "-")
    add("NQZ6", "NASDAQ 100 E-MINI DEC 26", "Cash and/or Derivatives", "Futures", FUTURES_WEIGHT, 25_000.0,
        "Chicago Mercantile Exchange", notional=FUND_NAV * 0.012)

    with path.open("w", newline="") as fh:
        fh.write("Sample Technology Fund (synthetic data for testing, not a real fund)\n")
        fh.write(f'Fund Holdings as of,"{AS_OF:%d/%b/%Y}"\n')
        fh.write('Inception Date,"-"\n')
        fh.write('Shares Outstanding,"-"\n')
        fh.write('Stock,"-"\nBond,"-"\nCash,"-"\nOther,"-"\n')
        fh.write("\xa0\n")
        writer = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
        fh.write("\xa0\n")
        fh.write('"Weights, prices and the fund itself are made up for testing."\n')


def simulate_returns(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A calm market with one month of stress, and a fund that runs a little hotter."""
    df = 5
    shocks = rng.standard_t(df, n) / np.sqrt(df / (df - 2))  # unit variance, fat tails
    vol = np.full(n, 0.0105)
    stress_start = n - 160  # inside the last 250 days, so the backtest sees it
    vol[stress_start : stress_start + 30] = 0.026
    vol[stress_start + 30 : stress_start + 45] = 0.017  # calming down
    reference = 0.0004 + vol * shocks
    fund = 1.08 * reference + rng.normal(0.0, 0.0025, n)
    return fund, reference


def write_prices(path: pathlib.Path, returns: np.ndarray, start_price: float, rng: np.random.Generator) -> None:
    dates = pd.bdate_range(end=AS_OF, periods=len(returns) + 1)
    close = start_price * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, len(close)))
    spread = np.abs(rng.normal(0, 0.004, len(close)))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(200_000, 900_000, len(close))
    frame = pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), "Open": open_.round(4), "High": high.round(4),
         "Low": low.round(4), "Close": close.round(4), "Volume": volume}
    )
    frame.to_csv(path, index=False)


COUNTERPARTIES = """\
# Fictional names and exposures for testing. Not real institutions or positions.
type,body,exposure_pct,is_credit_institution
deposit,Liffey Bank plc,3.0,yes
deposit,Corrib Bank AG,18.5,yes
otc,Boyne Trust Bank,4.0,yes
otc,Boyne Trust Bank,5.5,yes
otc,Northwind Markets Ltd,5.5,no
otc,Corrib Bank AG,2.0,yes
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--days", type=int, default=780, help="number of daily returns")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fund, reference = simulate_returns(rng, args.days)
    write_holdings(OUT / "SAMPLE_holdings.csv")
    write_prices(OUT / "SAMPLE_prices.csv", fund, 100.0, rng)
    write_prices(OUT / "SAMPLE_reference.csv", reference, 1000.0, rng)
    (OUT / "counterparties_example.csv").write_text(COUNTERPARTIES)
    print(f"Wrote samples to {OUT}")


if __name__ == "__main__":
    main()
