"""Rule sets, VaR parameters and the Finding record."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Diversification rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """One set of UCITS diversification limits. All limits are fractions of NAV."""

    name: str
    label: str
    single_issuer_limit: float
    exceptional_limit: float | None  # one issuer may go this high (Article 53(2))
    aggregate_threshold: float | None  # holdings above this count toward the 40% test
    aggregate_limit: float | None  # the 40% in "5/10/40"
    deposit_limit: float
    otc_credit_institution_limit: float
    otc_other_limit: float
    combined_limit: float | None  # securities + deposits + OTC with one body
    source: str


STANDARD = Regime(
    name="standard",
    label="Standard UCITS",
    single_issuer_limit=0.10,
    exceptional_limit=None,
    aggregate_threshold=0.05,
    aggregate_limit=0.40,
    deposit_limit=0.20,
    otc_credit_institution_limit=0.10,
    otc_other_limit=0.05,
    combined_limit=0.20,
    source="UCITS Directive 2009/65/EC, Article 52",
)

INDEX_TRACKER = Regime(
    name="index_tracker",
    label="Index-tracking UCITS",
    single_issuer_limit=0.20,
    exceptional_limit=None,
    aggregate_threshold=None,
    aggregate_limit=None,
    deposit_limit=0.20,
    otc_credit_institution_limit=0.10,
    otc_other_limit=0.05,
    combined_limit=None,  # not run for index trackers, see Limitations in the README
    source="UCITS Directive 2009/65/EC, Article 53(1)",
)

INDEX_TRACKER_EXCEPTIONAL = Regime(
    name="index_tracker_exceptional",
    label="Index-tracking UCITS, 35% exception",
    single_issuer_limit=0.20,
    exceptional_limit=0.35,
    aggregate_threshold=None,
    aggregate_limit=None,
    deposit_limit=0.20,
    otc_credit_institution_limit=0.10,
    otc_other_limit=0.05,
    combined_limit=None,
    source="UCITS Directive 2009/65/EC, Article 53(2)",
)

REGIMES: dict[str, Regime] = {
    r.name: r for r in (STANDARD, INDEX_TRACKER, INDEX_TRACKER_EXCEPTIONAL)
}

# Asset classes that count toward the issuer limits in this version.
# Cash, FX, futures and money-market fund units fall under other rules.
INVESTABLE_ASSET_CLASSES = frozenset({"equity", "fixed income"})

# ---------------------------------------------------------------------------
# Global exposure (VaR) rules
# CESR's Guidelines on Risk Measurement and the Calculation of Global Exposure
# and Counterparty Risk for UCITS (CESR/10-788)
# ---------------------------------------------------------------------------

VAR_CONFIDENCE = 0.99  # one-tailed
VAR_HORIZON_DAYS = 20  # one month of business days
VAR_OBSERVATION_DAYS = 250  # at least one year of daily history
ABSOLUTE_VAR_LIMIT = 0.20  # 20-day 99% VaR may not exceed 20% of NAV
RELATIVE_VAR_LIMIT = 2.0  # fund VaR may not exceed twice the reference portfolio's
BACKTEST_WINDOW = 250  # most recent business days used for backtesting
BACKTEST_NOTIFY_ABOVE = 4  # more overshootings than this at 99% must be reported

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

STATUSES = ("ok", "warning", "breach")


@dataclass(frozen=True)
class Finding:
    """The result of testing one thing against one limit.

    value and limit are on the same scale: fractions of NAV for the
    diversification and absolute VaR tests, a ratio for relative VaR.
    """

    rule: str
    subject: str
    value: float
    limit: float
    status: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    @property
    def headroom(self) -> float:
        return self.limit - self.value

    @property
    def utilization(self) -> float:
        return self.value / self.limit if self.limit else float("nan")
