"""Runs the checks end to end and collects their findings."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .backtest import (
    basel_zone,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    notification_required,
)
from .data import Counterparties, Holdings
from .exposure import check_absolute_var, check_relative_var
from .issuers import aggregate_by_issuer
from .limits import (
    check_aggregate_40,
    check_combined,
    check_deposits,
    check_otc_counterparties,
    check_single_issuer,
)
from .models import (
    BACKTEST_NOTIFY_ABOVE,
    BACKTEST_WINDOW,
    INVESTABLE_ASSET_CLASSES,
    REGIMES,
    STATUSES,
    VAR_CONFIDENCE,
    VAR_HORIZON_DAYS,
    VAR_OBSERVATION_DAYS,
    Finding,
    Regime,
)
from .var import (
    exception_hits,
    historical_var,
    parametric_var,
    rolling_var_forecasts,
    scale_to_horizon,
)

_RANK = {status: i for i, status in enumerate(STATUSES)}


def worst_status(findings: list[Finding]) -> str:
    """The most serious status in a list of findings ("ok" if the list is empty)."""
    return max((f.status for f in findings), key=_RANK.__getitem__, default="ok")


def status_counts(findings: list[Finding]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for f in findings:
        counts[f.status] += 1
    return counts


def resolve_regime(regime: Regime | str) -> Regime:
    if isinstance(regime, Regime):
        return regime
    try:
        return REGIMES[regime]
    except KeyError:
        raise ValueError(f"unknown regime {regime!r}, choose from {', '.join(REGIMES)}") from None


# ---------------------------------------------------------------------------
# Diversification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitResult:
    fund: str
    as_of: dt.date | None
    regime: Regime
    exposures: pd.Series  # issuer -> weight, largest first
    excluded: pd.Series  # asset class -> weight, for holdings the issuer limits do not cover
    findings: list[Finding]
    has_counterparties: bool
    warn_ratio: float

    @property
    def status(self) -> str:
        return worst_status(self.findings)

    @property
    def counts(self) -> dict[str, int]:
        return status_counts(self.findings)

    def by_rule(self, rule: str) -> list[Finding]:
        return [f for f in self.findings if f.rule == rule]


def run_limit_checks(
    holdings: Holdings,
    regime: Regime | str,
    counterparties: Counterparties | None = None,
    warn_ratio: float = 0.9,
) -> LimitResult:
    """Every diversification check for one holdings file under one regime."""
    regime = resolve_regime(regime)
    frame = holdings.frame
    exposures = aggregate_by_issuer(frame)

    findings: list[Finding] = []
    findings += check_single_issuer(exposures, regime, warn_ratio)
    findings += check_aggregate_40(exposures, regime, warn_ratio)
    if counterparties is not None:
        findings += check_deposits(counterparties.deposits, regime, warn_ratio)
        findings += check_otc_counterparties(counterparties.otc, regime, warn_ratio)
        findings += check_combined(exposures, counterparties.deposits, counterparties.otc, regime, warn_ratio)

    if frame.empty:
        excluded = pd.Series(dtype=float, name="weight")
    else:
        asset_class = frame["asset_class"].astype(str).str.strip()
        not_counted = ~asset_class.str.lower().isin(INVESTABLE_ASSET_CLASSES)
        excluded = frame.loc[not_counted].groupby(asset_class[not_counted])["weight"].sum().rename("weight")

    return LimitResult(
        fund=holdings.fund,
        as_of=holdings.as_of,
        regime=regime,
        exposures=exposures,
        excluded=excluded,
        findings=findings,
        has_counterparties=counterparties is not None,
        warn_ratio=warn_ratio,
    )


def compare_regimes(holdings: Holdings, regimes: list[str] | None = None, warn_ratio: float = 0.9) -> list[LimitResult]:
    """The same holdings tested under several regimes, for side-by-side output."""
    names = regimes or list(REGIMES)
    return [run_limit_checks(holdings, name, None, warn_ratio) for name in names]


# ---------------------------------------------------------------------------
# Global exposure and backtesting
# ---------------------------------------------------------------------------

VAR_METHODS = {"historical": historical_var, "parametric": parametric_var}


@dataclass(frozen=True)
class Backtest:
    returns: pd.Series  # daily returns over the backtest window
    forecasts: pd.Series  # the one-day VaR each of those mornings
    hits: pd.Series  # 1 where the loss beat the forecast
    exceptions: int
    observations: int
    kupiec: tuple[float, float]  # (LR, p-value)
    independence: tuple[float, float]
    conditional: tuple[float, float]
    zone: str
    notify: bool
    confidence: float

    @property
    def expected(self) -> float:
        return self.observations * (1 - self.confidence)


@dataclass(frozen=True)
class VarResult:
    fund: str
    as_of: dt.date
    approach: str  # "absolute" or "relative"
    method: str  # "historical" or "parametric"
    confidence: float
    horizon: int
    window: int
    var_1d: float  # the model in use
    var_horizon: float
    alt_method: str  # the other model, shown for comparison
    alt_var_1d: float
    alt_var_horizon: float
    reference_name: str | None
    reference_var_horizon: float | None
    finding: Finding  # the global exposure limit test
    backtest: Backtest
    backtest_finding: Finding  # overshootings against the reporting threshold
    warn_ratio: float

    @property
    def findings(self) -> list[Finding]:
        return [self.finding, self.backtest_finding]

    @property
    def status(self) -> str:
        return worst_status(self.findings)


def _clean_returns(returns: pd.Series) -> pd.Series:
    out = pd.Series(returns, dtype=float).dropna().sort_index()
    return out[~out.index.duplicated(keep="last")]


def run_var_checks(
    fund: str,
    returns: pd.Series,
    reference_returns: pd.Series | None = None,
    *,
    approach: str | None = None,
    method: str = "historical",
    window: int = VAR_OBSERVATION_DAYS,
    confidence: float = VAR_CONFIDENCE,
    horizon: int = VAR_HORIZON_DAYS,
    backtest_days: int = BACKTEST_WINDOW,
    warn_ratio: float = 0.9,
    reference_name: str | None = None,
) -> VarResult:
    """Today's VaR against the global exposure limit, plus a backtest of the model.

    approach defaults to "relative" when reference returns are given and
    "absolute" otherwise.
    """
    if method not in VAR_METHODS:
        raise ValueError(f"method must be one of {', '.join(VAR_METHODS)}, got {method!r}")
    approach = approach or ("relative" if reference_returns is not None else "absolute")
    if approach not in ("absolute", "relative"):
        raise ValueError(f"approach must be 'absolute' or 'relative', got {approach!r}")
    if approach == "relative" and reference_returns is None:
        raise ValueError("the relative approach needs reference portfolio returns")

    returns = _clean_returns(returns)
    needed = window + backtest_days
    if len(returns) < needed:
        raise ValueError(
            f"need at least {needed} daily returns ({window} to estimate VaR, "
            f"then {backtest_days} to backtest it), got {len(returns)}"
        )

    alt_method = "parametric" if method == "historical" else "historical"
    recent = returns.iloc[-window:]
    var_1d = VAR_METHODS[method](recent, confidence)
    alt_var_1d = VAR_METHODS[alt_method](recent, confidence)
    var_h = scale_to_horizon(var_1d, horizon)
    alt_var_h = scale_to_horizon(alt_var_1d, horizon)

    reference_var_h = None
    if approach == "relative":
        joined = pd.concat([returns.rename("fund"), _clean_returns(reference_returns).rename("ref")], axis=1, join="inner").dropna()
        if len(joined) < window:
            raise ValueError(f"fund and reference share only {len(joined)} dates, need {window}")
        tail = joined.iloc[-window:]
        fund_var_h = scale_to_horizon(VAR_METHODS[method](tail["fund"], confidence), horizon)
        reference_var_h = scale_to_horizon(VAR_METHODS[method](tail["ref"], confidence), horizon)
        finding = check_relative_var(fund_var_h, reference_var_h, warn_ratio)
    else:
        finding = check_absolute_var(var_h, warn_ratio)

    forecasts = rolling_var_forecasts(returns, window, confidence, method)
    hits = exception_hits(returns, forecasts).iloc[-backtest_days:]
    exceptions = int(hits.sum())
    hit_list = [int(h) for h in hits]
    backtest = Backtest(
        returns=returns.loc[hits.index],
        forecasts=forecasts.loc[hits.index],
        hits=hits,
        exceptions=exceptions,
        observations=len(hits),
        kupiec=kupiec_pof(exceptions, len(hits), confidence),
        independence=christoffersen_independence(hit_list),
        conditional=conditional_coverage(hit_list, confidence),
        zone=basel_zone(exceptions),
        notify=notification_required(exceptions),
        confidence=confidence,
    )

    if backtest.notify:
        bt_status, bt_note = "breach", f"{backtest.zone} zone; more than {BACKTEST_NOTIFY_ABOVE} exceptions must be reported"
    elif exceptions == BACKTEST_NOTIFY_ABOVE:
        bt_status, bt_note = "warning", f"{backtest.zone} zone; one more exception triggers reporting"
    else:
        bt_status, bt_note = "ok", f"{backtest.zone} zone"
    backtest_finding = Finding(
        "backtest_overshootings",
        f"{method} VaR, last {len(hits)} days",
        float(exceptions),
        float(BACKTEST_NOTIFY_ABOVE),
        bt_status,
        bt_note,
    )

    as_of = returns.index[-1]
    as_of = as_of.date() if hasattr(as_of, "date") else as_of
    return VarResult(
        fund=fund,
        as_of=as_of,
        approach=approach,
        method=method,
        confidence=confidence,
        horizon=horizon,
        window=window,
        var_1d=var_1d,
        var_horizon=var_h,
        alt_method=alt_method,
        alt_var_1d=alt_var_1d,
        alt_var_horizon=alt_var_h,
        reference_name=reference_name,
        reference_var_horizon=reference_var_h,
        finding=finding,
        backtest=backtest,
        backtest_finding=backtest_finding,
        warn_ratio=warn_ratio,
    )
