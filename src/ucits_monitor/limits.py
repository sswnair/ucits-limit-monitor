"""UCITS diversification checks.

Each check returns a list of Finding objects, one per item tested, including
the ones that pass. The report decides what to show.
"""

from __future__ import annotations

import pandas as pd

from .models import Finding, Regime

TOLERANCE = 1e-9  # values this close to a threshold count as on it


def classify(value: float, limit: float, warn_ratio: float = 0.9) -> str:
    """Label a value against its limit.

    Returns "breach" when the value is above the limit, "warning" when it is
    at or above warn_ratio * limit, and "ok" otherwise. A value exactly at the
    limit is a warning, since the limits are "no more than" rules. Values
    within TOLERANCE of either threshold count as on it, so floating-point
    noise cannot turn a value at the limit into a breach.

    Raises ValueError unless 0 < warn_ratio <= 1.
    """
    if not 0 < warn_ratio <= 1:
        raise ValueError(f"warn_ratio must be in (0, 1], got {warn_ratio}")
    if value > limit + TOLERANCE:
        return "breach"
    if value >= warn_ratio * limit - TOLERANCE:
        return "warning"
    return "ok"


def check_single_issuer(exposures: pd.Series, regime: Regime, warn_ratio: float = 0.9) -> list[Finding]:
    """One finding per issuer against the single-issuer limit.

    exposures is a Series of issuer -> weight (fraction of NAV), as returned
    by aggregate_by_issuer. Each issuer is tested against
    regime.single_issuer_limit: 10% for a standard UCITS, 20% for an index
    tracker.

    When the regime has an exceptional limit (35% under Article 53(2)), one
    issuer may use it: the largest, and only if it is above the normal limit.
    That finding carries the higher limit and a note saying so. Every other
    issuer keeps the normal limit.

    Returns an empty list for an empty Series.
    """
    findings: list[Finding] = []
    if exposures.empty:
        return findings
    ranked = exposures.sort_values(ascending=False)
    largest = ranked.index[0]
    for issuer, weight in ranked.items():
        limit = regime.single_issuer_limit
        note = ""
        if (
            regime.exceptional_limit is not None
            and issuer == largest
            and weight > regime.single_issuer_limit + TOLERANCE
        ):
            limit = regime.exceptional_limit
            note = "Uses the one-issuer 35% exception (Article 53(2))"
        findings.append(
            Finding("single_issuer", str(issuer), float(weight), limit, classify(float(weight), limit, warn_ratio), note)
        )
    return findings


def check_aggregate_40(exposures: pd.Series, regime: Regime, warn_ratio: float = 0.9) -> list[Finding]:
    """The 40% part of the 5/10/40 rule.

    Issuers weighing more than regime.aggregate_threshold (5%) may not add up
    to more than regime.aggregate_limit (40%). "More than" is strict: an
    issuer at exactly 5%, or within TOLERANCE of it, does not count.

    Returns one finding with subject "issuers above 5%", the total as its
    value and the count in its note, for example "5 issuers above 5%".
    Returns an empty list when the regime has no aggregate limit, as for
    index trackers.
    """
    if regime.aggregate_limit is None or regime.aggregate_threshold is None:
        return []
    above = exposures[exposures > regime.aggregate_threshold + TOLERANCE]
    total = float(above.sum())
    status = classify(total, regime.aggregate_limit, warn_ratio)
    note = f"{len(above)} issuers above {regime.aggregate_threshold:.0%}"
    return [Finding("aggregate_5_10_40", "issuers above 5%", total, regime.aggregate_limit, status, note)]


def check_deposits(deposits: pd.Series, regime: Regime, warn_ratio: float = 0.9) -> list[Finding]:
    """One finding per bank against the deposit limit, regime.deposit_limit
    (20% of NAV with any one institution).

    deposits is a Series of institution -> deposit (fraction of NAV).
    """
    return [
        Finding("deposits", str(body), float(v), regime.deposit_limit, classify(float(v), regime.deposit_limit, warn_ratio))
        for body, v in deposits.items()
    ]


def check_otc_counterparties(otc: pd.DataFrame, regime: Regime, warn_ratio: float = 0.9) -> list[Finding]:
    """One finding per OTC derivative counterparty.

    otc has one row per trade, with the columns counterparty, exposure
    (fraction of NAV) and is_credit_institution. Trades with the same
    counterparty are added together first. The limit is
    regime.otc_credit_institution_limit (10%) for credit institutions and
    regime.otc_other_limit (5%) for anyone else.
    """
    if otc.empty:
        return []
    grouped = otc.groupby("counterparty").agg(exposure=("exposure", "sum"), credit=("is_credit_institution", "first"))
    findings = []
    for cp, row in grouped.iterrows():
        limit = regime.otc_credit_institution_limit if bool(row["credit"]) else regime.otc_other_limit
        value = float(row["exposure"])
        findings.append(Finding("otc_counterparty", str(cp), value, limit, classify(value, limit, warn_ratio)))
    return findings


def check_combined(
    exposures: pd.Series,
    deposits: pd.Series,
    otc: pd.DataFrame,
    regime: Regime,
    warn_ratio: float = 0.9,
) -> list[Finding]:
    """Securities, deposits and OTC exposure to the same body, taken together.

    Each piece can sit inside its own limit while the three together go over
    regime.combined_limit (20%). OTC trades are summed per counterparty first.
    Names line up across the three sources because all of them are issuer keys.

    Returns one finding per body that appears in at least two of the sources,
    sorted by name; a body in one source only is covered by that source's own
    check. The note lists the sources in the order securities, deposits, otc,
    joined by " + ", for example "deposits + otc". Returns an empty list when
    the regime has no combined limit.
    """
    if regime.combined_limit is None:
        return []
    otc_by_cp = otc.groupby("counterparty")["exposure"].sum() if not otc.empty else pd.Series(dtype=float)
    sources = {"securities": exposures, "deposits": deposits, "otc": otc_by_cp}
    bodies = set().union(*(s.index for s in sources.values()))
    findings = []
    for body in sorted(bodies):
        present = [name for name, s in sources.items() if body in s.index]
        if len(present) < 2:
            continue
        total = float(sum(sources[name][body] for name in present))
        status = classify(total, regime.combined_limit, warn_ratio)
        findings.append(Finding("combined", str(body), total, regime.combined_limit, status, " + ".join(present)))
    return findings
