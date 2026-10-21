"""Global exposure limits for UCITS funds that use VaR.

A fund uses one of two approaches:
* Absolute VaR: the fund's 20-day 99% VaR may not exceed 20% of NAV. Used by
  funds with no meaningful benchmark, such as absolute return funds.
* Relative VaR: the fund's VaR may not exceed twice the VaR of an unleveraged
  reference portfolio. Used by funds managed against a benchmark.
"""

from __future__ import annotations

from .limits import classify
from .models import ABSOLUTE_VAR_LIMIT, RELATIVE_VAR_LIMIT, Finding


def check_absolute_var(var_20d: float, warn_ratio: float = 0.9) -> Finding:
    """The 20-day 99% VaR against the absolute limit, ABSOLUTE_VAR_LIMIT
    (20% of NAV). Returns a Finding with rule "absolute_var" and subject "fund".
    """
    return Finding("absolute_var", "fund", float(var_20d), ABSOLUTE_VAR_LIMIT, classify(float(var_20d), ABSOLUTE_VAR_LIMIT, warn_ratio))


def check_relative_var(fund_var_20d: float, reference_var_20d: float, warn_ratio: float = 0.9) -> Finding:
    """The fund's VaR against twice the reference portfolio's.

    The value tested is the ratio of the two VaRs, against RELATIVE_VAR_LIMIT
    (2.0): a ratio of 1.5 means the fund carries one and a half times the risk
    of its benchmark. The Finding has rule "relative_var", subject "fund vs
    reference portfolio" and a note giving both VaRs, for example
    "fund 12.40% vs reference 9.10%".

    Raises ValueError when the reference VaR is zero or negative.
    """
    if reference_var_20d <= 0:
        raise ValueError(f"reference VaR must be positive, got {reference_var_20d}")
    ratio = float(fund_var_20d) / float(reference_var_20d)
    note = f"fund {fund_var_20d:.2%} vs reference {reference_var_20d:.2%}"
    return Finding("relative_var", "fund vs reference portfolio", ratio, RELATIVE_VAR_LIMIT, classify(ratio, RELATIVE_VAR_LIMIT, warn_ratio), note)
