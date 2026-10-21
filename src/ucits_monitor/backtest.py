"""Backtests of the VaR model.

At 99% a sound model is exceeded on about 1 day in 100, or 2.5 days in a
250-day year. Too many exceptions means the model understates risk and too
few means it overstates it. Exceptions should also be spread out; clusters
mean the model is slow to react when volatility jumps.

The chi-squared p-values have closed forms, so SciPy is not needed: with one
degree of freedom p = erfc(sqrt(LR / 2)), with two p = exp(-LR / 2).
"""

from __future__ import annotations

import math
from typing import Sequence

from .models import BACKTEST_NOTIFY_ABOVE


def _xlogy(x: float, y: float) -> float:
    """x * log(y), with 0 * log(0) taken as 0."""
    return x * math.log(y) if x > 0 else 0.0


def kupiec_pof(exceptions: int, observations: int, confidence: float = 0.99) -> tuple[float, float]:
    """Kupiec's proportion-of-failures test of the number of exceptions.

        p       = 1 - confidence               expected exception rate
        p_hat   = exceptions / observations    observed rate
        ln L(q) = (observations - exceptions) * ln(1 - q) + exceptions * ln(q)
        LR      = -2 * (ln L(p) - ln L(p_hat))

    LR is chi-squared with one degree of freedom. Returns (LR, p_value); a
    p-value below 0.05 rejects the model at the 5% level. 0 * ln(0) is taken
    as 0, so zero exceptions and an exception every day both work, and LR is
    floored at 0.

    Zero exceptions in 250 days gives a p-value of about 0.025, so a model
    with no exceptions at all is rejected too.

    Raises ValueError unless observations > 0 and
    0 <= exceptions <= observations.
    """
    if observations <= 0 or not 0 <= exceptions <= observations:
        raise ValueError(f"need 0 <= exceptions <= observations and observations > 0, got {exceptions}/{observations}")
    p = 1 - confidence
    x, t = exceptions, observations
    p_hat = x / t
    ll_null = _xlogy(t - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(t - x, 1 - p_hat) + _xlogy(x, p_hat)
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))


def christoffersen_independence(hits: Sequence[int]) -> tuple[float, float]:
    """Christoffersen's test of whether exceptions cluster.

    Each pair of consecutive days falls into one of four counts:
        n00  no exception, then no exception
        n01  no exception, then an exception
        n10  exception, then no exception
        n11  exception, then exception

        pi01 = n01 / (n00 + n01)    chance of an exception after a quiet day
        pi11 = n11 / (n10 + n11)    chance of an exception after an exception
        pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

        ln L0 = (n00 + n10) * ln(1 - pi) + (n01 + n11) * ln(pi)
        ln L1 = n00 * ln(1 - pi01) + n01 * ln(pi01) + n10 * ln(1 - pi11) + n11 * ln(pi11)
        LR    = -2 * (ln L0 - ln L1), chi-squared with one degree of freedom

    A probability whose denominator is zero is taken as 0, and 0 * ln(0) as
    0. With no exceptions after the first day there is nothing to test, and
    the result is (0.0, 1.0). Returns (LR, p_value), with LR floored at 0.
    """
    h = [int(v) for v in hits]
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(h[:-1], h[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    if n01 + n11 == 0:
        return 0.0, 1.0
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))


def conditional_coverage(hits: Sequence[int], confidence: float = 0.99) -> tuple[float, float]:
    """Both tests at once: the right number of exceptions, and no clustering.

    LR_cc = LR_pof + LR_ind, where LR_pof is Kupiec's statistic for the number
    of hits over the number of days and LR_ind is Christoffersen's. LR_cc is
    chi-squared with two degrees of freedom. Returns (LR_cc, p_value) with
    p_value = exp(-LR_cc / 2).
    """
    h = [int(v) for v in hits]
    lr_pof, _ = kupiec_pof(sum(h), len(h), confidence)
    lr_ind, _ = christoffersen_independence(h)
    lr = lr_pof + lr_ind
    return lr, math.exp(-lr / 2)


def basel_zone(exceptions: int) -> str:
    """The Basel traffic light for 250 days at 99%: 0 to 4 exceptions is
    "green", 5 to 9 is "yellow" and 10 or more is "red"."""
    if exceptions <= 4:
        return "green"
    if exceptions <= 9:
        return "yellow"
    return "red"


def notification_required(exceptions: int, threshold: int = BACKTEST_NOTIFY_ABOVE) -> bool:
    """True when the overshootings have to be reported to the regulator: under
    the UCITS rules, more than 4 in the last 250 days at 99%."""
    return exceptions > threshold
