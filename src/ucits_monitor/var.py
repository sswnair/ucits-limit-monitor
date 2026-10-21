"""Value at Risk.

Conventions
-----------
* Returns are daily simple returns: 0.01 means +1%.
* A loss is a return with the sign flipped: a -3% day is a loss of 0.03.
* VaR is a positive fraction of NAV: 0.025 means that on 99% of days the fund
  should lose no more than 2.5%.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def _clean(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    return values[~np.isnan(values)]


def historical_var(returns, confidence: float = 0.99) -> float:
    """One-day VaR by historical simulation.

    VaR is the k-th worst loss in the sample, where
    k = ceil(n * (1 - confidence)) for n returns: the third-worst day out of
    250 at 99%. n * (1 - confidence) is rounded to 10 decimals before the
    ceiling, so floating-point error cannot push k up by a day.

    NaNs are ignored. When the k-th worst day is a gain, VaR is 0.0. Raises
    ValueError when no returns are left or confidence is not strictly between
    0 and 1.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    losses = -_clean(returns)
    n = len(losses)
    if n == 0:
        raise ValueError("no returns to compute VaR from")
    k = max(1, math.ceil(round(n * (1 - confidence), 10)))
    worst_first = np.sort(losses)[::-1]
    return max(float(worst_first[k - 1]), 0.0)


def parametric_var(returns, confidence: float = 0.99) -> float:
    """One-day VaR assuming normally distributed returns.

    VaR = z * sigma, where z is the standard normal quantile at the
    confidence level (2.326 at 99%) and sigma is the sample standard
    deviation (ddof=1). The mean is left out, since over a single day it is
    too small to matter.

    NaNs are ignored. Raises ValueError with fewer than two returns, or when
    confidence is not strictly between 0 and 1.
    """
    values = _clean(returns)
    if len(values) < 2:
        raise ValueError("need at least two returns for a standard deviation")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    z = NormalDist().inv_cdf(confidence)
    return float(z * np.std(values, ddof=1))


def scale_to_horizon(var_1d: float, days: int = 20) -> float:
    """Scale a one-day VaR to a longer horizon by the square root of time.

    VaR(days) = VaR(1 day) * sqrt(days); UCITS uses 20 business days. The
    rule assumes daily returns are independent, so it understates risk when
    volatility clusters. Raises ValueError when days is below 1.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    return float(var_1d * math.sqrt(days))


def rolling_var_forecasts(
    returns: pd.Series,
    window: int = 250,
    confidence: float = 0.99,
    method: str = "historical",
) -> pd.Series:
    """The one-day VaR that was available each morning.

    The forecast for day t comes from the `window` returns before it,
    positions t - window to t - 1. Day t's own return is never part of its
    forecast; letting it in is look-ahead bias, and it flatters the backtest.

    Returns a Series on the same index as returns, named "var_<method>", with
    NaN for the first `window` days. method is "historical" or "parametric";
    anything else raises ValueError.
    """
    functions = {"historical": historical_var, "parametric": parametric_var}
    if method not in functions:
        raise ValueError(f"method must be one of {sorted(functions)}, got {method!r}")
    fn = functions[method]
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(window, len(values)):
        out[t] = fn(values[t - window : t], confidence)
    return pd.Series(out, index=returns.index, name=f"var_{method}")


def exception_hits(returns: pd.Series, forecasts: pd.Series) -> pd.Series:
    """1 on the days the loss beat that morning's VaR, 0 otherwise.

    The loss is the return with its sign flipped, and it has to be strictly
    greater than the forecast to count as an exception (an "overshooting").
    The two Series are matched on their dates, and days without a forecast
    are dropped. Returns integers, named "hit".
    """
    aligned = pd.concat([returns.rename("r"), forecasts.rename("v")], axis=1).dropna()
    return ((-aligned["r"]) > aligned["v"]).astype(int).rename("hit")
