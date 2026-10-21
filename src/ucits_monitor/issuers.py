"""Grouping holdings by issuer.

UCITS limits apply to the issuer, not the line item. Alphabet's class A and
class C shares are two holdings but one issuer, and they count together.
"""

from __future__ import annotations

import re

import pandas as pd

from .models import INVESTABLE_ASSET_CLASSES

# Names the cleanup below cannot reconcile on its own, keyed by ticker.
ISSUER_OVERRIDES: dict[str, str] = {
    "GOOG": "ALPHABET INC",
    "GOOGL": "ALPHABET INC",
    "BRKB": "BERKSHIRE HATHAWAY INC",
    "BRK.B": "BERKSHIRE HATHAWAY INC",
    "FOX": "FOX CORP",
    "FOXA": "FOX CORP",
    "NWS": "NEWS CORP",
    "NWSA": "NEWS CORP",
}

_SHARE_CLASS = re.compile(
    r"\b(CLASS|CL|SERIES|SER)\s+[A-Z0-9]\b"
    r"|\b(REGISTERED|REG|ADR|GDR|ORD|ORDINARY|SHS|SHARES|NPV|NON VOTING|PREF|PREFERRED|UNITS?)\b"
)
_NON_WORD = re.compile(r"[^A-Z0-9& ]+")
_SPACES = re.compile(r"\s+")


def issuer_key(name: str, ticker: str | None = None) -> str:
    """Normalize a security name to an issuer key.

    >>> issuer_key("ALPHABET INC CLASS A")
    'ALPHABET INC'
    >>> issuer_key("Alphabet Inc. Class C")
    'ALPHABET INC'
    """
    if ticker:
        override = ISSUER_OVERRIDES.get(str(ticker).strip().upper())
        if override:
            return override
    key = str(name or "").upper().replace("-", " ")
    key = _NON_WORD.sub(" ", key)
    key = _SHARE_CLASS.sub(" ", key)
    return _SPACES.sub(" ", key).strip()


def aggregate_by_issuer(holdings: pd.DataFrame) -> pd.Series:
    """Total weight per issuer, largest first.

    holdings needs the columns issuer (an issuer key, see issuer_key),
    asset_class and weight (fraction of NAV). Only rows whose asset class is
    in INVESTABLE_ASSET_CLASSES count, compared case-insensitively and
    ignoring surrounding spaces, so cash, FX and futures lines are left out.
    Rows that share an issuer are added together.

    Returns a Series indexed by issuer and named "weight". An empty input, or
    one with nothing investable in it, gives an empty Series.
    """
    if holdings.empty:
        return pd.Series(dtype=float, name="weight")
    mask = holdings["asset_class"].astype(str).str.strip().str.lower().isin(INVESTABLE_ASSET_CLASSES)
    totals = holdings.loc[mask].groupby("issuer")["weight"].sum()
    return totals.sort_values(ascending=False).rename("weight")
