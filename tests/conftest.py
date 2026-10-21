"""Shared test fixtures.

The tests run straight from a checkout: the line below puts src/ on the path,
so `pytest` works before or after `pip install -e .`.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ucits_monitor.data import (  # noqa: E402
    SAMPLE_COUNTERPARTIES,
    SAMPLE_HOLDINGS,
    SAMPLE_PRICES,
    SAMPLE_REFERENCE,
    read_counterparties,
    read_holdings_file,
    read_prices_file,
    simple_returns,
)


@pytest.fixture
def sample_holdings():
    return read_holdings_file(SAMPLE_HOLDINGS, fund="SAMPLE")


@pytest.fixture
def sample_counterparties():
    return read_counterparties(SAMPLE_COUNTERPARTIES)


@pytest.fixture
def sample_returns():
    return simple_returns(read_prices_file(SAMPLE_PRICES))


@pytest.fixture
def sample_reference_returns():
    return simple_returns(read_prices_file(SAMPLE_REFERENCE))
