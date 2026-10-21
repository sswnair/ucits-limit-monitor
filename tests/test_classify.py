"""Status labels against a limit."""

import pytest

from ucits_monitor.limits import classify


@pytest.mark.parametrize(
    "value, limit, expected",
    [
        (0.05, 0.10, "ok"),
        (0.0, 0.10, "ok"),
        (0.0899, 0.10, "ok"),
        (0.09, 0.10, "warning"),  # exactly 90% of the limit
        (0.10, 0.10, "warning"),  # at the limit is allowed
        (0.1001, 0.10, "breach"),
        (1.9, 2.0, "warning"),  # works for ratios too
        (2.01, 2.0, "breach"),
    ],
)
def test_labels(value, limit, expected):
    assert classify(value, limit) == expected


def test_floating_point_noise_at_the_limit_is_not_a_breach():
    assert classify(0.1 + 0.2, 0.3) == "warning"


def test_warn_ratio_moves_the_warning_band():
    assert classify(0.085, 0.10, warn_ratio=0.8) == "warning"
    assert classify(0.079, 0.10, warn_ratio=0.8) == "ok"


def test_warn_ratio_of_one_warns_only_at_the_limit():
    assert classify(0.10, 0.10, warn_ratio=1.0) == "warning"
    assert classify(0.0999, 0.10, warn_ratio=1.0) == "ok"


@pytest.mark.parametrize("bad", [0, -0.5, 1.01, 2])
def test_bad_warn_ratio_raises(bad):
    with pytest.raises(ValueError):
        classify(0.05, 0.10, warn_ratio=bad)
