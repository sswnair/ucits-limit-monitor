"""End to end on the synthetic sample."""

import pytest

from ucits_monitor.cli import main
from ucits_monitor.engine import compare_regimes, run_limit_checks, run_var_checks
from ucits_monitor.models import INDEX_TRACKER, STANDARD
from ucits_monitor.report import result_frame, update_breach_log, write_outputs


def finding(result, rule, subject=None):
    matches = [f for f in result.findings if f.rule == rule and (subject is None or f.subject == subject)]
    assert len(matches) == 1, f"expected one {rule} finding for {subject}, got {len(matches)}"
    return matches[0]


def test_sample_as_a_standard_ucits(sample_holdings):
    result = run_limit_checks(sample_holdings, STANDARD)
    agg = finding(result, "aggregate_5_10_40")
    assert agg.value == pytest.approx(0.419)
    assert agg.status == "breach"
    assert agg.note == "6 issuers above 5%"
    assert finding(result, "single_issuer", "NVIDIA CORP").status == "warning"
    assert finding(result, "single_issuer", "ALPHABET INC").value == pytest.approx(0.051)
    assert result.status == "breach"
    assert result.excluded.to_dict() == pytest.approx({"Cash": 0.003, "Futures": 0.001})


def test_sample_as_an_index_tracker(sample_holdings):
    result = run_limit_checks(sample_holdings, INDEX_TRACKER)
    assert result.status == "ok"
    assert result.by_rule("aggregate_5_10_40") == []


def test_sample_counterparties(sample_holdings, sample_counterparties):
    result = run_limit_checks(sample_holdings, STANDARD, sample_counterparties)
    assert finding(result, "otc_counterparty", "NORTHWIND MARKETS LTD").status == "breach"
    assert finding(result, "otc_counterparty", "BOYNE TRUST BANK").value == pytest.approx(0.095)
    combined = finding(result, "combined", "CORRIB BANK AG")
    assert combined.value == pytest.approx(0.205)
    assert combined.status == "breach"
    assert result.counts == {"ok": 100, "warning": 3, "breach": 3}


def test_compare_regimes(sample_holdings):
    statuses = {r.regime.name: r.status for r in compare_regimes(sample_holdings)}
    assert statuses == {"standard": "breach", "index_tracker": "ok", "index_tracker_exceptional": "ok"}


def test_sample_var_relative(sample_returns, sample_reference_returns):
    result = run_var_checks("SAMPLE", sample_returns, sample_reference_returns)
    assert result.approach == "relative"
    assert result.finding.rule == "relative_var"
    assert result.finding.value == pytest.approx(1.0548, abs=1e-3)
    assert result.finding.status == "ok"
    bt = result.backtest
    assert bt.observations == 250
    assert bt.exceptions == 7
    assert bt.zone == "yellow"
    assert bt.notify is True
    assert bt.kupiec[1] < 0.05
    assert bt.independence[1] < 0.01  # the exceptions bunch up in the stress month
    assert result.backtest_finding.status == "breach"


def test_sample_var_absolute(sample_returns):
    result = run_var_checks("SAMPLE", sample_returns)
    assert result.approach == "absolute"
    assert result.var_horizon == pytest.approx(0.1875, abs=1e-3)
    assert result.finding.status == "warning"


def test_normal_model_misses_fat_tails(sample_returns):
    historical = run_var_checks("SAMPLE", sample_returns, method="historical")
    parametric = run_var_checks("SAMPLE", sample_returns, method="parametric")
    assert parametric.backtest.exceptions > historical.backtest.exceptions
    assert parametric.backtest.zone == "red"


def test_var_needs_enough_history(sample_returns):
    with pytest.raises(ValueError):
        run_var_checks("SAMPLE", sample_returns.iloc[:400])


def test_outputs(tmp_path, sample_holdings, sample_counterparties, sample_returns, sample_reference_returns):
    limits = run_limit_checks(sample_holdings, STANDARD, sample_counterparties)
    var = run_var_checks("SAMPLE", sample_returns, sample_reference_returns)
    paths = write_outputs(tmp_path, limits, var, title="Sample Technology Fund", sample=True)
    page = paths["report"].read_text()
    assert "Sample Technology Fund" in page and "5/10/40 rule" in page and "Synthetic sample data" in page

    frame = result_frame(limits, var)
    assert len(frame) == len(limits.findings) + 2
    log = update_breach_log(frame, paths["breach_log"])  # the same run, logged twice
    assert not log.duplicated(subset=["date", "fund", "regime", "rule", "subject"]).any()
    assert set(log["status"]) == {"warning", "breach"}


def test_command_line(tmp_path):
    assert main(["report", "--sample", "--out", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*_report.html"))
    assert main(["report", "--sample", "--out", str(tmp_path), "--fail-on-breach"]) == 2
