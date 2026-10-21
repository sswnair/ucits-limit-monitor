"""Command line interface.

    ucits-monitor report --sample            everything, on the synthetic sample
    ucits-monitor limits --fund CNDX         diversification checks on today's holdings
    ucits-monitor var --fund CNDX            VaR global exposure and backtest
    ucits-monitor compare --fund CNDX        the same holdings under every regime
    ucits-monitor funds                      the funds in config/funds.json

Run any command with --help for its options.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass

from . import __version__
from .data import (
    SAMPLE_COUNTERPARTIES,
    SAMPLE_HOLDINGS,
    SAMPLE_PRICES,
    SAMPLE_REFERENCE,
    Fund,
    load_funds,
    load_holdings,
    load_prices,
    read_counterparties,
    read_holdings_file,
    read_prices_file,
    simple_returns,
)
from .engine import LimitResult, VarResult, compare_regimes, resolve_regime, run_limit_checks, run_var_checks
from .models import REGIMES, Finding
from .report import RULE_LABELS, STATUS_LABELS, fmt_date, fmt_p, fmt_pct, fmt_value, plural, write_outputs

SAMPLE_TITLE = "Sample Technology Fund"
SAMPLE_REFERENCE_NAME = "sample reference index"

EXIT_OK, EXIT_ERROR, EXIT_BREACH = 0, 1, 2


class UsageError(Exception):
    """A problem with the command line or the inputs, shown without a traceback."""


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_COLORS = {"breach": "\033[31m", "warning": "\033[33m", "ok": "\033[32m"}


def _use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _status(status: str) -> str:
    text = STATUS_LABELS[status].upper().ljust(7)
    return f"{_COLORS[status]}{text}\033[0m" if _use_color() else text


def _print_findings(findings: list[Finding], show_all: bool) -> None:
    order = {"breach": 0, "warning": 1, "ok": 2}
    shown = findings if show_all else [f for f in findings if f.status != "ok"]
    for f in sorted(shown, key=lambda f: (order[f.status], f.rule, f.subject)):
        note = f"  {f.note}" if f.note else ""
        print(
            f"  {_status(f.status)}  {RULE_LABELS.get(f.rule, f.rule):<18} {f.subject[:30]:<30} "
            f"{fmt_value(f.rule, f.value):>8}  limit {fmt_value(f.rule, f.limit):>7}{note}"
        )
    counts = {s: sum(f.status == s for f in findings) for s in ("breach", "warning", "ok")}
    print(
        f"  {plural(counts['breach'], 'breach', 'breaches')}, "
        f"{plural(counts['warning'], 'warning', 'warnings')}, {counts['ok']} within limits"
    )


def _print_limits(result: LimitResult, title: str, show_all: bool) -> None:
    print(f"\n{title} · holdings as of {fmt_date(result.as_of)} · {result.regime.label}")
    _print_findings(result.findings, show_all)
    if not result.exposures.empty:
        name, weight = result.exposures.index[0], float(result.exposures.iloc[0])
        print(f"  Largest issuer: {name} {fmt_pct(weight)}")


def _print_var(result: VarResult, title: str, show_all: bool) -> None:
    bt = result.backtest
    print(
        f"\n{title} · prices to {fmt_date(result.as_of)} · {result.approach} VaR, {result.method}, "
        f"{result.confidence:.0%} / {result.horizon} days"
    )
    print(f"  {result.horizon}-day VaR {fmt_pct(result.var_horizon)} ({result.alt_method}: {fmt_pct(result.alt_var_horizon)})")
    if result.reference_var_horizon is not None:
        print(f"  Reference portfolio {fmt_pct(result.reference_var_horizon)}")
    print(
        f"  Backtest: {bt.exceptions} exceptions in {bt.observations} days, {bt.zone} zone, "
        f"Kupiec p {fmt_p(bt.kupiec[1])}, clustering p {fmt_p(bt.independence[1])}"
    )
    _print_findings(result.findings, True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    label: str
    title: str
    fund: Fund | None
    sample: bool


def _resolve(args: argparse.Namespace) -> Inputs:
    if args.sample:
        return Inputs("SAMPLE", SAMPLE_TITLE, None, True)
    if args.fund:
        funds = load_funds()
        fund = funds.get(args.fund.upper())
        if fund is None:
            raise UsageError(f"{args.fund} is not in config/funds.json. Configured: {', '.join(funds) or 'none'}")
        return Inputs(fund.ticker, fund.name, fund, False)
    if any(getattr(args, name, None) for name in ("holdings_file", "prices_file")):
        return Inputs("FILE", "Fund from file", None, False)
    raise UsageError("Choose what to check: --sample, --fund TICKER, or --holdings-file / --prices-file.")


def _holdings(args: argparse.Namespace, inputs: Inputs):
    if args.holdings_file:
        return read_holdings_file(args.holdings_file, fund=inputs.label)
    if inputs.sample:
        return read_holdings_file(SAMPLE_HOLDINGS, fund=inputs.label)
    if inputs.fund is not None:
        return load_holdings(inputs.fund, use_cache=not args.no_cache)
    raise UsageError("No holdings: pass --fund, --sample or --holdings-file.")


def _regime(args: argparse.Namespace, inputs: Inputs):
    if args.regime:
        return resolve_regime(args.regime)
    if inputs.fund is not None:
        return resolve_regime(inputs.fund.regime)
    return resolve_regime("standard")


def _counterparties(args: argparse.Namespace, inputs: Inputs):
    if args.counterparties:
        return read_counterparties(args.counterparties)
    if inputs.sample:
        return read_counterparties(SAMPLE_COUNTERPARTIES)
    return None


def _returns(args: argparse.Namespace, inputs: Inputs):
    if args.prices_file:
        prices = read_prices_file(args.prices_file)
    elif inputs.sample:
        prices = read_prices_file(SAMPLE_PRICES)
    elif inputs.fund is not None:
        prices = load_prices(inputs.fund.price_symbol, use_cache=not args.no_cache)
    else:
        raise UsageError("No prices: pass --fund, --sample or --prices-file.")

    reference, reference_name = None, None
    if args.approach != "absolute":
        if args.reference_file:
            reference, reference_name = read_prices_file(args.reference_file), "reference portfolio"
        elif inputs.sample:
            reference, reference_name = read_prices_file(SAMPLE_REFERENCE), SAMPLE_REFERENCE_NAME
        elif inputs.fund is not None and inputs.fund.reference_symbol:
            reference = load_prices(inputs.fund.reference_symbol, use_cache=not args.no_cache)
            reference_name = inputs.fund.reference_name or inputs.fund.reference_symbol
        if args.approach == "relative" and reference is None:
            raise UsageError("The relative approach needs a reference: pass --reference-file or set reference_symbol.")
    return simple_returns(prices), (simple_returns(reference) if reference is not None else None), reference_name


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_limits(args, inputs):
    return run_limit_checks(_holdings(args, inputs), _regime(args, inputs), _counterparties(args, inputs), args.warn)


def _run_var(args, inputs, series=None):
    returns, reference, reference_name = series or _returns(args, inputs)
    return run_var_checks(
        inputs.label,
        returns,
        reference,
        approach=args.approach,
        method=args.method,
        window=args.window,
        warn_ratio=args.warn,
        reference_name=reference_name,
    )


def _finish(args, inputs, limit_result=None, var_result=None) -> int:
    paths = write_outputs(args.out, limit_result, var_result, title=inputs.title, sample=inputs.sample)
    print(f"\nReport:     {paths['report']}")
    print(f"Findings:   {paths['findings']}")
    print(f"Breach log: {paths['breach_log']}")
    statuses = [r.status for r in (limit_result, var_result) if r is not None]
    return EXIT_BREACH if args.fail_on_breach and "breach" in statuses else EXIT_OK


def cmd_limits(args) -> int:
    inputs = _resolve(args)
    result = _run_limits(args, inputs)
    _print_limits(result, inputs.title, args.all)
    return _finish(args, inputs, limit_result=result)


def cmd_var(args) -> int:
    inputs = _resolve(args)
    result = _run_var(args, inputs)
    _print_var(result, inputs.title, args.all)
    return _finish(args, inputs, var_result=result)


def cmd_report(args) -> int:
    inputs = _resolve(args)
    limits = _run_limits(args, inputs)
    _print_limits(limits, inputs.title, args.all)
    var = None
    try:
        series = _returns(args, inputs)
    except (UsageError, OSError, ValueError) as exc:  # no prices: still report the limits
        print(f"\nSkipped the VaR checks, no usable prices: {exc}", file=sys.stderr)
    else:
        var = _run_var(args, inputs, series)
        _print_var(var, inputs.title, args.all)
    return _finish(args, inputs, limits, var)


def cmd_compare(args) -> int:
    inputs = _resolve(args)
    holdings = _holdings(args, inputs)
    results = compare_regimes(holdings, args.regimes, args.warn)
    print(f"\n{inputs.title} · holdings as of {fmt_date(holdings.as_of)} · the same holdings under each rule set\n")
    print(f"  {'Rule set':<38} {'Status':<8} {'Largest issuer':<34} {'5/10/40':>8}")
    for r in results:
        largest = ""
        if not r.exposures.empty:
            largest = f"{r.exposures.index[0][:24]} {fmt_pct(float(r.exposures.iloc[0]))}"
        agg = r.by_rule("aggregate_5_10_40")
        agg_text = fmt_pct(agg[0].value) if agg else "n/a"
        print(f"  {r.regime.label:<38} {_status(r.status)}  {largest:<34} {agg_text:>8}")
    return EXIT_OK


def cmd_funds(args) -> int:
    funds = load_funds()
    if not funds:
        print("No funds configured in config/funds.json.")
    for f in funds.values():
        ref = f", reference {f.reference_symbol}" if f.reference_symbol else ""
        print(f"  {f.ticker:<6} {f.name}  [{f.regime}, prices {f.price_symbol}{ref}]")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucits-monitor",
        description="UCITS diversification and VaR global exposure checks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def target(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("what to check")
        g.add_argument("--fund", help="ticker from config/funds.json, e.g. CNDX")
        g.add_argument("--sample", action="store_true", help="use the synthetic sample files (works offline)")
        g.add_argument("--no-cache", action="store_true", help="download again even if today's file is cached")
        g.add_argument("--warn", default=0.9, type=float, help="warn at this share of a limit (default: 0.9)")

    def output(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("output")
        g.add_argument("--out", default="output", type=pathlib.Path, help="folder for reports (default: output)")
        g.add_argument("--all", action="store_true", help="print every finding, not only warnings and breaches")
        g.add_argument("--fail-on-breach", action="store_true", help=f"exit with code {EXIT_BREACH} if anything breaches")

    def holdings_options(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("holdings")
        g.add_argument("--regime", choices=list(REGIMES), help="rule set (default: the fund's, else standard)")
        g.add_argument("--holdings-file", type=pathlib.Path, help="an iShares-format holdings CSV on disk")
        g.add_argument("--counterparties", type=pathlib.Path, help="CSV of deposits and OTC exposures")

    def var_options(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("VaR")
        g.add_argument("--prices-file", type=pathlib.Path, help="a Date/Close CSV for the fund")
        g.add_argument("--reference-file", type=pathlib.Path, help="a Date/Close CSV for the reference portfolio")
        g.add_argument("--approach", choices=["absolute", "relative"], help="default: relative if a reference exists")
        g.add_argument("--method", choices=["historical", "parametric"], default="historical")
        g.add_argument("--window", type=int, default=250, help="days of history per VaR estimate (default: 250)")

    p = sub.add_parser("limits", help="diversification checks")
    target(p)
    holdings_options(p)
    output(p)
    p.set_defaults(func=cmd_limits)

    p = sub.add_parser("var", help="VaR global exposure and backtest")
    target(p)
    var_options(p)
    output(p)
    p.set_defaults(func=cmd_var)

    p = sub.add_parser("report", help="both, in one HTML report")
    target(p)
    holdings_options(p)
    var_options(p)
    output(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="the same holdings under every regime")
    target(p)
    p.add_argument("--holdings-file", type=pathlib.Path, help="an iShares-format holdings CSV on disk")
    p.add_argument("--regimes", nargs="+", choices=list(REGIMES), help="default: all")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("funds", help="list the configured funds")
    p.set_defaults(func=cmd_funds)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (UsageError, ValueError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # output piped into something like `head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK
    except OSError as exc:
        if type(exc).__module__.startswith("requests") or isinstance(exc, (ConnectionError, TimeoutError)):
            print(f"\nCould not fetch data: {exc}\nTry again later, or use --sample or local files.", file=sys.stderr)
        else:
            print(f"\nError: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
