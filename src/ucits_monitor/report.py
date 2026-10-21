"""Output: a findings CSV, a running breach log, and a one-page HTML report.

The HTML report is a single file with no external resources, so it can be
emailed or archived as is. Charts are inline SVG.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import pathlib

import pandas as pd

from .engine import LimitResult, VarResult, status_counts, worst_status
from .models import BACKTEST_NOTIFY_ABOVE, Finding

RULE_LABELS = {
    "single_issuer": "Single issuer",
    "aggregate_5_10_40": "5/10/40 rule",
    "deposits": "Deposits",
    "otc_counterparty": "OTC counterparty",
    "combined": "Combined exposure",
    "absolute_var": "Absolute VaR",
    "relative_var": "Relative VaR",
    "backtest_overshootings": "VaR backtest",
}
RATIO_RULES = {"relative_var"}
COUNT_RULES = {"backtest_overshootings"}
STATUS_LABELS = {"ok": "OK", "warning": "Warning", "breach": "Breach"}
ZONE_STATUS = {"green": "ok", "yellow": "warning", "red": "breach"}
SEVERITY = {"breach": 0, "warning": 1, "ok": 2}

COLUMNS = ["date", "fund", "regime", "rule", "subject", "value", "limit", "utilization", "headroom", "status", "note"]
LOG_KEY = ["date", "fund", "regime", "rule", "subject"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _minus(text: str) -> str:
    return text.replace("-", "−")


def fmt_pct(value: float, digits: int = 2) -> str:
    return _minus(f"{value:.{digits}%}")


def fmt_value(rule: str, value: float) -> str:
    if rule in RATIO_RULES:
        return _minus(f"{value:.2f}x")
    if rule in COUNT_RULES:
        return _minus(f"{value:.0f}")
    return fmt_pct(value)


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def fmt_p(p: float) -> str:
    """A p-value, without pretending to precision below 0.001."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_date(day: dt.date | None) -> str:
    return f"{day:%b} {day.day}, {day.year}" if day else "date unknown"


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------


def findings_frame(findings: list[Finding], *, date: dt.date | None, fund: str, regime: str) -> pd.DataFrame:
    rows = [
        {
            "date": date.isoformat() if date else "",
            "fund": fund,
            "regime": regime,
            "rule": f.rule,
            "subject": f.subject,
            "value": f.value,
            "limit": f.limit,
            "utilization": f.utilization,
            "headroom": f.headroom,
            "status": f.status,
            "note": f.note,
        }
        for f in findings
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def result_frame(limit_result: LimitResult | None = None, var_result: VarResult | None = None) -> pd.DataFrame:
    """Every finding from one run, one row each."""
    frames = []
    if limit_result is not None:
        frames.append(
            findings_frame(limit_result.findings, date=limit_result.as_of, fund=limit_result.fund, regime=limit_result.regime.name)
        )
    if var_result is not None:
        frames.append(findings_frame(var_result.findings, date=var_result.as_of, fund=var_result.fund, regime="global_exposure"))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def update_breach_log(frame: pd.DataFrame, path: pathlib.Path) -> pd.DataFrame:
    """Add this run's warnings and breaches to a running log.

    Running the same day twice does not create duplicates: one row per
    date, fund, regime, rule and subject, keeping the latest.
    """
    path = pathlib.Path(path)
    new = frame.loc[frame["status"] != "ok"]
    parts = []
    if path.exists():
        parts.append(pd.read_csv(path, dtype={"date": str}, keep_default_na=False))
    if not new.empty:
        parts.append(new)
    parts = [p for p in parts if not p.empty]
    log = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    log = log.drop_duplicates(subset=LOG_KEY, keep="last")
    log = log.sort_values(["date", "fund", "rule", "subject"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


# ---------------------------------------------------------------------------
# Icons and small pieces
# ---------------------------------------------------------------------------


def icon(status: str, size: int = 14, x: float | None = None, y: float | None = None) -> str:
    """A status icon. Its shape carries the status as well as its color, so color is never the only cue."""
    pos = f' x="{x:.1f}" y="{y:.1f}"' if x is not None else ""
    if status == "ok":
        glyph = '<circle cx="8" cy="8" r="7.5" fill="currentColor"/><path d="M4.6 8.3l2.2 2.2 4.6-4.8" fill="none" style="stroke:var(--surface)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    elif status == "warning":
        glyph = '<path d="M8 .9l7.3 13.4H.7z" fill="currentColor" stroke-linejoin="round"/><rect x="7.1" y="5.2" width="1.8" height="5" rx=".9" style="fill:var(--surface)"/><circle cx="8" cy="12.2" r="1" style="fill:var(--surface)"/>'
    else:
        glyph = '<path d="M5.1.6h5.8l4.5 4.5v5.8l-4.5 4.5H5.1L.6 10.9V5.1z" fill="currentColor"/><path d="M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2" style="stroke:var(--surface)" stroke-width="1.8" stroke-linecap="round"/>'
    return f'<svg class="icon icon-{status}"{pos} width="{size}" height="{size}" viewBox="0 0 16 16" aria-hidden="true">{glyph}</svg>'


def chip(status: str, label: str | None = None) -> str:
    return f'<span class="chip">{icon(status)}<span>{_esc(label or STATUS_LABELS[status])}</span></span>'


def meter(utilization: float, status: str, caption: str) -> str:
    fill = max(0.0, min(utilization, 1.0)) * 100
    return (
        f'<div class="meter" role="img" aria-label="{_esc(caption)}">'
        f'<div class="meter-track"><div class="meter-fill meter-{status}" style="width:{fill:.1f}%"></div></div>'
        f'<p class="meter-caption">{_esc(caption)}</p></div>'
    )


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude


def _ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    step = _nice_step(hi - lo, target)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


def _tick_label(v: float) -> str:
    if abs(v) < 1e-12:
        return "0%"
    digits = 0 if abs(v * 100 - round(v * 100)) < 1e-9 else 1
    return fmt_pct(v, digits)


def _bar_path(x0: float, y: float, length: float, height: float, radius: float = 4.0) -> str:
    """A horizontal bar, square at the baseline and rounded at the data end."""
    r = min(radius, height / 2, max(length, 0) / 2)
    x1 = x0 + max(length, 0)
    return (
        f"M{x0:.1f},{y:.1f}H{x1 - r:.1f}A{r:.1f},{r:.1f} 0 0 1 {x1:.1f},{y + r:.1f}"
        f"V{y + height - r:.1f}A{r:.1f},{r:.1f} 0 0 1 {x1 - r:.1f},{y + height:.1f}H{x0:.1f}Z"
    )


def _short(name: str, limit: int = 30) -> str:
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def issuer_chart(result: LimitResult, top: int = 15) -> str:
    """Largest issuers against the single-issuer limit. Under the standard
    regime, the issuers that count toward the 40% test are highlighted."""
    regime = result.regime
    singles = {f.subject: f for f in result.by_rule("single_issuer")}
    exposures = result.exposures.head(top)
    if exposures.empty:
        return '<p class="empty">No holdings to chart.</p>'

    width, label_w, value_w, row_h, bar_h, top_pad, axis_h = 720, 196, 92, 26, 14, 34, 26
    x0, x1 = label_w + 10, width - value_w
    limits_shown = [regime.single_issuer_limit]
    exceptional = next((f for f in singles.values() if f.limit != regime.single_issuer_limit), None)
    if exceptional is not None:
        limits_shown.append(exceptional.limit)
    xmax = max(max(limits_shown), float(exposures.max())) * 1.08
    scale = (x1 - x0) / xmax
    height = top_pad + row_h * len(exposures) + axis_h
    plot_bottom = top_pad + row_h * len(exposures)
    threshold = regime.aggregate_threshold

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Largest {len(exposures)} issuers by weight against the {fmt_pct(regime.single_issuer_limit, 0)} limit">'
    ]
    for t in _ticks(0, xmax):
        x = x0 + t * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top_pad - 6}" y2="{plot_bottom}"/>')
        parts.append(f'<text class="axis-text" x="{x:.1f}" y="{plot_bottom + 17}" text-anchor="middle">{_tick_label(t)}</text>')

    for i, (issuer, weight) in enumerate(exposures.items()):
        y = top_pad + i * row_h + (row_h - bar_h) / 2
        finding = singles.get(issuer)
        status = finding.status if finding else "ok"
        limit = finding.limit if finding else regime.single_issuer_limit
        counted = threshold is not None and weight > threshold + 1e-9
        cls = "bar-main" if (threshold is None or counted) else "bar-muted"
        length = weight * scale
        tip = json.dumps([[fmt_pct(weight), issuer], [fmt_pct(limit), "limit"], [STATUS_LABELS[status], "status"]])
        parts.append(f'<g class="hit" tabindex="0" data-tip="{_esc(tip)}">')
        parts.append(f'<rect class="hit-area" x="0" y="{top_pad + i * row_h}" width="{width}" height="{row_h}"/>')
        parts.append(
            f'<text class="label" x="{label_w}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{_esc(_short(issuer))}</text>'
        )
        parts.append(f'<path class="{cls}" d="{_bar_path(x0, y, length, bar_h)}"/>')
        tx = x0 + length + 6
        parts.append(f'<text class="value" x="{tx:.1f}" y="{y + bar_h / 2 + 4:.1f}">{fmt_pct(weight)}</text>')
        if status != "ok":
            parts.append(icon(status, 13, tx + 46, y + (bar_h - 13) / 2))
        parts.append("</g>")

    def vline(value: float, cls: str, label: str, anchor: str = "middle") -> None:
        x = x0 + value * scale
        parts.append(f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{top_pad - 8}" y2="{plot_bottom}"/>')
        parts.append(f'<text class="ref-text" x="{x:.1f}" y="{top_pad - 14}" text-anchor="{anchor}">{_esc(label)}</text>')

    if threshold is not None:
        vline(threshold, "threshold-line", f"{fmt_pct(threshold, 0)} threshold")
    vline(regime.single_issuer_limit, "limit-line", f"{fmt_pct(regime.single_issuer_limit, 0)} limit")
    if exceptional is not None:
        vline(exceptional.limit, "limit-line", f"{fmt_pct(exceptional.limit, 0)} for one issuer", "end")
    parts.append(f'<line class="baseline" x1="{x0}" x2="{x0}" y1="{top_pad - 6}" y2="{plot_bottom}"/>')
    parts.append("</svg>")

    legend = ""
    if threshold is not None:
        legend = (
            '<div class="legend">'
            '<span><i class="key key-main"></i>Above 5%, counts toward the 40% test</span>'
            '<span><i class="key key-muted"></i>5% or less</span>'
            '<span><i class="key-line key-limit"></i>Limit</span>'
            "</div>"
        )
    return f'<div class="chart">{legend}<div class="chart-scroll">{"".join(parts)}</div><div class="tip" hidden></div></div>'


def backtest_chart(result: VarResult) -> str:
    """Each day's return against that morning's one-day VaR, exceptions marked."""
    bt = result.backtest
    returns = bt.returns.to_numpy(dtype=float)
    forecasts = bt.forecasts.to_numpy(dtype=float)
    hits = bt.hits.to_numpy(dtype=int)
    dates = [d.date() if hasattr(d, "date") else d for d in bt.returns.index]
    n = len(returns)
    if n == 0:
        return '<p class="empty">No backtest data.</p>'

    width, height, left, right, top, bottom = 720, 280, 56, 14, 14, 30
    pw, ph = width - left - right, height - top - bottom
    lo = min(float(returns.min()), float(-forecasts.max()))
    hi = max(float(returns.max()), 0.0)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    step = pw / n

    def X(i: float) -> float:
        return left + (i + 0.5) * step

    def Y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * ph

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Daily returns against one-day VaR over {n} days, {bt.exceptions} exceptions">'
    ]
    for t in _ticks(lo, hi):
        y = Y(t)
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis-text" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_tick_label(t)}</text>')

    month = None
    for i, day in enumerate(dates):
        if day.month != month:
            if month is not None and X(i) < width - right - 20:
                label = f"{day:%b}" if day.month != 1 else f"{day:%b} {day.year}"
                parts.append(f'<line class="tick" x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{top + ph}" y2="{top + ph + 4}"/>')
                parts.append(f'<text class="axis-text" x="{X(i):.1f}" y="{top + ph + 18}" text-anchor="middle">{label}</text>')
            month = day.month

    bar_w = max(step - 1.0, 0.8)
    zero = Y(0.0)
    for i, r in enumerate(returns):
        y, h = (Y(r), zero - Y(r)) if r >= 0 else (zero, Y(r) - zero)
        cls = "col-hit" if hits[i] else "col"
        parts.append(f'<rect class="{cls}" x="{X(i) - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h, 0.5):.2f}"/>')
    parts.append(f'<line class="baseline" x1="{left}" x2="{width - right}" y1="{zero:.1f}" y2="{zero:.1f}"/>')

    path = "M" + "L".join(f"{X(i):.1f},{Y(-v):.1f}" for i, v in enumerate(forecasts))
    parts.append(f'<path class="var-line" d="{path}"/>')
    for i in range(n):
        if hits[i]:
            parts.append(f'<circle class="dot-hit" cx="{X(i):.1f}" cy="{Y(returns[i]):.1f}" r="4.5"/>')

    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{top}" y2="{top + ph}" visibility="hidden"/>')
    parts.append(
        f'<rect class="overlay" x="{left}" y="{top}" width="{pw}" height="{ph}" tabindex="0" '
        f'aria-label="Use the arrow keys to step through the days"/>'
    )
    parts.append("</svg>")

    data = {
        "left": left,
        "step": step,
        "dates": [fmt_date(d) for d in dates],
        "returns": [fmt_pct(r) for r in returns],
        "vars": [fmt_pct(v) for v in forecasts],
        "hits": [int(h) for h in hits],
    }
    payload = json.dumps(data).replace("</", "<\\/")
    legend = (
        '<div class="legend">'
        '<span><i class="key key-col"></i>Daily return</span>'
        f'<span><i class="key-line key-var"></i>One-day VaR, {result.confidence:.0%} (shown as a loss)</span>'
        f'<span><i class="key-dot"></i>Exception ({bt.exceptions})</span>'
        "</div>"
    )
    return (
        f'<div class="chart" data-backtest>{legend}<div class="chart-scroll">{"".join(parts)}</div>'
        f'<script type="application/json" class="chart-data">{payload}</script><div class="tip" hidden></div></div>'
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _findings_table(findings: list[Finding], empty: str) -> str:
    if not findings:
        return f'<p class="empty">{_esc(empty)}</p>'
    rows = sorted(findings, key=lambda f: (SEVERITY[f.status], -f.utilization if f.limit else 0, f.rule, f.subject))
    body = "".join(
        "<tr>"
        f"<td>{chip(f.status)}</td>"
        f"<td>{_esc(RULE_LABELS.get(f.rule, f.rule))}</td>"
        f"<td>{_esc(f.subject)}</td>"
        f'<td class="num">{fmt_value(f.rule, f.value)}</td>'
        f'<td class="num">{fmt_value(f.rule, f.limit)}</td>'
        f'<td class="num">{fmt_value(f.rule, f.headroom)}</td>'
        f"<td>{_esc(f.note)}</td>"
        "</tr>"
        for f in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Status</th><th>Rule</th><th>Subject</th>'
        '<th class="num">Value</th><th class="num">Limit</th><th class="num">Headroom</th><th>Note</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _issuer_table(result: LimitResult) -> str:
    singles = {f.subject: f for f in result.by_rule("single_issuer")}
    threshold = result.regime.aggregate_threshold
    rows = []
    for issuer, weight in result.exposures.items():
        f = singles.get(issuer)
        counted = "Yes" if threshold is not None and weight > threshold + 1e-9 else ""
        rows.append(
            "<tr>"
            f"<td>{_esc(issuer)}</td>"
            f'<td class="num">{fmt_pct(weight)}</td>'
            f'<td class="num">{fmt_pct(f.limit) if f else ""}</td>'
            f'<td class="num">{fmt_pct(f.headroom) if f else ""}</td>'
            + (f"<td>{counted}</td>" if threshold is not None else "")
            + f"<td>{chip(f.status) if f else ''}</td>"
            "</tr>"
        )
    counted_head = "<th>In 40% test</th>" if threshold is not None else ""
    return (
        f'<details><summary>All {len(rows)} issuers as a table</summary><div class="table-wrap"><table>'
        '<thead><tr><th>Issuer</th><th class="num">Weight</th><th class="num">Limit</th>'
        f'<th class="num">Headroom</th>{counted_head}<th>Status</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></details>"
    )


def _excess(value: float) -> str:
    """Loss beyond VaR, without rounding a real exception down to 0.00%."""
    return fmt_pct(value) if value >= 0.00005 else "< 0.01%"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<tr><th scope="row">{_esc(k)}</th><td class="num">{v}</td></tr>' for k, v in rows)
    return f'<div class="table-wrap"><table class="kv"><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--page:#f5f3ed;--surface:#fcfbf8;--ink:#0b1f33;--ink-2:#4a5563;--muted:#6b7280;
--grid:#e6e3da;--axis:#c9c5b9;--rule:#c5a363;--main:#00897b;--deemph:#cfcbbf;--critical:#a1352a;
--ok:#1c6350;--warning:#8a6414;--breach:#a1352a;--track:#ebe8df;--border:rgba(11,31,51,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#111417;--surface:#1b1e22;
--ink:#eef0f2;--ink-2:#b9c0c8;--muted:#8a929b;--grid:#2b3036;--axis:#3a4048;--main:#22a094;--deemph:#4a5058;
--critical:#e06a5c;--ok:#3fae8a;--warning:#e0b04a;--breach:#e06a5c;--track:#2b3036;--border:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#111417;--surface:#1b1e22;--ink:#eef0f2;--ink-2:#b9c0c8;--muted:#8a929b;
--grid:#2b3036;--axis:#3a4048;--main:#22a094;--deemph:#4a5058;--critical:#e06a5c;--ok:#3fae8a;--warning:#e0b04a;
--breach:#e06a5c;--track:#2b3036;--border:rgba(255,255,255,.10)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 48px}
.kicker{margin:0;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2)}
h1{margin:4px 0 6px;font-size:28px;line-height:1.2}
h2{margin:0 0 4px;font-size:19px}
.meta{margin:0;color:var(--ink-2);font-size:14px}
header{padding-bottom:18px;border-bottom:2px solid var(--rule);margin-bottom:20px}
.banner{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--rule);padding:10px 14px;border-radius:6px;margin:0 0 20px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:24px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-label{margin:0 0 6px;font-size:13px;color:var(--ink-2)}
.tile-value{margin:0 0 8px;font-size:26px;font-weight:600;line-height:1.15}
.tile-sub{margin:6px 0 0;font-size:13px;color:var(--ink-2)}
section.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 18px 14px;margin-bottom:16px}
.section-intro{margin:0 0 12px;color:var(--ink-2);font-size:14px}
.chip{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;font-size:13px;font-weight:600}
.icon{flex:none}.icon-ok{color:var(--ok)}.icon-warning{color:var(--warning)}.icon-breach{color:var(--breach)}
.meter-track{height:8px;background:var(--track);border-radius:4px;overflow:hidden}
.meter-fill{height:100%;border-radius:4px}.meter-ok{background:var(--main)}.meter-warning{background:var(--warning)}.meter-breach{background:var(--critical)}
.meter-caption{margin:6px 0 0;font-size:13px;color:var(--ink-2)}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
thead th{font-size:12px;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis)}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.kv th{font-weight:400;color:var(--ink-2);width:60%}
details{margin-top:10px}summary{cursor:pointer;color:var(--ink-2);font-size:14px}
.empty{color:var(--ink-2);font-size:14px}
.chart{position:relative;margin:4px 0 8px}
.chart-scroll{overflow-x:auto}
.chart-svg{display:block;width:100%;min-width:620px;height:auto;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key{display:inline-block;width:12px;height:12px;border-radius:3px}
.key-main{background:var(--main)}.key-muted{background:var(--deemph)}.key-col{background:var(--deemph)}
.key-line{display:inline-block;width:16px;height:0;border-top:2px solid var(--main)}
.key-limit{border-top:2px dashed var(--critical)}
.key-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--critical);box-shadow:0 0 0 2px var(--surface)}
.grid{stroke:var(--grid);stroke-width:1}
.baseline,.tick{stroke:var(--axis);stroke-width:1}
.axis-text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.label{fill:var(--ink);font-size:12px}
.value{fill:var(--ink-2);font-size:11.5px;font-variant-numeric:tabular-nums}
.ref-text{fill:var(--ink-2);font-size:11px}
.bar-main{fill:var(--main)}.bar-muted{fill:var(--deemph)}
.limit-line{stroke:var(--critical);stroke-width:1.5;stroke-dasharray:5 3}
.threshold-line{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3}
.hit-area{fill:transparent}.hit{outline:none}
.hit:hover .hit-area,.hit:focus .hit-area{fill:var(--grid);opacity:.55}
.col{fill:var(--deemph)}.col-hit{fill:var(--critical)}
.var-line{fill:none;stroke:var(--main);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot-hit{fill:var(--critical);stroke:var(--surface);stroke-width:2}
.xhair{stroke:var(--ink-2);stroke-width:1}
.overlay{fill:transparent;outline:none}
.overlay:focus{stroke:var(--axis);stroke-width:1}
.tip{position:absolute;z-index:2;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;box-shadow:0 4px 14px rgba(0,0,0,.12);font-size:13px;min-width:150px}
.tip-row{display:flex;gap:8px;align-items:baseline}
.tip-row strong{font-variant-numeric:tabular-nums}.tip-row span{color:var(--ink-2)}
footer{color:var(--ink-2);font-size:13px;margin-top:22px}
footer p{margin:0 0 8px}
@media print{body{background:#fff}.tip{display:none}section.card,.tile{break-inside:avoid}}
@media (max-width:560px){h1{font-size:23px}.tile-value{font-size:22px}th,td{padding:6px 7px}}
"""

JS = """
(function(){
  function showTip(chart, rows, px, py){
    var tip = chart.querySelector('.tip');
    tip.replaceChildren();
    rows.forEach(function(r){
      var row = document.createElement('div'); row.className = 'tip-row';
      var v = document.createElement('strong'); v.textContent = r[0];
      var l = document.createElement('span'); l.textContent = r[1];
      row.append(v, l); tip.append(row);
    });
    tip.hidden = false;
    var box = chart.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    var x = Math.min(Math.max(px + 14, 0), box.width - w), y = Math.max(py - h - 12, 0);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip(chart){ chart.querySelector('.tip').hidden = true; }
  document.querySelectorAll('.chart').forEach(function(chart){
    chart.querySelectorAll('.hit').forEach(function(g){
      var rows = JSON.parse(g.getAttribute('data-tip'));
      function at(e){ var b = chart.getBoundingClientRect(); showTip(chart, rows, e.clientX - b.left, e.clientY - b.top); }
      g.addEventListener('pointermove', at);
      g.addEventListener('pointerleave', function(){ hideTip(chart); });
      g.addEventListener('focus', function(){
        var b = chart.getBoundingClientRect(), r = g.getBoundingClientRect();
        showTip(chart, rows, r.left - b.left + r.width * 0.45, r.top - b.top + 4);
      });
      g.addEventListener('blur', function(){ hideTip(chart); });
    });
    if (!chart.hasAttribute('data-backtest')) return;
    var data = JSON.parse(chart.querySelector('.chart-data').textContent);
    var svg = chart.querySelector('svg'), overlay = chart.querySelector('.overlay'), xhair = chart.querySelector('.xhair');
    var n = data.dates.length, current = n - 1;
    function place(i){
      current = Math.max(0, Math.min(n - 1, i));
      var x = data.left + (current + 0.5) * data.step;
      xhair.setAttribute('x1', x); xhair.setAttribute('x2', x); xhair.setAttribute('visibility', 'visible');
      var rows = [[data.returns[current], 'return, ' + data.dates[current]], [data.vars[current], 'VaR that morning']];
      if (data.hits[current]) rows.push(['Exception', 'the loss was bigger than VaR']);
      var pt = svg.createSVGPoint(); pt.x = x; pt.y = 20;
      var s = pt.matrixTransform(svg.getScreenCTM()), b = chart.getBoundingClientRect();
      showTip(chart, rows, s.x - b.left, s.y - b.top + 70);
    }
    overlay.addEventListener('pointermove', function(e){
      var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      var p = pt.matrixTransform(svg.getScreenCTM().inverse());
      place(Math.floor((p.x - data.left) / data.step));
    });
    overlay.addEventListener('pointerleave', function(){ xhair.setAttribute('visibility', 'hidden'); hideTip(chart); });
    overlay.addEventListener('focus', function(){ place(current); });
    overlay.addEventListener('blur', function(){ xhair.setAttribute('visibility', 'hidden'); hideTip(chart); });
    overlay.addEventListener('keydown', function(e){
      if (e.key === 'ArrowLeft') { place(current - 1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { place(current + 1); e.preventDefault(); }
    });
  });
})();
"""


def _limit_tiles(result: LimitResult) -> list[str]:
    counts = result.counts
    tiles = [
        '<div class="tile"><p class="tile-label">Diversification</p>'
        f'<p class="tile-value">{chip(result.status)}</p>'
        f'<p class="tile-sub">{plural(counts["breach"], "breach", "breaches")}, '
        f'{plural(counts["warning"], "warning", "warnings")}, {counts["ok"]} within limits</p></div>'
    ]
    agg = result.by_rule("aggregate_5_10_40")
    if agg:
        f = agg[0]
        tiles.append(
            '<div class="tile"><p class="tile-label">5/10/40 rule</p>'
            f'<p class="tile-value">{fmt_pct(f.value, 1)}</p>'
            + meter(f.utilization, f.status, f"of a 40% cap, {f.note}")
            + "</div>"
        )
    elif not result.exposures.empty:
        name, weight = result.exposures.index[0], float(result.exposures.iloc[0])
        f = next(x for x in result.by_rule("single_issuer") if x.subject == name)
        tiles.append(
            '<div class="tile"><p class="tile-label">Largest issuer</p>'
            f'<p class="tile-value">{fmt_pct(weight, 1)}</p>'
            + meter(f.utilization, f.status, f"{_short(name, 26)}, limit {fmt_pct(f.limit, 0)}")
            + "</div>"
        )
    return tiles


def _var_tiles(result: VarResult) -> list[str]:
    f, bt = result.finding, result.backtest
    if result.approach == "relative":
        exposure = (
            '<div class="tile"><p class="tile-label">Global exposure, relative VaR</p>'
            f'<p class="tile-value">{fmt_value(f.rule, f.value)}</p>'
            + meter(f.utilization, f.status, "times the reference portfolio's VaR, cap 2.00x")
            + "</div>"
        )
    else:
        exposure = (
            f'<div class="tile"><p class="tile-label">Global exposure, {result.horizon}-day VaR</p>'
            f'<p class="tile-value">{fmt_pct(f.value)}</p>'
            + meter(f.utilization, f.status, "of NAV, cap 20%")
            + "</div>"
        )
    backtest = (
        '<div class="tile"><p class="tile-label">Backtest, last 250 days</p>'
        f'<p class="tile-value">{bt.exceptions} exceptions</p>'
        f'<p class="tile-sub">{chip(ZONE_STATUS[bt.zone], bt.zone.capitalize() + " zone")}</p>'
        f'<p class="tile-sub">About {bt.expected:.1f} expected at {result.confidence:.0%}</p></div>'
    )
    return [exposure, backtest]


def _limit_sections(result: LimitResult) -> str:
    regime = result.regime
    intro = (
        f"Weights are summed by issuer, so share classes count together. "
        f"Limit per issuer: {fmt_pct(regime.single_issuer_limit, 0)}."
    )
    if regime.aggregate_limit is not None:
        intro += " Issuers above 5% may not add up to more than 40%."
    if regime.exceptional_limit is not None:
        intro += f" One issuer may go up to {fmt_pct(regime.exceptional_limit, 0)}."
    excluded = ""
    if not result.excluded.empty:
        listed = ", ".join(f"{_esc(k)} {fmt_pct(v)}" for k, v in result.excluded.items())
        excluded = f'<p class="section-intro">Not counted toward issuer limits: {listed}.</p>'
    html_parts = [
        '<section class="card"><h2>Issuer concentration</h2>'
        f'<p class="section-intro">{intro}</p>{issuer_chart(result)}{excluded}{_issuer_table(result)}</section>'
    ]
    if result.has_counterparties:
        rows = [f for f in result.findings if f.rule in ("deposits", "otc_counterparty", "combined")]
        html_parts.append(
            '<section class="card"><h2>Deposits and OTC counterparties</h2>'
            '<p class="section-intro">Deposits: 20% per bank. OTC derivatives: 10% per counterparty for credit '
            "institutions, 5% for others. Securities, deposits and OTC exposure to one body: 20% combined.</p>"
            f'{_findings_table(rows, "No deposits or OTC exposures supplied.")}</section>'
        )
    return "".join(html_parts)


def _var_sections(result: VarResult) -> str:
    bt, f = result.backtest, result.finding
    rows = [
        (f"One-day VaR, {result.method} ({result.window} days)", fmt_pct(result.var_1d)),
        (f"{result.horizon}-day VaR, square root of time", fmt_pct(result.var_horizon)),
        (f"{result.horizon}-day VaR, {result.alt_method}, for comparison", fmt_pct(result.alt_var_horizon)),
    ]
    if result.approach == "relative":
        name = result.reference_name or "reference portfolio"
        rows += [
            (f"{result.horizon}-day VaR of the {name}", fmt_pct(result.reference_var_horizon)),
            ("Ratio, fund to reference (cap 2.00x)", f"{fmt_value(f.rule, f.value)} {chip(f.status)}"),
        ]
    else:
        rows.append(("Limit, share of NAV", f"{fmt_pct(f.limit)} {chip(f.status)}"))
    exposure = (
        '<section class="card"><h2>Global exposure</h2>'
        f'<p class="section-intro">{result.confidence:.0%} one-tailed VaR from the last {result.window} daily returns, '
        f"scaled to {result.horizon} business days. "
        + (
            "Relative approach: the fund's VaR may not exceed twice the VaR of an unleveraged reference portfolio."
            if result.approach == "relative"
            else "Absolute approach: VaR may not exceed 20% of NAV."
        )
        + f"</p>{_kv_table(rows)}</section>"
    )
    stats = [
        ("Exceptions (loss bigger than that morning's VaR)", f"{bt.exceptions} of {bt.observations}"),
        ("Expected at this confidence", f"{bt.expected:.1f}"),
        ("Kupiec test, right number of exceptions (p-value)", fmt_p(bt.kupiec[1])),
        ("Christoffersen test, no clustering (p-value)", fmt_p(bt.independence[1])),
        ("Conditional coverage, both together (p-value)", fmt_p(bt.conditional[1])),
        ("Basel traffic light", chip(ZONE_STATUS[bt.zone], bt.zone.capitalize())),
        (f"Report to the regulator (more than {BACKTEST_NOTIFY_ABOVE})", "Yes" if bt.notify else "No"),
    ]
    exception_rows = "".join(
        f'<tr><td>{fmt_date(d.date() if hasattr(d, "date") else d)}</td><td class="num">{fmt_pct(r)}</td>'
        f'<td class="num">{fmt_pct(v)}</td><td class="num">{_excess(-r - v)}</td></tr>'
        for d, r, v, h in zip(bt.returns.index, bt.returns, bt.forecasts, bt.hits)
        if h
    )
    exceptions = (
        f'<details><summary>The {bt.exceptions} exceptions as a table</summary><div class="table-wrap"><table>'
        '<thead><tr><th>Date</th><th class="num">Return</th><th class="num">VaR</th><th class="num">Loss beyond VaR</th></tr></thead>'
        f"<tbody>{exception_rows}</tbody></table></div></details>"
        if bt.exceptions
        else ""
    )
    backtest = (
        '<section class="card"><h2>Backtest</h2>'
        f'<p class="section-intro">Each morning\'s one-day VaR, built only from the {result.window} days before it, '
        "against that day's return. A p-value below 0.05 rejects the model at the 5% level.</p>"
        f"{backtest_chart(result)}{_kv_table(stats)}{exceptions}</section>"
    )
    return exposure + backtest


def render_html(
    limit_result: LimitResult | None = None,
    var_result: VarResult | None = None,
    *,
    title: str | None = None,
    sample: bool = False,
    run_at: dt.datetime | None = None,
) -> str:
    if limit_result is None and var_result is None:
        raise ValueError("nothing to report")
    fund = (limit_result or var_result).fund
    title = title or fund
    run_at = run_at or dt.datetime.now()

    meta = []
    if limit_result is not None:
        meta.append(f"Holdings as of {fmt_date(limit_result.as_of)}")
        meta.append(f"{limit_result.regime.label} ({limit_result.regime.source.replace('UCITS Directive ', '')})")
    if var_result is not None:
        meta.append(f"Prices to {fmt_date(var_result.as_of)}")
    meta.append(f"Run on {fmt_date(run_at.date())} at {run_at:%H:%M}")

    all_findings = (limit_result.findings if limit_result else []) + (var_result.findings if var_result else [])
    issues = [f for f in all_findings if f.status != "ok"]
    tiles = (_limit_tiles(limit_result) if limit_result else []) + (_var_tiles(var_result) if var_result else [])
    banner = (
        '<p class="banner"><strong>Synthetic sample data.</strong> The fund, its holdings, prices and '
        "counterparties are made up for testing. They describe no real fund.</p>"
        if sample
        else ""
    )
    counts = status_counts(all_findings)
    summary_status = worst_status(all_findings)
    page_title = f"{title}: UCITS limit check, {STATUS_LABELS[summary_status].lower()}"

    body = (
        f"<main><header><p class=\"kicker\">UCITS limit monitor</p><h1>{_esc(title)}</h1>"
        f'<p class="meta">{_esc(" · ".join(meta))}</p></header>{banner}'
        f'<div class="tiles">{"".join(tiles)}</div>'
        '<section class="card"><h2>Issues to look at</h2>'
        f'<p class="section-intro">{plural(counts["breach"], "breach", "breaches")} and '
        f'{plural(counts["warning"], "warning", "warnings")}. '
        f"A warning means a value has reached {(limit_result or var_result).warn_ratio:.0%} of its limit.</p>"
        f'{_findings_table(issues, "Nothing above the warning level.")}</section>'
        + (_limit_sections(limit_result) if limit_result else "")
        + (_var_sections(var_result) if var_result else "")
        + "<footer>"
        "<p>Diversification rules: UCITS Directive 2009/65/EC, Articles 52 and 53. Global exposure and backtesting: "
        "CESR's Guidelines on Risk Measurement and the Calculation of Global Exposure and Counterparty Risk "
        "for UCITS (CESR/10-788).</p>"
        "<p>Cash, FX and futures lines are not counted toward issuer limits in this version. National rules and a "
        "fund's own prospectus can be stricter. This is a portfolio project, not compliance software.</p>"
        "</footer></main>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(page_title)}</title><style>{CSS}</style></head>"
        f"<body>{body}<script>{JS}</script></body></html>"
    )


def write_outputs(
    out_dir: pathlib.Path | str,
    limit_result: LimitResult | None = None,
    var_result: VarResult | None = None,
    *,
    title: str | None = None,
    sample: bool = False,
) -> dict[str, pathlib.Path]:
    """Write the findings CSV, update the breach log and write the HTML report."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = limit_result or var_result
    day = (limit_result.as_of if limit_result else None) or (var_result.as_of if var_result else None)
    stem = f"{first.fund}_{day.isoformat() if day else 'undated'}"

    frame = result_frame(limit_result, var_result)
    findings_path = out / f"{stem}_findings.csv"
    frame.to_csv(findings_path, index=False)
    log_path = out / "breach_log.csv"
    update_breach_log(frame, log_path)
    report_path = out / f"{stem}_report.html"
    report_path.write_text(render_html(limit_result, var_result, title=title, sample=sample), encoding="utf-8")
    return {"findings": findings_path, "breach_log": log_path, "report": report_path}
