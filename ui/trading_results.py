"""The Trading Compliance dashboard of a finished trading run, from ``views.json`` ``days[day]["dashboard"]``.

The data block is built by :mod:`reporting.trading_dashboard` (its docstring has
the schema and the definitions: TNA and current cash at the T-1 close, trade
values as in the trading report, post-trade cash as an estimate). This module
only formats it, top to bottom:

1. KPI strip: proposed trades (funds, buys, sells, net, post-trade cash) and the
   compliance outcome (staying compliant, breaking, curing, still failing).
2. Compliance impact matrix: funds with proposed trades × rules, each cell the
   before → after transition, with a legend and an "only rules that change or
   fail" toggle.
3. Rules at risk: per rule, the funds that break, still fail or are cured.
4. Trading summary table (one row per trading fund plus totals) with the cash
   columns and the footnoted definitions.
5. Three small-multiple charts (:mod:`ui.charts`).
6. The fund drill-down (:mod:`ui.trading_drilldown`).

The ``*_html`` builders are plain functions (unit-tested); every value is escaped.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import streamlit as st

from ui import charts
from ui import components as ui
from ui import results

# Transition -> (css class, glyph, text). OK stays quiet; BREAKS is the loud one.
TRANSITION_STYLE: Dict[str, Tuple[str, str, str]] = {
    "BREAKS": ("breaks", "✕", "BREAKS"),
    "FAILS": ("fails", "✕", "FAILS"),
    "STILL_FAILS": ("still", "!", "STILL FAILS"),
    "ERROR": ("error", "!", "ERROR"),
    "CURES": ("cures", "✓", "CURES"),
    "OK": ("ok", "✓", "PASS"),
    "N/A": ("na", "", "N/A"),
}
LEGEND_ORDER = ("BREAKS", "STILL_FAILS", "CURES", "OK", "FAILS", "ERROR", "N/A")
IMPACT_STYLE: Dict[str, Tuple[str, str]] = {
    "BREAKS": ("breaks", "Breaks"), "STILL_FAILS": ("still", "Still failing"), "ERROR": ("error", "Error"),
    "CURES": ("cures", "Cures"), "COMPLIANT": ("ok", "Compliant"), "UNKNOWN": ("na", "Unknown"),
}
ASSETS = charts.ASSETS
_SIDE_TEXT = {"BREAKS": "out of compliance", "CURES": "into compliance"}


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def transition_chip(code: Optional[str], title: str = "", *, quiet_ok: bool = True) -> str:
    """A before → after transition chip (``quiet_ok``: PASS → PASS is a faint tick, not a chip)."""
    code = code if code in TRANSITION_STYLE else "N/A"
    css, glyph, text = TRANSITION_STYLE[code]
    tip = f' title="{ui.esc(title)}"' if title else ""
    if code == "OK" and quiet_ok:
        return f'<span class="fm-tx ok quiet"{tip} aria-label="passes before and after">✓</span>'
    if code == "N/A" and quiet_ok:
        return f'<span class="fm-tx na quiet"{tip}>n/a</span>'
    mark = f'<b aria-hidden="true">{glyph}</b>' if glyph else ""
    return f'<span class="fm-tx {css}"{tip}>{mark}{ui.esc(text)}</span>'


def impact_chip(impact: Optional[str]) -> str:
    css, text = IMPACT_STYLE.get(str(impact), ("na", str(impact or "—")))
    return f'<span class="fm-tx {css} impact">{ui.esc(text)}</span>'


def _money_cell(value: object, *, signed: bool = False, sub: str = "", css: str = "") -> str:
    exact = ui.money(value, signed=signed, compact=False)
    text = ui.money(value, signed=signed)
    shown = ui.DASH if text == "—" else ui.esc(text)
    extra = f'<span class="sub">{ui.esc(sub)}</span>' if sub else ""
    return f'<td class="{ui.esc(("num " + css).strip())}" title="{ui.esc(exact)}">{shown}{extra}</td>'


def _num(value: object) -> Optional[float]:
    return charts._num(value)


def _asset(fund: Mapping, asset: str) -> Mapping:
    return (fund.get("assets") or {}).get(asset) or {}


# ---------------------------------------------------------------------------
# 1. KPI strip
# ---------------------------------------------------------------------------

def _tile(label: str, value: str, note: str = "", tone: str = "", glyph: str = "") -> str:
    icon = f'<span class="ic" aria-hidden="true">{ui.esc(glyph)}</span>' if glyph else ""
    sub = f'<div class="s">{ui.esc(note)}</div>' if note else ""
    return (f'<div class="fm-kpi {ui.esc(tone)}"><div class="l">{icon}{ui.esc(label)}</div>'
            f'<div class="v">{ui.esc(value)}</div>{sub}</div>')


def _by_class(kpis: Mapping, side: str) -> str:
    parts = [f"{charts.ASSET_LABELS[a].lower()} {ui.money(kpis.get(f'{a}_{side}_value'))}"
             for a in ASSETS if _num(kpis.get(f"{a}_{side}_value"))]
    return " · ".join(parts) or "none"


def kpi_html(dashboard: Mapping) -> str:
    """Two rows of tiles: the proposed trades, then the compliance outcome over the funds with trades."""
    k = dashboard.get("kpis") or {}
    totals = dashboard.get("totals") or {}
    traded, selected = k.get("funds_with_trades") or 0, k.get("funds_selected") or 0
    without = k.get("funds_without_trades") or 0
    net = _num(k.get("net_value")) or 0.0
    trades = [
        _tile("Funds with trades", str(traded),
              f"of {selected} selected" + (f" · {without} without trades" if without else "")),
        _tile("Proposed buys", ui.money(k.get("buy_value")), _by_class(k, "buy")),
        _tile("Proposed sells", ui.money(k.get("sell_value")), _by_class(k, "sell")),
        _tile("Net (buys − sells)", ui.money(net, signed=True),
              ("net buying" if net > 0 else "net selling" if net < 0 else "balanced")
              + f" · gross {ui.pct(totals.get('gross_pct_tna'))} of TNA"),
        _tile("Post-trade cash (est.)", ui.money(totals.get("post_cash")),
              f"from {ui.money(totals.get('cash'))} today · "
              + (f"{k.get('funds_negative_cash')} fund(s) below zero" if k.get("funds_negative_cash")
                 else "no fund below zero"),
              tone="warn" if k.get("funds_negative_cash") else ""),
    ]
    breaking, curing, still = (k.get("funds_breaking") or 0, k.get("funds_curing") or 0,
                               k.get("funds_still_failing") or 0)
    outcome = [
        _tile("Staying compliant", str(k.get("funds_compliant") or 0), "every rule passes before and after",
              tone="ok" if k.get("funds_compliant") else "", glyph="✓"),
        _tile("Going out of compliance", str(breaking),
              f"{k.get('rule_breaks') or 0} rule(s) pass before and fail after" if breaking else "no rule breaks",
              tone="breaks" if breaking else "", glyph="✕"),
        _tile("Coming into compliance", str(curing),
              f"{k.get('rule_cures') or 0} failing rule(s) cured" if curing else "no rule cured",
              tone="cures" if curing else "", glyph="✓"),
        _tile("Still failing", str(still),
              f"{k.get('rule_still_fails') or 0} rule(s) fail before and after" if still else "none",
              tone="still" if still else "", glyph="!"),
    ]
    if k.get("funds_error"):
        outcome.append(_tile("Check errors", str(k["funds_error"]), "a check raised an error", tone="error",
                             glyph="!"))
    return (f'<div class="fm-kpis"><div class="fm-kpi-group">Proposed trades</div><div class="fm-kpi-row">'
            f'{"".join(trades)}</div><div class="fm-kpi-group">Compliance after the trades</div>'
            f'<div class="fm-kpi-row">{"".join(outcome)}</div></div>')


# ---------------------------------------------------------------------------
# 2. Compliance impact matrix
# ---------------------------------------------------------------------------

def _rule_label(rule: Mapping) -> str:
    return results.SHORT_LABELS.get(str(rule.get("key")), str(rule.get("label") or rule.get("key") or ""))


def matrix_rules(dashboard: Mapping, only_changes: bool) -> List[Mapping]:
    rules = list((dashboard.get("matrix") or {}).get("rules") or [])
    return [r for r in rules if r.get("changes_or_fails")] if only_changes else rules


def matrix_html(dashboard: Mapping, only_changes: bool = True) -> str:
    """Funds with proposed trades (rows, breaks first) × rules: the before → after transition per cell."""
    matrix = dashboard.get("matrix") or {}
    rows = list(matrix.get("rows") or [])
    rules = matrix_rules(dashboard, only_changes)
    if not rows:
        return ui.empty("No proposed trades", "None of the selected funds has staged trades on this date.")
    if not rules:
        return ui.callout("ok", "Every rule passes (or is N/A) before and after the trades for every fund with "
                                "proposed trades. Turn off the toggle to see all the rules.")
    head = "".join(f'<th class="c" title="{ui.esc(r.get("title") or r.get("label"))}">{ui.esc(_rule_label(r))}</th>'
                   for r in rules)
    body = []
    for row in rows:
        fund = str(row.get("fund") or "")
        cells = []
        for rule in rules:
            cell = (row.get("cells") or {}).get(rule.get("key")) or {}
            code = cell.get("transition") or "N/A"
            tip = (f"{fund} · {rule.get('title') or rule.get('label')}: {cell.get('before') or '—'} before → "
                   f"{cell.get('after') or '—'} after")
            cells.append(f'<td class="c">{transition_chip(code, tip)}</td>')
        css = ' class="hl-bad"' if row.get("impact") == "BREAKS" else ""
        body.append(f'<tr{css}><td class="fund">{ui.esc(fund)}</td><td class="c">{impact_chip(row.get("impact"))}</td>'
                    f'{"".join(cells)}</tr>')
    table = (f'<div class="fm-tablewrap"><table class="fm-table fm-matrix fm-impact"><thead><tr><th>Fund</th>'
             f'<th class="c">Impact</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return ui.panel(table + legend_html(dashboard) + _others_note(dashboard))


def legend_html(dashboard: Mapping) -> str:
    meanings = ((dashboard.get("definitions") or {}).get("transitions") or {})
    items = []
    for code in LEGEND_ORDER:
        if code in ("FAILS", "ERROR") and not _seen(dashboard, code):
            continue
        items.append(f'<span>{transition_chip(code, quiet_ok=False)}<em>{ui.esc(meanings.get(code, ""))}</em></span>')
    return f'<div class="fm-txlegend">{"".join(items)}</div>'


def _seen(dashboard: Mapping, code: str) -> bool:
    return any(cell.get("transition") == code for row in (dashboard.get("matrix") or {}).get("rows") or []
               for cell in (row.get("cells") or {}).values())


def _others_note(dashboard: Mapping) -> str:
    notes = []
    for entry in dashboard.get("no_trade_funds") or []:
        text = str(entry.get("fund"))
        failed = entry.get("failed_after") or []
        if failed:
            text += f" (fails {', '.join(results.SHORT_LABELS.get(f, f) for f in failed)})"
        if _num(entry.get("report_trade_value")):
            text += f" — the report shows {ui.money(entry['report_trade_value'])} from its share columns"
        notes.append(text)
    out = ""
    if notes:
        out += f'<div class="fm-tablecap"><b>No proposed trades:</b> {ui.esc("; ".join(notes))}.</div>'
    missing = [str(e.get("fund")) for e in dashboard.get("unavailable_funds") or []]
    if missing:
        out += f'<div class="fm-tablecap"><b>No result:</b> {ui.esc(", ".join(missing))} (see the errors above).</div>'
    return out


# ---------------------------------------------------------------------------
# 3. Rules at risk
# ---------------------------------------------------------------------------

_RISK_GROUPS = (("breaks", "Breaks", "breaks"), ("fails", "Fails", "fails"), ("still_fails", "Still fails", "still"),
                ("errors", "Error", "error"), ("cures", "Cures", "cures"))


def rules_at_risk_html(dashboard: Mapping) -> str:
    """Per rule with any break, failure, error or cure after the trades: a count bar and the funds, grouped."""
    risks = [r for r in dashboard.get("rules_at_risk") or [] if any(r.get(g) for g, _, _ in _RISK_GROUPS)]
    total = (dashboard.get("kpis") or {}).get("funds_with_trades") or 0
    if not risks:
        return ui.callout("ok", "No rule fails, breaks or errors after the proposed trades.")
    rows = []
    for risk in risks:
        segments, groups = [], []
        for group, label, css in _RISK_GROUPS:
            funds = list(risk.get(group) or [])
            if not funds:
                continue
            if total:
                segments.append(f'<i class="{css}" style="width:{100 * len(funds) / total:.2f}%" '
                                f'title="{ui.esc(label)}: {len(funds)}"></i>')
            chips = "".join(f"<span>{ui.esc(f)}</span>" for f in funds)
            groups.append(f'<div class="grp {css}"><em>{transition_chip(_group_code(group), quiet_ok=False)}'
                          f"</em><div class=\"funds\">{chips}</div></div>")
        failing = len(set(risk.get("breaks") or []) | set(risk.get("still_fails") or []) | set(risk.get("fails") or []))
        count = f"{failing} of {total} fail after" if total else f"{failing} fail after"
        rows.append(f'<div class="fm-risk"><div class="head"><b>{ui.esc(risk.get("title") or risk.get("label"))}</b>'
                    f'<span>{ui.esc(count)}</span><div class="bar" role="img" aria-label="{ui.esc(count)}">'
                    f'{"".join(segments)}</div></div><div class="body">{"".join(groups)}</div></div>')
    return ui.panel(f'<div class="fm-risks">{"".join(rows)}</div>')


def _group_code(group: str) -> str:
    return {"breaks": "BREAKS", "fails": "FAILS", "still_fails": "STILL_FAILS", "errors": "ERROR",
            "cures": "CURES"}[group]


# ---------------------------------------------------------------------------
# 4. Trading summary table
# ---------------------------------------------------------------------------

def _has_class(dashboard: Mapping, asset: str) -> bool:
    totals = ((dashboard.get("totals") or {}).get("assets") or {}).get(asset) or {}
    return bool(_num(totals.get("gross_value")))


def _units(value: object, unit: str) -> str:
    number = _num(value)
    if not number:
        return ""
    word = {"contracts": "contracts", "face": "face"}.get(unit, "shares")
    return f"{ui.quantity(number)} {word}"


def _report_only(a: Mapping) -> bool:
    """The class has a trade value in the report (from the ex-post share columns) but no staged trade rows."""
    return not a.get("staged", True) and bool(_num(a.get("gross_value")))


def _class_cells(a: Mapping, asset: str, *, total: bool = False) -> List[str]:
    """Equity / options: buys, sells, net (options with contracts beneath). Treasury: one net cell."""
    unit = a.get("unit") or charts.ASSET_UNITS.get(asset, "shares")
    if not total and _report_only(a):
        tip = (f"The trading report shows {ui.money(a.get('gross_value'), compact=False)} from the ex-post share "
               "columns, but there are no staged trade rows; not in the cash estimate.")
        dash = f'<td class="num{{css}}" title="{ui.esc(tip)}">{ui.DASH}</td>'
        if asset == "treasury":
            return [dash.format(css=" grp")]
        return [dash.format(css=" grp"), dash.format(css=""),
                f'<td class="num" title="{ui.esc(tip)}">{ui.DASH}<span class="sub">report only</span></td>']
    if asset == "treasury":
        parts = [f"buys {ui.money(a.get('buy_value'))}" if _num(a.get("buy_value")) else "",
                 f"sells {ui.money(a.get('sell_value'))}" if _num(a.get("sell_value")) else ""]
        return [_money_cell(a.get("net_value"), signed=True, css="grp", sub=" · ".join(p for p in parts if p))]
    quantities = asset != "equity"
    net_sub = "" if total or not _num(a.get("net_pct_tna")) else ui.pct(a.get("net_pct_tna"), signed=True) + " TNA"
    return [_money_cell(a.get("buy_value"), css="grp", sub=_units(a.get("buy_quantity"), unit) if quantities else ""),
            _money_cell(a.get("sell_value"), sub=_units(a.get("sell_quantity"), unit) if quantities else ""),
            _money_cell(a.get("net_value"), signed=True, sub=net_sub)]


def _fund_row(fund: Mapping, classes: Sequence[str]) -> str:
    cash = fund.get("cash") or {}
    neg = bool(cash.get("negative"))
    post_sub = "" if cash.get("post") is None else f"{ui.pct(cash.get('post_pct_tna'))} TNA"
    cells = [f'<td class="fund">{ui.esc(fund.get("fund"))}{_flags(fund)}</td>', _money_cell(fund.get("tna")),
             _money_cell(cash.get("current"), css="grp"),
             _money_cell(cash.get("post"), sub=post_sub, css="neg" if neg else ""),
             f'<td class="num grp">{ui.esc(ui.pct(fund.get("gross_pct_tna")))}</td>']
    for asset in classes:
        cells += _class_cells(_asset(fund, asset), asset)
    css = ' class="hl-bad"' if fund.get("impact") == "BREAKS" else ""
    return f"<tr{css}>{''.join(cells)}</tr>"


def _flags(fund: Mapping) -> str:
    """Small badges after the fund name: ⚠ negative estimated cash; a count of data notes (hover lists them)."""
    cash = fund.get("cash") or {}
    out = []
    if cash.get("negative"):
        out.append('<span class="fm-ico bad" title="Estimated post-trade cash is below zero" '
                   'aria-label="negative cash">⚠</span>')
    notes = [str(w) for w in fund.get("warnings") or [] if "negative" not in str(w).lower()]
    if _num(cash.get("excluded_value")):
        notes.append(f"{ui.money(cash.get('excluded_value'), compact=False)} of trades left out of the cash estimate.")
    if notes:
        out.append(f'<span class="fm-ico warn" title="{ui.esc(" | ".join(notes))}" '
                   f'aria-label="{len(notes)} data notes">{len(notes)}</span>')
    return "".join(out)


def summary_table_html(dashboard: Mapping) -> str:
    """One row per fund with proposed trades, plus totals; amounts compact (hover for the exact figure)."""
    funds = list(dashboard.get("funds") or [])
    if not funds:
        return ""
    classes = [a for a in ASSETS if a == "equity" or _has_class(dashboard, a)]
    group = "".join(f'<th colspan="{1 if a == "treasury" else 3}" class="c grp">{ui.esc(charts.ASSET_LABELS[a])}'
                    f'{" premium" if a == "options" else ""}</th>' for a in classes)
    sub = "".join('<th class="num grp">Net</th>' if a == "treasury" else
                  '<th class="num grp">Buys</th><th class="num">Sells</th><th class="num">Net</th>' for a in classes)
    head = (f'<thead><tr class="top"><th rowspan="2">Fund</th><th rowspan="2" class="num">TNA</th>'
            f'<th colspan="2" class="c grp">Cash</th><th rowspan="2" class="num grp">Traded<br><span>% TNA</span></th>'
            f'{group}</tr><tr><th class="num grp">Current<br><span>T-1 close</span></th>'
            f'<th class="num">Post-trade<br><span>estimate</span></th>{sub}</tr></thead>')
    body = "".join(_fund_row(f, classes) for f in funds)
    body += _totals_row(dashboard, classes)
    table = f'<div class="fm-tablewrap"><table class="fm-table fm-summary">{head}<tbody>{body}</tbody></table></div>'
    return ui.panel(table + definitions_html(dashboard))


def _totals_row(dashboard: Mapping, classes: Sequence[str]) -> str:
    t = dashboard.get("totals") or {}
    cash_funds = t.get("cash_funds")
    negative = t.get("funds_negative_cash") or 0
    cells = [f'<td class="fund">Total<span class="sub">{ui.esc(t.get("funds") or 0)} funds</span></td>',
             _money_cell(t.get("tna")),
             _money_cell(t.get("cash"), css="grp", sub=f"{cash_funds} funds" if cash_funds else ""),
             _money_cell(t.get("post_cash"), sub=f"{negative} below 0" if negative else ""),
             f'<td class="num grp">{ui.esc(ui.pct(t.get("gross_pct_tna")))}</td>']
    for asset in classes:
        cells += _class_cells((t.get("assets") or {}).get(asset) or {}, asset, total=True)
    return f'<tr class="total">{"".join(cells)}</tr>'


def definitions_html(dashboard: Mapping) -> str:
    d = dashboard.get("definitions") or {}
    cash_date = dashboard.get("cash_date") or "T-1"
    notes = [("TNA", d.get("tna")), (f"Current cash ({cash_date})", d.get("current_cash")),
             ("Trade value", d.get("trade_value")), ("Net", d.get("net_value")),
             ("Post-trade cash (estimate)", d.get("post_cash"))]
    text = "".join(f"<p><b>{ui.esc(label)}.</b> {ui.esc(body)}</p>" for label, body in notes if body)
    text += ("<p>Amounts are compact ($M = millions); hover a figure for the exact amount. Option buys and sells are "
             "premium, with contracts beneath; treasuries are at price per 100 face. A dash with <i>report only</i>: "
             "the trading report shows a value from the ex-post share columns but there are no staged trades. "
             "After a fund's name, ⚠ marks negative estimated cash and an amber number counts its data notes "
             "(duplicate rows, trades left out of the estimate, missing cash); hover them for the details, or see "
             "the fund drill-down.</p>")
    return f'<details class="fm-foot fm-defs" open><summary class="t">Definitions</summary>{text}</details>'


# ---------------------------------------------------------------------------
# 5. Charts
# ---------------------------------------------------------------------------

def charts_html(dashboard: Mapping) -> str:
    funds = list(dashboard.get("funds") or [])
    if not funds:
        return ""
    figures = [charts.net_by_asset_chart(funds), charts.cash_dumbbell_chart(funds), charts.pct_tna_chart(funds)]
    return f'<div class="fm-charts-wrap"><div class="fm-charts">{"".join(f for f in figures if f)}</div></div>'


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def render_dashboard(view: Mapping, *, key: str, run: str, requested: Sequence[str] = ()) -> None:
    """The whole dashboard of one trading day (the downloads follow it in :func:`ui.panels.results_panel`)."""
    from ui import trading_drilldown  # noqa: PLC0415 (keeps this module importable on its own)

    dashboard = view.get("dashboard") or {}
    kpis = dashboard.get("kpis") or {}
    aside = (f"{dashboard.get('date') or view.get('date')} · cash and TNA at the {dashboard.get('cash_date') or 'T-1'} "
             f"close · {kpis.get('funds_with_trades', 0)} of {kpis.get('funds_selected', 0)} funds with proposed trades")
    ui.show(ui.heading("Trading compliance", aside) + kpi_html(dashboard))
    errors = _fund_errors(view)
    if errors:
        ui.show(ui.callout("error", f"{len(errors)} fund error{'s' if len(errors) != 1 else ''} on this day:", errors))
    if not dashboard.get("funds"):
        ui.show(ui.empty("No proposed trades", "None of the selected funds has staged trades on this date, so there "
                                               "is nothing to compare before and after.") + _others_note(dashboard))
        return
    head, toggle = st.columns([2.3, 1.5], vertical_alignment="bottom")
    head.html(ui.heading("Compliance impact", "funds with proposed trades × rules · before → after"))
    only_key = f"{key}-tc-only-{run}"
    st.session_state.setdefault(only_key, True)
    with toggle:
        only = st.toggle("Only changes and failures", key=only_key,
                         help="Show only the rules that change or fail for at least one fund")
    ui.show(matrix_html(dashboard, only))
    ui.show(ui.heading("Rules at risk", "the funds failing each rule after the trades") + rules_at_risk_html(dashboard))
    ui.show(ui.heading("Trading summary", "proposed trades and cash per fund") + summary_table_html(dashboard))
    ui.show(ui.heading("Trades at a glance", "same fund order in each chart; the table above has the numbers"))
    ui.show_svg(charts_html(dashboard))
    trading_drilldown.render(view, dashboard, key=key, run=run)


def _fund_errors(view: Mapping) -> List[str]:
    """One line per fund error of the day (already redacted by :func:`ui.history.load_views`)."""
    out = []
    for fund, messages in (view.get("fund_errors") or {}).items():
        for message in messages or []:
            out.append(f"{fund}: {message}")
    return out
