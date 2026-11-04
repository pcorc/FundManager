"""The Trading Compliance dashboard of a finished trading run, from ``views.json`` ``days[day]["dashboard"]``.

The data block is built by :mod:`reporting.trading_dashboard` (its docstring has
the schema and the definitions: TNA and current cash at the T-1 close, trade
values as in the trading report, post-trade cash as an estimate). This module
only formats it:

* Above the tabs: the heading with what ex-ante / ex-post mean, the KPI strip
  (proposed trades on the cash estimate's "priced" basis, post-trade cash, and
  the compliance outcome: staying compliant, breaking, deepening a breach,
  curing, still failing) and one data-quality note for the whole page.
* Tab "Compliance impact": the matrix (funds with proposed trades × rules, each
  cell the before → after transition; a failing rule's trend: worse / better /
  moves / unchanged) and Rules at risk (per rule, the funds that break, worsen,
  still fail or are cured). A click on a matrix row or cell, or on a fund in
  Rules at risk, opens that fund (and rule) in the drill-down tab.
* Tab "Trading summary": the per-fund table plus totals, its definitions, and
  three small-multiple charts (:mod:`ui.charts`).
* Tab "Fund drill-down" (:mod:`ui.trading_drilldown`).

The jump from a matrix cell to the drill-down is a hidden ``st.pills`` (the
real widget, with a callback that sets the drill-down's keys and the active
tab) that a small static script clicks for the HTML cells; the drill-down's
own pickers are the keyboard / no-script path. The ``*_html`` builders are
plain functions (unit-tested); every value is escaped.
"""
from __future__ import annotations

import re
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
# A still-failing rule's trend -> (css class, text after "STILL FAILS").
TREND_STYLE: Dict[str, Tuple[str, str]] = {"worse": ("worse", "worse"), "better": ("better", "better"),
                                           "moves": ("moves", "moves"), "same": ("same", "")}
LEGEND_ORDER = ("BREAKS", "STILL_FAILS", "CURES", "OK", "FAILS", "ERROR", "N/A")
IMPACT_STYLE: Dict[str, Tuple[str, str]] = {
    "BREAKS": ("breaks", "Breaks"), "STILL_FAILS": ("still", "Still failing"), "ERROR": ("error", "Error"),
    "CURES": ("cures", "Cures"), "COMPLIANT": ("ok", "Compliant"), "UNKNOWN": ("na", "Unknown"),
}
ASSETS = charts.ASSETS
TABS = ("Compliance impact", "Trading summary", "Fund drill-down")
DRILL_TAB = TABS[2]
BEFORE_AFTER = ("Before = ex-ante: the holdings as they stand · After = ex-post: the same holdings assuming the "
                "staged but unexecuted trades execute")
_MISSING_TIP = ("No custodian {what} for the T-1 close: figures are not estimated, and rule results may rest on "
                "missing data")

# A static script (no data in it): a click (or Enter) on an element with data-fm-jump clicks the matching option of
# the hidden jump pills, whose callback opens the drill-down tab on that fund and rule.
JUMP_SCRIPT = """<script>
(function () {
  if (window.fmJumpReady) { return; }
  window.fmJumpReady = true;
  function go(el) {
    var box = document.querySelector('[class*="st-key-tcjump"]');
    if (!box) { return; }
    var fund = el.getAttribute('data-fm-jump'), rule = el.getAttribute('data-fm-rule');
    var want = rule ? fund + ' #' + rule : fund, hit = null, plain = null;
    box.querySelectorAll('button').forEach(function (b) {
      var text = (b.innerText || b.textContent || '').trim();
      if (text === want) { hit = b; }
      if (text === fund) { plain = b; }
    });
    var target = hit || plain;
    if (!target) { return; }
    target.click();
    var tabs = el.closest('[data-testid="stTabs"]') || document.querySelector('[data-testid="stTabs"]');
    if (tabs) { tabs.scrollIntoView({block: 'start', behavior: 'smooth'}); }
  }
  function find(e) { return e.target && e.target.closest ? e.target.closest('[data-fm-jump]') : null; }
  document.addEventListener('click', function (e) { var el = find(e); if (el) { e.preventDefault(); go(el); } });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') { return; }
    var el = find(e);
    if (el) { e.preventDefault(); go(el); }
  });
})();
</script>"""


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def transition_chip(code: Optional[str], title: str = "", *, quiet_ok: bool = True, trend: Optional[str] = None) -> str:
    """A before → after transition chip (``quiet_ok``: PASS → PASS is a faint tick, not a chip).

    ``trend`` (STILL_FAILS only) adds "· worse" / "· better" / "· moves" inside the chip.
    """
    code = code if code in TRANSITION_STYLE else "N/A"
    css, glyph, text = TRANSITION_STYLE[code]
    tip = f' title="{ui.esc(title)}"' if title else ""
    if code == "OK" and quiet_ok:
        return f'<span class="fm-tx ok quiet"{tip} aria-label="passes before and after">✓</span>'
    if code == "N/A" and quiet_ok:
        return f'<span class="fm-tx na quiet"{tip}>n/a</span>'
    mark = f'<b aria-hidden="true">{glyph}</b>' if glyph else ""
    extra = ""
    if code == "STILL_FAILS" and trend in TREND_STYLE:
        trend_css, word = TREND_STYLE[trend]
        css += f" {trend_css}"
        extra = f'<i class="tr">· {ui.esc(word)}</i>' if word else ""
    return f'<span class="fm-tx {css}"{tip}>{mark}{ui.esc(text)}{extra}</span>'


def impact_chip(impact: Optional[str], worsens: bool = False) -> str:
    css, text = IMPACT_STYLE.get(str(impact), ("na", str(impact or "—")))
    if impact == "STILL_FAILS" and worsens:
        css, text = "still worse", "Worsens"
    return f'<span class="fm-tx {css} impact">{ui.esc(text)}</span>'


def _money_cell(value: object, *, signed: bool = False, sub: str = "", css: str = "", title: str = "") -> str:
    exact = title or ui.money(value, signed=signed, compact=False)
    text = ui.money(value, signed=signed)
    shown = ui.DASH if text == "—" else ui.esc(text)
    extra = f'<span class="sub">{ui.esc(sub)}</span>' if sub else ""
    return f'<td class="{ui.esc(("num " + css).strip())}" title="{ui.esc(exact)}">{shown}{extra}</td>'


def _num(value: object) -> Optional[float]:
    return charts._num(value)


def _asset(fund: Mapping, asset: str) -> Mapping:
    return (fund.get("assets") or {}).get(asset) or {}


def rule_name(rule: Mapping) -> str:
    """The one display name of a rule on this tab: the settings tile's (catalog) label, e.g. 'GICS Concentration'."""
    return str(rule.get("label") or rule.get("title") or rule.get("key") or rule.get("rule") or "")


def missing_icon(fund: Mapping) -> str:
    """A small ∅ badge when the fund's custodian TNA or cash is missing (its results may rest on missing data)."""
    missing = list(fund.get("data_missing") or [])
    if not missing:
        return ""
    what = " and ".join({"tna": "TNA", "cash": "cash"}.get(m, m) for m in missing)
    return (f'<span class="fm-ico na" title="{ui.esc(_MISSING_TIP.format(what=what))}" '
            f'aria-label="missing data">∅</span>')


# ---------------------------------------------------------------------------
# 1. KPI strip and data notes
# ---------------------------------------------------------------------------

def _tile(label: str, value: str, note: str = "", tone: str = "", glyph: str = "") -> str:
    icon = f'<span class="ic" aria-hidden="true">{ui.esc(glyph)}</span>' if glyph else ""
    sub = f'<div class="s">{ui.esc(note)}</div>' if note else ""
    return (f'<div class="fm-kpi {ui.esc(tone)}"><div class="l">{icon}{ui.esc(label)}</div>'
            f'<div class="v">{ui.esc(value)}</div>{sub}</div>')


def _priced(k: Mapping, field: str) -> object:
    """A priced KPI figure (schema 2); the report's figure for an older run."""
    value = k.get(f"priced_{field}")
    return k.get(field) if value is None else value


def _by_class(k: Mapping, side: str) -> str:
    parts = []
    for a in ASSETS:
        value = k.get(f"{a}_priced_{side}_value", k.get(f"{a}_{side}_value"))
        if _num(value):
            parts.append(f"{charts.ASSET_LABELS[a].lower()} {ui.money(value)}")
    return " · ".join(parts) or "none"


def _report_note(k: Mapping, field: str) -> str:
    """'report $301.9M' when the report's figure differs from the priced one shown."""
    report, priced = _num(k.get(field)), _num(k.get(f"priced_{field}"))
    if report is None or priced is None or abs(report - priced) < 0.5:
        return ""
    return f" · report {ui.money(report)} incl. flagged rows"


def kpi_html(dashboard: Mapping) -> str:
    """Two rows of five tiles: the proposed trades (priced basis), then the compliance outcome."""
    k = dashboard.get("kpis") or {}
    totals = dashboard.get("totals") or {}
    cash_date = dashboard.get("cash_date") or "T-1"
    traded, selected = k.get("funds_with_trades") or 0, k.get("funds_selected") or 0
    without = k.get("funds_without_trades") or 0
    net = _num(_priced(k, "net_value")) or 0.0
    gross_pct = (totals.get("priced") or {}).get("gross_pct_tna", totals.get("gross_pct_tna"))
    net_note = "net buying" if net > 0 else "net selling" if net < 0 else "balanced"
    if _num(gross_pct) is not None:
        net_note += f" · gross {ui.pct(gross_pct)} of TNA"
    negative, going = k.get("funds_negative_cash") or 0, k.get("funds_going_negative")
    before = k.get("funds_negative_before")
    cash_note = f"from {ui.money(totals.get('cash'))} at the {cash_date} close" if totals.get("cash") is not None \
        else "no custodian cash loaded"
    if negative and going is not None:
        parts = [f"{going} go below zero" if going else "", f"{before} already negative" if before else ""]
        cash_note += " · " + " · ".join(p for p in parts if p)
    elif negative:
        cash_note += f" · {negative} fund(s) below zero"
    elif totals.get("cash") is not None:
        cash_note += " · no fund below zero"
    trades = [
        _tile("Funds with trades", str(traded),
              f"of {selected} selected" + (f" · {without} without trades" if without else "")),
        _tile("Proposed buys", ui.money(_priced(k, "buy_value")), _by_class(k, "buy") + _report_note(k, "buy_value")),
        _tile("Proposed sells", ui.money(_priced(k, "sell_value")),
              _by_class(k, "sell") + _report_note(k, "sell_value")),
        _tile("Net (buys − sells)", ui.money(net, signed=True), net_note + _report_note(k, "net_value")),
        _tile("Post-trade cash (est.)", ui.money(totals.get("post_cash")), cash_note,
              tone="neg" if negative else "", glyph="⚠" if negative else ""),
    ]
    breaking, curing, still = (k.get("funds_breaking") or 0, k.get("funds_curing") or 0,
                               k.get("funds_still_failing") or 0)
    worsening = k.get("funds_worsening") or 0
    outcome = [
        _tile("Staying compliant", str(k.get("funds_compliant") or 0),
              "every rule passes before and after" if k.get("funds_compliant") else "none",
              tone="ok" if k.get("funds_compliant") else "", glyph="✓"),
        _tile("Going out of compliance", str(breaking),
              f"{k.get('rule_breaks') or 0} rule(s) pass before and fail after" if breaking else "no rule breaks",
              tone="breaks" if breaking else "", glyph="✕"),
        _tile("Breach deepens", str(worsening),
              f"{k.get('rule_worsens') or 0} failing rule(s) get worse" if worsening
              else "no failing rule gets worse", tone="worse" if worsening else "", glyph="▲"),
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
    return (f'<div class="fm-kpis"><div class="fm-kpi-group">Proposed trades · priced, each trade once</div>'
            f'<div class="fm-kpi-row">{"".join(trades)}</div><div class="fm-kpi-group">Compliance after the trades'
            f'</div><div class="fm-kpi-row">{"".join(outcome)}</div></div>')


def data_notes_html(dashboard: Mapping) -> str:
    """One page-level note on the data behind the figures ('' when there is nothing to say)."""
    k = dashboard.get("kpis") or {}
    notes = k.get("data_notes") or {}
    if not notes:
        return ""
    parts = []
    dup = notes.get("duplicate") or {}
    if dup.get("count"):
        parts.append(f"{dup['count']} duplicate trade row(s) worth {ui.money(dup.get('value'))} in "
                     f"{len(dup.get('funds') or [])} fund(s), counted once")
    restated = notes.get("restates_position") or {}
    if restated.get("count"):
        parts.append(f"{restated['count']} staged quantit{'y' if restated['count'] == 1 else 'ies'} that only restate "
                     f"the position held ({ui.money(restated.get('value'))}: "
                     f"{', '.join(restated.get('funds') or [])}), likely not trades")
    unpriced = notes.get("no_price") or {}
    if unpriced.get("count"):
        parts.append(f"{unpriced['count']} trade(s) without a usable price (the report values them at quantity × "
                     "multiplier)")
    unstaged = notes.get("unstaged") or {}
    if unstaged.get("funds"):
        parts.append(f"{ui.money(unstaged.get('value'))} of report trade value without staged rows in "
                     f"{len(unstaged['funds'])} fund(s) (ex-post position changes, mostly options; not in the "
                     "cash estimate)")
    if notes.get("data_missing"):
        parts.append(f"no custodian TNA or cash for {', '.join(notes['data_missing'])} (marked ∅)")
    if notes.get("flex"):
        parts.append(f"staged FLEX option rows the analysis does not value: {', '.join(notes['flex'])}")
    if not parts:
        return ""
    funds = k.get("funds_with_warnings") or 0
    head = (f"Data notes on {funds} of {k.get('funds_with_trades') or 0} funds with trades. The tiles, charts and "
            "post-trade cash leave flagged trades out; the summary table's buy / sell columns are the report's. "
            "Each fund's notes are in its drill-down.")
    return f'<div class="fm-dq">{ui.callout("warn", head, parts)}</div>'


# ---------------------------------------------------------------------------
# 2. Compliance impact matrix
# ---------------------------------------------------------------------------

def _rule_short(rule: Mapping) -> str:
    return results.SHORT_LABELS.get(str(rule.get("key")), rule_name(rule))


def matrix_rules(dashboard: Mapping, only_changes: bool) -> List[Mapping]:
    rules = list((dashboard.get("matrix") or {}).get("rules") or [])
    return [r for r in rules if r.get("changes_or_fails")] if only_changes else rules


def _cell_tip(fund: str, rule: Mapping, cell: Mapping) -> str:
    tip = f"{fund} · {rule_name(rule)}: {cell.get('before') or '—'} before → {cell.get('after') or '—'} after"
    key = cell.get("key") or {}
    if key and key.get("before") != key.get("after"):
        tip += f" · {key.get('label')} {key.get('before') or '—'} → {key.get('after') or '—'}"
    changed = cell.get("changed")
    if changed:
        tip += f" · {changed} figure{'s' if changed != 1 else ''} change"
    elif changed == 0:
        tip += " · no figure changes"
    trend = cell.get("trend")
    if trend in ("worse", "better"):
        tip += f" · the trades make it {trend}"
    return tip + " · click to open it in the drill-down"


def matrix_html(dashboard: Mapping, only_changes: bool = True) -> str:
    """Funds with proposed trades (rows, breaks first) × rules: the before → after transition per cell."""
    matrix = dashboard.get("matrix") or {}
    rows = list(matrix.get("rows") or [])
    rules = matrix_rules(dashboard, only_changes)
    all_rules = list(matrix.get("rules") or [])
    if not rows:
        return ui.empty("No proposed trades", "None of the selected funds has staged trades on this date.")
    if not rules:
        return ui.callout("ok", "Every rule passes (or is N/A) before and after the trades for every fund with "
                                "proposed trades. Turn off the toggle to see all the rules.")
    funds = {str(f.get("fund")): f for f in dashboard.get("funds") or []}
    head = "".join(f'<th class="c" title="{ui.esc(rule_name(r))}">{ui.esc(_rule_short(r))}</th>' for r in rules)
    body = []
    for row in rows:
        fund = str(row.get("fund") or "")
        info = funds.get(fund) or {}
        cells = []
        for rule in rules:
            cell = (row.get("cells") or {}).get(rule.get("key")) or {}
            code = cell.get("transition") or "N/A"
            index = all_rules.index(rule)
            cells.append(f'<td class="c jump" data-fm-jump="{ui.esc(fund)}" data-fm-rule="{index}" tabindex="0" '
                         f'role="button">{transition_chip(code, _cell_tip(fund, rule, cell), trend=cell.get("trend"))}'
                         f"</td>")
        css = ' class="hl-bad"' if row.get("impact") == "BREAKS" else ""
        body.append(f'<tr{css}><td class="fund jump" data-fm-jump="{ui.esc(fund)}" tabindex="0" role="button" '
                    f'title="Open {ui.esc(fund)} in the drill-down">{ui.esc(fund)}{missing_icon(info)}</td>'
                    f'<td class="c">{impact_chip(row.get("impact"), bool(info.get("worsens")))}</td>'
                    f'{"".join(cells)}</tr>')
    table = (f'<div class="fm-tablewrap"><table class="fm-table fm-matrix fm-impact"><thead><tr><th>Fund</th>'
             f'<th class="c">Impact</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return ui.panel(table + legend_html(dashboard) + _others_note(dashboard))


def legend_html(dashboard: Mapping) -> str:
    meanings = ((dashboard.get("definitions") or {}).get("transitions") or {})
    trends = ((dashboard.get("definitions") or {}).get("trends") or {})
    items = []
    for code in LEGEND_ORDER:
        if code in ("FAILS", "ERROR") and not _seen(dashboard, code):
            continue
        items.append(f'<span>{transition_chip(code, quiet_ok=False)}<em>{ui.esc(meanings.get(code, ""))}</em></span>')
        if code == "STILL_FAILS":
            for trend in ("worse", "better"):
                if _seen_trend(dashboard, trend):
                    items.append(f'<span>{transition_chip(code, quiet_ok=False, trend=trend)}'
                                 f'<em>{ui.esc(trends.get(trend, ""))}</em></span>')
    hint = ('<span class="hint">Click a fund or a cell to open it in the drill-down; hover a cell for the key figure '
            'before → after.</span>')
    return f'<div class="fm-txlegend">{"".join(items)}{hint}</div>'


def _cells(dashboard: Mapping):
    for row in (dashboard.get("matrix") or {}).get("rows") or []:
        yield from (row.get("cells") or {}).values()


def _seen(dashboard: Mapping, code: str) -> bool:
    return any(cell.get("transition") == code for cell in _cells(dashboard))


def _seen_trend(dashboard: Mapping, trend: str) -> bool:
    return any(cell.get("trend") == trend for cell in _cells(dashboard))


def _no_trade_rows(dashboard: Mapping) -> List[List[object]]:
    rules = {str(r.get("key")): r for r in (dashboard.get("matrix") or {}).get("rules") or []}
    out = []
    for entry in dashboard.get("no_trade_funds") or []:
        failed = [rule_name(rules[f]) if f in rules else results.SHORT_LABELS.get(f, f)
                  for f in entry.get("failed_after") or []]
        report = (f"{ui.money(entry['report_trade_value'])} from the ex-post share columns"
                  if _num(entry.get("report_trade_value")) else "")
        out.append([entry.get("fund"), ui.pill(entry.get("status_after"), small=True), ", ".join(failed) or "—",
                    report or "—"])
    return out


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


def no_trades_html(dashboard: Mapping) -> str:
    """The empty state: no selected fund has staged trades; a small table of those funds' results instead."""
    body = ui.empty("No proposed trades", "None of the selected funds has staged trades on this date, so there "
                                          "is nothing to compare before and after.")
    rows = _no_trade_rows(dashboard)
    if rows:
        table = ui.data_table(["Fund", "Rules (ex-post)", "Failing", "Report trade value"], rows, raw=(1,))
        body += ui.heading("Selected funds", "compliance of the holdings as they stand (no trades to apply)")
        body += ui.panel(table)
    missing = [str(e.get("fund")) for e in dashboard.get("unavailable_funds") or []]
    if missing:
        body += ui.callout("error", f"No result for {', '.join(missing)} (see the errors above).")
    return body


# ---------------------------------------------------------------------------
# 3. Rules at risk
# ---------------------------------------------------------------------------

_RISK_GROUPS = (("breaks", "Breaks", "breaks"), ("fails", "Fails", "fails"), ("still_fails", "Still fails", "still"),
                ("errors", "Error", "error"), ("cures", "Cures", "cures"))


def rules_at_risk_html(dashboard: Mapping) -> str:
    """Per rule with any break, failure, error or cure after the trades: a count bar and the funds, grouped.

    Each fund is a link-like chip (``data-fm-jump``) that opens that fund and rule in the drill-down.
    """
    risks = [r for r in dashboard.get("rules_at_risk") or [] if any(r.get(g) for g, _, _ in _RISK_GROUPS)]
    total = (dashboard.get("kpis") or {}).get("funds_with_trades") or 0
    if not risks:
        return ui.callout("ok", "No rule fails, breaks or errors after the proposed trades.")
    keys = [str(r.get("key")) for r in (dashboard.get("matrix") or {}).get("rules") or []]
    missing = {str(f.get("fund")) for f in dashboard.get("funds") or [] if f.get("data_missing")}
    rows = []
    for risk in risks:
        index = keys.index(str(risk.get("rule"))) if str(risk.get("rule")) in keys else None
        rule_attr = f' data-fm-rule="{index}"' if index is not None else ""
        trends = risk.get("trends") or {}
        segments, groups = [], []
        for group, label, css in _RISK_GROUPS:
            funds = list(risk.get(group) or [])
            if not funds:
                continue
            if group == "still_fails":
                worse = [f for f in funds if trends.get(f) == "worse"]
                if worse and total:
                    segments.append(f'<i class="worse" style="width:{100 * len(worse) / total:.2f}%" '
                                    f'title="Still fails, worse: {len(worse)}"></i>')
                rest = len(funds) - len(worse)
                if rest and total:
                    segments.append(f'<i class="still" style="width:{100 * rest / total:.2f}%" '
                                    f'title="{ui.esc(label)}: {rest}"></i>')
            elif total:
                segments.append(f'<i class="{css}" style="width:{100 * len(funds) / total:.2f}%" '
                                f'title="{ui.esc(label)}: {len(funds)}"></i>')
            chips = []
            for f in funds:
                trend = trends.get(f) if group == "still_fails" else None
                cls = ' class="worse"' if trend == "worse" else ""
                mark = '<b aria-hidden="true">▲</b>' if trend == "worse" else ""
                gap = (' <i class="miss" aria-label="missing data" title="No custodian TNA or cash for the T-1 '
                       'close: this result may rest on missing data">∅</i>') if f in missing else ""
                tip = f"Open {f} · {rule_name(risk)} in the drill-down" + (" (the trades make it worse)"
                                                                            if trend == "worse" else "")
                chips.append(f'<span{cls} data-fm-jump="{ui.esc(f)}"{rule_attr} tabindex="0" role="button" '
                             f'title="{ui.esc(tip)}">{mark}{ui.esc(f)}{gap}</span>')
            groups.append(f'<div class="grp {css}"><em>{transition_chip(_group_code(group), quiet_ok=False)}'
                          f'</em><div class="funds">{"".join(chips)}</div></div>')
        failing = len(set(risk.get("breaks") or []) | set(risk.get("still_fails") or []) | set(risk.get("fails") or []))
        count = f"{failing} of {total} fail after" if total else f"{failing} fail after"
        worse_count = len(risk.get("worsens") or [])
        if worse_count:
            count += f" · {worse_count} worse"
        rows.append(f'<div class="fm-risk"><div class="head"><b>{ui.esc(rule_name(risk))}</b>'
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


_REPORT_ONLY_TIP = ("The trading report shows {value} for this class from the ex-post share columns (nav -> iiv "
                    "shares), but there are no staged trade rows; it is not in the cash estimate. The fund's "
                    "drill-down lists these position changes.")


def _side_sub(a: Mapping, side: str, unit: str, quantities: bool) -> str:
    unpriced = _num(a.get(f"unpriced_{side}_value"))
    value = _num(a.get(f"{side}_value")) or 0.0
    if unpriced and abs(unpriced - value) < 0.5:
        return "unpriced"
    if unpriced:
        return f"incl. {ui.money(unpriced)} unpriced"
    return _units(a.get(f"{side}_quantity"), unit) if quantities else ""


def _class_cells(a: Mapping, asset: str, *, total: bool = False) -> List[str]:
    """Equity / options: buys, sells, net (options with contracts beneath). Treasury: one net cell."""
    unit = a.get("unit") or charts.ASSET_UNITS.get(asset, "shares")
    if not total and _report_only(a):
        tip = _REPORT_ONLY_TIP.format(value=ui.money(a.get("gross_value"), compact=False))
        value = (f'<td class="num muted{{css}}" title="{ui.esc(tip)}">{ui.esc(ui.money(a.get("gross_value")))}'
                 f'<span class="sub">in report · not staged</span></td>')
        dash = f'<td class="num" title="{ui.esc(tip)}">{ui.DASH}</td>'
        if asset == "treasury":
            return [value.format(css=" grp")]
        return [value.format(css=" grp"), dash, dash]
    if asset == "treasury":
        parts = [f"buys {ui.money(a.get('buy_value'))}" if _num(a.get("buy_value")) else "",
                 f"sells {ui.money(a.get('sell_value'))}" if _num(a.get("sell_value")) else ""]
        return [_money_cell(a.get("net_value"), signed=True, css="grp", sub=" · ".join(p for p in parts if p))]
    quantities = asset != "equity"
    net_sub = "" if total or not _num(a.get("net_pct_tna")) else ui.pct(a.get("net_pct_tna"), signed=True) + " TNA"
    if total and _num(a.get("unstaged_value")):
        net_sub = f"+ {ui.money(a.get('unstaged_value'))} not staged"
    return [_money_cell(a.get("buy_value"), css="grp", sub=_side_sub(a, "buy", unit, quantities)),
            _money_cell(a.get("sell_value"), sub=_side_sub(a, "sell", unit, quantities)),
            _money_cell(a.get("net_value"), signed=True, sub=net_sub)]


def _fund_row(fund: Mapping, classes: Sequence[str]) -> str:
    cash = fund.get("cash") or {}
    neg = bool(cash.get("negative"))
    post_sub = "" if cash.get("post") is None or _num(cash.get("post_pct_tna")) is None \
        else f"{ui.pct(cash.get('post_pct_tna'))} TNA"
    tna = _num(fund.get("tna")) or None
    tna_cell = (_money_cell(tna) if tna else
                f'<td class="num muted" title="{ui.esc(_MISSING_TIP.format(what="TNA"))}">no data</td>')
    current_sub = "" if cash.get("custodian_t") is None else f"T {ui.money(cash.get('custodian_t'))}"
    cells = [f'<td class="fund">{ui.esc(fund.get("fund"))}{_flags(fund)}</td>', tna_cell,
             _money_cell(cash.get("current"), css="grp", sub=current_sub),
             _money_cell(cash.get("post"), sub=post_sub, css="neg" if neg else ""),
             f'<td class="num grp">{ui.esc(ui.pct(fund.get("gross_pct_tna")) if tna else "—")}</td>']
    for asset in classes:
        cells += _class_cells(_asset(fund, asset), asset)
    css = ' class="hl-bad"' if fund.get("impact") == "BREAKS" else ""
    return f"<tr{css}>{''.join(cells)}</tr>"


def _flags(fund: Mapping) -> str:
    """Small badges after the fund name: ⚠ negative estimated cash; ∅ missing data; a count of data notes."""
    cash = fund.get("cash") or {}
    out = []
    if cash.get("negative"):
        tip = ("The trades take estimated cash below zero" if cash.get("goes_negative")
               else "Cash is already below zero before the trades" if cash.get("negative_before")
               else "Estimated post-trade cash is below zero")
        out.append(f'<span class="fm-ico bad" title="{ui.esc(tip)}" aria-label="negative cash">⚠</span>')
    out.append(missing_icon(fund))
    notes = [str(w) for w in fund.get("warnings") or [] if "below zero" not in str(w).lower()
             and "negative" not in str(w).lower()]
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
    cash_date = dashboard.get("cash_date") or "T-1"
    classes = [a for a in ASSETS if a == "equity" or _has_class(dashboard, a)]
    group = "".join(f'<th colspan="{1 if a == "treasury" else 3}" class="c grp">{ui.esc(charts.ASSET_LABELS[a])}'
                    f'{" premium" if a == "options" else ""}</th>' for a in classes)
    sub = "".join('<th class="num grp">Net</th>' if a == "treasury" else
                  '<th class="num grp">Buys</th><th class="num">Sells</th><th class="num">Net</th>' for a in classes)
    head = (f'<thead><tr class="top"><th rowspan="2">Fund</th><th rowspan="2" class="num">TNA</th>'
            f'<th colspan="2" class="c grp">Cash</th><th rowspan="2" class="num grp">Traded<br><span>% TNA</span></th>'
            f'{group}</tr><tr><th class="num grp">Current<br><span>{ui.esc(cash_date)} close</span></th>'
            f'<th class="num">Post-trade<br><span>estimate</span></th>{sub}</tr></thead>')
    body = "".join(_fund_row(f, classes) for f in funds)
    body += _totals_row(dashboard, classes)
    table = (f'<div class="fm-tablewrap fm-sumwrap"><table class="fm-table fm-summary">{head}<tbody>{body}</tbody>'
             f"</table></div>")
    return ui.panel(table + definitions_html(dashboard))


def _totals_row(dashboard: Mapping, classes: Sequence[str]) -> str:
    t = dashboard.get("totals") or {}
    cash_funds = t.get("cash_funds")
    negative = t.get("funds_negative_cash") or 0
    tna_funds, funds = t.get("tna_funds"), t.get("funds") or 0
    tna_sub = f"{tna_funds} funds" if tna_funds is not None and tna_funds != funds else ""
    cells = [f'<td class="fund">Total<span class="sub">{ui.esc(funds)} funds</span></td>',
             _money_cell(t.get("tna"), sub=tna_sub),
             _money_cell(t.get("cash"), css="grp", sub=f"{cash_funds} funds" if cash_funds else ""),
             _money_cell(t.get("post_cash"), sub=f"{negative} below 0" if negative else ""),
             f'<td class="num grp">{ui.esc(ui.pct(t.get("gross_pct_tna")))}</td>']
    for asset in classes:
        cells += _class_cells((t.get("assets") or {}).get(asset) or {}, asset, total=True)
    return f'<tr class="total">{"".join(cells)}</tr>'


def definitions_html(dashboard: Mapping) -> str:
    d = dashboard.get("definitions") or {}
    cash_date = dashboard.get("cash_date") or "T-1"
    notes = [("TNA", d.get("tna")), (f"Current cash ({cash_date} close)", d.get("current_cash")),
             (f"T ({dashboard.get('date') or 'trade date'}) beneath current cash", d.get("custodian_t")),
             ("Trade value", d.get("trade_value")), ("Net", d.get("net_value")),
             ("Post-trade cash (estimate)", d.get("post_cash"))]
    text = "".join(f"<p><b>{ui.esc(label)}.</b> {ui.esc(body)}</p>" for label, body in notes if body)
    text += ("<p>Amounts are compact ($M = millions); hover a figure for the exact amount. Option buys and sells are "
             "premium, with contracts beneath (<i>unpriced</i>: no usable price, so the report's figure is contracts "
             "× multiplier, not dollars); treasuries are at price per 100 face. <i>In report · not staged</i>: the "
             "trading report shows a value from the ex-post share columns but there are no staged trade rows (the "
             "drill-down lists those position changes). After a fund's name, ⚠ marks negative estimated cash, ∅ "
             "missing custodian TNA or cash, and an amber number counts its data notes (duplicate rows, trades left "
             "out of the estimate); hover them for the details, or see the fund drill-down.</p>")
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

def keys(key: str, run: str) -> Dict[str, str]:
    """The session keys of one dashboard (the drill-down reads and the jump callback writes them)."""
    return {"tabs": f"{key}-tctabs-{run}", "only": f"{key}-tc-only-{run}", "jump": f"{key}-tcjump-{run}",
            "fund": f"{key}-tcfund-{run}", "rule": f"{key}-tcrule-{run}", "focus": f"{key}-tcfocus-{run}"}


def jump_options(dashboard: Mapping) -> List[str]:
    """Every fund, then 'FUND #i' for each matrix rule i (the hidden jump pills' options, what the script clicks)."""
    funds = [str(f.get("fund")) for f in dashboard.get("funds") or []]
    count = len((dashboard.get("matrix") or {}).get("rules") or [])
    return funds + [f"{fund} #{i}" for fund in funds for i in range(count)]


def apply_jump(dashboard: Mapping, names: Mapping[str, str]) -> None:
    """The jump pills' callback: open the drill-down tab on the picked fund (and rule, shown first)."""
    choice = st.session_state.get(names["jump"])
    st.session_state[names["jump"]] = None  # the same fund can be picked again
    if not choice:
        return
    fund, _, index = str(choice).partition(" #")
    rules = (dashboard.get("matrix") or {}).get("rules") or []
    rule = str(rules[int(index)].get("key")) if index.isdigit() and int(index) < len(rules) else None
    st.session_state[names["fund"]] = fund
    st.session_state[names["focus"]] = rule
    from ui import trading_drilldown  # noqa: PLC0415 (it imports this module)
    st.session_state[names["rule"]] = rule if rule in trading_drilldown.funds_by_rule(dashboard) and \
        fund in trading_drilldown.funds_by_rule(dashboard)[rule] else trading_drilldown.ALL_FUNDS
    st.session_state[names["tabs"]] = DRILL_TAB


def render_dashboard(view: Mapping, *, key: str, run: str, requested: Sequence[str] = ()) -> None:
    """The whole dashboard of one trading day (the downloads follow it in :func:`ui.panels.results_panel`)."""
    from ui import trading_drilldown  # noqa: PLC0415 (keeps this module importable on its own)

    dashboard = view.get("dashboard") or {}
    kpis = dashboard.get("kpis") or {}
    aside = (f"{dashboard.get('date') or view.get('date')} · cash and TNA at the {dashboard.get('cash_date') or 'T-1'} "
             f"close · {kpis.get('funds_with_trades', 0)} of {kpis.get('funds_selected', 0)} funds with proposed trades")
    ui.show(ui.heading("Trading compliance", aside) + f'<p class="fm-beforeafter">{ui.esc(BEFORE_AFTER)}</p>')
    errors = _fund_errors(view)
    if errors:
        ui.show(ui.callout("error", f"{len(errors)} fund error{'s' if len(errors) != 1 else ''} on this day:", errors))
    if not dashboard.get("funds"):
        ui.show(no_trades_html(dashboard))
        return
    ui.show(kpi_html(dashboard) + data_notes_html(dashboard))
    names = keys(key, run)
    with st.container(key="tcjump_" + re.sub(r"\W", "_", f"{key}_{run}")):  # hidden (CSS): no layout gap
        st.pills("Open in the drill-down", jump_options(dashboard), key=names["jump"], on_change=apply_jump,
                 args=(dashboard, names), label_visibility="collapsed")
        st.html(JUMP_SCRIPT, unsafe_allow_javascript=True)
    if st.session_state.get(names["tabs"]) not in TABS:
        st.session_state[names["tabs"]] = TABS[0]
    impact, summary, drill = st.tabs(list(TABS), key=names["tabs"], on_change="rerun")
    with impact:
        head, toggle = st.columns([2.3, 1.5], vertical_alignment="bottom")
        head.html(ui.heading("Compliance impact", "funds with proposed trades × rules · before → after"))
        st.session_state.setdefault(names["only"], True)
        with toggle:
            only = st.toggle("Only changes and failures", key=names["only"],
                             help="Show only the rules that change or fail for at least one fund")
        ui.show(matrix_html(dashboard, only))
        ui.show(ui.heading("Rules at risk", "the funds failing each rule after the trades · click one to open it")
                + rules_at_risk_html(dashboard))
    with summary:
        ui.show(ui.heading("Trading summary", "proposed trades and cash per fund") + summary_table_html(dashboard))
        ui.show(ui.heading("Trades at a glance", "same fund order in each chart; the table above has the numbers"))
        ui.show_svg(charts_html(dashboard))
    with drill:
        trading_drilldown.render(view, dashboard, key=key, run=run)


def _fund_errors(view: Mapping) -> List[str]:
    """One line per fund error of the day (already redacted by :func:`ui.history.load_views`)."""
    out = []
    for fund, messages in (view.get("fund_errors") or {}).items():
        for message in messages or []:
            out.append(f"{fund}: {message}")
    return out
