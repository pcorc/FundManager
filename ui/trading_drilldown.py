"""The fund drill-down of the Trading Compliance dashboard (one fund with proposed trades at a time).

Top to bottom: a rule filter and the fund picker (funds ordered breaks first);
the fund header (impact, status change, TNA, trades, cash); its data notes; the
cash waterfall beside the Summary Metrics before → after (the compliance
checks' own trade-date figures, not trade-adjusted); one comparison card per
rule (before / after side by side, changed rows and GICS table rows
highlighted, the rule's key figure and trend; the rule opened from the matrix
first, then failing, breaking and changing rules; the unchanged ones folded);
the trade blotter per asset class, plus the ex-post position changes without
staged rows; and, folded, the full rule cards as in the PDF.

The matrix and Rules at risk open a fund here through
:func:`ui.trading_results.apply_jump`, which sets the fund, rule filter and
focus keys this module reads.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence, Set

import streamlit as st

from ui import charts
from ui import components as ui
from ui import results
from ui import trading_results as tr

ALL_FUNDS = "__all__"
ALL_FUNDS_LABEL = "All funds with proposed trades"
FILTER_LABEL = "Funds by rule (failing or cured)"
_SHOWN_FIRST = ("BREAKS", "STILL_FAILS", "FAILS", "ERROR", "CURES")
_CHANGE_TEXT = {"OUT_OF_COMPLIANCE": "Out of compliance", "INTO_COMPLIANCE": "Into compliance",
                "STILL_IN_BREACH": "Still in breach", "UNCHANGED": "Unchanged"}
_EFFECT = {"open": "opens", "close": "closes", "increase": "adds to", "reduce": "reduces", "flip": "flips"}
_FLAG_TEXT = {"no_price": ("no price", "No usable price: the report values it at quantity x multiplier; "
                                       "left out of the cash estimate"),
              "restates_position": ("restates position", "The staged quantity equals the position already held "
                                                         "and the ex-post position is unchanged; left out of the "
                                                         "cash estimate"),
              "duplicate": ("duplicate row", "Repeats another trade row exactly (duplicate holdings rows); the "
                                             "trading report counts it too, the cash estimate counts the trade "
                                             "once")}
_NUMBER = re.compile(r"^[−\-+]?\$?[\d,]*\.?\d+%?$")
_TABLE_ROWS = 8  # changed table rows shown per GICS table (the rest are counted)
METRICS_NOTE = ("Cash and TNA here are the trade-date custodian figures the compliance checks use, on both sides; "
                "they are not adjusted for the trades. The waterfall has the estimated post-trade cash.")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def fund_header_html(fund: Mapping) -> str:
    cash = fund.get("cash") or {}
    change = str(fund.get("status_change") or "UNCHANGED")
    tna = charts._num(fund.get("tna")) or None
    stats = [("TNA", ui.money(tna) if tna else "no data", "" if tna else "warnish"),
             ("Trades", f"{fund.get('trade_count') or 0}", ""),
             ("Buys", ui.money(fund.get("buy_value")), ""),
             ("Sells", ui.money(fund.get("sell_value")), "")]
    if tna:
        stats.append(("Traded", f"{ui.pct(fund.get('gross_pct_tna'))} of TNA", ""))
    stats.append(("Violations", f"{fund.get('violations_before', '—')} → {fund.get('violations_after', '—')}",
                  "bad" if (fund.get("violations_after") or 0) > (fund.get("violations_before") or 0) else ""))
    if cash.get("current") is None:
        stats.append(("Cash", "not estimated (no custodian cash)", "warnish"))
    else:
        stats.append((f"Cash ({cash.get('date') or 'T-1'} close)",
                      f"{ui.money(cash.get('current'))} → {ui.money(cash.get('post'))} est.",
                      "bad" if cash.get("negative") else ""))
    if cash.get("custodian_t") is not None:
        stats.append((f"Cash ({cash.get('custodian_t_date') or 'T'}, used by the checks)",
                      ui.money(cash.get("custodian_t")), ""))
    return (f'<div class="fm-fundhead"><div class="who"><span class="t">{ui.esc(fund.get("fund"))}</span>'
            f'{tr.impact_chip(fund.get("impact"), bool(fund.get("worsens")))}{tr.missing_icon(fund)}'
            f'<span class="chg">{ui.esc(_CHANGE_TEXT.get(change, change.title()))}</span></div>{ui.stats(stats)}</div>')


def _cell(value: object, status: Optional[str]) -> str:
    if status:
        return ui.pill(status, str(value) if value not in (None, "") else status, small=True)
    return ui.cell_text(value, numeric=_is_number(value))


def _is_number(value: object) -> bool:
    return bool(_NUMBER.match(str(value or "").strip().replace(" ", "")))


def _as_float(value: object) -> Optional[float]:
    text = str(value or "").strip().replace(",", "").replace("$", "").replace("%", "").replace("−", "-")
    try:
        return float(text)
    except ValueError:
        return None


def _direction(before: object, after: object) -> str:
    b, a = _as_float(before), _as_float(after)
    if b is None or a is None or a == b:
        return ""
    return '<i class="up" aria-label="up">▲</i>' if a > b else '<i class="down" aria-label="down">▼</i>'


def diff_rows_html(rows: Sequence[Mapping]) -> str:
    out = []
    for row in rows:
        changed = bool(row.get("changed"))
        after = _cell(row.get("after"), row.get("status_after"))
        if changed and not row.get("status_after"):
            after = _direction(row.get("before"), row.get("after")) + f"<b>{after}</b>"
        css = ' class="chg"' if changed else ""
        out.append(f"<tr{css}><th>{ui.esc(row.get('label'))}</th>"
                   f'<td class="num">{_cell(row.get("before"), row.get("status_before"))}</td>'
                   f'<td class="num">{after}</td></tr>')
    return "".join(out)


def comparison_table(rows: Sequence[Mapping], before: str = "Before · ex-ante", after: str = "After · ex-post") -> str:
    if not rows:
        return ""
    return (f'<table class="fm-cmp"><thead><tr><th></th><th class="num">{ui.esc(before)}</th>'
            f'<th class="num">{ui.esc(after)}</th></tr></thead><tbody>{diff_rows_html(rows)}</tbody></table>')


def table_diff_html(table: Mapping) -> str:
    """The changed rows of one rule table (GICS weights): one line per category, each column before → after."""
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    if not rows:
        return ""
    lines = []
    for row in rows[:_TABLE_ROWS]:
        before, after = row.get("before"), row.get("after")
        if before is None or after is None:
            lines.append({"label": row.get("key"), "before": "—" if before is None else "listed",
                          "after": "—" if after is None else "listed", "changed": True})
            continue
        parts = []
        for i, (old, new) in enumerate(zip(before, after)):
            if old != new:
                name = columns[i + 1] if i + 1 < len(columns) else f"col {i + 2}"
                parts.append((name, old, new))
        for name, old, new in parts or [("", "", "")]:
            lines.append({"label": f"{row.get('key')}" + (f" · {name}" if name else ""),
                          "before": old, "after": new, "changed": True})
    more = len(rows) - _TABLE_ROWS
    title = f"{table.get('title')}: {len(rows)} of {table.get('total') or len(rows)} rows change"
    body = comparison_table(lines)
    if more > 0:
        body += f'<div class="fm-tablecap">and {more} more changed row{"s" if more != 1 else ""} (see the full card)</div>'
    return f'<div class="fm-sub">{ui.esc(title)}</div>{body}'


def _key_line(rule: Mapping) -> str:
    key = rule.get("key") or {}
    if not key or key.get("before") == key.get("after"):
        return ""
    trend = rule.get("trend")
    word = {"worse": "worse after the trades", "better": "better after the trades"}.get(str(trend), "")
    return (f'<div class="fm-keyfig {ui.esc(trend or "")}"><span>{ui.esc(key.get("label"))}</span>'
            f'<b>{ui.esc(key.get("before") or "—")} → {ui.esc(key.get("after") or "—")}</b>'
            f"<em>{ui.esc(word)}</em></div>")


def rule_compare_html(rule: Mapping, focus: bool = False) -> str:
    """One rule before → after: the band names it with the transition; rows side by side, changes highlighted."""
    rows = list(rule.get("rows") or [])
    details = list(rule.get("details") or [])
    body = _key_line(rule) + comparison_table(rows)
    for table in rule.get("tables") or []:
        body += table_diff_html(table)
    if details:
        moved = sum(1 for d in details if d.get("changed"))
        label = f"More detail ({len(details)}" + (f", {moved} changed)" if moved else ")")
        body += ui.details(label, comparison_table(details), open_=False)
    if not body:
        body = '<div class="fm-reason">No figures to compare for this rule.</div>'
    changed = rule.get("changed_rows") or 0
    sub = f"{changed} changed" if changed else "no change"
    title = tr.rule_name(rule)
    code = str(rule.get("transition"))
    # The transition chip already says FAIL → FAIL / PASS → FAIL; the before / after pills are shown only when it
    # does not (N/A, errors).
    pills = ""
    if code in ("N/A", "ERROR", "FAILS"):
        pills = (f'{ui.pill(rule.get("before"), small=True)}<span class="arrow">→</span>'
                 f'{ui.pill(rule.get("after"), small=True)}')
    band = (f'<div class="fm-band"><span>{ui.esc(title)} <span class="sub">{ui.esc(sub)}</span></span>'
            f'<span class="fm-band-tx">{pills}{tr.transition_chip(code, quiet_ok=False, trend=rule.get("trend"))}'
            f"</span></div>")
    css = tr.TRANSITION_STYLE.get(code, ("na",))[0] + (" focus" if focus else "")
    return f'<div class="fm-card fm-cmpcard t-{ui.esc(css)}">{band}<div class="fm-body">{body}</div></div>'


def split_rules(rules: Sequence[Mapping], focus: Optional[str] = None) -> tuple:
    """(shown, folded): the focused rule first, failing / breaking / curing rules and rules whose values move;
    the rest folded."""
    shown = [r for r in rules if r.get("transition") in _SHOWN_FIRST or not r.get("unchanged") or r.get("rule") == focus]
    shown.sort(key=lambda r: r.get("rule") != focus)
    folded = [r for r in rules if r not in shown]
    return shown, folded


def folded_rules_html(rules: Sequence[Mapping]) -> str:
    if not rules:
        return ""
    rows = [[tr.rule_name(r), ui.pill(r.get("before"), small=True), ui.pill(r.get("after"), small=True)]
            for r in rules]
    table = ui.data_table(["Rule", "Before", "After"], rows, raw=(1, 2))
    return (f'<div class="fm-card fm-folded"><details><summary>{len(rules)} rule{"s" if len(rules) != 1 else ""} '
            f"unchanged by the trades</summary>{table}</details></div>")


def _qty(value: object) -> str:
    return ui.quantity(value) if value is not None else "—"


def _position(trade: Mapping) -> str:
    before, after = trade.get("before_qty"), trade.get("after_qty")
    if before is None:
        return ui.DASH
    effect = _EFFECT.get(str(trade.get("effect")), "")
    tip = f' title="Ex-post holdings show {ui.esc(_qty(trade.get("ex_post_qty")))}"' if trade.get("ex_post_qty") is not None else ""
    note = f'<span class="sub">{ui.esc(effect)}</span>' if effect else ""
    return f'<span{tip}>{ui.esc(_qty(before))} → {ui.esc(_qty(after))}</span>{note}'


def _trade_flags(trade: Mapping) -> str:
    chips = []
    for flag in trade.get("flags") or []:
        text, tip = _FLAG_TEXT.get(str(flag), (str(flag), ""))
        css = "na" if flag == "duplicate" else "warn"
        chips.append(f'<span class="fm-flag {css}" title="{ui.esc(tip)}">{ui.esc(text)}</span>')
    return "".join(chips)


def _price_text(price: object, asset: str) -> str:
    if price is None:
        return ui.DASH
    return ui.esc(f"{float(price):,.4f}".rstrip("0").rstrip(".") if asset != "equity" else f"{float(price):,.2f}")


def blotter_html(fund: Mapping) -> List[str]:
    """One card per asset class with staged trades: ticker, side, quantity, price, est. value, % TNA, position."""
    cards = []
    for group in fund.get("blotter") or []:
        trades = list(group.get("trades") or [])
        if not trades:
            continue
        asset = str(group.get("asset_class") or "")
        unit = group.get("unit") or charts.ASSET_UNITS.get(asset, "shares")
        qty_head = {"contracts": "Contracts", "face": "Face"}.get(unit, "Shares")
        rows = []
        for trade in trades:
            side = str(trade.get("side") or "")
            css = "" if trade.get("in_cash", True) else ' class="out"'
            tip = "" if trade.get("in_cash", True) else ' title="Left out of the cash estimate"'
            rows.append(f'<tr{css}{tip}><td class="fund">{ui.esc(trade.get("ticker"))}</td>'
                        f'<td><span class="fm-side {"buy" if side == "BUY" else "sell"}">{ui.esc(side)}</span></td>'
                        f'<td class="num">{ui.esc(_qty(trade.get("quantity")))}</td>'
                        f'<td class="num">{_price_text(trade.get("price"), asset)}</td>'
                        f'<td class="num" title="{ui.esc(ui.money(trade.get("value"), compact=False))}">'
                        f'{ui.esc(ui.money(trade.get("value"), compact=False))}</td>'
                        f'<td class="num">{ui.esc(ui.pct(trade.get("pct_tna")))}</td>'
                        f'<td class="num">{_position(trade)}</td><td>{_trade_flags(trade)}</td></tr>')
        head = ("<thead><tr><th>Ticker</th><th>Side</th>"
                f'<th class="num">{ui.esc(qty_head)}</th><th class="num">Price</th><th class="num">Est. value</th>'
                '<th class="num">% TNA</th><th class="num">Position before → after</th><th>Flags</th></tr></thead>')
        table = (f'<div class="fm-tablewrap fm-scroll"><table class="fm-table fm-blotter">{head}<tbody>'
                 f'{"".join(rows)}</tbody></table></div>')
        sub = (f"{len(trades)} trade{'s' if len(trades) != 1 else ''} · buys {ui.money(group.get('buy_value'))}"
               f" · sells {ui.money(group.get('sell_value'))}")
        cards.append(ui.card(group.get("label") or charts.ASSET_LABELS.get(asset, asset), table, subtitle=sub,
                             plain=True))
    return cards


def _rows_flag(change: Mapping) -> str:
    rows = int(change.get("rows") or 1)
    if rows <= 1:
        return ""
    return (f'<span class="fm-flag na" title="The ex-post holdings list this position {rows} times (duplicate rows); '
            f'shown once here, the report counts each">duplicate row ×{rows}</span>')


def unstaged_html(fund: Mapping) -> List[str]:
    """One card per asset class the report values without staged rows: the ex-post nav → iiv share changes."""
    cards = []
    for group in fund.get("unstaged") or []:
        changes = list(group.get("changes") or [])
        if not changes:
            continue
        asset = str(group.get("asset_class") or "")
        unit = group.get("unit") or charts.ASSET_UNITS.get(asset, "shares")
        qty_head = {"contracts": "Contracts", "face": "Face"}.get(unit, "Shares")
        rows = []
        for change in changes:
            delta = charts._num(change.get("change")) or 0.0
            rows.append(f'<tr><td class="fund">{ui.esc(change.get("ticker"))}</td>'
                        f'<td><span class="fm-side {"buy" if delta > 0 else "sell"}">'
                        f'{"ADDS" if delta > 0 else "REDUCES"}</span></td>'
                        f'<td class="num">{ui.esc(ui.quantity(delta))}</td>'
                        f'<td class="num">{_price_text(change.get("price"), asset)}</td>'
                        f'<td class="num">{ui.esc(ui.money(change.get("value"), compact=False))}</td>'
                        f'<td class="num">{ui.esc(_qty(change.get("before_qty")))} → '
                        f'{ui.esc(_qty(change.get("after_qty")))}</td><td>{_rows_flag(change)}</td></tr>')
        head = ("<thead><tr><th>Ticker</th><th>Change</th>"
                f'<th class="num">{ui.esc(qty_head)}</th><th class="num">Price</th><th class="num">Value</th>'
                '<th class="num">NAV → IIV shares</th><th>Flags</th></tr></thead>')
        table = (f'<div class="fm-tablewrap fm-scroll"><table class="fm-table fm-blotter">{head}<tbody>'
                 f'{"".join(rows)}</tbody></table></div>')
        caption = ('<div class="fm-tablecap">In the ex-post holdings (and so in the ex-post checks and the '
                   f"report's {ui.esc(ui.money(group.get('value')))} trade value) but with no staged trade rows; "
                   "not in the cash estimate.</div>")
        sub = f"{len(changes)} position change{'s' if len(changes) != 1 else ''} · in report, not staged"
        title = f"{group.get('label') or charts.ASSET_LABELS.get(asset, asset)} · position changes without staged rows"
        cards.append(ui.card(title, table + caption, subtitle=sub, plain=True))
    return cards


def metrics_card_html(fund: Mapping) -> str:
    rows = list(fund.get("metrics") or [])
    body = comparison_table(rows) or '<div class="fm-reason">No summary metrics for this fund.</div>'
    footer = f'<div class="fm-foot"><p>{ui.esc(METRICS_NOTE)}</p></div>' if rows else ""
    return ui.card("Summary Metrics", body, subtitle="before → after the trades · compliance inputs", footer=footer)


def waterfall_html(fund: Mapping) -> str:
    cash = fund.get("cash") or {}
    if cash.get("current") is None:
        return ui.card("Cash waterfall", '<div class="fm-reason">No custodian cash was loaded for this fund, so '
                                         "post-trade cash is not estimated.</div>")
    excluded = cash.get("excluded") or []
    note = ""
    if excluded:
        note = (f"{len(excluded)} trade{'s' if len(excluded) != 1 else ''} ({ui.money(cash.get('excluded_value'))}) "
                "left out: no usable price, a staged quantity that only restates the position held, or a duplicate "
                "row.")
    source = {"ex_ante_previous": f"custodian cash at the {cash.get('date') or 'T-1'} close",
              "ex_ante_current": "custodian cash on the trade date (no T-1 figure)",
              "ex_post_current": "custodian cash from the ex-post pass (no ex-ante figure)"}.get(
        str(cash.get("source")), "custodian cash")
    note = f"Starts from {source}; an estimate (no settlement timing, fees or other flows). " + note
    chart = charts.waterfall_chart(fund.get("waterfall") or [], title="Cash waterfall", note=note.strip())
    return f'<div class="fm-card fm-wf">{chart}</div>'


def warnings_html(fund: Mapping) -> str:
    warnings = [str(w) for w in fund.get("warnings") or []]
    if not warnings:
        return ""
    note = ui.callout("warn", f"Data notes for {fund.get('fund')} ({len(warnings)}):", warnings)
    return f'<div class="fm-dq">{note}</div>'


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def funds_by_rule(dashboard: Mapping) -> Dict[str, Set[str]]:
    """{rule key: the funds that break, fail, still fail, error or are cured on it} (rules with any)."""
    out = {}
    for risk in dashboard.get("rules_at_risk") or []:
        hit = (set(risk.get("breaks") or []) | set(risk.get("still_fails") or []) | set(risk.get("fails") or [])
               | set(risk.get("errors") or []) | set(risk.get("cures") or []))
        if hit:
            out[str(risk.get("rule"))] = hit
    return out


def rule_filter_options(dashboard: Mapping) -> List[str]:
    """[ALL_FUNDS] + the rule keys of Rules at risk (shown with :func:`rule_filter_label`)."""
    return [ALL_FUNDS] + list(funds_by_rule(dashboard))


def rule_filter_label(dashboard: Mapping, choice: str) -> str:
    if choice == ALL_FUNDS:
        return ALL_FUNDS_LABEL
    risk = next((r for r in dashboard.get("rules_at_risk") or [] if str(r.get("rule")) == choice), {})
    return f"{tr.rule_name(risk) or choice} · {len(funds_by_rule(dashboard).get(choice, ()))}"


def funds_for_filter(dashboard: Mapping, choice: str) -> List[str]:
    funds = [str(f.get("fund")) for f in dashboard.get("funds") or []]
    hit = funds_by_rule(dashboard).get(choice)
    return funds if choice == ALL_FUNDS or hit is None else [f for f in funds if f in hit]


def render(view: Mapping, dashboard: Mapping, *, key: str, run: str) -> None:
    funds = {str(f.get("fund")): f for f in dashboard.get("funds") or []}
    if not funds:
        return
    names = tr.keys(key, run)
    ui.show(ui.heading("Fund drill-down", "pick a fund, or click one in the matrix or Rules at risk"))
    options = rule_filter_options(dashboard)
    if st.session_state.get(names["rule"]) not in options:
        st.session_state[names["rule"]] = ALL_FUNDS
    pick, _ = st.columns([1, 1.6])
    choice = pick.selectbox(FILTER_LABEL, options, key=names["rule"],
                            format_func=lambda c: rule_filter_label(dashboard, c))
    fund_names = funds_for_filter(dashboard, choice) or list(funds)
    if st.session_state.get(names["fund"]) not in fund_names:
        st.session_state[names["fund"]] = fund_names[0]
    name = st.pills("Fund", fund_names, key=names["fund"], required=True, label_visibility="collapsed") \
        or fund_names[0]
    fund = funds[name]
    focus = st.session_state.get(names["focus"])
    ui.show(fund_header_html(fund) + warnings_html(fund))
    left, right = st.columns(2, gap="medium")
    with left:  # SVG: rendered through Markdown (st.html strips it)
        if (fund.get("cash") or {}).get("current") is None:
            ui.show(waterfall_html(fund))
        else:
            ui.show_svg(waterfall_html(fund))
    with right:
        ui.show(metrics_card_html(fund))
    shown, folded = split_rules(fund.get("rules") or [], focus)
    ui.show(ui.heading("Rules before → after", "failing, breaking and changing rules first")
            + ui.grid([rule_compare_html(r, focus=r.get("rule") == focus) for r in shown])
            + folded_rules_html(folded))
    cards = blotter_html(fund) + unstaged_html(fund)
    count = sum(len(g.get("trades") or []) for g in fund.get("blotter") or [])
    ui.show(ui.heading("Trade blotter", f"{count} proposed trade{'s' if count != 1 else ''} · grouped by asset class")
            + ("".join(f'<div class="fm-blot">{c}</div>' for c in cards) if cards
               else ui.empty("No staged trades", "This fund has no staged trade rows.")))
    with st.expander("Full rule cards (as in the PDF)", icon=":material/description:"):
        side = st.segmented_control("Holdings", [results.EX_POST, results.EX_ANTE], key=f"{key}-tcside-{run}",
                                    required=True, default=results.EX_POST, label_visibility="collapsed")
        which = "compliance_ex_ante" if side == results.EX_ANTE else "compliance"
        cards = results.fund_cards_html(view, name, which)
        ui.show(ui.grid(cards) if cards else ui.empty("No rule cards", "No rule results for this fund."))
