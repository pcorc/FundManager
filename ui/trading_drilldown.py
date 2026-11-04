"""The fund drill-down of the Trading Compliance dashboard (one fund with proposed trades at a time).

Top to bottom: a rule filter and the fund picker (funds ordered breaks first);
the fund header (impact, status change, TNA, trades, cash); its data notes; the
cash waterfall beside the Summary Metrics before → after; one comparison card
per rule (before / after side by side, changed rows highlighted; failing,
breaking and changing rules first, the unchanged ones folded); the trade
blotter per asset class; and, folded, the full rule cards as in the PDF.
"""
from __future__ import annotations

import re
from typing import List, Mapping, Optional, Sequence

import streamlit as st

from ui import charts
from ui import components as ui
from ui import results
from ui import trading_results as tr

ALL_FUNDS = "All funds with proposed trades"
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
                                             "trading report counts it too")}
_NUMBER = re.compile(r"^[−\-+]?\$?[\d,]*\.?\d+%?$")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def fund_header_html(fund: Mapping) -> str:
    cash = fund.get("cash") or {}
    change = str(fund.get("status_change") or "UNCHANGED")
    stats = [("TNA", ui.money(fund.get("tna")), ""),
             ("Trades", f"{fund.get('trade_count') or 0}", ""),
             ("Buys", ui.money(fund.get("buy_value")), ""),
             ("Sells", ui.money(fund.get("sell_value")), ""),
             ("Traded", f"{ui.pct(fund.get('gross_pct_tna'))} of TNA", ""),
             ("Violations", f"{fund.get('violations_before', '—')} → {fund.get('violations_after', '—')}",
              "bad" if (fund.get("violations_after") or 0) > (fund.get("violations_before") or 0) else ""),
             ("Cash", f"{ui.money(cash.get('current'))} → {ui.money(cash.get('post'))} est.",
              "bad" if cash.get("negative") else "")]
    return (f'<div class="fm-fundhead"><div class="who"><span class="t">{ui.esc(fund.get("fund"))}</span>'
            f'{tr.impact_chip(fund.get("impact"))}<span class="chg">{ui.esc(_CHANGE_TEXT.get(change, change.title()))}'
            f'</span></div>{ui.stats(stats)}</div>')


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


def rule_compare_html(rule: Mapping) -> str:
    """One rule before → after: the band names it with the transition; rows side by side, changes highlighted."""
    rows = list(rule.get("rows") or [])
    details = list(rule.get("details") or [])
    body = comparison_table(rows)
    if details:
        moved = sum(1 for d in details if d.get("changed"))
        label = f"More detail ({len(details)}" + (f", {moved} changed)" if moved else ")")
        body += ui.details(label, comparison_table(details), open_=False)
    if not body:
        body = '<p class="fm-reason">No figures to compare for this rule.</p>'
    changed = rule.get("changed_rows") or 0
    sub = f"{changed} changed" if changed else "no change"
    title = rule.get("title") or rule.get("label") or rule.get("rule")
    band = (f'<div class="fm-band"><span>{ui.esc(title)} <span class="sub">{ui.esc(sub)}</span></span>'
            f'<span class="fm-band-tx">{ui.pill(rule.get("before"), small=True)}<span class="arrow">→</span>'
            f'{ui.pill(rule.get("after"), small=True)}{tr.transition_chip(rule.get("transition"), quiet_ok=False)}'
            f"</span></div>")
    return f'<div class="fm-card fm-cmpcard t-{ui.esc(tr.TRANSITION_STYLE.get(str(rule.get("transition")), ("na",))[0])}">{band}<div class="fm-body">{body}</div></div>'


def split_rules(rules: Sequence[Mapping]) -> tuple:
    """(shown, folded): failing / breaking / curing rules and rules whose values move first; the rest folded."""
    shown = [r for r in rules if r.get("transition") in _SHOWN_FIRST or not r.get("unchanged")]
    folded = [r for r in rules if r not in shown]
    return shown, folded


def folded_rules_html(rules: Sequence[Mapping]) -> str:
    if not rules:
        return ""
    rows = [[r.get("title") or r.get("label"), ui.pill(r.get("before"), small=True), ui.pill(r.get("after"), small=True)]
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
            price = trade.get("price")
            price_text = ui.DASH if price is None else ui.esc(f"{float(price):,.4f}".rstrip("0").rstrip(".")
                                                              if asset != "equity" else f"{float(price):,.2f}")
            css = "" if trade.get("in_cash", True) else ' class="out"'
            tip = "" if trade.get("in_cash", True) else ' title="Left out of the cash estimate"'
            rows.append(f'<tr{css}{tip}><td class="fund">{ui.esc(trade.get("ticker"))}</td>'
                        f'<td><span class="fm-side {"buy" if side == "BUY" else "sell"}">{ui.esc(side)}</span></td>'
                        f'<td class="num">{ui.esc(_qty(trade.get("quantity")))}</td><td class="num">{price_text}</td>'
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


def metrics_card_html(fund: Mapping) -> str:
    rows = list(fund.get("metrics") or [])
    body = comparison_table(rows) or '<p class="fm-reason">No summary metrics for this fund.</p>'
    return ui.card("Summary Metrics", body, subtitle="before → after the trades")


def waterfall_html(fund: Mapping) -> str:
    cash = fund.get("cash") or {}
    if cash.get("current") is None:
        return ui.card("Cash waterfall", '<p class="fm-reason">No custodian cash was loaded for this fund, so '
                                         "post-trade cash is not estimated.</p>")
    excluded = cash.get("excluded") or []
    note = ""
    if excluded:
        note = (f"{len(excluded)} trade{'s' if len(excluded) != 1 else ''} ({ui.money(cash.get('excluded_value'))}) "
                "left out: no usable price, or a staged quantity that only restates the position held.")
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
    return ui.callout("warn", f"Data notes for {fund.get('fund')} ({len(warnings)}):", warnings)


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def rule_filter_options(dashboard: Mapping) -> List[str]:
    options = [ALL_FUNDS]
    for risk in dashboard.get("rules_at_risk") or []:
        failing = (set(risk.get("breaks") or []) | set(risk.get("still_fails") or []) | set(risk.get("fails") or [])
                   | set(risk.get("errors") or []) | set(risk.get("cures") or []))
        if failing:
            options.append(f"{risk.get('title') or risk.get('label')} · {len(failing)}")
    return options


def funds_for_filter(dashboard: Mapping, choice: str) -> List[str]:
    funds = [str(f.get("fund")) for f in dashboard.get("funds") or []]
    if choice == ALL_FUNDS:
        return funds
    for risk in dashboard.get("rules_at_risk") or []:
        if choice.startswith(f"{risk.get('title') or risk.get('label')} · "):
            hit = (set(risk.get("breaks") or []) | set(risk.get("still_fails") or []) | set(risk.get("fails") or [])
                   | set(risk.get("errors") or []) | set(risk.get("cures") or []))
            return [f for f in funds if f in hit]
    return funds


def render(view: Mapping, dashboard: Mapping, *, key: str, run: str) -> None:
    funds = {str(f.get("fund")): f for f in dashboard.get("funds") or []}
    if not funds:
        return
    ui.show(ui.heading("Fund drill-down", "pick a fund; narrow the list to the funds failing a rule"))
    options = rule_filter_options(dashboard)
    filter_key = f"{key}-tcrule-{run}"
    if st.session_state.get(filter_key) not in options:
        st.session_state[filter_key] = ALL_FUNDS
    pick, _ = st.columns([1, 1.6])
    choice = pick.selectbox("Funds failing a rule", options, key=filter_key)
    names = funds_for_filter(dashboard, choice) or list(funds)
    fund_key = f"{key}-tcfund-{run}"
    if st.session_state.get(fund_key) not in names:
        st.session_state[fund_key] = names[0]
    name = st.pills("Fund", names, key=fund_key, required=True, label_visibility="collapsed") or names[0]
    fund = funds[name]
    ui.show(fund_header_html(fund) + warnings_html(fund))
    left, right = st.columns(2, gap="medium")
    with left:  # SVG: rendered through Markdown (st.html strips it)
        ui.show_svg(waterfall_html(fund))
    with right:
        ui.show(metrics_card_html(fund))
    shown, folded = split_rules(fund.get("rules") or [])
    ui.show(ui.heading("Rules before → after", "failing, breaking and changing rules first")
            + ui.grid([rule_compare_html(r) for r in shown]) + folded_rules_html(folded))
    cards = blotter_html(fund)
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
