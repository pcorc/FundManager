"""A finished run's results on the page, rendered from its ``views.json`` (schema in ``reporting.result_views``).

Per day: an overview matrix (funds × checks, PASS / FAIL / N/A pills, plus NAV
and holdings recon when they ran; the trading overview first on a trading
day), then for a chosen fund the rule tables as cards in a two-column grid
(header band with the rule title and status, label / value rows as in the PDF,
details, GICS tables, notes and footnotes), the NAV card, the holdings
reconciliation and, on a trading day, the before / after checks and the trade
activity. The builders (``*_html``) are plain functions returning HTML.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import streamlit as st

from ui import components as ui

SHORT_LABELS = {
    "gics_compliance": "GICS", "prospectus_80pct_policy": "80% Policy", "diversification_40act_check": "40 Act",
    "diversification_IRS_check": "IRS", "diversification_IRC_check": "IRC", "max_15pct_illiquid_sai": "Illiquid",
    "real_estate_check": "Real Estate", "commodities_check": "Commod.", "twelve_d1a_other_inv_cos": "12d1-1",
    "twelve_d2_insurance_cos": "12d-2", "twelve_d3_sec_biz": "12d-3",
}
SUMMARY_KEY = "summary_metrics"
EX_POST, EX_ANTE = "After trades (ex-post)", "Before trades (ex-ante)"


def rule_keys(day: Mapping) -> List[str]:
    """The day's test rules in PDF order (the summary sheet is not a test)."""
    keys = [k for k in day.get("rules") or {} if k != SUMMARY_KEY]
    for fund_rules in (day.get("compliance") or {}).values():  # a rule missing from "rules" still shows
        keys += [k for k in fund_rules if k != SUMMARY_KEY and k not in keys]
    return keys


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

IRC_KEY = "diversification_IRC_check"
IRC_NOTE = "IRC diversification applies to VIT funds only; no fund in this run is one, so it has no column."


def matrix_html(day: Mapping, requested: Sequence[str] = ()) -> str:
    """Funds × checks with status pills (overall first; NAV and holdings when they ran).

    ``requested`` (the run's compliance tests) explains a picked IRC check that has no column.
    """
    compliance = day.get("compliance") or {}
    summary = day.get("compliance_summary") or {}
    nav, holdings = day.get("nav") or {}, day.get("holdings") or {}
    keys = rule_keys(day)
    columns = ["Fund"] + (["Overall"] if compliance else []) + [SHORT_LABELS.get(k, k) for k in keys]
    columns += (["NAV"] if nav else []) + (["Holdings"] if holdings else [])
    rows = []
    for fund in day.get("funds") or []:
        row: List[str] = [fund]
        if compliance:
            info = summary.get(fund) or {}
            failed = len(info.get("failed") or [])
            row.append(ui.pill(info.get("status"), title=f"{failed} failed" if failed else None, small=True))
        for key in keys:
            view = (compliance.get(fund) or {}).get(key)
            row.append(ui.pill(view.get("status"), title=view.get("reason"), small=True) if view else ui.pill(None))
        if nav:
            row.append(ui.pill((nav.get(fund) or {}).get("status"), small=True))
        if holdings:
            row.append(ui.pill((holdings.get(fund) or {}).get("status"), small=True))
        rows.append(row)
    if not rows:
        return ui.empty("No funds", "This day has no fund results.")
    caption = IRC_NOTE if compliance and IRC_KEY in requested and IRC_KEY not in keys else ""
    return ui.panel(ui.data_table(columns, rows, raw=range(1, len(columns)), css="fm-matrix", caption=caption))


def rule_card_html(view: Mapping, footnotes: Sequence[str] = ()) -> str:
    """One rule's card: the PDF metric rows, then details, tables, notes and the rule's footnotes."""
    body = ""
    if view.get("reason") and not any(r.get("reason") == view.get("reason") for r in view.get("rows") or []):
        body += f'<p class="fm-reason">{ui.esc(view["reason"])}</p>'
    body += ui.kv_table(view.get("rows") or [])
    detail_rows = view.get("details") or []
    if detail_rows:
        body += ui.details(f"More detail ({len(detail_rows)})", ui.kv_table(detail_rows))
    for table in view.get("tables") or []:
        count = table.get("total") or len(table.get("rows") or [])
        body += ui.details(f"{table.get('title') or 'Table'} ({count})",
                           ui.data_table(table.get("columns") or [], table.get("rows") or [], total=table.get("total")))
    for note in view.get("notes") or []:
        body += f'<div class="fm-notes">{ui.esc(note)}</div>'
    return ui.card(view.get("title") or view.get("label") or "Rule", body, status=view.get("status"),
                   footer=ui.footnotes("Footnotes", footnotes))


def nav_card_html(nav: Mapping) -> str:
    """The NAV card, shown across the grid: summary and checks beside the G/L sections, then the tables."""
    left = ui.kv_table(nav.get("summary") or [])
    checks = nav.get("checks") or []
    if checks:
        left += ui.subhead("Checks") + ui.kv_table(
            [{"label": c.get("label"), "value": c.get("status"), "status": c.get("status")} for c in checks])
    if nav.get("rows"):
        left += ui.details("Reconciliation detail", ui.kv_table(nav["rows"]))
    right = ""
    for n, section in enumerate(nav.get("sections") or []):
        head = ui.subhead(section.get("title") or "")
        right += (head.replace('class="fm-sub"', 'class="fm-sub first"', 1) if n == 0 else head)
        right += ui.kv_table(section.get("rows") or [])
    if nav.get("gl"):
        right += ui.details("G/L walk", ui.kv_table(nav["gl"]))
    body = ui.split(left, right) if right else left
    for table in nav.get("tables") or []:
        body += ui.data_table(table.get("columns") or [], table.get("rows") or [], total=table.get("total"),
                              title=table.get("title") or "")
    return ui.card("NAV Reconciliation", body, status=nav.get("status"))


def holdings_html(holdings: Mapping) -> str:
    """The holdings reconciliation: break counts as chips, then each section's discrepancy tables."""
    chips = []
    for item in holdings.get("counts") or []:
        value = item.get("value") or 0
        cls = "" if not value else "bad" if item.get("kind") == "holdings" else "warnish"
        chips.append((item.get("label") or "", value, cls))
    body = ui.stats(chips)
    for section in holdings.get("sections") or []:
        tables = section.get("tables") or []
        rows = sum(t.get("total") or len(t.get("rows") or []) for t in tables)
        extra = [(c.get("label") or "", c.get("value"), "") for c in section.get("counts") or []]
        inner = ui.stats(extra) if extra else ""
        inner += "".join(ui.data_table(t.get("columns") or [], t.get("rows") or [], total=t.get("total"),
                                       title=t.get("title") or "") for t in tables)
        body += ui.details(f"{section.get('title') or section.get('key')} · {rows} row{'s' if rows != 1 else ''}",
                           inner, open_=True)
    subtitle = (f"{holdings.get('holdings_breaks', 0)} holdings breaks · "
                f"{holdings.get('price_breaks', 0)} price breaks")
    return ui.card("Holdings Reconciliation", body, status=holdings.get("status"), subtitle=subtitle)


# The trading PDF's row colours: into compliance green, out of compliance red, still in breach amber.
_HIGHLIGHT = {"into": "hl-good", "out": "hl-bad", "breach": "hl-breach"}
_CHANGE_STATUS = {"into": "PASS", "out": "FAIL", "breach": "WARN"}


def trading_overview_html(trading: Mapping) -> str:
    overview = trading.get("overview") or {}
    rows, classes = [], []
    for row in overview.get("rows") or []:
        change = str(row.get("status_change") or "UNCHANGED")
        highlight = row.get("highlight") or ""
        rows.append([row.get("fund"), ui.pill(_CHANGE_STATUS.get(highlight, "N/A"), change.replace("_", " ").title()),
                     row.get("violations_before"), row.get("violations_after"), row.get("net_change")])
        classes.append(_HIGHLIGHT.get(highlight, ""))
    totals = overview.get("totals") or {}
    if rows and totals:
        rows.append(["Total", "", totals.get("violations_before"), totals.get("violations_after"),
                     totals.get("net_change")])
        classes.append("total")
    summary = trading.get("summary") or {}
    caption = (f"{summary.get('total_funds_traded', '?')} of {summary.get('total_funds_analyzed', '?')} funds traded. "
               "Green: back into compliance · red: out of compliance · amber: still in breach.")
    columns = overview.get("columns") or ["Fund", "Status Change", "Violations Before", "Violations After",
                                          "Net Change"]
    return ui.panel(ui.data_table(columns, rows, row_classes=classes, raw=(1,), caption=caption))


def trading_checks_html(checks: Sequence[Mapping]) -> str:
    rows = [[c.get("label"), ui.pill(c.get("before")), ui.pill(c.get("after"))] for c in checks]
    classes = ["hl-breach" if c.get("changed") else "" for c in checks]
    return ui.card("Checks Before / After Trades", ui.data_table(["Check", "Before", "After"], rows, raw=(1, 2),
                                                                  row_classes=classes))


def trading_activity_html(trading: Mapping, fund: str) -> str:
    """The fund's trade activity by asset class (metrics down, asset classes across) and its trades."""
    rows = [r for r in (trading.get("activity") or {}).get("rows") or [] if r.get("fund") == fund]
    body = ""
    if rows:
        labels = [cell.get("label") for cell in rows[0].get("cells") or []]
        table = [[label] + [next((c.get("value") for c in r.get("cells") or [] if c.get("label") == label), "")
                            for r in rows] for label in labels]
        body += ui.data_table(["Metric"] + [r.get("asset_class") or "" for r in rows], table)
    trades = (trading.get("activity_details") or {}).get(fund) or []
    if trades:
        body += ui.details(f"Trades ({len(trades)})", ui.data_table(
            ["Asset Class", "Direction", "Ticker", "Quantity", "Market Value"],
            [[t.get("asset_class"), t.get("direction"), t.get("ticker"), t.get("quantity"), t.get("market_value")]
             for t in trades]))
    if not body:
        body = '<p class="fm-reason">No trade activity for this fund on this day.</p>'
    return ui.card("Trade Activity", body)


def fund_cards_html(day: Mapping, fund: str, side: str = "compliance") -> List[str]:
    """The rule cards of ``fund`` (``side`` = ``compliance`` or ``compliance_ex_ante``), summary sheet first."""
    rules = day.get("rules") or {}
    views: Mapping[str, Mapping] = (day.get(side) or {}).get(fund) or {}
    order = [SUMMARY_KEY] + rule_keys(day)
    keys = [k for k in order if k in views] + [k for k in views if k not in order]
    return [rule_card_html(views[k], (rules.get(k) or {}).get("footnotes") or []) for k in keys]


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------

def _pick(label: str, options: Sequence[str], key: str, *, pills: bool = True, show_label: bool = False) -> str:
    """A single choice that always holds one of ``options`` (kept per run folder in ``key``)."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    if pills:
        return st.pills(label, list(options), key=key, required=True,
                        label_visibility="visible" if show_label else "collapsed") or options[0]
    return st.selectbox(label, list(options), key=key)


def _first_failing(day: Mapping) -> Optional[str]:
    summary = day.get("compliance_summary") or {}
    for fund in day.get("funds") or []:
        if (summary.get(fund) or {}).get("status") in ("FAIL", "ERROR"):
            return fund
    return None


def render_views(views: Mapping, *, key: str, run: str, day: Optional[str] = None,
                 requested: Sequence[str] = ()) -> bool:
    """The results of one run; ``False`` when ``views`` holds no day (the caller then shows the downloads only).

    ``day`` (the Reports page's Report date) pins the tables to that day, so they always match the downloads
    filtered to it; without it (or for a day the views do not hold) a Report date picker chooses.
    """
    days: Dict[str, Mapping] = dict(views.get("days") or {})
    if not days:
        return False
    names = list(days)
    if day in days:
        picked = day
    elif len(names) == 1:
        picked = names[0]
    else:
        picked = _pick("Report date", names[::-1], f"{key}-day-{run}", pills=len(names) <= 10, show_label=True)
    view = days[picked]
    trading = view.get("kind") == "trading"
    if trading and view.get("dashboard"):
        from ui import trading_results  # noqa: PLC0415 (trading_results imports this module)

        trading_results.render_dashboard(view, key=key, run=run, requested=requested)
        return True
    if trading and view.get("dashboard_error"):
        ui.show(ui.callout("warn", f"The trading dashboard could not be built ({view['dashboard_error']}); "
                                   "the report's overview is shown instead."))
    funds = list(view.get("funds") or [])
    ui.show(ui.heading("Results", f"{picked} · {len(funds)} fund{'s' if len(funds) != 1 else ''}"
                       + (" · trading day" if trading else "")))
    if trading and view.get("trading"):
        ui.show(trading_overview_html(view["trading"]))
    if view.get("compliance") or view.get("nav") or view.get("holdings"):
        ui.show(matrix_html(view, requested))
    if not funds:
        return True
    fund_key = f"{key}-fund-{run}"
    if fund_key not in st.session_state:
        st.session_state[fund_key] = _first_failing(view) or funds[0]
    ui.show(ui.heading("Fund detail", "pick a fund"))
    fund = _pick("Fund", funds, fund_key)
    for message in (view.get("fund_errors") or {}).get(fund) or []:
        ui.show(ui.callout("error", message))
    side = "compliance"
    if trading and view.get("compliance_ex_ante"):
        choice = st.segmented_control("Holdings", [EX_POST, EX_ANTE], key=f"{key}-side-{run}", required=True,
                                      default=EX_POST, label_visibility="collapsed")
        side = "compliance_ex_ante" if choice == EX_ANTE else "compliance"
    cards, wide = [], []  # rule cards in rows of two; the activity and NAV cards across both columns
    if trading and view.get("trading"):
        checks = (view["trading"].get("checks") or {}).get(fund) or []
        if checks:
            cards.append(trading_checks_html(checks))
        wide.append(trading_activity_html(view["trading"], fund))
    cards += fund_cards_html(view, fund, side)
    if (view.get("nav") or {}).get(fund):
        wide.append(nav_card_html(view["nav"][fund]))
    if cards or wide:
        ui.show(ui.grid(cards, wide))
    if (view.get("holdings") or {}).get(fund):
        ui.show(holdings_html(view["holdings"][fund]))
    return True
