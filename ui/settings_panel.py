"""The New run settings panel (left pane): Date Range, Select Fund, Operations, Compliance Checks, Options, buttons.

The fund and check tiles are checkboxes keyed ``nrf_<ticker>`` / ``nrr_<rule>``
(styled as tiles by :mod:`ui.theme`). Their truth is the lists
``new_run.FUNDS_KEY`` / ``new_run.RULES_KEY``: each tile's state is set from the
list before it is drawn and its ``on_change`` callback updates the list, so a
group chip, *Select All*, the filter (which hides tiles) and RESET all act on
the same picks.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Sequence

import streamlit as st

from ui import components as ui
from ui import new_run as nr

FUND_COLUMNS = 3
OP_LABELS = {"compliance": "Compliance checks", "reconciliation": "Holdings reconciliation",
             "nav_reconciliation": "NAV reconciliation"}
_ACRONYMS = {"ETF", "VIT", "NAV", "OMS"}
_SMALL_WORDS = {"AND", "OF", "THE"}
_WEEKDAYS_HELP = "Weekends snap to business days. There is no holiday calendar: a market holiday still runs."


def group_label(name: str) -> str:
    """``CLOSED_END_FUNDS`` -> ``Closed End Funds``, ``ETF_FUNDS`` -> ``ETF Funds``."""
    words = [w for w in str(name).replace("-", "_").split("_") if w]
    out = []
    for i, word in enumerate(words):
        upper = word.upper()
        out.append(upper if upper in _ACRONYMS else word.lower() if i and upper in _SMALL_WORDS
                   else word.capitalize())
    return " ".join(out) or str(name)


# ---------------------------------------------------------------------------
# Callbacks (they run before the script, so they may change any widget's state)
# ---------------------------------------------------------------------------

def _toggle(list_key: str, value: str, widget_key: str) -> None:
    picked: List[str] = list(st.session_state.get(list_key) or [])
    if st.session_state.get(widget_key):
        if value not in picked:
            picked.append(value)
    else:
        picked = [v for v in picked if v != value]
    st.session_state[list_key] = picked


def group_click(members: Sequence[str]) -> None:
    """A group chip: tick all of its funds, or untick them all when they are all ticked already."""
    picked: List[str] = list(st.session_state.get(nr.FUNDS_KEY) or [])
    if set(members) <= set(picked):
        picked = [t for t in picked if t not in set(members)]
    else:
        picked += [t for t in members if t not in picked]
    st.session_state[nr.FUNDS_KEY] = picked


def clear_funds() -> None:
    st.session_state[nr.FUNDS_KEY] = []


def select_all_rules(keys: Sequence[str]) -> None:
    """The Select All box: every check ticked, or none."""
    st.session_state[nr.RULES_KEY] = list(keys) if st.session_state.get("nr_rules_all") else []


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _dates(c: "nr.Choices", locked: bool) -> None:
    today = date.today()
    trading = c.kind == nr.KIND_TRADING
    st.html(ui.section("Date Range", aside="weekends snap to business days"))
    c.date_kind = st.segmented_control(
        "Dates", [nr.AS_OF, nr.RANGE], key=nr.KEYS["date_kind"], required=True, disabled=locked,
        format_func={nr.AS_OF: "Single day", nr.RANGE: "Date range"}.get, label_visibility="collapsed",
        width="stretch")
    if c.date_kind == nr.AS_OF:
        c.as_of = st.date_input("As of date", max_value=today, key=nr.KEYS["as_of"], disabled=locked,
                                help=_WEEKDAYS_HELP, format="YYYY/MM/DD")
        return
    start, end = st.columns(2)
    c.date_range = (start.date_input("Start Date", max_value=today, key=nr.START_KEY, disabled=locked,
                                     help=_WEEKDAYS_HELP, format="YYYY/MM/DD"),
                    end.date_input("End Date", max_value=today, key=nr.END_KEY, disabled=locked,
                                   format="YYYY/MM/DD"))
    if trading:  # stacked is EOD compliance only: show what will run
        st.session_state[nr.KEYS["date_mode"]] = nr.PER_DAY
    c.date_mode = st.segmented_control(
        "Output", [nr.PER_DAY, nr.STACKED], key=nr.KEYS["date_mode"], required=True,
        disabled=locked or trading, on_change=_stacked_turns_pdf_off, width="stretch",
        help="Stacked writes one compliance workbook over the whole range (End of day only).")
    if c.date_mode == nr.STACKED and not trading:
        c.daily_reports = st.checkbox("Also write daily reports", key=nr.KEYS["daily_reports"], disabled=locked)


def _stacked_turns_pdf_off() -> None:
    """``on_change`` of the Output control: like the CLI's time_series preset, a stacked run defaults to no PDF."""
    if st.session_state.get(nr.KEYS["date_mode"]) == nr.STACKED:
        st.session_state[nr.KEYS["create_pdf"]] = False


def _funds(catalog: dict, c: "nr.Choices", locked: bool) -> None:
    funds: List[str] = list(catalog["funds"])
    known = set(funds)
    picked = [t for t in st.session_state.get(nr.FUNDS_KEY) or [] if t in known]
    st.session_state[nr.FUNDS_KEY] = picked
    st.html(ui.section("Select Fund", len(picked), aside=f"{len(funds)} available"))
    cohorts: Dict[str, List[str]] = {n: [t for t in m if t in known] for n, m in catalog["cohorts"].items()}
    with st.container(key="nr_chips", horizontal=True, gap="small"):
        for name in sorted(cohorts, key=lambda n: (n != "ALL_FUNDS", n)):
            members = cohorts[name]
            if not members:
                continue
            full = set(members) <= set(picked)
            st.button(f"{group_label(name)} · {len(members)}", key=f"nr_chip_{name}", disabled=locked,
                      type="primary" if full else "secondary", on_click=group_click, args=(members,),
                      help=("Untick" if full else "Tick") + f" the {len(members)} funds of {name}")
        st.button("Clear", key="nr_chip__clear", disabled=locked or not picked, type="tertiary",
                  on_click=clear_funds)
    query = st.text_input("Filter funds", key=nr.FILTER_KEY, disabled=locked, placeholder="Search tickers…",
                          label_visibility="collapsed").strip().upper()
    shown = [t for t in funds if query in t] if query else funds
    if not shown:
        st.html(f'<p class="fm-hint">No ticker matches “{ui.esc(query)}”.</p>')
    cols = st.columns(FUND_COLUMNS, gap="small")
    for i, ticker in enumerate(shown):
        key = f"nrf_{ticker}"
        st.session_state[key] = ticker in picked
        cols[i % FUND_COLUMNS].checkbox(ticker, key=key, disabled=locked, width="stretch",
                                        on_change=_toggle, args=(nr.FUNDS_KEY, ticker, key))
    c.funds = picked


def _operations(c: "nr.Choices", locked: bool) -> None:
    st.html(ui.section("Operations"))
    c.kind = st.segmented_control("Run type", [nr.KIND_EOD, nr.KIND_TRADING], key=nr.KEYS["kind"], required=True,
                                  disabled=locked, width="stretch", label_visibility="collapsed",
                                  help="Trading compliance cannot be combined with the end-of-day operations.")
    if c.kind == nr.KIND_TRADING:
        st.html('<p class="fm-hint">Compares each fund\'s compliance before and after the day\'s trades '
                "(ex-ante vs ex-post). It runs on its own, with the checks below.</p>")
        return
    c.eod_ops = [op for op, label in nr.EOD_OPS.items()
                 if st.checkbox(OP_LABELS.get(op, label), key=f"nr_op_{op}", disabled=locked, width="stretch")]


def _rules(catalog: dict, c: "nr.Choices", locked: bool) -> None:
    rules = nr.check_rules(catalog)
    keys = [key for key, _ in rules]
    picked = list(st.session_state.get(nr.RULES_KEY) or [])
    head, every = st.columns([2.2, 1], vertical_alignment="center")
    head.html(ui.section("Compliance Checks", len([k for k in picked if k in keys])))
    st.session_state["nr_rules_all"] = set(keys) <= set(picked)
    every.checkbox("Select All", key="nr_rules_all", disabled=locked, on_change=select_all_rules, args=(keys,))
    for key, label in rules:
        widget = f"nrr_{key}"
        st.session_state[widget] = key in picked
        st.checkbox(label, key=widget, disabled=locked, width="stretch", on_change=_toggle,
                    args=(nr.RULES_KEY, key, widget))
    c.rules_mode, c.rules = nr.rules_choice(catalog, picked)


def _options(c: "nr.Choices", locked: bool) -> None:
    st.html(ui.section("Options"))
    pdf, email = st.columns(2)
    c.create_pdf = pdf.checkbox("Create PDF", key=nr.KEYS["create_pdf"], disabled=locked)
    c.send_email = email.checkbox("Send email", key=nr.KEYS["send_email"], disabled=locked,
                                  help="Off by default; needs a second confirmation for each run.")
    try:
        _, _, stacked = nr.run_dates(c)
        tag_hint = f"blank = {nr.default_tag(nr.operations_for(c), stacked)}"
    except nr.InvalidRequest:
        tag_hint = "optional, e.g. backdated"
    c.tag = st.text_input("Output tag", key=nr.KEYS["tag"], disabled=locked, max_chars=40, placeholder=tag_hint,
                          help="Goes into the report file names (blank = the CLI's name for this run type).")
    if c.send_email:
        st.html(ui.callout("warn", "This run will send the notification emails to their usual recipients "
                                   "(or open Outlook drafts on the server, depending on FUNDMANAGER_EMAIL_MODE)."))
        c.email_confirmed = st.checkbox("Yes, send the emails for this run", key=nr.KEYS["email_confirmed"],
                                        disabled=locked)


def form(catalog: dict, locked: bool) -> "nr.Choices":
    """Render every section (disabled while a run is active) and return what they hold."""
    c = nr.Choices(cohorts=[], exclude=[])
    # Run type first in the data flow (dates and checks depend on it), but drawn under the funds.
    c.kind = st.session_state.get(nr.KEYS["kind"]) or nr.KIND_EOD
    _dates(c, locked)
    st.html(ui.divider())
    _funds(catalog, c, locked)
    st.html(ui.divider())
    _operations(c, locked)
    if nr.uses_rules(c):
        st.html(ui.divider())
        _rules(catalog, c, locked)
    st.html(ui.divider())
    _options(c, locked)
    return c


def run_label(c: "nr.Choices") -> str:
    """The Run button's label for the picked operations."""
    ops = nr.operations_for(c)
    if "trading_compliance" in ops:
        return "Run trading compliance"
    if ops == ("compliance",):
        return "Run compliance checks"
    if "compliance" in ops:
        return "Run checks & recons"
    return "Run reconciliation" if ops else "Run"


def buttons(catalog: dict, c: "nr.Choices", *, can_run: bool, locked: bool) -> bool:
    """RUN (primary) and RESET, pinned at the foot of the panel (``nr_actions``); returns whether RUN was clicked."""
    with st.container(key="nr_actions"):
        st.html(ui.divider())
        run, reset = st.columns([2.3, 1], gap="small")
        clicked = run.button(run_label(c).upper(), type="primary", disabled=not can_run, key="nr_run",
                             width="stretch", icon=":material/play_arrow:")
        reset.button("RESET", key="nr_reset", disabled=locked, width="stretch", on_click=nr.reset_form,
                     args=(catalog,), help="Back to the defaults: no fund, every check, Compliance, the previous "
                                           "business day.")
    return clicked
