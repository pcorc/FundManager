"""The New run settings panel (left pane): Date Range, Select Fund, Operations, Compliance Checks, Options, buttons.

The fund and check tiles are checkboxes keyed ``nrf_<ticker>`` / ``nrr_<rule>``
(styled as tiles by :mod:`ui.theme`). Their truth is the lists
``new_run.FUNDS_KEY`` / ``new_run.RULES_KEY``: each tile's state is set from the
list before it is drawn and its ``on_change`` callback updates the list, so a
group chip, *Select All*, the filter (which hides tiles) and RESET all act on
the same picks.

The Select Fund, Compliance Checks and Options sections take a :class:`FormKeys`
(the session keys they use), so the Trading Compliance page
(:mod:`ui.trading_page`) draws the same sections on its own picks (``tcf_`` /
``tcr_`` tiles). New run is end-of-day only; trading has its own page.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import streamlit as st

from ui import components as ui
from ui import new_run as nr

FUND_COLUMNS = 3
OP_LABELS = {"compliance": "Compliance checks", "reconciliation": "Holdings reconciliation",
             "nav_reconciliation": "NAV reconciliation"}
_ACRONYMS = {"ETF", "VIT", "NAV", "OMS"}
_SMALL_WORDS = {"AND", "OF", "THE"}
_WEEKDAYS_HELP = "Weekends snap to business days. There is no holiday calendar: a market holiday still runs."
TRADING_HINT = ("Trading compliance (the checks before and after the staged trades) has its own page: "
                "Trading Compliance, in the top bar.")


@dataclass(frozen=True)
class FormKeys:
    """The session keys one page's fund / check / option sections use (tile prefixes are styled by the theme)."""

    funds: str
    rules: str
    filter: str
    fund_tile: str
    rule_tile: str
    chips: str
    chip: str
    rules_all: str
    pdf: str
    email: str
    email_ok: str
    tag: str


# Literal keys (new_run imports this module, so its constants are not set yet); new_run.KEYS has the same names.
NR_KEYS = FormKeys(funds="nr_funds", rules="nr_rules", filter="nr_fund_filter", fund_tile="nrf_", rule_tile="nrr_",
                   chips="nr_chips", chip="nr_chip_", rules_all="nr_rules_all", pdf="nr_pdf", email="nr_email",
                   email_ok="nr_email_ok", tag="nr_tag")


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


def group_click(members: Sequence[str], list_key: str = "nr_funds") -> None:
    """A group chip: tick all of its funds, or untick them all when they are all ticked already."""
    picked: List[str] = list(st.session_state.get(list_key) or [])
    if set(members) <= set(picked):
        picked = [t for t in picked if t not in set(members)]
    else:
        picked += [t for t in members if t not in picked]
    st.session_state[list_key] = picked


def clear_funds(list_key: str = "nr_funds") -> None:
    st.session_state[list_key] = []


def select_all_rules(keys: Sequence[str], list_key: str = "nr_rules", all_key: str = "nr_rules_all") -> None:
    """The Select All box: every check ticked, or none."""
    st.session_state[list_key] = list(keys) if st.session_state.get(all_key) else []


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _dates(c: "nr.Choices", locked: bool) -> None:
    today = date.today()
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
    c.date_mode = st.segmented_control(
        "Output", [nr.PER_DAY, nr.STACKED], key=nr.KEYS["date_mode"], required=True,
        disabled=locked, on_change=_stacked_turns_pdf_off, width="stretch",
        help="Stacked writes one compliance workbook over the whole range.")
    if c.date_mode == nr.STACKED:
        c.daily_reports = st.checkbox("Also write daily reports", key=nr.KEYS["daily_reports"], disabled=locked)


def _stacked_turns_pdf_off() -> None:
    """``on_change`` of the Output control: like the CLI's time_series preset, a stacked run defaults to no PDF."""
    if st.session_state.get(nr.KEYS["date_mode"]) == nr.STACKED:
        st.session_state[nr.KEYS["create_pdf"]] = False


def tile_marks(prefix: str, tickers: Sequence[str]) -> str:
    """A style block putting a small accent dot on the named tiles (only letter-and-digit tickers are used)."""
    safe = [t for t in tickers if str(t).isalnum()]
    if not safe:
        return ""
    labels = ", ".join(f".st-key-{prefix}{t} label" for t in safe)
    dots = ", ".join(f".st-key-{prefix}{t} label::after" for t in safe)
    # In the tile's right padding, so the ticker keeps its full width.
    return (f"<style>{labels} {{ position: relative; }} {dots} {{ content: \"\"; position: absolute; right: .32rem; "
            "top: 50%; margin-top: -3px; width: 6px; height: 6px; border-radius: 50%; "
            "background: var(--fm-accent-text); }</style>")


def funds_section(catalog: dict, c: "nr.Choices", locked: bool, keys: FormKeys = NR_KEYS, *,
                  extra_chips: Sequence[Tuple[str, Sequence[str], str]] = (), marked: Sequence[str] = (),
                  note: str = "") -> None:
    """Select Fund: group chips (``extra_chips`` first, as ``(label, members, help)``), the search, the tiles.

    ``marked`` tickers get a small dot on their tile (the Trading page marks the funds with staged trades);
    ``note`` is a hint line under the chips.
    """
    funds: List[str] = list(catalog["funds"])
    known = set(funds)
    picked = [t for t in st.session_state.get(keys.funds) or [] if t in known]
    st.session_state[keys.funds] = picked
    st.html(ui.section("Select Fund", len(picked), aside=f"{len(funds)} available"))
    cohorts: Dict[str, List[str]] = {n: [t for t in m if t in known] for n, m in catalog["cohorts"].items()}
    with st.container(key=keys.chips, horizontal=True, gap="small"):
        for n, (label, members, tip) in enumerate(extra_chips):
            members = [t for t in members if t in known]
            if members:
                full = set(members) <= set(picked)
                st.button(f"{label} · {len(members)}", key=f"{keys.chip}_extra{n}", disabled=locked,
                          type="primary" if full else "secondary", on_click=group_click,
                          args=(members, keys.funds), help=tip)
        for name in sorted(cohorts, key=lambda n: (n != "ALL_FUNDS", n)):
            members = cohorts[name]
            if not members:
                continue
            full = set(members) <= set(picked)
            st.button(f"{group_label(name)} · {len(members)}", key=f"{keys.chip}{name}", disabled=locked,
                      type="primary" if full else "secondary", on_click=group_click, args=(members, keys.funds),
                      help=("Untick" if full else "Tick") + f" the {len(members)} funds of {name}")
        st.button("Clear", key=f"{keys.chip}_clear", disabled=locked or not picked, type="tertiary",
                  on_click=clear_funds, args=(keys.funds,))
    if note:
        st.html(f'<p class="fm-hint">{ui.esc(note)}</p>')
    query = st.text_input("Filter funds", key=keys.filter, disabled=locked, placeholder="Search tickers…",
                          label_visibility="collapsed").strip().upper()
    shown = [t for t in funds if query in t] if query else funds
    if not shown:
        st.html(f'<p class="fm-hint">No ticker matches “{ui.esc(query)}”.</p>')
    marks = tile_marks(keys.fund_tile, [t for t in shown if t in set(marked)])
    if marks:
        st.html(marks)
    cols = st.columns(FUND_COLUMNS, gap="small")
    for i, ticker in enumerate(shown):
        key = f"{keys.fund_tile}{ticker}"
        st.session_state[key] = ticker in picked
        cols[i % FUND_COLUMNS].checkbox(ticker, key=key, disabled=locked, width="stretch",
                                        on_change=_toggle, args=(keys.funds, ticker, key))
    c.funds = picked


def _operations(c: "nr.Choices", locked: bool) -> None:
    st.html(ui.section("Operations", aside="end of day"))
    c.kind = nr.KIND_EOD
    c.eod_ops = [op for op, label in nr.EOD_OPS.items()
                 if st.checkbox(OP_LABELS.get(op, label), key=f"nr_op_{op}", disabled=locked, width="stretch")]
    st.html(f'<p class="fm-hint">{ui.esc(TRADING_HINT)}</p>')


def rules_section(catalog: dict, c: "nr.Choices", locked: bool, keys: FormKeys = NR_KEYS) -> None:
    """Compliance Checks: Select All and one tile per rule."""
    rules = nr.check_rules(catalog)
    rule_keys = [key for key, _ in rules]
    picked = list(st.session_state.get(keys.rules) or [])
    head, every = st.columns([2.2, 1], vertical_alignment="center")
    head.html(ui.section("Compliance Checks", len([k for k in picked if k in rule_keys])))
    st.session_state[keys.rules_all] = set(rule_keys) <= set(picked)
    every.checkbox("Select All", key=keys.rules_all, disabled=locked, on_change=select_all_rules,
                   args=(rule_keys, keys.rules, keys.rules_all))
    for key, label in rules:
        widget = f"{keys.rule_tile}{key}"
        st.session_state[widget] = key in picked
        st.checkbox(label, key=widget, disabled=locked, width="stretch", on_change=_toggle,
                    args=(keys.rules, key, widget))
    c.rules_mode, c.rules = nr.rules_choice(catalog, picked)


def options_section(c: "nr.Choices", locked: bool, keys: FormKeys = NR_KEYS) -> None:
    """Options: Create PDF, Send email (plus its confirmation tick) and the output tag."""
    st.html(ui.section("Options"))
    pdf, email = st.columns(2)
    c.create_pdf = pdf.checkbox("Create PDF", key=keys.pdf, disabled=locked)
    c.send_email = email.checkbox("Send email", key=keys.email, disabled=locked,
                                  help="Off by default; needs a second confirmation for each run.")
    try:
        _, _, stacked = nr.run_dates(c)
        tag_hint = f"blank = {nr.default_tag(nr.operations_for(c), stacked)}"
    except nr.InvalidRequest:
        tag_hint = "optional, e.g. backdated"
    c.tag = st.text_input("Output tag", key=keys.tag, disabled=locked, max_chars=40, placeholder=tag_hint,
                          help="Goes into the report file names (blank = the CLI's name for this run type).")
    if c.send_email:
        st.html(ui.callout("warn", "This run will send the notification emails to their usual recipients "
                                   "(or open Outlook drafts on the server, depending on FUNDMANAGER_EMAIL_MODE)."))
        c.email_confirmed = st.checkbox("Yes, send the emails for this run", key=keys.email_ok, disabled=locked)


def form(catalog: dict, locked: bool) -> "nr.Choices":
    """Render every section (disabled while a run is active) and return what they hold."""
    c = nr.Choices(cohorts=[], exclude=[], kind=nr.KIND_EOD)  # trading runs on its own page
    _dates(c, locked)
    st.html(ui.divider())
    funds_section(catalog, c, locked)
    st.html(ui.divider())
    _operations(c, locked)
    if nr.uses_rules(c):
        st.html(ui.divider())
        rules_section(catalog, c, locked)
    st.html(ui.divider())
    options_section(c, locked)
    return c


def run_label(c: "nr.Choices") -> str:
    """The Run button's label for the picked operations."""
    ops = nr.operations_for(c)
    if "trading_compliance" in ops:
        return "Run trading check"
    if ops == ("compliance",):
        return "Run compliance checks"
    if "compliance" in ops:
        return "Run checks & recons"
    return "Run reconciliation" if ops else "Run"


_RESET_HELP = "Back to the defaults: no fund, every check, Compliance, the previous business day."


def buttons(catalog: dict, c: "nr.Choices", *, can_run: bool, locked: bool, prefix: str = "nr",
            reset: Optional[Callable[[dict], None]] = None, reset_help: str = _RESET_HELP) -> bool:
    """RUN (primary) and RESET, pinned at the foot of the panel (``<prefix>_actions``); True when RUN was clicked."""
    with st.container(key=f"{prefix}_actions"):
        st.html(ui.divider())
        run, reset_col = st.columns([2.3, 1], gap="small")
        clicked = run.button(run_label(c).upper(), type="primary", disabled=not can_run, key=f"{prefix}_run",
                             width="stretch", icon=":material/play_arrow:")
        reset_col.button("RESET", key=f"{prefix}_reset", disabled=locked, width="stretch",
                         on_click=reset or nr.reset_form, args=(catalog,), help=reset_help)
    return clicked
