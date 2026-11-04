"""The Trading Compliance page: check the funds' proposed (staged, unexecuted) trades before they execute.

Left, the settings (the New run look): Trade date, Select Fund (group chips, a
"Staged trades" quick pick and a dot on the tiles of the funds that have staged
trades on the date, the search and the tiles), Compliance Checks, Options and
RUN TRADING CHECK / RESET. Right: before a run the preview (what ex-ante and
ex-post mean, the counts and the check guide); during a run the shared
progress panel; after it the dashboard (:mod:`ui.trading_results`) with the
downloads and Publish (:func:`ui.panels.results_panel`).

A trading run is the ``trading_compliance`` operation of one business day,
built with :func:`ui.new_run.build_request` (so it is validated by
``processing.api.validate`` like every run) and started by :mod:`ui.runner`
under the same one-run-per-output-root lock: while any run is active the form
is locked and that run's progress is shown. Email is off by default, needs the
second confirmation tick, and both are cleared once a run starts. Re-run of a
trading run (Run history) loads it here (``tc_prefill``), email off.

The staged-trades lookup (:func:`processing.api.staged_trades`, read-only, a
few seconds) is cached per date; when it fails the page says so and the funds
are picked by hand (a fund without staged trades is reported as "no proposed
trades").
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from processing import api
from processing.api import RunRequest
from ui import components as ui
from ui import history, new_run as nr, panels, preview, runner, settings_panel as sp
from utilities.workdays import business_day_offset, ensure_business_day

_log = logging.getLogger("fundmanager.ui")

PAGE = "Trading Compliance"
DATE_KEY, WATCH_KEY, PREFILL_KEY = "tc_date", "tc_watch", "tc_prefill"
NOTICE_KEY, NOTICE_ERROR_KEY, STARTED_KEY = "tc_notice", "tc_notice_error", "tc_started"
KEYS = sp.FormKeys(funds="tc_funds", rules="tc_rules", filter="tc_fund_filter", fund_tile="tcf_", rule_tile="tcr_",
                   chips="tc_chips", chip="tc_chip_", rules_all="tc_rules_all", pdf="tc_pdf", email="tc_email",
                   email_ok="tc_email_ok", tag="tc_tag")
CONFIRM_KEYS = (KEYS.email_ok,)  # never kept: a confirmation lasts only while it is on screen
KEPT_KEYS = (DATE_KEY, KEYS.funds, KEYS.rules, KEYS.filter, KEYS.pdf, KEYS.email, KEYS.tag)
STAGED_RETRY_SECONDS = 60
STAGED_KEY = "tc_staged"  # {date: lookup}: this session's copy, so a dashboard click never waits on the DB
STAGED_TTL = 300
EX_ANTE_TEXT = ("Ex-ante: the holdings as they stand on the trade date, before the trades (the checks use the "
                "trade-date holdings and custodian data; the dashboard's cash estimate starts from the previous "
                "business day's close).")
EX_POST_TEXT = "Ex-post: the same holdings assuming the staged but unexecuted trades execute."


def default_date(today: Optional[date] = None) -> date:
    """Today, or the Friday before on a weekend: proposed trades are checked on their trade date."""
    return ensure_business_day(today or date.today())


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def defaults(catalog: dict, today: Optional[date] = None) -> Dict[str, object]:
    """A fresh form: the trade date, no fund, every check, PDF on, email off."""
    return {DATE_KEY: default_date(today), KEYS.funds: [], KEYS.rules: [k for k, _ in nr.check_rules(catalog)],
            KEYS.filter: "", KEYS.pdf: True, KEYS.email: False, KEYS.email_ok: False, KEYS.tag: ""}


def keep_form_state() -> None:
    """Keep the trading picks while their widgets are not rendered (another page); see ``new_run.keep_form_state``."""
    for key in KEPT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def set_defaults(catalog: dict) -> None:
    for key, value in defaults(catalog).items():
        st.session_state.setdefault(key, value)


def clear_email() -> None:
    st.session_state[KEYS.email] = False
    for key in CONFIRM_KEYS:
        st.session_state[key] = False


def reset_form(catalog: dict) -> None:
    """RESET callback: every pick back to its default (email off)."""
    for key, value in defaults(catalog).items():
        st.session_state[key] = value
    clear_email()


def request_rerun(run_dir: Path) -> None:
    """Re-run of a trading run (Run history): load its request into this page and switch to it."""
    st.session_state[PREFILL_KEY] = str(run_dir)
    st.session_state["page"] = PAGE


def apply_prefill(catalog: dict) -> None:
    run_dir = st.session_state.pop(PREFILL_KEY, None)
    if not run_dir:
        return
    try:
        request = history.load_request(Path(run_dir))
    except (OSError, ValueError, TypeError) as exc:
        st.session_state[NOTICE_ERROR_KEY] = f"Could not load the request of {Path(run_dir).name}: {exc}"
        return
    choices = nr.choices_from_request(catalog, request)
    st.session_state[DATE_KEY] = request.start_date
    st.session_state[KEYS.funds] = nr.selected_funds(catalog, choices)
    st.session_state[KEYS.rules] = ([k for k, _ in nr.check_rules(catalog)] if choices.rules_mode == nr.ALL_RULES
                                    else list(choices.rules))
    st.session_state[KEYS.pdf] = request.create_pdf
    st.session_state[KEYS.tag] = request.output_tag if request.output_tag != "trading_compliance" else ""
    st.session_state[KEYS.filter] = ""
    clear_email()
    notice = f"Loaded the settings of {Path(run_dir).name}. Email is off; check the picks and press Run."
    if request.end_date and request.end_date > request.start_date:
        notice += (f" The original run covered {request.start_date} to {request.end_date}; this page checks one "
                   f"trade date at a time, so {request.start_date} is loaded.")
    st.session_state[NOTICE_KEY] = notice


# ---------------------------------------------------------------------------
# Staged trades (read-only DB lookup, cached per date)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=STAGED_TTL, show_spinner=False)
def _load_staged(day: str) -> dict:
    try:
        return {"funds": api.staged_trades(date.fromisoformat(day)), "error": None, "at": time.time()}
    except Exception as exc:  # the DB is down or misconfigured: pick funds by hand
        _log.warning("Staged trades lookup failed: %s", type(exc).__name__)  # no message: it may name the DB
        return {"funds": {}, "error": type(exc).__name__, "at": time.time()}


def staged_trades(day: date, *, keep: bool = False) -> dict:
    """``{"funds": {fund: counts}, "error": name | None}``; a failure is retried after a minute.

    The session keeps its own copy per date: with ``keep`` (a finished run's dashboard is on screen, so a click on
    it must not wait on the DB) that copy is reused however old; otherwise it is refreshed after
    :data:`STAGED_TTL` seconds. The lookup itself runs without a page-wide spinner.
    """
    memo = st.session_state.setdefault(STAGED_KEY, {})
    found = memo.get(day.isoformat())
    if found is not None and not found.get("error") and (keep or time.time() - found.get("at", 0) < STAGED_TTL):
        return found
    found = _load_staged(day.isoformat())
    if found.get("error") and time.time() - found.get("at", 0) > STAGED_RETRY_SECONDS:
        _load_staged.clear()
        found = _load_staged(day.isoformat())
    memo[day.isoformat()] = found
    return found


def staged_note(day: date, found: dict, catalog: dict) -> Tuple[List[str], str]:
    """(funds with staged trades that can be picked, the hint line under the chips)."""
    if found.get("error"):
        return [], (f"Could not look up the staged trades ({found['error']}); pick the funds or groups. "
                    "Funds without trades are reported as 'no proposed trades'.")
    known = set(catalog["funds"])
    funds = sorted(f for f, counts in (found.get("funds") or {}).items() if f in known and (counts or {}).get("total"))
    if not funds:
        return [], f"No staged trades found for {day}."
    return funds, f"{len(funds)} fund{'s' if len(funds) != 1 else ''} have staged trades on {day} (marked •)."


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

def form(catalog: dict, locked: bool, keep_staged: bool = False) -> Tuple["nr.Choices", List[str]]:
    """The settings panel; returns the picks and the funds with staged trades on the date."""
    today = date.today()
    st.html(ui.section("Trade Date", aside="proposed trades are checked on their trade date"))
    day = st.date_input("Trade date", key=DATE_KEY, max_value=today, disabled=locked, format="YYYY/MM/DD",
                        label_visibility="collapsed",
                        help="Weekends snap to the Friday before. Current cash and TNA come from the previous "
                             "business day's close.")
    run_day = ensure_business_day(day or default_date())
    st.html(f'<p class="fm-hint">Cash and TNA from the {business_day_offset(run_day, -1)} close · '
            f"ex-ante vs ex-post on {run_day}</p>")
    st.html(ui.divider())
    staged, note = staged_note(run_day, staged_trades(run_day, keep=keep_staged), catalog)
    c = nr.Choices(cohorts=[], exclude=[], kind=nr.KIND_TRADING, date_kind=nr.AS_OF, as_of=day)
    chips = [("Staged trades", staged, f"Tick the {len(staged)} funds with staged trades on {run_day}")] if staged else []
    sp.funds_section(catalog, c, locked, KEYS, extra_chips=chips, marked=staged, note=note)
    st.html(ui.divider())
    sp.rules_section(catalog, c, locked, KEYS)
    st.html(ui.divider())
    sp.options_section(c, locked, KEYS)
    return c, staged


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_html(catalog: dict, c: "nr.Choices", staged: Sequence[str], problem: Optional[str],
                 request: Optional[RunRequest], notes: Sequence[str]) -> str:
    """Cards (trade date, funds, checks, passes), what ex-ante / ex-post mean, then the problem or the summary."""
    counts = preview.counts(catalog, c)
    try:
        day, _, _ = nr.run_dates(c)
        day_note = f"{day:%A} · cash from {business_day_offset(day, -1)}"
    except nr.InvalidRequest:
        day, day_note = None, ""
    picked = nr.selected_funds(catalog, c) if c.funds else []
    with_trades = [f for f in picked if f in set(staged)]
    funds_note = ", ".join(picked[:6]) + (f" +{len(picked) - 6}" if len(picked) > 6 else "")
    if staged and picked:
        funds_note = f"{len(with_trades)} with staged trades · " + funds_note
    cards = [(f"{day:%b} {day.day}" if day else "—", "Trade Date", day_note),
             (counts.funds, "Funds Selected", funds_note),
             (counts.checks, "Checks Selected", "" if counts.checks else "none picked"),
             (2, "Passes", "ex-ante + ex-post")]
    body = ui.heading("Trading check preview") + ui.metric_cards(cards, accent=(1,))
    body += (f'<div class="fm-passes"><div><b>Before · ex-ante</b><span>{ui.esc(EX_ANTE_TEXT)}</span></div>'
             f'<div class="arrow" aria-hidden="true">→</div><div><b>After · ex-post</b>'
             f'<span>{ui.esc(EX_POST_TEXT)}</span></div></div>')
    estimate = None
    if counts.fund_days and counts.days:
        rates = preview._history_rates(str(runner.output_root()))
        seconds, from_history = preview.estimate_seconds(counts.fund_days, counts.days, rates)
        estimate = (seconds, from_history, len(rates))
    body += preview.sentence(counts, estimate)
    if problem:
        body += ui.callout("error", problem)
    elif request is not None:
        for note in notes:
            body += ui.callout("info", note)
        missing = [f for f in picked if f not in set(staged)] if staged else []
        if missing:
            body += ui.callout("info", f"{len(missing)} of the picked funds have no staged trades on this date; "
                                       "they will be listed as 'no proposed trades': " + ", ".join(missing[:12])
                               + (" …" if len(missing) > 12 else ""))
        body += (f'<p class="fm-hint" style="text-align:center">{ui.esc(ui.summary_text(runner.summarize(request)))}'
                 f"  ·  {ui.esc('Output tag “' + request.output_tag + '”')}</p>")
    return body


def next_run_html(catalog: dict, c: "nr.Choices", problem: Optional[str], request: Optional[RunRequest]) -> str:
    """After a run, a one-line summary of what RUN would start next (the dashboard keeps the space)."""
    counts = preview.counts(catalog, c)
    try:
        day = str(nr.run_dates(c)[0])
    except nr.InvalidRequest:
        day = "—"
    what = f"{counts.funds} fund{'s' if counts.funds != 1 else ''} · {day} · {counts.checks} checks"
    state = (f'<span class="bad">{ui.esc(problem)}</span>' if problem
             else '<span class="ok">ready: press Run trading check</span>' if request else "")
    return f'<div class="fm-nextrun"><span>Next check</span><b>{ui.esc(what)}</b>{state}</div>'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _start(request: RunRequest, user: str, root: Path) -> None:
    try:
        st.session_state[WATCH_KEY] = str(runner.start_run(request, user=user, root=root))
    except runner.RunLocked as exc:
        st.session_state[NOTICE_ERROR_KEY] = str(exc)
    except OSError as exc:
        st.session_state[NOTICE_ERROR_KEY] = f"Could not start the run: {exc}"
    else:
        st.session_state[STARTED_KEY] = True
    st.rerun()


def render(catalog: dict, user: str) -> None:
    """Settings | preview, progress or the dashboard of the last trading run started (or reattached) here."""
    st.title("Trading Compliance")
    root = runner.output_root()
    active = runner.active_run(root)
    running = active is not None and not panels.is_finished(active)
    foreign = running and not history.is_trading(active)  # an end-of-day run holds the lock
    if running and not foreign:
        st.session_state[WATCH_KEY] = str(active)  # reattach after a refresh, or show another user's run
    if st.session_state.pop(STARTED_KEY, False):
        clear_email()  # each emailing run needs its own confirmation
    set_defaults(catalog)
    apply_prefill(catalog)

    watch = st.session_state.get(WATCH_KEY)
    showing = bool(watch) and Path(watch).is_dir() and panels.is_finished(Path(watch))  # a dashboard is on screen
    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        with st.container(key="tc_settings"):
            choices, staged = form(catalog, locked=running, keep_staged=showing)
            request, problem = nr.evaluate(catalog, choices)
            if sp.buttons(catalog, choices, can_run=not running and request is not None, locked=running, prefix="tc",
                          reset=reset_form, reset_help="Back to the defaults: today's trades, no fund, every check, "
                                                       "email off."):
                _start(request, user, root)
    with right:
        for key, kind in ((NOTICE_KEY, "info"), (NOTICE_ERROR_KEY, "error")):
            if key in st.session_state:
                preview.notice(kind, st.session_state.pop(key))
        if foreign:
            ui.show(ui.callout("info", "An end-of-day run is in progress (New run); the form is locked until it "
                                       "finishes."))
            panels.progress_panel(str(active), user)
        watch = st.session_state.get(WATCH_KEY)
        live = bool(watch) and Path(watch).is_dir() and not panels.is_finished(Path(watch))
        if live:
            panels.progress_panel(watch, user)
        finished = bool(watch) and Path(watch).is_dir() and not live
        notes = nr.date_notes(choices, request) if request is not None else []
        if finished:
            ui.show(next_run_html(catalog, choices, None if running else problem, request))
            panels.results_panel(Path(watch), user, key="tc")
        else:
            ui.show(preview_html(catalog, choices, staged, None if running else problem, request, notes))
            if not live:
                preview.rule_guide(catalog, choices)
