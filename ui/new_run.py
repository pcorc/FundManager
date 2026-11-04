"""The New run page (end of day): settings on the left; on the right a live run's progress, the preview, then results.

Trading compliance has its own page (:mod:`ui.trading_page`), which reuses
:class:`Choices` and :func:`build_request` with ``kind=KIND_TRADING``; Re-run of a
trading run is routed there.

The settings panel (:mod:`ui.settings_panel`) holds plain widgets (not an
``st.form``) so the counts and the validation update as you pick.
:func:`build_request` turns the picks into a ``RunRequest`` and
``processing.api.validate`` checks it in-process before the Run button is
enabled; problems show inline in the preview (:mod:`ui.preview`). Only one run
happens per output root, so while any run is active the controls are disabled
and its progress panel is shown (a refreshed browser reattaches through the
lock file).

The picked funds and compliance checks live in session state as lists
(``nr_funds``, ``nr_rules``); their checkbox tiles are synced from those lists
on every run, so the group chips, *Select All*, the ticker filter, RESET and
Re-run can change them.

Email: a run sends nothing unless *Send email* and its confirmation are
ticked, and both are cleared once a run starts.

Dates are snapped to business days here (weekends only; there is no holiday
calendar), so the request, the preview, history and the Reports page all name
the days that actually run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from config.run_configurations import FULL_COMPLIANCE_TESTS
from processing.api import InvalidRequest, RunRequest, select_funds, validate
from ui import history, panels, preview, runner, settings_panel
from utilities.workdays import business_day_offset, ensure_business_day, iter_business_days

EOD_OPS = {"compliance": "Compliance", "reconciliation": "Holdings recon", "nav_reconciliation": "NAV recon"}
RECON_OPS = ("reconciliation", "nav_reconciliation")
KIND_EOD, KIND_TRADING = "End of day", "Trading compliance"
ALL_RULES, CHOOSE_RULES = "All rules", "Choose rules"
AS_OF, RANGE = "As of", "Range"
PER_DAY, STACKED = "One report per day", "One stacked workbook"
# The tag `python main.py <mode>` uses when none is given, so UI and CLI reports get the same file names.
_CLI_TAGS = {("compliance",): "compliance", ("reconciliation",): "reconciliation", ("nav_reconciliation",): "nav"}


@dataclass
class Choices:
    """What the form holds, independent of Streamlit (so it can be unit-tested and pre-filled).

    Empty ``funds`` and ``cohorts`` mean every fund (the API's rule); the page itself always sends the
    ticked tickers and refuses an empty pick.
    """

    cohorts: List[str] = field(default_factory=list)
    funds: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    kind: str = KIND_EOD
    eod_ops: List[str] = field(default_factory=lambda: ["compliance"])
    rules_mode: str = ALL_RULES
    rules: List[str] = field(default_factory=list)
    date_kind: str = AS_OF
    as_of: Optional[date] = None
    date_range: Tuple[date, ...] = ()
    date_mode: str = PER_DAY
    daily_reports: bool = False
    create_pdf: bool = True
    send_email: bool = False
    email_confirmed: bool = False  # the "Yes, send" tick
    tag: str = ""


def default_as_of(today: Optional[date] = None) -> date:
    """The previous business day."""
    return business_day_offset(today or date.today(), -1)


def next_business_day(value: date) -> date:
    """``value`` if it is Mon-Fri, else the following Monday."""
    return value if value.weekday() < 5 else business_day_offset(value, 1)


def selected_funds(catalog: dict, choices: Choices) -> List[str]:
    """The tickers the picks resolve to (same rule as ``processing.api.select_funds``)."""
    cohorts = catalog["cohorts"]
    if choices.cohorts:
        return select_funds([set(cohorts.get(name, ())) for name in choices.cohorts], choices.funds, choices.exclude)
    return select_funds(funds=choices.funds or catalog["funds"], exclude=choices.exclude)


def check_rules(catalog: dict) -> List[Tuple[str, str]]:
    """``(key, label)`` of the compliance checks offered as tiles: the tests of 'All rules' less the summary sheet."""
    return [(r["key"], r["label"]) for r in catalog["rule_catalog"]
            if r["key"] in FULL_COMPLIANCE_TESTS and r["key"] != "summary_metrics"]


def default_tag(operations: Sequence[str], stacked: bool) -> str:
    """The output tag the CLI preset for these operations uses (``eod``, ``compliance``, ``time_series`` …)."""
    if "trading_compliance" in operations:
        return "trading_compliance"
    if stacked:
        return "time_series"
    return _CLI_TAGS.get(tuple(operations), "eod")


def operations_for(choices: Choices) -> Tuple[str, ...]:
    return ("trading_compliance",) if choices.kind == KIND_TRADING else tuple(
        op for op in EOD_OPS if op in choices.eod_ops)


def uses_rules(choices: Choices) -> bool:
    """Whether the compliance checks apply (Trading, or End of day with Compliance ticked)."""
    return choices.kind == KIND_TRADING or "compliance" in choices.eod_ops


def run_dates(choices: Choices) -> Tuple[date, date, bool]:
    """(start, end, stacked) snapped to business days; raises ``InvalidRequest`` for an empty or reversed range.

    An as-of weekend day runs the Friday before; a range starts on its first and ends on its last business day.
    """
    if choices.date_kind == RANGE:
        if len(choices.date_range) != 2 or None in choices.date_range:
            raise InvalidRequest("Pick both a start and an end date for the range.")
        picked_start, picked_end = choices.date_range
        if picked_end < picked_start:
            raise InvalidRequest(f"The end date {picked_end} is before the start date {picked_start}.")
        start, end = next_business_day(picked_start), ensure_business_day(picked_end)
        if end < start:
            raise InvalidRequest(f"There are no business days between {picked_start} and {picked_end}.")
        return start, end, choices.date_mode == STACKED and choices.kind == KIND_EOD
    day = ensure_business_day(choices.as_of or default_as_of())
    return day, day, False


def build_request(catalog: dict, choices: Choices) -> RunRequest:
    """The ``RunRequest`` for ``choices``, validated; raises ``InvalidRequest`` with a readable message."""
    operations = operations_for(choices)
    trading = choices.kind == KIND_TRADING
    if not operations:
        raise InvalidRequest("Pick at least one operation.")
    if not selected_funds(catalog, choices):
        raise InvalidRequest("The fund selection is empty: the chosen groups, funds and exclusions leave no fund.")
    start, end, stacked = run_dates(choices)
    if stacked and not choices.daily_reports:
        if "compliance" not in operations:
            raise InvalidRequest("A stacked run writes only the compliance workbook: tick Compliance, or "
                                 "'Also write daily reports' for the recon reports.")
        if any(op in operations for op in RECON_OPS):
            raise InvalidRequest("A stacked run writes only the compliance workbook, so the recon reports would be "
                                 "lost: untick Holdings / NAV recon, or tick 'Also write daily reports'.")
    tests: Sequence[str] = tuple(FULL_COMPLIANCE_TESTS)  # "All rules" = the RunRequest default
    if (trading or "compliance" in operations) and choices.rules_mode == CHOOSE_RULES:
        if not choices.rules:
            raise InvalidRequest("Pick at least one compliance check, or use Select All.")
        tests = tuple(choices.rules)
    if choices.send_email and not choices.email_confirmed:
        raise InvalidRequest("Tick the confirmation to send notification emails, or switch email off.")
    request = RunRequest(
        operations=operations, start_date=start, end_date=end,
        cohorts=tuple(choices.cohorts), funds=tuple(choices.funds), exclude=tuple(choices.exclude),
        compliance_tests=tuple(tests), date_mode="stacked" if stacked else "per_day",
        generate_daily_reports=bool(stacked and choices.daily_reports), create_pdf=choices.create_pdf,
        send_email=bool(choices.send_email and choices.email_confirmed),
        output_tag=runner.safe_name(choices.tag, default="") or default_tag(operations, stacked),
    )
    return validate(request)


def business_days(start: date, end: date) -> List[date]:
    """The weekdays in ``[start, end]`` (the request's dates are already snapped to business days)."""
    return list(iter_business_days(start, end))


def choices_from_request(catalog: dict, request: RunRequest) -> Choices:
    """Pre-fill for Re-run. Email (and its confirmation) always comes back off; unknown cohorts/funds are dropped."""
    known = set(catalog["funds"])
    tickers = lambda values: [t for v in values for t in str(v).replace(",", " ").upper().split() if t in known]  # noqa: E731
    trading = "trading_compliance" in request.operations
    tests = set(request.compliance_tests)
    end = request.end_date or request.start_date
    ranged = request.date_mode == "stacked" or end != request.start_date
    return Choices(
        cohorts=[c for c in request.cohorts if c in catalog["cohorts"]],
        funds=tickers(request.funds), exclude=tickers(request.exclude),
        kind=KIND_TRADING if trading else KIND_EOD,
        eod_ops=[op for op in EOD_OPS if op in request.operations] or ["compliance"],
        rules_mode=ALL_RULES if not tests or tests == set(FULL_COMPLIANCE_TESTS) else CHOOSE_RULES,
        rules=[r["key"] for r in catalog["rule_catalog"] if r["key"] in tests],
        date_kind=RANGE if ranged else AS_OF,
        as_of=request.start_date,
        date_range=(request.start_date, end) if ranged else (),
        date_mode=STACKED if request.date_mode == "stacked" else PER_DAY,
        daily_reports=request.generate_daily_reports, create_pdf=request.create_pdf,
        tag=request.output_tag or "",
    )


def rules_choice(catalog: dict, picked: Sequence[str]) -> Tuple[str, List[str]]:
    """(rules_mode, rules) for the ticked checks: exactly every tile ticked is 'All rules'."""
    if set(picked) == {key for key, _ in check_rules(catalog)}:
        return ALL_RULES, []
    order = {r["key"]: i for i, r in enumerate(catalog["rule_catalog"])}
    return CHOOSE_RULES, sorted(dict.fromkeys(picked), key=lambda k: order.get(k, len(order)))


def evaluate(catalog: dict, choices: Choices) -> Tuple[Optional[RunRequest], Optional[str]]:
    """(request, None) when the picks make a valid run, else (None, the reason)."""
    if not choices.funds and not choices.cohorts:
        return None, "Pick at least one fund: tick its tile, or use a group chip."
    try:
        return build_request(catalog, choices), None
    except (InvalidRequest, ValueError) as exc:
        return None, str(exc)


def date_notes(choices: Choices, request: RunRequest) -> List[str]:
    """Say when a picked date is not the day that runs (weekends)."""
    picked = [choices.as_of] if choices.date_kind == AS_OF else list(choices.date_range)
    actual = [request.start_date] if choices.date_kind == AS_OF else [request.start_date, request.end_date]
    return [f"{p:%A} {p} is not a business day: the run uses {a}." for p, a in zip(picked, actual)
            if p is not None and p != a]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# Plain widget keys, by Choices field. The fund and rule picks are the lists FUNDS_KEY / RULES_KEY.
KEYS = {"kind": "nr_kind", "date_kind": "nr_date_kind", "as_of": "nr_as_of", "date_mode": "nr_date_mode",
        "daily_reports": "nr_daily", "create_pdf": "nr_pdf", "send_email": "nr_email",
        "email_confirmed": "nr_email_ok", "tag": "nr_tag"}
FUNDS_KEY, RULES_KEY, START_KEY, END_KEY, FILTER_KEY = "nr_funds", "nr_rules", "nr_start", "nr_end", "nr_fund_filter"
OP_KEYS = tuple(f"nr_op_{op}" for op in EOD_OPS)
WATCH_KEY = "nr_watch"  # the run this page shows (started here, or reattached by ui.app.reattach_active_run)
CONFIRM_KEYS = ("nr_email_ok",)  # never kept: a confirmation lasts only while it is on screen
KEPT_KEYS = (tuple(k for k in KEYS.values() if k not in CONFIRM_KEYS) + OP_KEYS
             + (FUNDS_KEY, RULES_KEY, START_KEY, END_KEY, FILTER_KEY))
PANE_KEY, RESULTS, GUIDE = "nr_pane", "Results", "Check guide"


def keep_form_state() -> None:
    """Keep the picks while their widgets are hidden (another page, Trading, As of…); call at the top of each run.

    Streamlit drops the state of a keyed widget that is not rendered in a run; re-assigning the key makes
    it plain session state for that run, so it survives. Confirmation ticks are deliberately not kept.
    """
    for key in KEPT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def request_rerun(run_dir: Path) -> None:
    """Button callback for Re-run: load that run's request into its form (a trading run's on the Trading page)."""
    if history.is_trading(Path(run_dir)):
        from ui import trading_page  # noqa: PLC0415 (trading_page imports this module)

        trading_page.request_rerun(Path(run_dir))
        return
    st.session_state["nr_prefill"] = str(run_dir)
    st.session_state["page"] = "New run"


def defaults(catalog: dict, today: Optional[date] = None) -> Dict[str, object]:
    """A fresh form: no fund ticked, every check ticked, Compliance, the previous business day, email off."""
    as_of = default_as_of(today or date.today())
    base = Choices()
    values: Dict[str, object] = {key: getattr(base, name) for name, key in KEYS.items()}
    values.update({KEYS["as_of"]: as_of, START_KEY: business_day_offset(as_of, -4), END_KEY: as_of,
                   FUNDS_KEY: [], RULES_KEY: [key for key, _ in check_rules(catalog)], FILTER_KEY: ""})
    values.update({f"nr_op_{op}": op == "compliance" for op in EOD_OPS})
    return values


def set_defaults(catalog: dict) -> None:
    """Seed the state of widgets not set yet, so widgets never get both a ``value=`` and a Session State value."""
    for key, value in defaults(catalog).items():
        st.session_state.setdefault(key, value)


def reset_form(catalog: dict) -> None:
    """RESET button callback: every pick back to its default (email off)."""
    for key, value in defaults(catalog).items():
        st.session_state[key] = value
    clear_email()


def clear_email() -> None:
    """Untick Send email and its confirmation (before their widgets are rendered in this run)."""
    st.session_state[KEYS["send_email"]] = False
    for key in CONFIRM_KEYS:
        st.session_state[key] = False


def _apply_prefill(catalog: dict) -> None:
    run_dir = st.session_state.pop("nr_prefill", None)
    if not run_dir:
        return
    try:
        choices = choices_from_request(catalog, history.load_request(Path(run_dir)))
    except (OSError, ValueError, TypeError) as exc:
        st.session_state["nr_notice_error"] = f"Could not load the request of {Path(run_dir).name}: {exc}"
        return
    for name, key in KEYS.items():
        st.session_state[key] = getattr(choices, name)
    if choices.date_range:
        st.session_state[START_KEY], st.session_state[END_KEY] = choices.date_range
    for op in EOD_OPS:
        st.session_state[f"nr_op_{op}"] = op in choices.eod_ops
    st.session_state[FUNDS_KEY] = selected_funds(catalog, choices)  # groups and exclusions become ticks
    st.session_state[RULES_KEY] = ([key for key, _ in check_rules(catalog)] if choices.rules_mode == ALL_RULES
                                   else list(choices.rules))
    st.session_state[FILTER_KEY] = ""
    clear_email()
    st.session_state["nr_notice"] = (f"Loaded the settings of {Path(run_dir).name}. "
                                     "Email is off; check the picks and press Run.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _start(request: RunRequest, user: str, root: Path) -> None:
    try:
        st.session_state[WATCH_KEY] = str(runner.start_run(request, user=user, root=root))
    except runner.RunLocked as exc:
        st.session_state["nr_notice_error"] = str(exc)
    except OSError as exc:
        st.session_state["nr_notice_error"] = f"Could not start the run: {exc}"
    else:
        st.session_state["nr_started"] = True
    st.rerun()


def render(catalog: dict, user: str) -> None:
    """The whole page: settings | preview, validation, Run, then the active run's progress or the last results."""
    st.title("New run")
    root = runner.output_root()
    active = runner.active_run(root)
    running = active is not None and not panels.is_finished(active)
    foreign = running and history.is_trading(active)  # a trading run (Trading Compliance page) holds the lock
    if running and not foreign:
        st.session_state[WATCH_KEY] = str(active)  # reattach after a refresh, or show another user's run
    if st.session_state.pop("nr_started", False):
        clear_email()  # each emailing run needs its own confirmation
        st.session_state[PANE_KEY] = RESULTS
    set_defaults(catalog)
    _apply_prefill(catalog)

    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        with st.container(key="nr_settings"):
            choices = settings_panel.form(catalog, locked=running)
            request, problem = evaluate(catalog, choices)
            if settings_panel.buttons(catalog, choices, can_run=not running and request is not None, locked=running):
                _start(request, user, root)
    with right:
        for key, kind in (("nr_notice", "info"), ("nr_notice_error", "error")):
            if key in st.session_state:
                preview.notice(kind, st.session_state.pop(key))
        if foreign:
            preview.notice("info", "A trading compliance run is in progress (Trading Compliance page); the form is "
                                   "locked until it finishes.")
            panels.progress_panel(str(active), user)
        watch = st.session_state.get(WATCH_KEY)
        live = bool(watch) and Path(watch).is_dir() and not panels.is_finished(Path(watch))
        if live:  # first in the pane, so it is in view the moment RUN is pressed
            panels.progress_panel(watch, user)
        preview.render(catalog, choices, request, None if running else problem,
                       date_notes(choices, request) if request is not None else [])
        if not live and watch and Path(watch).is_dir():
            st.session_state.setdefault(PANE_KEY, RESULTS)
            pane = st.segmented_control("Show", [RESULTS, GUIDE], key=PANE_KEY, required=True,
                                        label_visibility="collapsed")
            if pane == GUIDE:
                preview.rule_guide(catalog, choices)
            else:
                panels.results_panel(Path(watch), user, key="nr")
        elif not live:
            preview.rule_guide(catalog, choices)
