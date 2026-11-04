"""The New run page: pick funds, operations, rules and dates, validate inline, launch, watch, download.

Widgets are plain (not an ``st.form``) so the fund-count badge and the
validation message update as you pick. :func:`build_request` turns the picks
into a ``RunRequest`` and ``processing.api.validate`` checks it in-process
before the Run button is enabled. Only one run happens per output root, so
while any run is active the controls are disabled and its progress panel is
shown (a refreshed browser reattaches through the lock file).

Email: End-of-day runs send nothing unless *Send email* and its confirmation
are ticked, and both are cleared once a run starts. Trading compliance always
sends (or drafts) the trading-analysis email, because the backend's trading
path has no switch for it, so the page says so and requires the confirmation
before a trading run.

Dates are snapped to business days here (weekends only; there is no holiday
calendar), so the request, the caption, history and the Reports page all name
the days that actually run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from config.run_configurations import FULL_COMPLIANCE_TESTS
from processing.api import InvalidRequest, RunRequest, select_funds, validate
from ui import history, panels, runner
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
    """What the form holds, independent of Streamlit (so it can be unit-tested and pre-filled)."""

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
    send_email: bool = False  # End of day only: trading always sends
    email_confirmed: bool = False  # the "Yes, send" tick (EOD email, or the trading email)
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


def run_dates(choices: Choices) -> Tuple[date, date, bool]:
    """(start, end, stacked) snapped to business days; raises ``InvalidRequest`` for an empty or reversed range.

    An as-of weekend day runs the Friday before; a range starts on its first and ends on its last business day.
    """
    if choices.date_kind == RANGE:
        if len(choices.date_range) != 2:
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
            raise InvalidRequest("Pick at least one compliance rule, or choose 'All rules'.")
        tests = tuple(choices.rules)
    if trading and not choices.email_confirmed:
        raise InvalidRequest("Trading compliance always sends the trading-analysis email (the backend has no "
                             "switch to turn it off): tick the confirmation to run it.")
    if not trading and choices.send_email and not choices.email_confirmed:
        raise InvalidRequest("Tick the confirmation to send notification emails, or switch email off.")
    request = RunRequest(
        operations=operations, start_date=start, end_date=end,
        cohorts=tuple(choices.cohorts), funds=tuple(choices.funds), exclude=tuple(choices.exclude),
        compliance_tests=tuple(tests), date_mode="stacked" if stacked else "per_day",
        generate_daily_reports=bool(stacked and choices.daily_reports), create_pdf=choices.create_pdf,
        send_email=trading or bool(choices.send_email and choices.email_confirmed),  # trading: what really happens
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


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_KEYS = {"cohorts": "nr_cohorts", "funds": "nr_funds", "exclude": "nr_exclude", "kind": "nr_kind",
         "rules_mode": "nr_rules_mode", "rules": "nr_rules", "date_kind": "nr_date_kind", "as_of": "nr_as_of",
         "date_range": "nr_range", "date_mode": "nr_date_mode", "daily_reports": "nr_daily",
         "create_pdf": "nr_pdf", "send_email": "nr_email", "email_confirmed": "nr_email_ok", "tag": "nr_tag"}
_OP_KEYS = tuple(f"nr_op_{op}" for op in EOD_OPS)
_CONFIRM_KEYS = ("nr_email_ok", "nr_trading_ok")  # never kept: a confirmation lasts only while it is on screen
_KEPT_KEYS = tuple(k for k in _KEYS.values() if k not in _CONFIRM_KEYS) + _OP_KEYS


def keep_form_state() -> None:
    """Keep the picks while their widgets are hidden (another page, Trading, As of…); call at the top of each run.

    Streamlit drops the state of a keyed widget that is not rendered in a run; re-assigning the key makes
    it plain session state for that run, so it survives. Confirmation ticks are deliberately not kept.
    """
    for key in _KEPT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def request_rerun(run_dir: Path) -> None:
    """Button callback for Re-run: load that run's request into the form and switch to this page."""
    st.session_state["nr_prefill"] = str(run_dir)
    st.session_state["page"] = "New run"


def _apply_prefill(catalog: dict) -> None:
    run_dir = st.session_state.pop("nr_prefill", None)
    if not run_dir:
        return
    try:
        choices = choices_from_request(catalog, history.load_request(Path(run_dir)))
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"Could not load the request of {Path(run_dir).name}: {exc}")
        return
    for name, key in _KEYS.items():
        value = getattr(choices, name)
        if name == "date_range" and not value:
            continue
        st.session_state[key] = value
    for op in EOD_OPS:
        st.session_state[f"nr_op_{op}"] = op in choices.eod_ops
    _clear_email()
    st.info(f"Loaded the request of {Path(run_dir).name}. Email is off; check the picks and press Run.")


def _clear_email() -> None:
    """Untick Send email and both confirmations (before their widgets are rendered in this run)."""
    st.session_state["nr_email"] = False
    for key in _CONFIRM_KEYS:
        st.session_state[key] = False


def _prune(key: str, options: Sequence[str]) -> None:
    """Drop picks that are no longer offered (Streamlit rejects a value outside the options)."""
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]


def _set_defaults(today: date) -> None:
    """Seed the state of widgets not set yet, so widgets never get both a ``value=`` and a Session State value.

    Keys survive while their widget is hidden because :func:`keep_form_state` re-assigns them every run.
    """
    as_of = default_as_of(today)
    defaults = {**{key: getattr(Choices(), name) for name, key in _KEYS.items()},
                _KEYS["as_of"]: as_of, _KEYS["date_range"]: (as_of - timedelta(days=7), as_of)}
    defaults.update({f"nr_op_{op}": op == "compliance" for op in EOD_OPS})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _stacked_turns_pdf_off() -> None:
    """``on_change`` of the Output radio: like the CLI's time_series preset, a stacked run defaults to no PDF."""
    if st.session_state.get(_KEYS["date_mode"]) == STACKED:
        st.session_state[_KEYS["create_pdf"]] = False


def _form(catalog: dict, locked: bool) -> Choices:
    """Render the controls (disabled while a run is active) and return what they hold."""
    c = Choices()
    today = date.today()
    _set_defaults(today)
    cohorts: Dict[str, List[str]] = catalog["cohorts"]
    st.subheader("Funds")
    c.cohorts = st.multiselect("Fund groups", sorted(cohorts), key=_KEYS["cohorts"], disabled=locked,
                               format_func=lambda n: f"{n} ({len(cohorts[n])})")
    pool = sorted(set().union(*(cohorts[n] for n in c.cohorts))) if c.cohorts else list(catalog["funds"])
    _prune(_KEYS["funds"], pool)
    _prune(_KEYS["exclude"], pool)
    c.funds = st.multiselect("Funds" + (" (narrowed by the groups)" if c.cohorts else ""), pool,
                             key=_KEYS["funds"], disabled=locked,
                             placeholder="All funds in the groups" if c.cohorts else "All funds")
    c.exclude = st.multiselect("Exclude", pool, key=_KEYS["exclude"], disabled=locked)
    count = len(selected_funds(catalog, c))
    st.badge(f"{count} fund{'s' if count != 1 else ''} selected", color="blue" if count else "red")

    st.subheader("Operations")
    c.kind = st.radio("Run type", [KIND_EOD, KIND_TRADING], key=_KEYS["kind"], horizontal=True, disabled=locked,
                      help="Trading compliance cannot be combined with the end-of-day operations.")
    trading = c.kind == KIND_TRADING
    if not trading:
        cols = st.columns(len(EOD_OPS))
        c.eod_ops = [op for col, (op, label) in zip(cols, EOD_OPS.items())
                     if col.checkbox(label, key=f"nr_op_{op}", disabled=locked)]
    if trading or "compliance" in c.eod_ops:
        c.rules_mode = st.radio("Compliance rules", [ALL_RULES, CHOOSE_RULES], key=_KEYS["rules_mode"],
                                horizontal=True, disabled=locked)
        if c.rules_mode == CHOOSE_RULES:
            labels = {r["key"]: r["label"] for r in catalog["rule_catalog"]}
            c.rules = st.multiselect("Rules", list(labels), key=_KEYS["rules"], format_func=labels.get,
                                     disabled=locked)

    st.subheader("Dates")
    c.date_kind = st.radio("Dates", [AS_OF, RANGE], key=_KEYS["date_kind"], horizontal=True, disabled=locked,
                           label_visibility="collapsed")
    weekdays_help = "Weekends snap to business days. There is no holiday calendar: a market holiday still runs."
    if c.date_kind == AS_OF:
        c.as_of = st.date_input("As of", max_value=today, key=_KEYS["as_of"], disabled=locked, help=weekdays_help)
    else:
        c.date_range = tuple(st.date_input("Range (start – end)", max_value=today, key=_KEYS["date_range"],
                                           disabled=locked, help=weekdays_help))
        if trading:  # stacked is EOD compliance only: show what will run
            st.session_state[_KEYS["date_mode"]] = PER_DAY
        c.date_mode = st.radio("Output", [PER_DAY, STACKED], key=_KEYS["date_mode"], horizontal=True,
                               disabled=locked or trading, on_change=_stacked_turns_pdf_off,
                               help="Stacked writes one compliance workbook over the whole range (EOD only).")
        if c.date_mode == STACKED and not trading:
            c.daily_reports = st.checkbox("Also write daily reports", key=_KEYS["daily_reports"], disabled=locked)

    st.subheader("Options")
    cols = st.columns(3)
    c.create_pdf = cols[0].checkbox("Create PDF", key=_KEYS["create_pdf"], disabled=locked)
    if trading:
        c.send_email = cols[1].checkbox("Send email", value=True, key="nr_email_trading", disabled=True,
                                        help="Trading compliance always sends the trading-analysis email.")
    else:
        c.send_email = cols[1].checkbox("Send email", key=_KEYS["send_email"], disabled=locked)
    try:
        start, end, stacked = run_dates(c)
        tag_hint = f"blank = {default_tag(operations_for(c), stacked)}"
    except InvalidRequest:
        tag_hint = "optional, e.g. backdated"
    c.tag = cols[2].text_input("Output tag", key=_KEYS["tag"], disabled=locked, max_chars=40, placeholder=tag_hint,
                               help="Goes into the report file names (blank = the CLI's name for this run type).")
    if trading:
        st.warning("Trading compliance always sends the trading-analysis email to its usual recipients (or opens "
                   "an Outlook draft on the server, depending on FUNDMANAGER_EMAIL_MODE): the backend has no "
                   "switch to turn it off.")
        c.email_confirmed = st.checkbox("Yes, send the trading email for this run", key="nr_trading_ok",
                                        disabled=locked)
    elif c.send_email:
        st.warning("This run will send the notification emails to their usual recipients "
                   "(or open Outlook drafts on the server, depending on FUNDMANAGER_EMAIL_MODE).")
        c.email_confirmed = st.checkbox("Yes, send the emails for this run", key=_KEYS["email_confirmed"],
                                        disabled=locked)
    return c


def _date_notes(choices: Choices, request: RunRequest) -> List[str]:
    """Say when a picked date is not the day that runs (weekends)."""
    picked = [choices.as_of] if choices.date_kind == AS_OF else list(choices.date_range)
    actual = [request.start_date] if choices.date_kind == AS_OF else [request.start_date, request.end_date]
    return [f"{p:%A} {p} is not a business day: the run uses {a}." for p, a in zip(picked, actual)
            if p is not None and p != a]


def render(catalog: dict, user: str) -> None:
    """The whole page: form, validation, Run, then the active run's progress or the last run's results."""
    st.title("New run")
    root = runner.output_root()
    active = runner.active_run(root)
    running = active is not None and not panels.is_finished(active)
    if running:
        st.session_state["nr_watch"] = str(active)  # reattach after a refresh, or show another user's run
    if st.session_state.pop("nr_started", False):
        _clear_email()  # each emailing run needs its own confirmation
    _apply_prefill(catalog)
    choices = _form(catalog, locked=running)

    request, problem = None, None
    try:
        request = build_request(catalog, choices)
    except (InvalidRequest, ValueError) as exc:
        problem = str(exc)
    if problem and not running:
        st.error(problem)
    elif request is not None:
        for note in _date_notes(choices, request):
            st.info(note)
        days = business_days(request.start_date, request.end_date or request.start_date)
        runs = ", ".join(map(str, days)) if len(days) <= 5 else f"{days[0]} … {days[-1]}"
        st.caption(f"{runner.summarize(request)} · {len(days)} business day(s): {runs} · "
                   f"tag {request.output_tag}")
    if st.button("Run", type="primary", disabled=running or request is None, key="nr_run"):
        try:
            st.session_state["nr_watch"] = str(runner.start_run(request, user=user, root=root))
        except runner.RunLocked as exc:
            st.error(str(exc))
        except OSError as exc:
            st.error(f"Could not start the run: {exc}")
        else:
            st.session_state["nr_started"] = True
            st.rerun()

    watch = st.session_state.get("nr_watch")
    if watch and Path(watch).is_dir():
        st.divider()
        if panels.is_finished(Path(watch)):
            panels.results_panel(Path(watch), user, key="nr")
        else:
            panels.progress_panel(watch, user)
