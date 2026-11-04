"""The New run Configuration Preview: metric cards, the 'will run for' sentence, inline problems, the check guide.

:func:`counts` and :func:`estimate_seconds` are plain functions (unit-tested);
the time estimate uses the seconds per fund-day of past successful runs in the
output root (:func:`ui.history.fund_day_seconds`) and a heuristic before there
are any. The check guide describes each picked rule from
``reporting.result_views.rule_descriptions()`` (static, no DB).
"""
from __future__ import annotations

import functools
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from processing.api import InvalidRequest, RunRequest
from ui import components as ui
from ui import history, new_run as nr, runner

# Before any run history: ~10 s per day to load the data plus ~7 s per fund-day (measured on the live DB).
HEURISTIC_DAY_SECONDS, HEURISTIC_FUND_DAY_SECONDS = 10.0, 7.0
HISTORY_RUNS = 40  # recent runs the estimate looks at

OPERATION_GUIDE = {
    "reconciliation": ("Holdings Reconciliation",
                       "Compares the OMS (Vest) holdings with the custodian's and the index provider's.",
                       ["Holdings missing on either side, and quantity mismatches (shown in red)",
                        "Price breaks and index weight differences (shown in amber)",
                        "Equity, listed options, FLEX options and treasuries, per custodian"],
                       "Funds with a custodian holdings feed"),
    "nav_reconciliation": ("NAV Reconciliation",
                           "Rebuilds today's expected NAV from yesterday's net assets plus today's gains and "
                           "losses, and compares it with the custodian's NAV.",
                           ["NAV Good 2-Dec: expected and custodian NAV agree to the cent",
                            "NAV Good 4-Dec: they agree to four decimals",
                            "G/L components: equity, options, FLEX, treasuries, assignments, accruals, distributions"],
                           "Funds with a custodian NAV"),
    "trading_compliance": ("Trading Compliance",
                           "Runs the picked compliance checks twice, on the holdings before (ex-ante) and after "
                           "(ex-post) the day's trades, and lists the funds whose status changed.",
                           ["Funds moving into or out of a breach are highlighted",
                            "Trade activity per fund and asset class: trade value, % of net and total assets"],
                           "Funds that traded on the day"),
}


@dataclass(frozen=True)
class Counts:
    """What the preview cards and sentence show (``days`` is ``None`` while the dates are invalid)."""

    funds: int
    days: Optional[int]
    checks: int  # compliance rules picked (0 when compliance does not run)
    recons: int  # holdings / NAV recon operations
    operations: Tuple[str, ...]
    trading: bool

    @property
    def per_fund_day(self) -> int:
        return self.checks + self.recons

    @property
    def passes(self) -> int:
        return 2 if self.trading else 1

    @property
    def total(self) -> Optional[int]:
        return None if self.days is None else self.funds * self.days * self.per_fund_day * self.passes

    @property
    def fund_days(self) -> Optional[int]:
        return None if self.days is None else self.funds * self.days * self.passes


def counts(catalog: dict, choices: "nr.Choices") -> Counts:
    """The preview numbers for the current picks (they do not need a valid request)."""
    try:
        start, end, _ = nr.run_dates(choices)
        days: Optional[int] = len(nr.business_days(start, end))
    except InvalidRequest:
        days = None
    ops = nr.operations_for(choices)
    checks = 0
    if nr.uses_rules(choices):
        offered = {key for key, _ in nr.check_rules(catalog)}
        checks = len(offered) if choices.rules_mode == nr.ALL_RULES else len([r for r in choices.rules if r in offered])
    return Counts(funds=len(nr.selected_funds(catalog, choices)) if (choices.funds or choices.cohorts) else 0,
                  days=days, checks=checks, recons=len([op for op in ops if op in nr.RECON_OPS]),
                  operations=ops, trading="trading_compliance" in ops)


def estimate_seconds(fund_days: int, days: int, history_rates: Sequence[float] = ()) -> Tuple[float, bool]:
    """(seconds, from_history): the median seconds per fund-day of past runs, else the heuristic."""
    if history_rates:
        return statistics.median(history_rates) * fund_days, True
    return HEURISTIC_DAY_SECONDS * days + HEURISTIC_FUND_DAY_SECONDS * fund_days, False


def format_duration(seconds: float) -> str:
    """A rounded range: ``~ 30–50 seconds``, ``~ 2–3 minutes``, ``~ 1–2 hours``."""
    low, high = seconds * 0.8, seconds * 1.3
    if high < 60:
        lo, hi = max(int(round(low / 5.0)) * 5, 5), max(int(round(high / 5.0)) * 5, 10)
        return f"~ {lo}–{hi} seconds" if hi > lo else f"~ {hi} seconds"
    if high < 90 * 60:
        lo, hi = max(int(low // 60), 1), max(int(-(-high // 60)), 1)
        return f"~ {lo}–{hi} minutes" if hi > lo else f"~ {hi} minute{'s' if hi != 1 else ''}"
    lo, hi = max(low / 3600, 1.0), high / 3600
    return f"~ {lo:.0f}–{hi:.0f} hours" if round(hi) > round(lo) else f"~ {hi:.0f} hours"


@st.cache_data(ttl=120, show_spinner=False)
def _history_rates(root: str) -> List[float]:
    """Seconds per fund-day of the recent successful runs under ``root`` (cached briefly: it reads every folder)."""
    return history.fund_day_seconds(history.list_runs(Path(root), limit=HISTORY_RUNS))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def sentence(c: Counts, estimate: Optional[Tuple[float, bool, int]]) -> str:
    """The 'Your checks will run for: …' HTML."""
    if c.days is None or not c.funds or not c.per_fund_day:
        what = "the trade date, at least one fund and the checks" if c.trading else             "the dates, at least one fund and what to run"
        return f'<div class="fm-sentence">Pick {what}; the preview updates as you go.</div>'
    checks = _plural(c.per_fund_day, "check")
    if c.checks and c.recons:  # the card counts the rules; say where the other checks come from
        checks += f" ({_plural(c.checks, 'rule')} + {_plural(c.recons, 'recon')})"
    parts = [_plural(c.funds, "fund"), _plural(c.days, "day"), checks]
    if c.trading:
        parts.append("2 passes")
    what = []
    if c.checks:
        what.append(_plural(c.checks, "compliance rule"))
    what += [nr.EOD_OPS[op] for op in c.operations if op in nr.RECON_OPS]
    if c.trading:
        what.append("before and after the trades")
    line = " × ".join(parts)
    est = ""
    if estimate:
        seconds, from_history, runs = estimate
        source = f"estimate from {_plural(runs, 'past run')}" if from_history else "rough estimate"
        est = (f'<br><span class="est">Estimated completion time: {ui.esc(format_duration(seconds))} '
               f"({ui.esc(source)})</span>")
    return (f'<div class="fm-sentence">Your checks will run for:<br><b>{ui.esc(line)} '
            f'<span class="eq">=</span> {c.total:,} total operations</b><br>'
            f'<span class="est">{ui.esc(" + ".join(what))}</span>{est}</div>')


def notice(kind: str, message: str) -> None:
    ui.show(ui.callout(kind, message))


def render(catalog: dict, choices: "nr.Choices", request: Optional[RunRequest], problem: Optional[str],
           notes: Sequence[str]) -> None:
    """Heading, metric cards, the sentence, then the problem (or the weekend notes) inline."""
    c = counts(catalog, choices)
    ops_note = " · ".join(nr.EOD_OPS.get(op, "Trading") for op in c.operations) or "none picked"
    try:
        start, end, _ = nr.run_dates(choices)
        days_note = str(start) if start == end else f"{start} → {end}"
    except InvalidRequest:
        days_note = ""
    picked = nr.selected_funds(catalog, choices) if c.funds else []
    funds_note = ", ".join(picked[:6]) + (f" +{len(picked) - 6}" if len(picked) > 6 else "")
    cards = [(c.days if c.days is not None else "—", "Days Selected", days_note),
             (c.funds, "Funds Selected", funds_note),
             (c.checks, "Checks Selected", "" if c.checks else
              "none picked" if "compliance" in c.operations else "compliance off" if c.recons else ""),
             (len(c.operations), "Operations", ops_note)]
    estimate = None
    if c.fund_days and c.days:
        rates = _history_rates(str(runner.output_root()))
        seconds, from_history = estimate_seconds(c.fund_days, c.days, rates)
        estimate = (seconds, from_history, len(rates))
    body = ui.heading("Configuration Preview") + ui.metric_cards(cards, accent=(1,)) + sentence(c, estimate)
    if problem:
        body += ui.callout("error", problem)
    elif request is not None:
        for note in notes:
            body += ui.callout("info", note)
        tag = f"Output tag “{request.output_tag}”"
        body += f'<p class="fm-hint" style="text-align:center">{ui.esc(ui.summary_text(runner.summarize(request)))}  ·  {ui.esc(tag)}</p>'
    ui.show(body)


@functools.lru_cache(maxsize=1)
def _descriptions() -> Dict[str, dict]:
    from reporting.result_views import rule_descriptions  # noqa: PLC0415 (heavy import, only when shown)

    return rule_descriptions()


def guide_card(title: str, summary: str, conditions: Sequence[str], applies_to: str, footnotes: Sequence[str] = ()) -> str:
    body = f'<p class="fm-desc">{ui.esc(summary)}</p>'
    if conditions:
        body += '<ul class="fm-conds">' + "".join(f"<li>{ui.esc(x)}</li>" for x in conditions) + "</ul>"
    if applies_to:
        body += f'<div class="fm-applies">Applies to: {ui.esc(applies_to)}</div>'
    if footnotes:
        body += ui.details("Report footnotes", "".join(f'<p class="fm-desc">{ui.esc(n)}</p>' for n in footnotes))
    return ui.card(title, body)


def rule_guide(catalog: dict, choices: "nr.Choices") -> None:
    """Cards describing what each picked check and operation does (before a run)."""
    cards = []
    if nr.uses_rules(choices):
        descriptions = _descriptions()
        keys = [k for k, _ in nr.check_rules(catalog)] if choices.rules_mode == nr.ALL_RULES else choices.rules
        for key in keys:
            d = descriptions.get(key)
            if d:
                cards.append(guide_card(d["title"], d["summary"], d.get("conditions") or [], d.get("applies_to") or "",
                                        d.get("footnotes") or []))
    for op in nr.operations_for(choices):
        if op in OPERATION_GUIDE:
            title, summary, conditions, applies = OPERATION_GUIDE[op]
            cards.append(guide_card(title, summary, conditions, applies))
    if not cards:
        ui.show(ui.empty("Nothing picked yet", "Tick an operation and some checks to see what each one does."))
        return
    ui.show(ui.heading("What will be checked", f"{len(cards)} item{'s' if len(cards) != 1 else ''}")
            + ui.grid(cards))
