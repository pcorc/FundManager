"""CLI parsing helpers and parameter resolution for FundManager."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable as IterableABC
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Mapping, Optional, Sequence


@dataclass
class CLIOptions:
    """Raw options captured from the command line."""

    analysis_type: str
    as_of_date: date
    previous_date: Optional[date]
    funds: List[str]
    eod_reports: List[str]
    compliance_tests: List[str]
    output_dir: str
    create_pdf: bool
    ex_ante_date: Optional[date]
    ex_post_date: Optional[date]
    custodian_date: Optional[date]
    custodian_previous_date: Optional[date]


@dataclass
class EODRunParameters:
    """Resolved configuration for an EOD analysis run."""

    trade_date: date
    previous_trade_date: date
    operations: List[str]
    compliance_tests: List[str]
    create_pdf: bool


@dataclass
class TradingComplianceParameters:
    """Resolved configuration for the trading compliance comparison."""

    ex_ante_date: date
    ex_post_date: date
    vest_previous_date: date
    custodian_date: date
    custodian_previous_date: date
    compliance_tests: List[str]
    create_pdf: bool


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def parse_arguments(argv: Optional[Sequence[str]] = None) -> CLIOptions:
    """Parse command line arguments into a :class:`CLIOptions` payload."""

    parser = argparse.ArgumentParser(
        description="Run FundManager processing and generate compliance/reconciliation reports",
    )
    parser.add_argument(
        "--analysis-type",
        choices=["eod", "trading", "trading_compliance"],
        default="eod",
        help="Run either nightly end-of-day processing or intraday trading compliance",
    )
    parser.add_argument(
        "--as-of-date",
        type=_parse_date,
        default=date.today(),
        help="Business date to evaluate (default: today)",
    )
    parser.add_argument(
        "--previous-date",
        type=_parse_date,
        help="Explicit previous business date override (default: computed automatically)",
    )
    parser.add_argument(
        "--fund",
        action="append",
        default=[],
        help="Fund ticker to process (repeatable, supports comma-separated lists)",
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        default=["all"],
        choices=["all", "compliance", "holdings", "nav", "holdings_nav"],
        help="For EOD runs choose which report families to build",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EXPORT_PATH", "./reports"),
        help="Directory to write generated artefacts (default honours EXPORT_PATH)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation even when PDF helpers are installed",
    )
    parser.add_argument(
        "--compliance-test",
        action="append",
        default=[],
        help="Compliance test(s) to execute (repeatable, comma separated, default: all)",
    )
    parser.add_argument(
        "--ex-ante-date",
        type=_parse_date,
        help="Trading compliance: explicit ex-ante evaluation date (default: as-of-date)",
    )
    parser.add_argument(
        "--ex-post-date",
        type=_parse_date,
        help="Trading compliance: explicit ex-post evaluation date (default: ex-ante date)",
    )
    parser.add_argument(
        "--custodian-date",
        type=_parse_date,
        help="Trading compliance: custodian data availability date (default: ex-ante minus 1 biz day)",
    )
    parser.add_argument(
        "--custodian-prev-date",
        type=_parse_date,
        help="Trading compliance: previous custodian comparison date (default: custodian minus 1 biz day)",
    )

    raw_args = parser.parse_args(argv)

    analysis_type = raw_args.analysis_type.lower()
    if analysis_type == "trading":
        analysis_type = "trading_compliance"

    funds: List[str] = []
    for entry in raw_args.fund:
        for token in entry.replace(",", " ").split():
            token = token.strip().upper()
            if token:
                funds.append(token)

    compliance_tests = _parse_list_tokens(raw_args.compliance_test)

    return CLIOptions(
        analysis_type=analysis_type,
        as_of_date=raw_args.as_of_date,
        previous_date=raw_args.previous_date,
        funds=funds,
        eod_reports=list(raw_args.reports),
        compliance_tests=compliance_tests,
        output_dir=raw_args.output_dir,
        create_pdf=not raw_args.no_pdf,
        ex_ante_date=raw_args.ex_ante_date,
        ex_post_date=raw_args.ex_post_date,
        custodian_date=raw_args.custodian_date,
        custodian_previous_date=raw_args.custodian_prev_date,
    )


def resolve_eod_parameters(options: CLIOptions) -> EODRunParameters:
    operations = _determine_eod_operations(options.eod_reports)
    if not operations:
        raise ValueError("At least one report family must be selected for an EOD run")

    previous_date = options.previous_date or _business_day_offset(options.as_of_date, -1)
    return EODRunParameters(
        trade_date=options.as_of_date,
        previous_trade_date=previous_date,
        operations=operations,
        compliance_tests=list(options.compliance_tests),
        create_pdf=options.create_pdf,
    )


def resolve_trading_parameters(options: CLIOptions) -> TradingComplianceParameters:
    ex_ante_date = options.ex_ante_date or options.as_of_date
    ex_post_date = options.ex_post_date or ex_ante_date
    vest_previous = _business_day_offset(ex_ante_date, -1)
    custodian_date = options.custodian_date or _business_day_offset(ex_ante_date, -1)
    custodian_previous = options.custodian_previous_date or _business_day_offset(custodian_date, -1)

    return TradingComplianceParameters(
        ex_ante_date=ex_ante_date,
        ex_post_date=ex_post_date,
        vest_previous_date=vest_previous,
        custodian_date=custodian_date,
        custodian_previous_date=custodian_previous,
        compliance_tests=list(options.compliance_tests),
        create_pdf=options.create_pdf,
    )


def apply_overrides(options: CLIOptions, overrides: Optional[Mapping[str, object]]) -> CLIOptions:
    """Apply explicit overrides (e.g. from ``main`` keyword arguments)."""

    if not overrides:
        return options

    data = asdict(options)

    for key, value in overrides.items():
        if key not in data:
            raise KeyError(f"Unknown CLI option override: {key}")

        if key in {"as_of_date", "previous_date", "ex_ante_date", "ex_post_date", "custodian_date", "custodian_previous_date"}:
            if value is None:
                data[key] = None
            else:
                data[key] = _parse_date(value) if isinstance(value, str) else value
        elif key == "analysis_type":
            if value is None:
                raise ValueError("analysis_type override cannot be None")
            lowered = str(value).lower()
            data[key] = "trading_compliance" if lowered == "trading" else lowered
        elif key in {"funds", "eod_reports", "compliance_tests"}:
            data[key] = _normalise_override_sequence(key, value)
        elif key in {"create_pdf"}:
            data[key] = bool(value)
        elif key == "output_dir":
            data[key] = str(value)
        else:
            data[key] = value

    return CLIOptions(**data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    """Parse an ISO-like date string into a :class:`datetime.date`."""

    if isinstance(value, date):  # argparse may already pass date objects
        return value
    value = value.strip()
    if value.lower() == "today":
        return date.today()
    if value.lower() == "yesterday":
        return _business_day_offset(date.today(), -1)
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_list_tokens(entries: Sequence[str], *, preserve_all: bool = False) -> List[str]:
    """Split repeated/CSV entries into a unique, ordered list."""

    tokens: List[str] = []
    for entry in entries:
        if entry is None:
            continue
        for token in str(entry).replace(",", " ").split():
            cleaned = token.strip()
            if not cleaned:
                continue
            if not preserve_all and cleaned.lower() == "all":
                continue
            tokens.append(cleaned)

    seen: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen


def _normalise_override_sequence(name: str, value) -> List[str]:
    """Normalise override payloads that represent list-type options."""

    if value is None:
        return []

    preserve_all = name == "eod_reports"

    if isinstance(value, str) or not isinstance(value, IterableABC):
        iterable = [value]
    else:
        iterable = list(value)

    tokens: List[str] = []
    for item in iterable:
        if isinstance(item, str):
            tokens.extend(_parse_list_tokens([item], preserve_all=preserve_all))
        else:
            tokens.append(str(item))

    if name == "funds":
        tokens = [token.upper() for token in tokens]

    return tokens


def _business_day_offset(anchor: date, offset: int) -> date:
    """Shift ``anchor`` by ``offset`` business days (weekends only)."""

    if offset == 0:
        return anchor

    step = 1 if offset > 0 else -1
    remaining = abs(offset)
    current = anchor
    while remaining > 0:
        current += timedelta(days=step)
        if current.weekday() < 5:  # Monday=0 .. Friday=4
            remaining -= 1
    return current


def _determine_eod_operations(report_tokens: Iterable[str]) -> List[str]:
    normalized = {token.lower() for token in report_tokens}
    if not normalized or "all" in normalized:
        normalized.update({"compliance", "holdings", "nav"})
    if "holdings_nav" in normalized:
        normalized.update({"holdings", "nav"})

    desired = set()
    for token in normalized:
        if token == "compliance":
            desired.add("compliance")
        elif token in {"holdings", "reconciliation"}:
            desired.add("reconciliation")
        elif token in {"nav", "nav_reconciliation"}:
            desired.add("nav_reconciliation")

    operation_order = ["compliance", "reconciliation", "nav_reconciliation"]
    return [name for name in operation_order if name in desired]


__all__ = [
    "CLIOptions",
    "EODRunParameters",
    "TradingComplianceParameters",
    "parse_arguments",
    "resolve_eod_parameters",
    "resolve_trading_parameters",
    "apply_overrides",
]