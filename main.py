"""Command line entry point for the FundManager reporting suite.

This module wires together data loading, fund processing, and the
report-generation helpers under a flexible command line interface. It supports
nightly end-of-day (EOD) runs as well as intraday trading compliance
comparisons between ex-ante and ex-post checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from config.database import initialize_database
from config.fund_registry import FundRegistry
from processing.bulk_data_loader import BulkDataLoader
from processing.fund_manager import FundManager, ProcessingResults
from reporting.compliance_reporter import build_compliance_reports
from reporting.nav_recon_reporter import build_reconciliation_reports
from reporting.trade_compliance_reporter import build_trading_compliance_reports


# ---------------------------------------------------------------------------
# Data classes capturing parsed CLI options and resolved run-time parameters.
# ---------------------------------------------------------------------------


@dataclass
class CLIOptions:
    """Raw options captured from the command line."""

    analysis_type: str
    as_of_date: date
    previous_date: Optional[date]
    funds: List[str]
    eod_reports: List[str]
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
    create_pdf: bool


@dataclass
class TradingComplianceParameters:
    """Resolved configuration for the trading compliance comparison."""

    ex_ante_date: date
    ex_post_date: date
    vest_previous_date: date
    custodian_date: date
    custodian_previous_date: date
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

    return CLIOptions(
        analysis_type=analysis_type,
        as_of_date=raw_args.as_of_date,
        previous_date=raw_args.previous_date,
        funds=funds,
        eod_reports=list(raw_args.reports),
        output_dir=raw_args.output_dir,
        create_pdf=not raw_args.no_pdf,
        ex_ante_date=raw_args.ex_ante_date,
        ex_post_date=raw_args.ex_post_date,
        custodian_date=raw_args.custodian_date,
        custodian_previous_date=raw_args.custodian_prev_date,
    )


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


# ---------------------------------------------------------------------------
# Configuration resolution helpers
# ---------------------------------------------------------------------------


def resolve_eod_parameters(options: CLIOptions) -> EODRunParameters:
    operations = _determine_eod_operations(options.eod_reports)
    if not operations:
        raise ValueError("At least one report family must be selected for an EOD run")

    previous_date = options.previous_date or _business_day_offset(options.as_of_date, -1)
    return EODRunParameters(
        trade_date=options.as_of_date,
        previous_trade_date=previous_date,
        operations=operations,
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
        create_pdf=options.create_pdf,
    )


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


# ---------------------------------------------------------------------------
# Core orchestration helpers
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point used by ``python -m`` or direct execution."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fund_manager.main")

    options = parse_arguments(argv)
    output_dir = Path(options.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting %s run for %s (funds: %s)",
        options.analysis_type,
        options.as_of_date.isoformat(),
        ",".join(options.funds) if options.funds else "ALL",
    )

    session, base_cls = initialize_database()

    try:
        registry = FundRegistry.from_database(session, base_cls)
        registry = _filter_registry(registry, options.funds)
        if not registry.funds:
            logger.error("No funds available after applying filters")
            return 1

        if options.analysis_type == "trading_compliance":
            params = resolve_trading_parameters(options)
            logger.info(
                "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
                params.ex_ante_date,
                params.ex_post_date,
                params.vest_previous_date,
                params.custodian_date,
                params.custodian_previous_date,
            )
            results_ex_ante, results_ex_post, artefacts = _run_trading_compliance(
                session,
                base_cls,
                registry,
                params,
                output_dir,
            )

            _log_processing_summary(logger, "ex-ante", results_ex_ante.summary)
            _log_processing_summary(logger, "ex-post", results_ex_post.summary)
            _log_generated_paths(logger, _flatten_trading_paths(artefacts))

        else:  # EOD flow
            params = resolve_eod_parameters(options)
            logger.info(
                "EOD dates — trade date: %s, previous business date: %s",
                params.trade_date,
                params.previous_trade_date,
            )
            processing_results, artefacts = _run_eod_analysis(
                session,
                base_cls,
                registry,
                params,
                output_dir,
            )

            _log_processing_summary(logger, "eod", processing_results.summary)
            _log_generated_paths(logger, _flatten_eod_paths(artefacts))

    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    logger.info("Run completed successfully")
    return 0


def _filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return a registry containing only the requested funds (if any)."""

    if not funds:
        return registry

    missing = [fund for fund in funds if fund not in registry.funds]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")

    filtered = FundRegistry()
    filtered.funds = {fund: registry.funds[fund] for fund in funds}
    return filtered


def _run_eod_analysis(
    session,
    base_cls,
    registry: FundRegistry,
    params: EODRunParameters,
    output_dir: Path,
):
    operations = params.operations
    results = _process_funds(
        session,
        base_cls,
        registry,
        target_date=params.trade_date,
        operations=operations,
        analysis_type="eod",
    )

    compliance_payload = _extract_payload(results, "compliance_results")
    reconciliation_payload = _extract_payload(results, "reconciliation_results")
    nav_payload = _extract_payload(results, "nav_results")

    artefacts: MutableMapping[str, object] = {}

    if "compliance" in operations and compliance_payload:
        artefacts["compliance"] = build_compliance_reports(
            results,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            create_pdf=params.create_pdf,
        )

    if any(name in operations for name in ("reconciliation", "nav_reconciliation")):
        artefacts["reconciliation"] = build_reconciliation_reports(
            holdings_results=reconciliation_payload if "reconciliation" in operations else None,
            nav_results=nav_payload if "nav_reconciliation" in operations else None,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            create_pdf=params.create_pdf,
            compliance_results=compliance_payload if "compliance" in operations else None,
        )

    return results, artefacts


def _run_trading_compliance(
    session,
    base_cls,
    registry: FundRegistry,
    params: TradingComplianceParameters,
    output_dir: Path,
):
    ex_ante_results = _process_funds(
        session,
        base_cls,
        registry,
        target_date=params.ex_ante_date,
        operations=["compliance"],
        analysis_type="ex_ante",
    )
    ex_post_results = _process_funds(
        session,
        base_cls,
        registry,
        target_date=params.ex_post_date,
        operations=["compliance"],
        analysis_type="ex_post",
    )

    ante_payload = _extract_payload(ex_ante_results, "compliance_results")
    post_payload = _extract_payload(ex_post_results, "compliance_results")

    if not ante_payload and not post_payload:
        raise RuntimeError("Trading compliance run produced no compliance results to compare")

    artefacts = build_trading_compliance_reports(
        results_ex_ante=ante_payload,
        results_ex_post=post_payload,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        create_pdf=params.create_pdf,
    )

    return ex_ante_results, ex_post_results, artefacts


def _process_funds(
    session,
    base_cls,
    registry: FundRegistry,
    *,
    target_date: date,
    operations: Sequence[str],
    analysis_type: str,
) -> ProcessingResults:
    loader = BulkDataLoader(session, base_cls, registry)
    data_store = loader.load_all_data_for_date(target_date)
    manager = FundManager(registry, data_store, analysis_type=analysis_type)
    return manager.run_daily_operations(list(operations))


def _extract_payload(results: ProcessingResults, attribute: str) -> Dict[str, Mapping[str, object]]:
    payload: Dict[str, Mapping[str, object]] = {}
    for fund_name, fund_result in results.fund_results.items():
        value = getattr(fund_result, attribute, None)
        if value:
            payload[fund_name] = value
    return payload


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_processing_summary(logger: logging.Logger, label: str, summary: Mapping[str, object]) -> None:
    pretty = ", ".join(f"{key}={value}" for key, value in summary.items())
    logger.info("Processing summary (%s): %s", label, pretty)


def _log_generated_paths(logger: logging.Logger, paths: Mapping[str, str]) -> None:
    if not paths:
        logger.warning("No report artefacts were produced")
        return
    for label, path in sorted(paths.items()):
        logger.info("Generated %s -> %s", label, path)


def _flatten_eod_paths(artefacts: Mapping[str, object]) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    compliance = artefacts.get("compliance")
    if compliance is not None:
        excel_path = getattr(compliance, "excel_path", None)
        pdf_path = getattr(compliance, "pdf_path", None)
        if excel_path:
            paths["compliance_excel"] = excel_path
        if pdf_path:
            paths["compliance_pdf"] = pdf_path

    reconciliation = artefacts.get("reconciliation")
    if reconciliation is not None:
        holdings = getattr(reconciliation, "holdings", None)
        if holdings is not None:
            if getattr(holdings, "excel_path", None):
                paths["holdings_reconciliation_excel"] = holdings.excel_path  # type: ignore[attr-defined]
            if getattr(holdings, "pdf_path", None):
                paths["holdings_reconciliation_pdf"] = holdings.pdf_path  # type: ignore[attr-defined]

        nav = getattr(reconciliation, "nav", None)
        if nav is not None:
            if getattr(nav, "excel_path", None):
                paths["nav_reconciliation_excel"] = nav.excel_path  # type: ignore[attr-defined]
            if getattr(nav, "pdf_path", None):
                paths["nav_reconciliation_pdf"] = nav.pdf_path  # type: ignore[attr-defined]

        combined = getattr(reconciliation, "combined_reconciliation_pdf", None)
        if combined:
            paths["reconciliation_summary_pdf"] = combined

        full = getattr(reconciliation, "full_summary_pdf", None)
        if full:
            paths["oversight_summary_pdf"] = full

    return paths


def _flatten_trading_paths(artefacts) -> Dict[str, str]:  # type: ignore[no-untyped-def]
    paths: Dict[str, str] = {}
    if artefacts is None:
        return paths

    report = getattr(artefacts, "report", None)
    if report is not None:
        if getattr(report, "excel_path", None):
            paths["trading_compliance_excel"] = report.excel_path  # type: ignore[attr-defined]
        if getattr(report, "pdf_path", None):
            paths["trading_compliance_pdf"] = report.pdf_path  # type: ignore[attr-defined]

    return paths


if __name__ == "__main__":
    raise SystemExit(main())