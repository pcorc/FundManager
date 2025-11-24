"""High level orchestration helpers for FundManager processing modes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from config.fund_registry import FundRegistry
from processing.bulk_data_loader import BulkDataLoader, BulkDataStore
from processing.fund_manager import FundManager, ProcessingResults
from processing.fund import Fund
from reporting.compliance_reporter import (
    build_compliance_reports,
    build_compliance_reports_for_range,
)
from reporting.compliance_reporting import GeneratedComplianceReport
from reporting.reconciliation_reporter import build_reconciliation_reports
from reporting.nav_recon_reporter import build_nav_reconciliation_reports
from reporting.trade_compliance_reporter import (
    build_trading_compliance_reports,
    combine_trading_and_compliance_reports,
)

@dataclass(frozen=True)
class DataLoadRequest:
    """Describe a unique combination of target/previous dates to fetch."""

    label: str
    target_date: date
    previous_date: Optional[date] = None
    analysis_type: Optional[str] = None


@dataclass(frozen=True)
class RangeRunResults:
    """Aggregated artefacts for a multi-day EOD run."""

    results_by_date: "OrderedDict[date, ProcessingResults]"
    daily_artefacts: "OrderedDict[str, Mapping[str, object]]"
    stacked_compliance: Optional["GeneratedComplianceReport"]


def plan_eod_requests(params) -> Sequence[DataLoadRequest]:
    """Return the data requests needed for an EOD run."""

    return [
        DataLoadRequest(
            label="eod",
            target_date=params.trade_date,
            previous_date=params.previous_trade_date,
            analysis_type="eod",
        )
    ]


def plan_trading_requests(params) -> Sequence[DataLoadRequest]:
    """Return the data requests needed for trading compliance."""

    return [
        DataLoadRequest(
            label="ex_ante",
            target_date=params.ex_ante_date,
            previous_date=params.vest_previous_date,
            analysis_type="ex_ante",
        ),
        DataLoadRequest(
            label="ex_post",
            target_date=params.ex_post_date,
            previous_date=params.custodian_previous_date,
            analysis_type="ex_post",
        ),
    ]


def fetch_data_stores(
    session,
    base_cls,
    registry: FundRegistry,
    requests: Sequence[DataLoadRequest],
) -> Dict[str, BulkDataStore]:
    """Load all required data stores using minimal database calls."""

    loader = BulkDataLoader(session, base_cls, registry)
    cache: Dict[Tuple[date, Optional[date], Optional[str]], BulkDataStore] = {}
    stores: Dict[str, BulkDataStore] = {}

    for request in requests:
        key = (request.target_date, request.previous_date, request.analysis_type)
        if key not in cache:
            cache[key] = loader.load_all_data_for_date(
                request.target_date,
                previous_date=request.previous_date,
                analysis_type=request.analysis_type,

            )
        stores[request.label] = cache[key]

    return stores


def run_eod_mode(
    registry: FundRegistry,
    data_store: BulkDataStore,
    params,
    output_dir: Path,
    *,
    output_tag: str | None = None,
) -> Tuple[ProcessingResults, Mapping[str, object]]:
    """Execute the full EOD workflow for the provided data."""

    def _build_prefix(base: str) -> str:
        return f"{base}_{tag}" if tag else base

    results = _run_operations(
        registry,
        data_store,
        operations=params.operations,
        analysis_type="eod",
        compliance_tests=params.compliance_tests,
    )

    compliance_payload = _extract_payload(results, "compliance_results")
    reconciliation_payload = _extract_payload(results, "reconciliation_results")
    nav_payload = _extract_payload(results, "nav_results")

    artefacts: MutableMapping[str, object] = {}
    gics_mapping = getattr(data_store, "gics_mapping", None)

    if "compliance" in params.operations and compliance_payload:
        artefacts["compliance"] = build_compliance_reports(
            results,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            file_name_prefix=_build_prefix("compliance_results"),
            test_functions=params.compliance_tests or None,
            gics_mapping=gics_mapping,
            create_pdf=params.create_pdf,
        )

    if any(name in params.operations for name in ("reconciliation", "nav_reconciliation")):
        # Extract fund registry for property-based reconciliation reporting
        fund_registry = _extract_fund_registry(results)

        artefacts["reconciliation"] = build_nav_reconciliation_reports(
            holdings_results=reconciliation_payload if "reconciliation" in params.operations else None,
            nav_results=nav_payload if "nav_reconciliation" in params.operations else None,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            file_name_prefix=_build_prefix("reconciliation_summary"),
            create_pdf=params.create_pdf
        )

    return results, artefacts


def run_trading_mode(
    registry: FundRegistry,
    *,
    ex_ante_store: BulkDataStore,
    ex_post_store: BulkDataStore,
    params,
    output_dir: Path,
    output_tag: str | None = None,
):
    """Execute the trading compliance comparison."""

    def _build_prefix(base: str) -> str:
        return f"{base}_{output_tag}" if output_tag else base

    results_ex_ante = _run_operations(
        registry,
        ex_ante_store,
        operations=["compliance"],
        analysis_type="ex_ante",
        compliance_tests=params.compliance_tests,
    )
    results_ex_post = _run_operations(
        registry,
        ex_post_store,
        operations=["compliance"],
        analysis_type="ex_post",
        compliance_tests=params.compliance_tests,
    )

    ante_payload = _extract_payload(results_ex_ante, "compliance_results")
    post_payload = _extract_payload(results_ex_post, "compliance_results")

    if not ante_payload and not post_payload:
        raise RuntimeError("Trading compliance run produced no compliance results to compare")

    # Extract fund registry from ex_post results for property-based reporting
    fund_registry_ex_post = _extract_fund_registry(results_ex_post)

    # Extract traded funds information from the ex_post results
    # This should include the holdings data with trade_rebal information
    traded_funds_info = {}

    # Method 1: Extract from the compliance results if they contain fund objects
    for fund_name in post_payload.keys():
        fund_info = {}

        # Try to get the fund object or holdings data from results
        if fund_name in post_payload:
            fund_results = post_payload[fund_name]

            # Look for fund object
            if "fund_object" in fund_results:
                fund_info["fund_object"] = fund_results["fund_object"]
            elif "fund" in fund_results:
                fund_info["fund"] = fund_results["fund"]

            # Look for holdings data that might contain trade information
            for asset_type in ["equity", "options", "treasury"]:
                if asset_type in fund_results:
                    fund_info[asset_type] = fund_results[asset_type]

        if fund_info:
            traded_funds_info[fund_name] = fund_info

    trading_report = build_trading_compliance_reports(
        results_ex_ante=ante_payload,
        results_ex_post=post_payload,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        traded_funds_info=traded_funds_info,  # Pass the populated traded_funds_info
        fund_registry=fund_registry_ex_post,  # NEW: Pass fund registry for property-based reporting
        file_name_prefix=_build_prefix("trading_compliance_results"),
        create_pdf=params.create_pdf,
    )

    ex_post_compliance_report = None
    gics_mapping_ex_post = getattr(ex_post_store, "gics_mapping", None)

    if post_payload:
        ex_post_compliance_report = build_compliance_reports(
            results_ex_post,
            report_date=params.ex_post_date,
            output_dir=str(output_dir),
            file_name_prefix=_build_prefix("compliance_results_expost"),
            test_functions=params.compliance_tests or None,
            gics_mapping=gics_mapping_ex_post,
            create_pdf=params.create_pdf,
        )

    combined_report = combine_trading_and_compliance_reports(
        trading_report=trading_report,
        compliance_report=ex_post_compliance_report,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        file_name_prefix=_build_prefix("trading_compliance_results_combined"),
    )

    combined_artefacts = SimpleNamespace(
        report=trading_report,
        ex_post_compliance=ex_post_compliance_report,
        combined_report=combined_report,
    )


def run_eod_range_mode(
    session,
    base_cls,
    registry: FundRegistry,
    *,
    start_date: date,
    end_date: date,
    operations: Sequence[str],
    compliance_tests: Sequence[str],
    output_dir: Path,
    create_pdf: bool = True,
    generate_daily_reports: bool = True,
    output_tag: str | None = None,
) -> RangeRunResults:
    """Execute the EOD workflow for each business day in ``[start_date, end_date]``."""

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    loader = BulkDataLoader(session, base_cls, registry)
    results_by_date: "OrderedDict[date, ProcessingResults]" = OrderedDict()
    daily_artefacts: "OrderedDict[str, Mapping[str, object]]" = OrderedDict()

    latest_gics_mapping = None

    def _build_prefix(base: str) -> str:
        return f"{base}_{output_tag}" if output_tag else base

    for trade_date in _iter_business_days(start_date, end_date):
        previous_date = _previous_business_day(trade_date)
        data_store = loader.load_all_data_for_date(
            trade_date,
            previous_date=previous_date,
            analysis_type="eod",
        )
        latest_gics_mapping = getattr(data_store, "gics_mapping", None)

        day_results = _run_operations(
            registry,
            data_store,
            operations=operations,
            analysis_type="eod",
            compliance_tests=compliance_tests,
        )

        results_by_date[trade_date] = day_results

        if not generate_daily_reports:
            continue

        artefacts: Dict[str, object] = {}

        compliance_payload = _extract_payload(day_results, "compliance_results")
        reconciliation_payload = _extract_payload(day_results, "reconciliation_results")
        nav_payload = _extract_payload(day_results, "nav_results")

        if "compliance" in operations and compliance_payload:
            artefacts["compliance"] = build_compliance_reports(
                day_results,
                report_date=trade_date,
                output_dir=str(output_dir),
                file_name_prefix=_build_prefix("compliance_results"),
                test_functions=compliance_tests or None,
                gics_mapping=latest_gics_mapping,
                create_pdf=create_pdf,
            )

        if any(name in operations for name in ("reconciliation", "nav_reconciliation")):
            # Extract fund registry for property-based reconciliation reporting
            fund_registry = _extract_fund_registry(day_results)

            artefacts["reconciliation"] = build_nav_reconciliation_reports(
                holdings_results=reconciliation_payload if "reconciliation" in operations else None,
                nav_results=nav_payload if "nav_reconciliation" in operations else None,
                report_date=trade_date,
                output_dir=str(output_dir),
                create_pdf=create_pdf,
            )

        daily_artefacts[trade_date.isoformat()] = artefacts

        stacked_report = None
        if "compliance" in operations and results_by_date:
            stacked_report = build_compliance_reports_for_range(
                results_by_date.items(),
                output_dir=str(output_dir),
                output_tag=output_tag,
                test_functions=compliance_tests or None,
                gics_mapping=latest_gics_mapping,
                create_pdf=create_pdf,
            )

    return RangeRunResults(
        results_by_date=results_by_date,
        daily_artefacts=daily_artefacts,
        stacked_compliance=stacked_report,
    )


def flatten_eod_paths(artefacts: Mapping[str, object]) -> Dict[str, str]:
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

        nav = getattr(reconciliation, "nav", None)
        if nav is not None:
            if getattr(nav, "excel_path", None):
                paths["nav_reconciliation_excel"] = nav.excel_path  # type: ignore[attr-defined]

        combined = getattr(reconciliation, "combined_reconciliation_pdf", None)
        if combined:
            paths["reconciliation_summary_pdf"] = combined

    return paths


def flatten_trading_paths(artefacts) -> Dict[str, str]:  # type: ignore[no-untyped-def]
    paths: Dict[str, str] = {}
    if artefacts is None:
        return paths

    def _extract(name: str):
        if isinstance(artefacts, Mapping):
            return artefacts.get(name)
        return getattr(artefacts, name, None)

    def _append(prefix: str, report) -> None:
        if report is None:
            return
        excel_path = getattr(report, "excel_path", None)
        pdf_path = getattr(report, "pdf_path", None)
        if excel_path:
            paths[f"{prefix}_excel"] = excel_path
        if pdf_path:
            paths[f"{prefix}_pdf"] = pdf_path

    if hasattr(artefacts, "excel_path") or hasattr(artefacts, "pdf_path"):
        _append("trading_compliance", artefacts)
        return paths

    _append("trading_compliance", _extract("report"))
    _append("trading_ex_post_compliance", _extract("ex_post_compliance"))
    _append("trading_compliance_combined", _extract("combined_report"))

    return paths


def _run_operations(
    registry: FundRegistry,
    data_store: BulkDataStore,
    *,
    operations: Sequence[str],
    analysis_type: str,
    compliance_tests: Sequence[str],
) -> ProcessingResults:
    """Execute operations on all funds using FundManager."""
    manager = FundManager(registry, data_store, analysis_type=analysis_type)
    return manager.run_daily_operations(
        list(operations), compliance_tests=list(compliance_tests)
    )


def _extract_fund_registry(results: ProcessingResults) -> Dict[str, Fund]:
    """Extract Fund objects from ProcessingResults to create registry for reporting.

    This enables property-based logic throughout the reporting pipeline,
    allowing reports to leverage fund properties like is_private_fund,
    uses_index_flex, etc.

    Args:
        results: ProcessingResults containing fund_results with Fund objects

    Returns:
        Dictionary mapping fund names to Fund objects
    """
    fund_registry: Dict[str, Fund] = {}
    for fund_name, fund_result in results.fund_results.items():
        if fund_result.fund:
            fund_registry[fund_name] = fund_result.fund
    return fund_registry


def _iter_business_days(start: date, end: date):
    """Iterate over business days (Monday-Friday) in date range."""
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def _previous_business_day(anchor: date) -> date:
    """Return the previous business day before anchor date."""
    current = anchor - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _extract_payload(results: ProcessingResults, attribute: str) -> Dict[str, Mapping[str, object]]:
    """Extract a specific attribute from all fund results into a payload dictionary.

    Args:
        results: ProcessingResults containing fund_results
        attribute: Attribute name to extract (e.g., 'compliance_results', 'reconciliation_results')

    Returns:
        Dictionary mapping fund names to their extracted attribute values
    """
    payload: Dict[str, Mapping[str, object]] = {}
    for fund_name, fund_result in results.fund_results.items():
        value = getattr(fund_result, attribute, None)
        if value:
            payload[fund_name] = value
    return payload


__all__ = [
    "DataLoadRequest",
    "RangeRunResults",
    "plan_eod_requests",
    "plan_trading_requests",
    "fetch_data_stores",
    "run_eod_mode",
    "run_eod_range_mode",
    "run_trading_mode",
    "flatten_eod_paths",
    "flatten_trading_paths",
]