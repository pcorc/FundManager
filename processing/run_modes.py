"""High level orchestration helpers for FundManager processing modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from config.fund_registry import FundRegistry
from processing.bulk_data_loader import BulkDataLoader, BulkDataStore
from processing.fund_manager import FundManager, ProcessingResults
from reporting.compliance_reporter import build_compliance_reports
from reporting.reconciliation_reporter import build_reconciliation_reports
from reporting.nav_recon_reporter import build_nav_reconciliation_reports
from reporting.trade_compliance_reporter import build_trading_compliance_reports


@dataclass(frozen=True)
class DataLoadRequest:
    """Describe a unique combination of target/previous dates to fetch."""

    label: str
    target_date: date
    previous_date: Optional[date] = None


def plan_eod_requests(params) -> Sequence[DataLoadRequest]:
    """Return the data requests needed for an EOD run."""

    return [
        DataLoadRequest(
            label="eod",
            target_date=params.trade_date,
            previous_date=params.previous_trade_date,
        )
    ]


def plan_trading_requests(params) -> Sequence[DataLoadRequest]:
    """Return the data requests needed for trading compliance."""

    return [
        DataLoadRequest(
            label="ex_ante",
            target_date=params.ex_ante_date,
            previous_date=params.vest_previous_date,
        ),
        DataLoadRequest(
            label="ex_post",
            target_date=params.ex_post_date,
            previous_date=params.custodian_previous_date,
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
    cache: Dict[Tuple[date, Optional[date]], BulkDataStore] = {}
    stores: Dict[str, BulkDataStore] = {}

    for request in requests:
        key = (request.target_date, request.previous_date)
        if key not in cache:
            cache[key] = loader.load_all_data_for_date(
                request.target_date,
                previous_date=request.previous_date,
            )
        stores[request.label] = cache[key]

    return stores


def run_eod_mode(
    registry: FundRegistry,
    data_store: BulkDataStore,
    params,
    output_dir: Path,
) -> Tuple[ProcessingResults, Mapping[str, object]]:
    """Execute the full EOD workflow for the provided data."""

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

    if "compliance" in params.operations and compliance_payload:
        artefacts["compliance"] = build_compliance_reports(
            results,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            create_pdf=params.create_pdf,
        )

    if any(name in params.operations for name in ("reconciliation", "nav_reconciliation")):
        artefacts["reconciliation"] = build_nav_reconciliation_reports(
            holdings_results=reconciliation_payload if "reconciliation" in params.operations else None,
            nav_results=nav_payload if "nav_reconciliation" in params.operations else None,
            report_date=params.trade_date,
            output_dir=str(output_dir),
            create_pdf=params.create_pdf,
            compliance_results=compliance_payload if "compliance" in params.operations else None,
        )

    return results, artefacts


def run_trading_mode(
    registry: FundRegistry,
    *,
    ex_ante_store: BulkDataStore,
    ex_post_store: BulkDataStore,
    params,
    output_dir: Path,
):
    """Execute the trading compliance comparison."""

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

    artefacts = build_trading_compliance_reports(
        results_ex_ante=ante_payload,
        results_ex_post=post_payload,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        create_pdf=params.create_pdf,
    )

    return results_ex_ante, results_ex_post, artefacts


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


def flatten_trading_paths(artefacts) -> Dict[str, str]:  # type: ignore[no-untyped-def]
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


def _run_operations(
    registry: FundRegistry,
    data_store: BulkDataStore,
    *,
    operations: Sequence[str],
    analysis_type: str,
    compliance_tests: Sequence[str],
) -> ProcessingResults:
    manager = FundManager(registry, data_store, analysis_type=analysis_type)
    return manager.run_daily_operations(
        list(operations), compliance_tests=list(compliance_tests)
    )


def _extract_payload(results: ProcessingResults, attribute: str) -> Dict[str, Mapping[str, object]]:
    payload: Dict[str, Mapping[str, object]] = {}
    for fund_name, fund_result in results.fund_results.items():
        value = getattr(fund_result, attribute, None)
        if value:
            payload[fund_name] = value
    return payload


__all__ = [
    "DataLoadRequest",
    "plan_eod_requests",
    "plan_trading_requests",
    "fetch_data_stores",
    "run_eod_mode",
    "run_trading_mode",
    "flatten_eod_paths",
    "flatten_trading_paths",
]