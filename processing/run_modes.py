"""High level orchestration helpers for FundManager processing modes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import logging

from notifications.email_manager import EmailManager, EmailPayload

from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from processing.bulk_data_loader import BulkDataLoader, BulkDataStore
from processing.fund_manager import FundManager, ProcessingResults
from processing.fund import FundRegistry
from reporting.compliance_reporter import (
    build_compliance_reports,
    build_compliance_reports_for_range,
)
from reporting.compliance_reporting import GeneratedComplianceReport
from config.operation_config import OperationConfig, OperationMode
from reporting.nav_recon_reporter import build_nav_reconciliation_reports
from utilities.workdays import business_day_offset, iter_business_days
from reporting.trade_compliance_reporter import (
    build_trading_compliance_reports,
    combine_trading_and_compliance_reports,
)

def _subdir(base: Path, name: str) -> Path:
    """Return base / name, creating the directory if needed."""
    out = Path(base) / name
    out.mkdir(parents=True, exist_ok=True)
    return out

@dataclass
class EODRunParameters:
    """Resolved configuration for an EOD analysis run."""

    trade_date: date
    previous_trade_date: date
    operations: List[str]
    compliance_tests: List[str]
    output_tag: str | None
    create_pdf: bool
    send_email: bool = True


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
    output_tag: str | None


@dataclass(frozen=True)
class RangeRunResults:
    """Aggregated artefacts for a multi-day EOD run."""

    results_by_date: "OrderedDict[date, ProcessingResults]"
    daily_artefacts: "OrderedDict[str, Mapping[str, object]]"
    stacked_compliance: Optional["GeneratedComplianceReport"]


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
        return f"{base}_{output_tag}" if output_tag else base

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
            output_dir=str(_subdir(output_dir, "Compliance")),
            file_name_prefix=_build_prefix("compliance_results"),
            test_functions=params.compliance_tests or None,
            gics_mapping=gics_mapping,
            create_pdf=params.create_pdf,
        )

    if any(name in params.operations for name in ("reconciliation", "nav_reconciliation")):
        artefacts["reconciliation"] = build_nav_reconciliation_reports(
            holdings_results=reconciliation_payload if "reconciliation" in params.operations else None,
            nav_results=nav_payload if "nav_reconciliation" in params.operations else None,
            report_date=params.trade_date,
            output_dir=str(_subdir(output_dir, "Holdings and NAV Recon")),
            file_name_prefix=_build_prefix("reconciliation_summary"),
            create_pdf=params.create_pdf,
        )

    # Send the EOD notification email (HTML body + PDF/Excel attachments).
    if getattr(params, "send_email", True):
        try:
            EmailManager().send(
                _build_eod_email_payload(results, artefacts, params),
                OperationConfig.create(OperationMode.EOD),
            )
        except Exception:
            logging.getLogger(__name__).exception("EOD email notification failed")

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
        file_name_prefix=_build_prefix("trading_analysis"),
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

    try:
        EmailManager().send(
            _build_trading_email_payload(results_ex_ante, results_ex_post, combined_artefacts, params),
            OperationConfig.create(OperationMode.TRADING_COMPLIANCE),
        )
    except Exception:
        logging.getLogger(__name__).exception("Trading-analysis email notification failed")

    return results_ex_ante, results_ex_post, combined_artefacts

def run_eod_range_mode(
    session,
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

    loader = BulkDataLoader(session, registry)
    results_by_date: "OrderedDict[date, ProcessingResults]" = OrderedDict()
    daily_artefacts: "OrderedDict[str, Mapping[str, object]]" = OrderedDict()

    latest_gics_mapping = None
    stacked_report = None

    def _build_prefix(base: str) -> str:
        return f"{base}_{output_tag}" if output_tag else base

    for trade_date in iter_business_days(start_date, end_date):
        previous_date = business_day_offset(trade_date, -1)
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

            artefacts["reconciliation"] = build_nav_reconciliation_reports(
                holdings_results=reconciliation_payload if "reconciliation" in operations else None,
                nav_results=nav_payload if "nav_reconciliation" in operations else None,
                report_date=trade_date,
                output_dir=str(output_dir),
                create_pdf=create_pdf,
            )

        daily_artefacts[trade_date.isoformat()] = artefacts

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


# ---------------------------------------------------------------------------
# Email wiring
# ---------------------------------------------------------------------------
def _summary_count(summary: Mapping[str, object], *keys: str) -> int:
    """Pull a numeric count out of a holdings-summary subdict by trying ``keys``."""
    for k in keys:
        value = summary.get(k) if isinstance(summary, Mapping) else None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _build_eod_email_payload(
    results: ProcessingResults,
    artefacts: Mapping[str, object],
    params,
) -> EmailPayload:
    """Assemble the email payload for an EOD run."""
    nav_by_fund: Dict[str, Mapping[str, object]] = _extract_payload(results, "nav_results")
    # Convert NAVReconciliationResults objects → legacy dicts if needed.
    nav_summary: Dict[str, Mapping[str, object]] = {
        fund: (data.to_legacy_dict() if hasattr(data, "to_legacy_dict") else data)
        for fund, data in nav_by_fund.items()
    }

    # Holdings summary table — mirrors the PDF Holdings Summary section.
    recon_payload = _extract_payload(results, "reconciliation_results")
    holdings_summary: Dict[str, Mapping[str, object]] = {}
    for fund, payload in recon_payload.items():
        summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
        eq    = summary.get("custodian_equity", {}) or {}
        opt   = summary.get("custodian_option", {}) or {}
        flex  = summary.get("custodian_flex_option", {}) or {}
        treas = summary.get("custodian_treasury", {}) or {}
        idx   = summary.get("index_equity", {}) or {}
        holdings_summary[fund] = {
            "equity_holdings":      _summary_count(eq,    "holdings_discrepancies", "final_recon"),
            "equity_price":         _summary_count(eq,    "price_discrepancies", "price_discrepancies_T"),
            "option_holdings":      _summary_count(opt,   "holdings_discrepancies", "final_recon"),
            "option_price":         _summary_count(opt,   "price_discrepancies", "price_discrepancies_T"),
            "flex_option_holdings": _summary_count(flex,  "holdings_discrepancies", "final_recon"),
            "flex_option_price":    _summary_count(flex,  "price_discrepancies", "price_discrepancies_T"),
            "treasury_holdings":    _summary_count(treas, "holdings_discrepancies", "final_recon"),
            "treasury_price":       _summary_count(treas, "price_discrepancies", "price_discrepancies_T"),
            "index_holdings":       _summary_count(idx,   "holdings_discrepancies"),
        }

    gl_summary = {
        fund: {
            "Equity G/L":      data.get("Equity G/L", 0),
            "Option G/L":      data.get("Option G/L", 0),
            "Flex Option G/L": data.get("Flex Option G/L", 0),
            "Treasury G/L":    data.get("Treasury G/L", 0),
        }
        for fund, data in nav_summary.items()
    }

    paths = flatten_eod_paths(artefacts)
    attachment_keys = (
        "reconciliation_summary_pdf",
        "nav_reconciliation_excel",
        "holdings_reconciliation_excel",
        "compliance_pdf",
        "compliance_excel",
    )
    attachments = [Path(paths[k]) for k in attachment_keys if k in paths]

    return EmailPayload(
        date=str(params.trade_date),
        attachments=attachments,
        nav_summary=nav_summary,
        holdings_summary=holdings_summary,
        gl_summary=gl_summary,
    )


def _build_trading_email_payload(
    results_ex_ante: ProcessingResults,
    results_ex_post: ProcessingResults,
    artefacts,
    params,
) -> EmailPayload:
    """Assemble the email payload for a trading-analysis run."""
    # Pull the analyzer output that already powers the PDF / Excel.
    report = getattr(artefacts, "report", None)
    comparison = getattr(report, "comparison_data", None) or {}
    funds_payload = comparison.get("funds", {}) or {}

    # Compliance status changes: fund -> [{test, ante_status, post_status}, ...]
    compliance_changes: Dict[str, list] = {}
    for fund_name, fund_data in funds_payload.items():
        checks = (fund_data or {}).get("checks", {}) or {}
        changed_rows = []
        for check_name, check in checks.items():
            if not isinstance(check, Mapping):
                continue
            post_status = str(check.get("status_after", "") or "").upper()
            # Include the row if (a) the status changed between ex-ante and
            # ex-post, or (b) the fund is currently out of compliance on
            # ex-post regardless of whether it changed.
            if not check.get("changed") and post_status != "FAIL":
                continue
            changed_rows.append({
                "test": check_name,
                "ante_status": check.get("status_before", ""),
                "post_status": check.get("status_after", ""),
            })
        if changed_rows:
            compliance_changes[fund_name] = changed_rows

    # Trade activity summary: fund -> {asset_class: asset_trade_summary dict}
    trade_activity_summary: Dict[str, Mapping[str, Mapping[str, object]]] = {}
    for fund_name, fund_data in funds_payload.items():
        trade_info = (fund_data or {}).get("trade_info") or {}
        asset_summary = trade_info.get("asset_trade_summary") or {}
        if asset_summary:
            trade_activity_summary[fund_name] = asset_summary

    paths = flatten_trading_paths(artefacts)
    attachment_keys = (
        "trading_compliance_combined_pdf",
        "trading_compliance_combined_excel",
        "trading_compliance_pdf",
        "trading_compliance_excel",
        "trading_ex_post_compliance_pdf",
        "trading_ex_post_compliance_excel",
    )
    attachments = [Path(paths[k]) for k in attachment_keys if k in paths]

    return EmailPayload(
        date=str(params.ex_post_date),
        attachments=attachments,
        compliance_changes=compliance_changes,
        trade_activity_summary=trade_activity_summary,
    )


