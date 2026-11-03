"""High level orchestration helpers for FundManager processing modes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import logging

from notifications.email_manager import GL_COLUMNS, EmailManager, EmailPayload

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

_RECON_OPERATIONS = ("reconciliation", "nav_reconciliation")
# (summary key prefix, reconciliation summary section) for the holdings email table.
_HOLDINGS_EMAIL_SOURCES = (
    ("equity", "custodian_equity"),
    ("option", "custodian_option"),
    ("flex_option", "custodian_flex_option"),
    ("treasury", "custodian_treasury"),
)
_EOD_ATTACHMENT_KEYS = (
    "reconciliation_summary_pdf",
    "nav_reconciliation_excel",
    "holdings_reconciliation_excel",
    "compliance_pdf",
    "compliance_excel",
)
_TRADING_ATTACHMENT_KEYS = (
    "trading_compliance_combined_pdf",
    "trading_compliance_combined_excel",
    "trading_compliance_pdf",
    "trading_compliance_excel",
    "trading_ex_post_compliance_pdf",
    "trading_ex_post_compliance_excel",
)


def _subdir(base: Path, name: str) -> Path:
    """Return base / name, creating the directory if needed."""
    out = Path(base) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _prefix(base: str, output_tag: str | None) -> str:
    return f"{base}_{output_tag}" if output_tag else base


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


def _build_eod_artefacts(
    results: ProcessingResults,
    operations: Sequence[str],
    report_date: date,
    output_dir: Path,
    *,
    subdirs: bool,
    output_tag: str | None,
    compliance_tests: Sequence[str],
    gics_mapping,
    create_pdf: bool,
) -> Dict[str, object]:
    """One day's reports: ``subdirs`` (single-day) writes tagged files into subfolders, else root with untagged recon."""
    compliance_payload = _extract_payload(results, "compliance_results")
    reconciliation_payload = _extract_payload(results, "reconciliation_results")
    nav_payload = _extract_payload(results, "nav_results")

    artefacts: Dict[str, object] = {}
    if "compliance" in operations and compliance_payload:
        artefacts["compliance"] = build_compliance_reports(
            results,
            report_date=report_date,
            output_dir=str(_subdir(output_dir, "Compliance") if subdirs else output_dir),
            file_name_prefix=_prefix("compliance_results", output_tag),
            test_functions=compliance_tests or None,
            gics_mapping=gics_mapping,
            create_pdf=create_pdf,
        )

    if any(name in operations for name in _RECON_OPERATIONS):
        artefacts["reconciliation"] = build_nav_reconciliation_reports(
            holdings_results=reconciliation_payload if "reconciliation" in operations else None,
            nav_results=nav_payload if "nav_reconciliation" in operations else None,
            report_date=report_date,
            output_dir=str(_subdir(output_dir, "Holdings and NAV Recon") if subdirs else output_dir),
            file_name_prefix=_prefix("reconciliation_summary", output_tag if subdirs else None),
            create_pdf=create_pdf,
        )
    return artefacts


def run_eod_mode(
    registry: FundRegistry,
    data_store: BulkDataStore,
    params,
    output_dir: Path,
    *,
    output_tag: str | None = None,
) -> Tuple[ProcessingResults, Mapping[str, object]]:
    """Execute the full EOD workflow for the provided data."""
    results = _run_operations(registry, data_store, params.operations, "eod", params.compliance_tests)
    artefacts = _build_eod_artefacts(
        results,
        params.operations,
        params.trade_date,
        output_dir,
        subdirs=True,
        output_tag=output_tag,
        compliance_tests=params.compliance_tests,
        gics_mapping=getattr(data_store, "gics_mapping", None),
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
    results_ex_ante = _run_operations(registry, ex_ante_store, ["compliance"], "ex_ante", params.compliance_tests)
    results_ex_post = _run_operations(registry, ex_post_store, ["compliance"], "ex_post", params.compliance_tests)

    ante_payload = _extract_payload(results_ex_ante, "compliance_results")
    post_payload = _extract_payload(results_ex_post, "compliance_results")

    if not ante_payload and not post_payload:
        raise RuntimeError("Trading compliance run produced no compliance results to compare")

    # Compliance results hold only test-name keys plus the 'fund_object' FundManager adds.
    traded_funds_info = {
        fund_name: {"fund_object": fund_results["fund_object"]}
        for fund_name, fund_results in post_payload.items()
        if "fund_object" in fund_results
    }

    trading_report = build_trading_compliance_reports(
        results_ex_ante=ante_payload,
        results_ex_post=post_payload,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        traded_funds_info=traded_funds_info,
        file_name_prefix=_prefix("trading_analysis", output_tag),
        create_pdf=params.create_pdf,
    )

    ex_post_compliance_report = None
    if post_payload:
        ex_post_compliance_report = build_compliance_reports(
            results_ex_post,
            report_date=params.ex_post_date,
            output_dir=str(output_dir),
            file_name_prefix=_prefix("compliance_results_expost", output_tag),
            test_functions=params.compliance_tests or None,
            gics_mapping=getattr(ex_post_store, "gics_mapping", None),
            create_pdf=params.create_pdf,
        )

    combined_report = combine_trading_and_compliance_reports(
        trading_report=trading_report,
        compliance_report=ex_post_compliance_report,
        report_date=params.ex_post_date,
        output_dir=str(output_dir),
        file_name_prefix=_prefix("trading_compliance_results_combined", output_tag),
    )

    combined_artefacts = SimpleNamespace(
        report=trading_report,
        ex_post_compliance=ex_post_compliance_report,
        combined_report=combined_report,
    )

    # Always sent: trading runs have no send_email switch.
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

    for trade_date in iter_business_days(start_date, end_date):
        previous_date = business_day_offset(trade_date, -1)
        data_store = loader.load_all_data_for_date(
            trade_date,
            previous_date=previous_date,
            analysis_type="eod",
        )
        latest_gics_mapping = getattr(data_store, "gics_mapping", None)
        day_results = _run_operations(registry, data_store, operations, "eod", compliance_tests)
        results_by_date[trade_date] = day_results

        if generate_daily_reports:
            daily_artefacts[trade_date.isoformat()] = _build_eod_artefacts(
                day_results,
                operations,
                trade_date,
                output_dir,
                subdirs=False,
                output_tag=output_tag,
                compliance_tests=compliance_tests,
                gics_mapping=latest_gics_mapping,
                create_pdf=create_pdf,
            )

    # The stacked report uses the last day's GICS mapping.
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


def _add_report_paths(paths: Dict[str, str], prefix: str, report, kinds: Sequence[str] = ("excel", "pdf")) -> None:
    for kind in kinds:
        path = getattr(report, f"{kind}_path", None)
        if path:
            paths[f"{prefix}_{kind}"] = path


def flatten_eod_paths(artefacts: Mapping[str, object]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    _add_report_paths(paths, "compliance", artefacts.get("compliance"))
    reconciliation = artefacts.get("reconciliation")
    _add_report_paths(paths, "holdings_reconciliation", getattr(reconciliation, "holdings", None), ("excel",))
    _add_report_paths(paths, "nav_reconciliation", getattr(reconciliation, "nav", None), ("excel",))
    combined = getattr(reconciliation, "combined_reconciliation_pdf", None)
    if combined:
        paths["reconciliation_summary_pdf"] = combined
    return paths


def flatten_trading_paths(artefacts) -> Dict[str, str]:  # type: ignore[no-untyped-def]
    paths: Dict[str, str] = {}
    _add_report_paths(paths, "trading_compliance", getattr(artefacts, "report", None))
    _add_report_paths(paths, "trading_ex_post_compliance", getattr(artefacts, "ex_post_compliance", None))
    _add_report_paths(paths, "trading_compliance_combined", getattr(artefacts, "combined_report", None))
    return paths


def _run_operations(
    registry: FundRegistry,
    data_store: BulkDataStore,
    operations: Sequence[str],
    analysis_type: str,
    compliance_tests: Sequence[str],
) -> ProcessingResults:
    manager = FundManager(registry, data_store, analysis_type=analysis_type)
    return manager.run_daily_operations(list(operations), compliance_tests=list(compliance_tests))


def _extract_payload(results: ProcessingResults, attribute: str) -> Dict[str, Mapping[str, object]]:
    """``{fund: getattr(fund_result, attribute)}`` for funds where that attribute is truthy."""
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


def _attachments(paths: Mapping[str, str], keys: Sequence[str]) -> List[Path]:
    return [Path(paths[k]) for k in keys if k in paths]


def _build_eod_email_payload(
    results: ProcessingResults,
    artefacts: Mapping[str, object],
    params,
) -> EmailPayload:
    """Assemble the email payload for an EOD run."""
    # Convert NAVReconciliationResults objects → legacy dicts if needed.
    nav_summary: Dict[str, Mapping[str, object]] = {
        fund: (data.to_legacy_dict() if hasattr(data, "to_legacy_dict") else data)
        for fund, data in _extract_payload(results, "nav_results").items()
    }

    # Holdings summary table — mirrors the PDF Holdings Summary section.
    holdings_summary: Dict[str, Mapping[str, object]] = {}
    for fund, payload in _extract_payload(results, "reconciliation_results").items():
        summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
        sections = [summary.get(key, {}) or {} for _, key in _HOLDINGS_EMAIL_SOURCES]
        index = summary.get("index_equity", {}) or {}
        counts: Dict[str, int] = {}
        for (name, _), section in zip(_HOLDINGS_EMAIL_SOURCES, sections):
            counts[f"{name}_holdings"] = _summary_count(section, "holdings_discrepancies", "final_recon")
            counts[f"{name}_price"] = _summary_count(section, "price_discrepancies", "price_discrepancies_T")
        counts["index_holdings"] = _summary_count(index, "holdings_discrepancies")
        holdings_summary[fund] = counts

    gl_summary = {fund: {key: data.get(key, 0) for key in GL_COLUMNS} for fund, data in nav_summary.items()}

    return EmailPayload(
        date=str(params.trade_date),
        attachments=_attachments(flatten_eod_paths(artefacts), _EOD_ATTACHMENT_KEYS),
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
            # Include the row if the status changed between ex-ante and ex-post,
            # or the fund is out of compliance on ex-post regardless.
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
        asset_summary = ((fund_data or {}).get("trade_info") or {}).get("asset_trade_summary") or {}
        if asset_summary:
            trade_activity_summary[fund_name] = asset_summary

    return EmailPayload(
        date=str(params.ex_post_date),
        attachments=_attachments(flatten_trading_paths(artefacts), _TRADING_ATTACHMENT_KEYS),
        compliance_changes=compliance_changes,
        trade_activity_summary=trade_activity_summary,
    )
