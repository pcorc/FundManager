"""Convenience helpers for generating NAV and holdings reconciliation artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional
from pathlib import Path

from processing.fund_manager import ProcessingResults
from reporting.nav_recon_reporting import (
    GeneratedNAVReconciliationReport,
    generate_daily_operations_pdf,
    generate_nav_reconciliation_reports,
    generate_reconciliation_summary_pdf,
)
from reporting.reconciliation_reporting import (
    GeneratedReconciliationReport,
    generate_reconciliation_reports,
)


@dataclass
class GeneratedReconciliationArtefacts:
    """Bundle of holdings/NAV reconciliation outputs."""
    holdings: Optional[GeneratedReconciliationReport]
    nav: Optional[GeneratedNAVReconciliationReport]
    combined_reconciliation_pdf: Optional[str]
    full_summary_pdf: Optional[str]


def _extract_results(
        results: ProcessingResults | None,
        attribute: str,
) -> Mapping[str, Any]:
    if results is None:
        return {}
    payload: dict[str, Any] = {}
    for fund_name, fund_result in results.fund_results.items():
        value = getattr(fund_result, attribute, None)
        if value:
            payload[fund_name] = value
    return payload


def build_nav_reconciliation_reports(
        results: ProcessingResults | None = None,
        report_date: date | datetime | str = None,
        output_dir: str = "",
        *,
        create_pdf: bool = True,
        holdings_results: Mapping[str, Any] | None = None,
        nav_results: Mapping[str, Any] | None = None,
        compliance_results: Mapping[str, Any] | None = None,
) -> Optional[GeneratedReconciliationArtefacts]:
    """Build holdings/NAV reconciliation artefacts using provided payloads."""

    derived_holdings: Mapping[str, Any] = holdings_results or _extract_results(results, "reconciliation_results")
    derived_nav: Mapping[str, Any] = nav_results or _extract_results(results, "nav_results")
    derived_compliance: Mapping[str, Any] = compliance_results or _extract_results(results, "compliance_results")

    if not derived_holdings and not derived_nav:
        return None

    holdings_report: Optional[GeneratedReconciliationReport] = None
    if derived_holdings:
        holdings_report = generate_reconciliation_reports(
            derived_holdings,
            report_date,
            output_dir,
            create_pdf=create_pdf,
        )

    nav_report: Optional[GeneratedNAVReconciliationReport] = None
    if derived_nav:
        # Build the Excel path
        excel_path = str(Path(output_dir) / f"nav_reconciliation_{report_date}.xlsx")

        # Call generate_nav_reconciliation_reports with correct signature
        excel_output = generate_nav_reconciliation_reports(
            derived_nav,  # reconciliation_results
            str(report_date),  # date_str
            excel_path  # excel_path
        )

        pdf_output = None
        if create_pdf:
            # Generate PDF if requested - this would be a separate function
            # You might need to implement a PDF generation for NAV reports
            pass

        nav_report = GeneratedNAVReconciliationReport(
            excel_path=excel_output,
            pdf_path=pdf_output
        )

    combined_pdf: Optional[str] = None
    if create_pdf:
        combined_pdf = generate_reconciliation_summary_pdf(
            derived_holdings,
            derived_nav,
            report_date,
            output_dir,
        )

    full_summary_pdf: Optional[str] = None
    if create_pdf:
        full_summary_pdf = generate_daily_operations_pdf(
            derived_compliance,
            derived_holdings,
            derived_nav,
            report_date,
            output_dir,
        )

    return GeneratedReconciliationArtefacts(
        holdings=holdings_report,
        nav=nav_report,
        combined_reconciliation_pdf=combined_pdf,
        full_summary_pdf=full_summary_pdf,
    )