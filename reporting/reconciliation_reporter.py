"""High level orchestration for reconciliation report generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional

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
class GeneratedReconciliationSuite:
    """Bundle describing all reconciliation related outputs."""

    reconciliation: Optional[GeneratedReconciliationReport]
    nav_reconciliation: Optional[GeneratedNAVReconciliationReport]
    reconciliation_summary_pdf: Optional[str]
    daily_summary_pdf: Optional[str]


def build_reconciliation_reports(
    results: ProcessingResults,
    report_date: date | datetime | str,
    output_dir: str,
    *,
    compliance_results: Mapping[str, Any] | None = None,
    create_pdf: bool = True,
) -> GeneratedReconciliationSuite:
    """Generate holdings, NAV, and combined reconciliation artefacts."""

    reconciliation_payload = {
        fund_name: fund_result.reconciliation_results
        for fund_name, fund_result in results.fund_results.items()
        if fund_result.reconciliation_results
    }

    nav_payload = {
        fund_name: fund_result.nav_results
        for fund_name, fund_result in results.fund_results.items()
        if fund_result.nav_results
    }

    reconciliation_report = (
        generate_reconciliation_reports(
            reconciliation_payload,
            report_date,
            output_dir,
            create_pdf=create_pdf,
        )
        if reconciliation_payload
        else None
    )

    nav_report = (
        generate_nav_reconciliation_reports(
            nav_payload,
            report_date,
            output_dir,
            create_pdf=create_pdf,
        )
        if nav_payload
        else None
    )

    reconciliation_summary = (
        generate_reconciliation_summary_pdf(
            reconciliation_payload,
            nav_payload,
            report_date,
            output_dir,
        )
        if create_pdf
        else None
    )

    daily_summary = (
        generate_daily_operations_pdf(
            compliance_results or {},
            reconciliation_payload,
            nav_payload,
            report_date,
            output_dir,
        )
        if create_pdf
        else None
    )

    return GeneratedReconciliationSuite(
        reconciliation=reconciliation_report,
        nav_reconciliation=nav_report,
        reconciliation_summary_pdf=reconciliation_summary,
        daily_summary_pdf=daily_summary,
    )