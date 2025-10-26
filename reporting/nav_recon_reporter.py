"""Convenience helpers for generating NAV reconciliation artefacts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from processing.fund_manager import ProcessingResults
from reporting.nav_recon_reporting import (
    GeneratedNAVReconciliationReport,
    generate_nav_reconciliation_reports,
)


def build_nav_reconciliation_reports(
    results: ProcessingResults,
    report_date: date | datetime | str,
    output_dir: str,
    *,
    create_pdf: bool = True,
) -> Optional[GeneratedNAVReconciliationReport]:
    """Build NAV reconciliation Excel/PDF outputs from processed results."""

    nav_payload = {
        fund_name: fund_result.nav_results
        for fund_name, fund_result in results.fund_results.items()
        if fund_result.nav_results
    }

    if not nav_payload:
        return None

    return generate_nav_reconciliation_reports(
        nav_payload,
        report_date,
        output_dir,
        create_pdf=create_pdf,
    )