"""Convenience helpers for generating NAV and holdings reconciliation artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional
from pathlib import Path

from openpyxl import load_workbook

from reporting.combined_reconciliation_report import CombinedReconciliationReport
from reporting.nav_recon_reporting import (
    GeneratedNAVReconciliationReport,
    NAVReconciliationExcelReport,
    convert_nav_results_to_dicts,
)
from reporting.reconciliation_reporting import (
    GeneratedReconciliationReport,
    generate_reconciliation_reports,
)
from reporting.report_utils import normalize_report_date


@dataclass
class GeneratedReconciliationArtefacts:
    """Bundle of holdings/NAV reconciliation outputs."""
    holdings: Optional[GeneratedReconciliationReport]
    nav: Optional[GeneratedNAVReconciliationReport]
    combined_reconciliation_pdf: Optional[str]


def build_nav_reconciliation_reports(
        *,
        report_date: date | datetime | str = None,
        output_dir: str = "",
        file_name_prefix: str = "reconciliation_summary",
        create_pdf: bool = True,
        holdings_results: Mapping[str, Any] | None = None,
        nav_results: Mapping[str, Any] | None = None,
) -> Optional[GeneratedReconciliationArtefacts]:
    """Build holdings/NAV reconciliation artefacts using provided payloads."""
    holdings: Mapping[str, Any] = holdings_results or {}
    nav: Mapping[str, Any] = nav_results or {}

    if not holdings and not nav:
        return None

    date_str = str(report_date)
    file_stem = f"{file_name_prefix}_{date_str}"
    excel_path = Path(output_dir) / f"{file_stem}.xlsx"

    holdings_report: Optional[GeneratedReconciliationReport] = None
    if holdings:
        holdings_report = generate_reconciliation_reports(
            {date_str: holdings},
            report_date,
            output_dir,
            file_name_prefix=file_name_prefix,
        )

    nav_report: Optional[GeneratedNAVReconciliationReport] = None
    # The NAV sheets are appended to the holdings workbook only when this run just
    # wrote it; a workbook left by an earlier run is replaced, never appended to.
    holdings_xlsx = Path(holdings_report.excel_path) if holdings_report and holdings_report.excel_path else None
    if nav:
        nav = convert_nav_results_to_dicts(nav)
        NAVReconciliationExcelReport(
            results=nav,
            report_date=date_str,
            output_path=excel_path,
            workbook=load_workbook(excel_path) if holdings_xlsx == excel_path else None,
        )
        nav_report = GeneratedNAVReconciliationReport(excel_path=str(excel_path))

    combined_pdf: Optional[str] = None
    if create_pdf:
        pdf_path = Path(output_dir) / f"{file_stem}.pdf"
        CombinedReconciliationReport(
            {fund: payload for fund, payload in nav.items() if payload},
            {fund: payload for fund, payload in holdings.items() if payload},
            normalize_report_date(report_date),
            pdf_path,
        )
        combined_pdf = str(pdf_path)

    return GeneratedReconciliationArtefacts(
        holdings=holdings_report,
        nav=nav_report,
        combined_reconciliation_pdf=combined_pdf,
    )
