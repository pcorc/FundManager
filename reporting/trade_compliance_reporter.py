"""Orchestration utilities for trading compliance reporting."""

import logging
from copy import copy as copy_style
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from openpyxl import load_workbook
from PyPDF2 import PdfMerger

from reporting.compliance_reporting import GeneratedComplianceReport
from reporting.report_utils import ensure_dir, normalize_report_date
from reporting.trade_compliance_analyzer import TradingComplianceAnalyzer
from reporting.trade_compliance_reporting import (
    GeneratedTradingComplianceReport,
    generate_trading_excel_report,
    generate_trading_pdf_report,
)

logger = logging.getLogger(__name__)


def build_trading_compliance_reports(
    results_ex_ante: Mapping[str, Any],
    results_ex_post: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    traded_funds_info: Mapping[str, Mapping[str, Any]] | None = None,
    file_name_prefix: str = "trading_analysis",
    create_pdf: bool = True,
) -> Optional[GeneratedTradingComplianceReport]:
    """Generate trading compliance comparison reports."""

    if not (results_ex_ante or results_ex_post):
        return None

    comparison_data = TradingComplianceAnalyzer(
        results_ex_ante=results_ex_ante,
        results_ex_post=results_ex_post,
        date=report_date,
        traded_funds_info=traded_funds_info or {},
    ).analyze()

    date_str = normalize_report_date(report_date)
    output_path = ensure_dir(output_dir)
    excel_path = str(output_path / f"{file_name_prefix}_{date_str}.xlsx")
    pdf_path = str(output_path / f"{file_name_prefix}_{date_str}.pdf")

    generate_trading_excel_report(comparison_data, excel_path)
    if create_pdf:
        generate_trading_pdf_report(comparison_data, pdf_path)

    return GeneratedTradingComplianceReport(excel_path, pdf_path if create_pdf else None, comparison_data)

# Aggregated Excel/PDF artefacts for trading + ex-post compliance (same fields).
CombinedTradingComplianceReport = GeneratedComplianceReport


def combine_trading_and_compliance_reports(
    *,
    trading_report: Optional[GeneratedTradingComplianceReport],
    compliance_report: Optional[object],
    report_date: date | datetime | str,
    output_dir: str,
    file_name_prefix: str,
) -> Optional[CombinedTradingComplianceReport]:
    """Merge trading compliance and ex-post compliance outputs into single files."""

    trading_excel = getattr(trading_report, "excel_path", None)
    compliance_excel = getattr(compliance_report, "excel_path", None)
    trading_pdf = getattr(trading_report, "pdf_path", None)
    compliance_pdf = getattr(compliance_report, "pdf_path", None)

    date_str = normalize_report_date(report_date)
    output_path = ensure_dir(output_dir)

    combined_excel = None
    if trading_excel and compliance_excel:
        combined_excel = _merge_excel_workbooks(
            Path(trading_excel),
            Path(compliance_excel),
            output_path / f"{file_name_prefix}_{date_str}.xlsx",
        )

    # Merge whenever an ex-post compliance pass ran, even if its PDF failed to render
    # (pdf_path None): _merge_pdfs keeps the trading pages so the combined PDF still exists.
    combined_pdf = None
    if trading_pdf and compliance_report is not None:
        combined_pdf = _merge_pdfs(
            Path(trading_pdf),
            Path(compliance_pdf) if compliance_pdf else None,
            output_path / f"{file_name_prefix}_{date_str}.pdf",
        )

    if combined_excel or combined_pdf:
        return CombinedTradingComplianceReport(combined_excel, combined_pdf)

    return None


def _merge_excel_workbooks(
    trading_path: Path, compliance_path: Path, output_path: Path
) -> Optional[str]:
    missing = [str(p) for p in (trading_path, compliance_path) if not Path(p).is_file()]
    if missing:
        logger.warning("Combined trading workbook skipped; missing input: %s", ", ".join(missing))
        return None

    trading_wb = load_workbook(trading_path)
    compliance_wb = load_workbook(compliance_path)

    for sheet_name in compliance_wb.sheetnames:
        _append_sheet(compliance_wb[sheet_name], trading_wb)

    trading_wb.save(output_path)
    return str(output_path)


def _append_sheet(source_sheet, target_wb):
    target_sheet = target_wb.create_sheet(source_sheet.title)

    for row in source_sheet.iter_rows():
        for cell in row:
            target_cell = target_sheet.cell(row=cell.row, column=cell.col_idx, value=cell.value)
            if cell.has_style:
                target_cell._style = copy_style(cell._style)
            if cell.hyperlink:
                target_cell._hyperlink = copy_style(cell.hyperlink)
            if cell.comment:
                target_cell.comment = copy_style(cell.comment)

    for key, dimension in source_sheet.column_dimensions.items():
        target_sheet.column_dimensions[key].width = dimension.width

    for idx, dimension in source_sheet.row_dimensions.items():
        target_sheet.row_dimensions[idx].height = dimension.height

    for merged_cell in source_sheet.merged_cells.ranges:
        target_sheet.merge_cells(str(merged_cell))


def _merge_pdfs(
    trading_pdf: Path, compliance_pdf: Optional[Path], output_path: Path
) -> Optional[str]:
    """Merge the trading-compliance and ex-post compliance PDFs into one combined file.

    Skips any source PDF that is None or doesn't exist on disk (the compliance PDF can
    fail to render, which is logged and reported as pdf_path None) so the trading run
    still produces a useful combined output. Returns None if no source PDFs were available.
    """
    available = [p for p in (trading_pdf, compliance_pdf) if p is not None and Path(p).exists()]
    if not available:
        return None

    merger = PdfMerger()
    try:
        for path in available:
            merger.append(str(path))
        merger.write(output_path)
    finally:
        merger.close()

    return str(output_path)