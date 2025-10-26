from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from reporting.base_report_pdf import BaseReportPDF
from reporting.report_utils import (
    ensure_dataframe,
    format_number,
    normalize_compliance_results,
    normalize_nav_payload,
    normalize_reconciliation_payload,
    normalize_report_date,
    summarise_compliance_status,
    summarise_nav_differences,
    summarise_reconciliation_breaks,
)


@dataclass
class GeneratedNAVReconciliationReport:
    """File locations for generated NAV reconciliation artefacts."""

    excel_path: Optional[str]
    pdf_path: Optional[str]


class NAVReconciliationExcelReport:
    """Render NAV reconciliation data into an Excel workbook."""

    SUMMARY_COLUMNS = {
        "prior_nav": "Prior NAV",
        "current_nav": "Custodian NAV",
        "expected_nav": "Expected NAV",
        "difference": "Variance",
        "net_gain": "Net Gain/Loss",
        "dividends": "Dividends",
        "expenses": "Expenses",
        "distributions": "Distributions",
        "flows_adjustment": "Flow Adjustment",
    }

    def __init__(
        self,
        results: Mapping[str, Any],
        report_date: str,
        output_path: Path,
    ) -> None:
        self.results = normalize_nav_payload(results)
        self.report_date = report_date
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._export()

    # ------------------------------------------------------------------
    def _export(self) -> None:
        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            self._write_summary_sheet(writer)
            self._write_detail_sheets(writer)

    def _write_summary_sheet(self, writer: pd.ExcelWriter) -> None:
        rows: list[Dict[str, Any]] = []
        for fund_name, payload in sorted(self.results.items()):
            summary = payload.get("summary", {})
            if not summary:
                continue
            row = {"Fund": fund_name}
            for key, display_name in self.SUMMARY_COLUMNS.items():
                row[display_name] = summary.get(key)
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["Fund", *self.SUMMARY_COLUMNS.values()])
        df.to_excel(writer, sheet_name="Summary", index=False)

    def _write_detail_sheets(self, writer: pd.ExcelWriter) -> None:
        for fund_name, payload in sorted(self.results.items()):
            details = payload.get("details", {})
            for component, data in sorted(details.items()):
                df = ensure_dataframe(data)
                if df.empty:
                    continue
                sheet_name = self._make_sheet_name(fund_name, component)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _make_sheet_name(self, fund: str, component: str) -> str:
        base = f"{fund[:15]}-{component[:15]}"
        return base[:31]


class NAVReconciliationPDF(BaseReportPDF):
    """Generate a PDF summary for NAV reconciliation results."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.results = normalize_nav_payload(results)

    def render(self) -> None:
        self.add_title("NAV Reconciliation Summary", f"As of {self.report_date}")

        totals = summarise_nav_differences(self.results)
        if totals["funds"]:
            avg_diff = totals["absolute_difference"] / totals["funds"] if totals["funds"] else 0.0
            self.add_section_heading("Portfolio Overview")
            self.add_key_value_table(
                [
                    ("Funds Analysed", totals["funds"]),
                    ("Total Absolute Variance", format_number(totals["absolute_difference"], 4)),
                    ("Average Absolute Variance", format_number(avg_diff, 4)),
                ],
                header=("Metric", "Value"),
            )

        for fund_name, payload in sorted(self.results.items()):
            summary = payload.get("summary", {})
            if not summary:
                continue
            self.add_section_heading(fund_name)
            rows = [
                ("Prior NAV", format_number(summary.get("prior_nav"), 4)),
                ("Custodian NAV", format_number(summary.get("current_nav"), 4)),
                ("Expected NAV", format_number(summary.get("expected_nav"), 4)),
                ("Variance", format_number(summary.get("difference"), 4)),
                ("Net Gain/Loss", format_number(summary.get("net_gain"), 4)),
                ("Dividends", format_number(summary.get("dividends"), 4)),
                ("Expenses", format_number(summary.get("expenses"), 4)),
                ("Distributions", format_number(summary.get("distributions"), 4)),
                ("Flow Adjustment", format_number(summary.get("flows_adjustment"), 4)),
            ]
            self.add_key_value_table(rows, header=("Metric", "Value"))

        self.output()


class CombinedReconciliationPDF(BaseReportPDF):
    """Combined holdings + NAV reconciliation summary."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        reconciliation_results: Mapping[str, Any],
        nav_results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.reconciliation_results = normalize_reconciliation_payload(reconciliation_results)
        self.nav_results = normalize_nav_payload(nav_results)

    def render(self) -> None:
        self.add_title("Reconciliation Summary", f"As of {self.report_date}")

        if self.reconciliation_results:
            totals = summarise_reconciliation_breaks(self.reconciliation_results)
            self.add_section_heading("Holdings Reconciliation")
            rows = [
                (
                    recon_type.replace("_", " ").title(),
                    totals.get(recon_type, 0),
                )
                for recon_type in sorted(totals)
            ]
            if rows:
                self.add_table([
                    "Reconciliation",
                    "Total Breaks",
                ], rows, column_widths=[100, 40], alignments=["L", "R"])

            for fund_name, payload in sorted(self.reconciliation_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                self.add_section_heading(f"{fund_name} - Breaks")
                fund_rows = []
                for recon_type, metrics in sorted(summary.items()):
                    total_breaks = sum(
                        int(value)
                        for value in (metrics or {}).values()
                        if isinstance(value, (int, float))
                    )
                    fund_rows.append((recon_type.replace("_", " ").title(), total_breaks))
                if fund_rows:
                    self.add_key_value_table(fund_rows, header=("Reconciliation", "Breaks"))

        if self.nav_results:
            self.add_section_heading("NAV Reconciliation")
            totals = summarise_nav_differences(self.nav_results)
            avg_diff = (
                totals["absolute_difference"] / totals["funds"]
                if totals["funds"]
                else 0.0
            )
            self.add_key_value_table(
                [
                    ("Funds Analysed", totals["funds"]),
                    ("Total Absolute Variance", format_number(totals["absolute_difference"], 4)),
                    ("Average Absolute Variance", format_number(avg_diff, 4)),
                ],
                header=("Metric", "Value"),
            )

            detail_rows = []
            for fund_name, payload in sorted(self.nav_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                detail_rows.append(
                    (
                        fund_name,
                        format_number(summary.get("expected_nav"), 4),
                        format_number(summary.get("current_nav"), 4),
                        format_number(summary.get("difference"), 4),
                    )
                )
            if detail_rows:
                self.add_table(
                    ["Fund", "Expected NAV", "Custodian NAV", "Variance"],
                    detail_rows,
                    column_widths=[70, 35, 35, 35],
                    alignments=["L", "R", "R", "R"],
                )

        self.output()


class DailyOperationsSummaryPDF(BaseReportPDF):
    """Combined summary across compliance, reconciliation, and NAV."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        compliance_results: Mapping[str, Any],
        reconciliation_results: Mapping[str, Any],
        nav_results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.compliance_results = normalize_compliance_results(compliance_results or {})
        self.reconciliation_results = normalize_reconciliation_payload(reconciliation_results or {})
        self.nav_results = normalize_nav_payload(nav_results or {})

    def render(self) -> None:
        self.add_title("Daily Oversight Summary", f"As of {self.report_date}")

        if self.compliance_results:
            compliance_summary = summarise_compliance_status(self.compliance_results)
            self.add_section_heading("Compliance Overview")
            rows = [
                ("Funds Processed", compliance_summary["funds"]),
                ("Funds with Breaches", compliance_summary["funds_in_breach"]),
                ("Checks Evaluated", compliance_summary["total_checks"]),
                ("Checks Failed", compliance_summary["failed_checks"]),
            ]
            self.add_key_value_table(rows, header=("Metric", "Value"))

        if self.reconciliation_results:
            totals = summarise_reconciliation_breaks(self.reconciliation_results)
            self.add_section_heading("Holdings Reconciliation")
            rows = [
                (name.replace("_", " ").title(), totals.get(name, 0))
                for name in sorted(totals)
            ]
            if rows:
                self.add_table([
                    "Reconciliation",
                    "Total Breaks",
                ], rows, column_widths=[100, 40], alignments=["L", "R"])

            for fund_name, payload in sorted(self.reconciliation_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                self.add_section_heading(f"{fund_name} - Breaks")
                fund_rows = []
                for recon_type, metrics in sorted(summary.items()):
                    total_breaks = sum(
                        int(value)
                        for value in (metrics or {}).values()
                        if isinstance(value, (int, float))
                    )
                    fund_rows.append((recon_type.replace("_", " ").title(), total_breaks))
                if fund_rows:
                    self.add_key_value_table(fund_rows, header=("Reconciliation", "Breaks"))

        if self.nav_results:
            self.add_section_heading("NAV Reconciliation")
            totals = summarise_nav_differences(self.nav_results)
            avg_diff = (
                totals["absolute_difference"] / totals["funds"]
                if totals["funds"]
                else 0.0
            )
            self.add_key_value_table(
                [
                    ("Funds Analysed", totals["funds"]),
                    ("Total Absolute Variance", format_number(totals["absolute_difference"], 4)),
                    ("Average Absolute Variance", format_number(avg_diff, 4)),
                ],
                header=("Metric", "Value"),
            )

            detail_rows = []
            for fund_name, payload in sorted(self.nav_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                detail_rows.append(
                    (
                        fund_name,
                        format_number(summary.get("expected_nav"), 4),
                        format_number(summary.get("current_nav"), 4),
                        format_number(summary.get("difference"), 4),
                    )
                )
            if detail_rows:
                self.add_table(
                    ["Fund", "Expected NAV", "Custodian NAV", "Variance"],
                    detail_rows,
                    column_widths=[70, 35, 35, 35],
                    alignments=["L", "R", "R", "R"],
                )

        self.output()


def generate_nav_reconciliation_reports(
    results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "nav_reconciliation_results",
    create_pdf: bool = True,
) -> GeneratedNAVReconciliationReport:
    """Generate Excel and PDF NAV reconciliation reports."""

    normalized = normalize_nav_payload(results)
    if not normalized:
        return GeneratedNAVReconciliationReport(None, None)

    date_str = normalize_report_date(report_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    pdf_path = output_path / f"{file_name_prefix}_{date_str}.pdf"

    NAVReconciliationExcelReport(normalized, date_str, excel_path)

    pdf_result: Optional[str] = None
    if create_pdf:
        pdf = NAVReconciliationPDF(str(pdf_path), date_str, normalized)
        pdf.render()
        pdf_result = str(pdf_path)

    return GeneratedNAVReconciliationReport(str(excel_path), pdf_result)


def generate_reconciliation_summary_pdf(
    reconciliation_results: Mapping[str, Any],
    nav_results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name: str = "reconciliation_summary.pdf",
) -> Optional[str]:
    """Create the combined holdings + NAV reconciliation PDF."""

    recon_payload = normalize_reconciliation_payload(reconciliation_results)
    nav_payload = normalize_nav_payload(nav_results)
    if not recon_payload and not nav_payload:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / file_name

    pdf = CombinedReconciliationPDF(str(pdf_path), normalize_report_date(report_date), recon_payload, nav_payload)
    pdf.render()
    return str(pdf_path)


def generate_daily_operations_pdf(
    compliance_results: Mapping[str, Any],
    reconciliation_results: Mapping[str, Any],
    nav_results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name: str = "daily_operations_summary.pdf",
) -> Optional[str]:
    """Create the combined compliance + reconciliation + NAV PDF summary."""

    if not any([compliance_results, reconciliation_results, nav_results]):
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / file_name

    pdf = DailyOperationsSummaryPDF(
        str(pdf_path),
        normalize_report_date(report_date),
        compliance_results,
        reconciliation_results,
        nav_results,
    )
    pdf.render()
    return str(pdf_path)