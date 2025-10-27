"""Holdings reconciliation reporting utilities."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from reporting.base_report_pdf import BaseReportPDF
from reporting.report_utils import (
    ensure_dataframe,
    normalize_reconciliation_payload,
    normalize_report_date,
)

from reporting.base_report_pdf import BaseReportPDF
from reporting.report_utils import (
    ensure_dataframe,
    normalize_reconciliation_payload,
    normalize_report_date,
)


@dataclass
class GeneratedReconciliationReport:
    """Container for holdings reconciliation artefact locations."""

    excel_path: Optional[str]
    pdf_path: Optional[str]


class ReconciliationExcelReport:
    """Write reconciliation results to an Excel workbook."""

    def __init__(
        self,
        results: Mapping[str, Any],
        report_date: str,
        output_path: Path,
    ) -> None:
        self.results = normalize_reconciliation_payload(results)
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
            for recon_type, metrics in sorted(summary.items()):
                if isinstance(metrics, Mapping):
                    for metric_name, value in sorted(metrics.items()):
                        rows.append(
                            {
                                "Fund": fund_name,
                                "Reconciliation": recon_type,
                                "Metric": metric_name,
                                "Value": value,
                            }
                        )
                else:
                    rows.append(
                        {
                            "Fund": fund_name,
                            "Reconciliation": recon_type,
                            "Metric": "value",
                            "Value": metrics,
                        }
                    )

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["Fund", "Reconciliation", "Metric", "Value"])
        df.to_excel(writer, sheet_name="Summary", index=False)

    def _write_detail_sheets(self, writer: pd.ExcelWriter) -> None:
        for fund_name, payload in sorted(self.results.items()):
            details = payload.get("details", {})
            for recon_type, sections in sorted(details.items()):
                if isinstance(sections, Mapping):
                    for section_name, data in sorted(sections.items()):
                        df = ensure_dataframe(data)
                        if df.empty:
                            continue
                        sheet_name = self._make_sheet_name(fund_name, recon_type, section_name)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    df = ensure_dataframe(sections)
                    if df.empty:
                        continue
                    sheet_name = self._make_sheet_name(fund_name, recon_type, "details")
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _make_sheet_name(self, fund: str, recon_type: str, section: str) -> str:
        base = f"{fund[:10]}-{recon_type[:10]}-{section[:8]}"
        return base[:31]


class ReconciliationSummaryPDF(BaseReportPDF):
    """High-level summary of holdings reconciliation breaks."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.results = normalize_reconciliation_payload(results)

    def render(self) -> None:
        self.add_title("Holdings Reconciliation Summary", f"As of {self.report_date}")

        totals = self._aggregate_totals()
        if totals:
            self.add_section_heading("Breaks by Reconciliation")
            rows = [
                (name.replace("_", " ").title(), count)
                for name, count in sorted(totals.items())
            ]
            self.add_table(["Reconciliation", "Breaks"], rows, column_widths=[110, 30], alignments=["L", "R"])

        for fund_name, payload in sorted(self.results.items()):
            summary = payload.get("summary", {})
            if not summary:
                continue
            self.add_section_heading(fund_name)
            rows = []
            for recon_type, metrics in sorted(summary.items()):
                total_breaks = self._sum_numeric(metrics)
                rows.append((recon_type.replace("_", " ").title(), total_breaks))
            if rows:
                self.add_key_value_table(rows, header=("Reconciliation", "Breaks"))

        self.output()

    # ------------------------------------------------------------------
    def _aggregate_totals(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for payload in self.results.values():
            summary = payload.get("summary", {})
            for recon_type, metrics in summary.items():
                totals[recon_type] = totals.get(recon_type, 0) + self._sum_numeric(metrics)
        return totals

    def _sum_numeric(self, metrics: Any) -> int:
        if isinstance(metrics, Mapping):
            total = 0
            for value in metrics.values():
                if isinstance(value, (int, float)):
                    total += int(value)
            return total
        if isinstance(metrics, (int, float)):
            return int(metrics)
        return 0


def generate_reconciliation_reports(
    results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "reconciliation_results",
    create_pdf: bool = True,
) -> GeneratedReconciliationReport:
    """Generate holdings reconciliation Excel and PDF files."""

    normalized = normalize_reconciliation_payload(results)
    if not normalized:
        return GeneratedReconciliationReport(None, None)

    date_str = normalize_report_date(report_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    pdf_path = output_path / f"{file_name_prefix}_{date_str}.pdf"

    ReconciliationExcelReport(normalized, date_str, excel_path)

    pdf_result: Optional[str] = None
    if create_pdf:
        pdf = ReconciliationSummaryPDF(str(pdf_path), date_str, normalized)
        pdf.render()
        pdf_result = str(pdf_path)

    return GeneratedReconciliationReport(str(excel_path), pdf_result)