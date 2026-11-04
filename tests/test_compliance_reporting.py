"""ComplianceReportPDF cell logic on hand-built checker payloads (no DB, no PDF written).

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import Workbook, load_workbook  # noqa: E402

from reporting import compliance_reporter, compliance_reporting, trade_compliance_reporter  # noqa: E402
from reporting.compliance_reporting import _PDF_SECTIONS, ComplianceReportPDF  # noqa: E402

IRS = "diversification_IRS_check"


def _pdf() -> ComplianceReportPDF:
    """A renderer without running __init__ (which draws and saves the whole PDF)."""
    return ComplianceReportPDF.__new__(ComplianceReportPDF)


def _row(test_key: str, label: str) -> tuple:
    section = next(s for s in _PDF_SECTIONS if s[1] == test_key)
    return next(metric for metric in section[4] if metric[0] == label)


class IrsCondition2bTests(unittest.TestCase):
    """D9: the checker has no IRS 2b test, so the PDF reads N/A, as the workbook does, not blank."""

    def _cell(self, fund_data):
        _, fmt, source, field = _row(IRS, "Condition 2b")
        return _pdf()._metric_value(fund_data, IRS, fmt, source, field)

    def test_not_evaluated_reads_na(self):
        self.assertEqual(self._cell({IRS: {"is_compliant": True, "condition_IRS_1": True}}), "N/A")

    def test_a_2b_value_is_shown(self):
        self.assertEqual(self._cell({IRS: {"is_compliant": False, "condition_IRS_2_b": False}}), "FAIL")

    def test_no_irs_result_stays_blank(self):
        self.assertEqual(self._cell({}), "")


class IrcNonVitColumnTests(unittest.TestCase):
    """D11: in a run mixing a VIT fund with others, a non-VIT column of the IRC table reads N/A."""

    IRC = "diversification_IRC_check"
    VIT = {IRC: {"is_compliant": True, "condition_IRC_55": True, "calculations": {"top_1": 0.0419}}}

    def _cell(self, fund_data, label):
        _, fmt, source, field = _row(self.IRC, label)
        return _pdf()._cell_text(fund_data, self.IRC, fmt, source, field)

    def test_vit_fund_shows_its_result(self):
        self.assertEqual(self._cell(self.VIT, "Overall Status"), "PASS")
        self.assertEqual(self._cell(self.VIT, "Condition 55"), "PASS")
        self.assertEqual(self._cell(self.VIT, "Top 1 Exposure"), "4.19000%")

    def test_fund_without_irc_result_reads_na_not_zero(self):
        etf = {IRS: {"is_compliant": True}}
        for label in ("Overall Status", "Condition 55", "Top 1 Exposure", "Top 4 Exposure"):
            self.assertEqual(self._cell(etf, label), "N/A", label)


class StackedFileNameTests(unittest.TestCase):
    """D18: the stacked workbook is compliance_results[_<tag>]_<first>_to_<last>, the last date not doubled."""

    @staticmethod
    def _day():
        result = SimpleNamespace(compliance_results={IRS: {"is_compliant": True}})
        return SimpleNamespace(fund_results={"KNG": result})

    def _stem(self, days, **kwargs):
        with mock.patch.object(compliance_reporter, "generate_compliance_reports") as generate:
            compliance_reporter.build_compliance_reports_for_range(
                [(day, self._day()) for day in days], "out", **kwargs)
        return generate.call_args.kwargs["file_stem"]

    def test_range_with_tag(self):
        days = (date(2026, 10, 30), date(2026, 10, 26))
        self.assertEqual(self._stem(days, output_tag="cefs_ts"), "compliance_results_cefs_ts_2026-10-26_to_2026-10-30")

    def test_range_without_tag(self):
        days = (date(2026, 10, 26), date(2026, 10, 30))
        self.assertEqual(self._stem(days), "compliance_results_2026-10-26_to_2026-10-30")

    def test_single_day_keeps_the_range_form(self):
        """Never the daily report's name, which the same run may also write into the same folder."""
        self.assertEqual(self._stem((date(2026, 10, 30),), output_tag="t"), "compliance_results_t_2026-10-30_to_2026-10-30")


class ReportedPathsTests(unittest.TestCase):
    """D21: a report names only the files it wrote, and the trading merge skips a missing input."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _generate(self, sheet_data, pdf_error=None):
        report = SimpleNamespace(sheet_data=sheet_data, file_path=self.tmp / "c_2026-10-30.xlsx",
                                 file_stem="c_2026-10-30", results={})
        with mock.patch.object(compliance_reporting, "ComplianceReport", return_value=report),                 mock.patch.object(compliance_reporting, "ComplianceReportPDF", side_effect=pdf_error) as pdf:
            generated = compliance_reporting.generate_compliance_reports({}, "2026-10-30", str(self.tmp))
        return generated, pdf

    def test_written_files_are_reported(self):
        generated, pdf = self._generate({"Summary": object()})
        self.assertEqual(generated.excel_path, str(self.tmp / "c_2026-10-30.xlsx"))
        self.assertEqual(generated.pdf_path, str(self.tmp / "c_2026-10-30.pdf"))
        self.assertEqual(pdf.call_args.args[1], generated.pdf_path)

    def test_failed_pdf_is_not_reported(self):
        with self.assertLogs(compliance_reporting.logger, "ERROR"):
            generated, _ = self._generate({"Summary": object()}, pdf_error=ValueError("fpdf broke"))
        self.assertIsNone(generated.pdf_path)
        self.assertIsNotNone(generated.excel_path)

    def test_workbook_without_sheets_is_not_reported(self):
        generated, _ = self._generate({})
        self.assertIsNone(generated.excel_path)

    def _workbook(self, name, sheet):
        workbook = Workbook()
        workbook.active.title = sheet
        workbook.save(self.tmp / name)
        return str(self.tmp / name)

    def _combine(self, compliance_excel):
        return trade_compliance_reporter.combine_trading_and_compliance_reports(
            trading_report=SimpleNamespace(excel_path=self._workbook("t.xlsx", "Summary"), pdf_path=None),
            compliance_report=SimpleNamespace(excel_path=compliance_excel, pdf_path=None),
            report_date="2026-10-30", output_dir=str(self.tmp), file_name_prefix="combined")

    def test_merge_skips_a_missing_workbook(self):
        with self.assertLogs(trade_compliance_reporter.logger, "WARNING"):
            self.assertIsNone(self._combine(str(self.tmp / "never_written.xlsx")))
        self.assertFalse((self.tmp / "combined_2026-10-30.xlsx").exists())

    def test_merge_appends_the_compliance_sheets(self):
        combined = self._combine(self._workbook("c.xlsx", "IRS_Diversification"))
        with open(combined.excel_path, "rb") as handle:  # a closed handle, so the temp dir can be removed
            self.assertEqual(load_workbook(handle).sheetnames, ["Summary", "IRS_Diversification"])


if __name__ == "__main__":
    unittest.main()
