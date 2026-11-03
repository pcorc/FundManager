"""ComplianceReportPDF cell logic on hand-built checker payloads (no DB, no PDF written).

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    unittest.main()
