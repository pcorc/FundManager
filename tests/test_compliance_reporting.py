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


if __name__ == "__main__":
    unittest.main()
