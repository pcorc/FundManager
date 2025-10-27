"""Combined NAV and holdings reconciliation PDF report."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from fpdf import FPDF

from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer
from services.compliance_summary_extractor import extract_compliance_summary

logger = logging.getLogger(__name__)


class CombinedReconciliationPDF(FPDF, HoldingsReconciliationRenderer):
    """FPDF subclass with navigation helpers for the combined report."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.alias_nb_pages()
        self.section_links: Dict[str, int] = {}
        self.top_link = self.add_link()

    @property
    def pdf(self) -> "CombinedReconciliationPDF":  # pragma: no cover - convenience
        return self

    # ------------------------------------------------------------------
    def header(self) -> None:  # pragma: no cover - relies on FPDF state
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Combined Reconciliation Report", 0, 1, "C")

    def footer(self) -> None:  # pragma: no cover - relies on FPDF state
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")


class CombinedReconciliationReport:
    """Coordinator that builds the combined reconciliation PDF report."""

    def __init__(
        self,
        nav_reconciliation_results: Mapping[str, Mapping[str, Any]],
        holdings_reconciliation_results: Mapping[str, Mapping[str, Any]],
        nav_recon_summary: Mapping[str, Iterable[Mapping[str, Any]]],
        holdings_recon_summary: Mapping[str, Iterable[Mapping[str, Any]]],
        date: str,
        output_path: str | Path,
        *,
        compliance_results: Mapping[str, Any] | None = None,
    ) -> None:
        self.nav_results = nav_reconciliation_results or {}
        self.holdings_results = holdings_reconciliation_results or {}
        self.nav_summary = nav_recon_summary or {}
        self.holdings_summary = holdings_recon_summary or {}
        self.compliance_results = compliance_results or {}
        self.date = str(date)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.pdf = CombinedReconciliationPDF()
        self.fund_links = self._build_links()

        self._generate_report()
        self.pdf.output(str(self.output_path))
        logger.info("Combined reconciliation report generated at %s", self.output_path)

    # ------------------------------------------------------------------
    def _build_links(self) -> Dict[str, int]:
        funds = sorted(self._get_all_funds())
        links = {fund: self.pdf.add_link() for fund in funds}
        return links

    def _get_all_funds(self) -> set[str]:
        funds: set[str] = set()
        for fund_data in self.nav_results.values():
            funds.update(fund_data.keys())
        for fund_data in self.holdings_results.values():
            funds.update(fund_data.keys())
        return funds

    # ------------------------------------------------------------------
    def _generate_report(self) -> None:
        self._add_nav_summary_page()
        self._add_holdings_summary_page()
        if self.compliance_results:
            self._add_compliance_summary_page()
        self._add_gl_components_page()
        for fund in sorted(self._get_all_funds()):
            self._add_fund_detail_page(fund)

    def _add_nav_summary_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_link(self.pdf.top_link, page=self.pdf.page_no())
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "NAV Reconciliation Summary", ln=True, align="C")
        self.pdf.cell(0, 6, f"Date: {self.date}", ln=True, align="C")
        self.pdf.ln(5)
        self._add_nav_summary_table()

    def _add_nav_summary_table(self) -> None:
        cols = [
            ("Fund", 30),
            ("Expected TNA", 35),
            ("Custodian TNA", 35),
            ("TNA Diff ($)", 30),
            ("Expected NAV", 30),
            ("Custodian NAV", 30),
            ("NAV Diff (2-dec)", 30),
            ("NAV Diff (4-dec)", 30),
        ]

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 7, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=8)

        for funds_data in self.nav_results.values():
            for fund_name, nav_data in sorted(funds_data.items()):
                nav_data = nav_data or {}
                nav_diff = float(nav_data.get("NAV Diff ($)", nav_data.get("difference", 0.0)) or 0.0)
                ok2 = bool(nav_data.get("NAV Good (2 Digit)", abs(nav_diff) < 0.01))
                ok4 = bool(nav_data.get("NAV Good (4 Digit)", abs(nav_diff) < 0.0001))

                width = cols[0][1]
                x, y = self.pdf.get_x(), self.pdf.get_y()
                self.pdf.set_font("Arial", "BU", 8)
                self.pdf.set_text_color(0, 0, 200)
                self.pdf.cell(width, 7, fund_name, border=1, align="C")
                if fund_name in self.fund_links:
                    self.pdf.link(x, y, width, 7, self.fund_links[fund_name])
                self.pdf.set_font("Arial", size=8)
                self.pdf.set_text_color(0, 0, 0)

                def _fmt(value: Any, digits: int = 2) -> str:
                    try:
                        return f"{float(value):,.{digits}f}"
                    except (TypeError, ValueError):
                        return ""

                values = [
                    _fmt(nav_data.get("Expected TNA", nav_data.get("expected_tna")), 0),
                    _fmt(nav_data.get("Custodian TNA", nav_data.get("custodian_tna")), 0),
                    _fmt(nav_data.get("TNA Diff ($)", nav_data.get("tna_diff")), 0),
                    _fmt(nav_data.get("Expected NAV", nav_data.get("expected_nav"))),
                    _fmt(nav_data.get("Custodian NAV", nav_data.get("current_nav"))),
                    _fmt(nav_diff, 2),
                    _fmt(nav_diff, 4),
                ]

                for (label, width), value in zip(cols[1:], values, strict=False):
                    if label == "NAV Diff (2-dec)":
                        self.pdf.set_fill_color(200, 255, 200) if ok2 else self.pdf.set_fill_color(255, 200, 200)
                    elif label == "NAV Diff (4-dec)":
                        self.pdf.set_fill_color(200, 255, 200) if ok4 else self.pdf.set_fill_color(255, 200, 200)
                    else:
                        self.pdf.set_fill_color(255, 255, 255)
                    self.pdf.cell(width, 7, value, border=1, fill=True, align="R")
                self.pdf.ln()

    def _add_holdings_summary_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Holdings Reconciliation Summary", ln=True, align="C")
        self.pdf.cell(0, 6, f"Date: {self.date}", ln=True, align="C")
        self.pdf.ln(5)
        self._add_holdings_summary_table()

    def _add_holdings_summary_table(self) -> None:
        all_summaries: Dict[str, Mapping[str, Any]] = {}
        for summary_list in self.holdings_summary.values():
            for summary in summary_list:
                if isinstance(summary, Mapping) and "fund" in summary:
                    all_summaries[summary["fund"]] = summary.get("summary", {})

        if not all_summaries:
            self.pdf.cell(0, 8, "No holdings summary data available", ln=True)
            return

        cols = [
            ("Fund", 30),
            ("Idx Hold", 20),
            ("Idx Wgt", 20),
            ("Eq Hold", 20),
            ("Eq Qty", 20),
            ("Eq Pr T", 18),
            ("Eq Pr T-1", 18),
            ("Opt Hold", 20),
            ("Opt Qty", 20),
            ("Opt Pr T", 18),
            ("Opt Pr T-1", 18),
        ]

        self.pdf.set_font("Arial", "B", 7)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=7)

        for fund_name in sorted(all_summaries):
            summary = all_summaries[fund_name]
            x, y = self.pdf.get_x(), self.pdf.get_y()
            self.pdf.set_font("Arial", "BU", 7)
            self.pdf.set_text_color(0, 0, 200)
            self.pdf.cell(cols[0][1], 6, fund_name, border=1, align="C")
            if fund_name in self.fund_links:
                self.pdf.link(x, y, cols[0][1], 6, self.fund_links[fund_name])
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.set_font("Arial", size=7)

            values = [
                summary.get("index_equity", {}).get("holdings_discrepancies", 0),
                summary.get("index_equity", {}).get("significant_diffs", 0),
                summary.get("custodian_equity", {}).get("final_recon", 0),
                summary.get("custodian_equity", {}).get("raw_recon", 0),
                summary.get("custodian_equity", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_equity", {}).get("price_discrepancies_T1", 0),
                summary.get("custodian_option", {}).get("final_recon", 0),
                summary.get("custodian_option", {}).get("raw_recon", 0),
                summary.get("custodian_option", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_option", {}).get("price_discrepancies_T1", 0),
            ]

            for val, (_, width) in zip(values, cols[1:], strict=False):
                highlight = isinstance(val, (int, float)) and val > 0
                self.pdf.set_fill_color(255, 200, 200) if highlight else self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(width, 6, str(int(val)) if isinstance(val, (int, float)) else str(val), border=1, fill=True, align="C")
            self.pdf.ln()

    def _add_compliance_summary_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "Compliance Summary", ln=True, align="C")
        self.pdf.cell(0, 6, f"Date: {self.date}", ln=True, align="C")
        self.pdf.ln(5)

        summary = extract_compliance_summary(self.compliance_results)
        if not summary:
            self.pdf.cell(0, 8, "No compliance data available", ln=True)
            return

        cols = [
            ("Fund", 25),
            ("GICS", 18),
            ("80%", 18),
            ("40 Act", 18),
            ("IRS", 18),
            ("IRC", 18),
            ("Illiq", 18),
            ("RE", 18),
            ("Comm", 18),
            ("12d1", 18),
            ("12d2", 18),
            ("12d3", 18),
        ]

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 7, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=8)

        test_keys = ["GICS", "80%", "40 Act", "IRS", "IRC", "Illiquid", "RE", "Comm", "12d1", "12d2", "12d3"]
        for fund_name in sorted(summary):
            fund_summary = summary[fund_name]
            x, y = self.pdf.get_x(), self.pdf.get_y()
            self.pdf.set_font("Arial", "BU", 8)
            self.pdf.set_text_color(0, 0, 200)
            self.pdf.cell(cols[0][1], 6, fund_name, border=1, align="C")
            if fund_name in self.fund_links:
                self.pdf.link(x, y, cols[0][1], 6, self.fund_links[fund_name])
            self.pdf.set_font("Arial", size=8)
            self.pdf.set_text_color(0, 0, 0)

            for key, (_, width) in zip(test_keys, cols[1:], strict=False):
                result = fund_summary.get(key, "N/A")
                if result == "PASS":
                    self.pdf.set_fill_color(200, 255, 200)
                    self.pdf.set_text_color(0, 100, 0)
                elif result == "FAIL":
                    self.pdf.set_fill_color(255, 200, 200)
                    self.pdf.set_text_color(139, 0, 0)
                else:
                    self.pdf.set_fill_color(245, 245, 245)
                    self.pdf.set_text_color(100, 100, 100)
                self.pdf.cell(width, 6, result, border=1, fill=True, align="C")
                self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln()

        self.pdf.ln(5)
        self.pdf.set_font("Arial", "I", 8)
        self.pdf.cell(0, 5, "Legend: Green = PASS | Red = FAIL | Gray = N/A", ln=True)

    def _add_gl_components_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Gain/Loss Components by Fund", ln=True, align="C")
        self.pdf.ln(2)

        cols = [
            ("Fund", 25),
            ("Beg. TNA", 30),
            ("Eqt G/L", 24),
            ("Opt G/L", 24),
            ("Flex G/L", 24),
            ("Tsy G/L", 24),
            ("Assign G/L", 24),
            ("Accruals", 24),
            ("Other", 20),
            ("Expected TNA", 30),
            ("G/L Today", 26),
        ]

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=8)

        totals = {key: 0.0 for key in ["beg_tna", "eqt_gl", "opt_gl", "flex_gl", "tsy_gl", "assign_gl", "accruals", "other", "expected_tna", "gl_today"]}

        for funds in self.nav_results.values():
            for fund_name, nav in sorted(funds.items()):
                nav = nav or {}
                beg_tna = float(nav.get("Beginning TNA", nav.get("prior_nav", 0.0)) or 0.0)
                eqt_gl = float(nav.get("Equity G/L", nav.get("equity_gl", 0.0)) or 0.0)
                opt_gl = float(nav.get("Option G/L", nav.get("options_gl", 0.0)) or 0.0)
                flex_gl = float(nav.get("Flex Option G/L", nav.get("flex_options_gl", 0.0)) or 0.0)
                tsy_gl = float(nav.get("Treasury G/L", nav.get("treasury_gl", 0.0)) or 0.0)
                assign_gl = float(nav.get("Assignment G/L", nav.get("assignment_gl", 0.0)) or 0.0)
                accruals = float(nav.get("Accruals", nav.get("accruals", 0.0)) or 0.0)
                other = float(nav.get("Other", nav.get("other", 0.0)) or 0.0)
                expected_tna = float(nav.get("Expected TNA", nav.get("expected_tna", 0.0)) or 0.0)
                gl_today = eqt_gl + opt_gl + flex_gl + tsy_gl + assign_gl - accruals + other

                x, y = self.pdf.get_x(), self.pdf.get_y()
                self.pdf.set_font("Arial", "BU", 8)
                self.pdf.set_text_color(0, 0, 200)
                self.pdf.cell(cols[0][1], 6, fund_name, border=1, align="C")
                if fund_name in self.fund_links:
                    self.pdf.link(x, y, cols[0][1], 6, self.fund_links[fund_name])
                self.pdf.set_text_color(0, 0, 0)
                self.pdf.set_font("Arial", size=8)

                values = [
                    beg_tna,
                    eqt_gl,
                    opt_gl,
                    flex_gl,
                    tsy_gl,
                    assign_gl,
                    accruals,
                    other,
                    expected_tna,
                    gl_today,
                ]
                for value, (_, width) in zip(values, cols[1:], strict=False):
                    self.pdf.cell(width, 6, f"{value:,.0f}", border=1, align="R")
                self.pdf.ln()

                totals["beg_tna"] += beg_tna
                totals["eqt_gl"] += eqt_gl
                totals["opt_gl"] += opt_gl
                totals["flex_gl"] += flex_gl
                totals["tsy_gl"] += tsy_gl
                totals["assign_gl"] += assign_gl
                totals["accruals"] += accruals
                totals["other"] += other
                totals["expected_tna"] += expected_tna
                totals["gl_today"] += gl_today

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.cell(cols[0][1], 6, "TOTAL", border=1, fill=True, align="C")
        total_values = [
            totals["beg_tna"],
            totals["eqt_gl"],
            totals["opt_gl"],
            totals["flex_gl"],
            totals["tsy_gl"],
            totals["assign_gl"],
            totals["accruals"],
            totals["other"],
            totals["expected_tna"],
            totals["gl_today"],
        ]
        for value, (_, width) in zip(total_values, cols[1:], strict=False):
            self.pdf.cell(width, 6, f"{value:,.0f}", border=1, fill=True, align="R")
        self.pdf.ln()

    def _add_fund_detail_page(self, fund_name: str) -> None:
        self.pdf.add_page()
        if fund_name in self.fund_links:
            self.pdf.set_link(self.fund_links[fund_name], page=self.pdf.page_no())
        self._add_navigation_bar(fund_name)

        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 12, f"Fund: {fund_name} - Details", ln=True, fill=True)
        self.pdf.ln(5)

        nav_data = self._find_nav_data_for_fund(fund_name)
        if nav_data:
            self._add_nav_details(nav_data)

        holdings_data, date_str = self._find_holdings_data_for_fund(fund_name)
        if holdings_data:
            if self.pdf.get_y() > 200:
                self.pdf.add_page()
            else:
                self.pdf.ln(10)
            self.pdf.set_fill_color(230, 230, 230)
            self.pdf.set_font("Arial", "B", 14)
            self.pdf.cell(0, 10, "Holdings Reconciliation", ln=True, fill=True)
            self.pdf.ln(4)
            self.pdf.render_fund_holdings_section(fund_name, date_str or self.date, holdings_data)

    def _add_navigation_bar(self, current_fund: str) -> None:
        start_x = self.pdf.get_x()
        start_y = self.pdf.get_y()
        available_width = self.pdf.w - 2 * self.pdf.l_margin

        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.rect(start_x, start_y, available_width, 8, "F")

        self.pdf.set_font("Arial", "U", 9)
        self.pdf.set_text_color(0, 0, 200)
        self.pdf.cell(30, 8, "Summary", ln=False, align="C")
        self.pdf.link(start_x, start_y, 30, 8, self.pdf.top_link)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(5, 8, "|", ln=False, align="C")

        funds = sorted(self._get_all_funds())
        fund_link_width = 25
        separator_width = 5
        usable_width = available_width - 35
        funds_per_row = max(1, int(usable_width // (fund_link_width + separator_width)))

        if len(funds) <= funds_per_row:
            visible_funds = funds
        else:
            current_idx = funds.index(current_fund)
            start_idx = max(0, current_idx - 2)
            end_idx = min(len(funds), start_idx + funds_per_row)
            if end_idx - start_idx < funds_per_row and end_idx == len(funds):
                start_idx = max(0, len(funds) - funds_per_row)
            visible_funds = funds[start_idx:end_idx]
        for fund in visible_funds:
            if fund == current_fund:
                self.pdf.set_font("Arial", "B", 9)
                self.pdf.set_text_color(0, 0, 0)
            else:
                self.pdf.set_font("Arial", "U", 9)
                self.pdf.set_text_color(0, 0, 200)
            fund_x = self.pdf.get_x()
            display_name = fund[:8] if len(fund) > 8 else fund
            self.pdf.cell(fund_link_width, 8, display_name, ln=False, align="C")
            if fund != current_fund and fund in self.fund_links:
                self.pdf.link(fund_x, start_y, fund_link_width, 8, self.fund_links[fund])
        if len(funds) > len(visible_funds):
            self.pdf.set_text_color(100, 100, 100)
            self.pdf.set_font("Arial", "I", 8)
            start_idx = funds.index(visible_funds[0]) + 1
            end_idx = start_idx + len(visible_funds) - 1
            indicator = f"(Showing {start_idx}-{end_idx} of {len(funds)})"
            self.pdf.cell(fund_link_width, 8, indicator, ln=False, align="C")
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(10)

    def _find_nav_data_for_fund(self, fund_name: str) -> Mapping[str, Any] | None:
        for funds_data in self.nav_results.values():
            if fund_name in funds_data:
                return funds_data[fund_name]
        return None

    def _find_holdings_data_for_fund(self, fund_name: str) -> tuple[Mapping[str, Any] | None, str | None]:
        for date_str, funds_data in self.holdings_results.items():
            if fund_name in funds_data:
                return funds_data[fund_name], date_str
        return None, None

    def _add_nav_details(self, nav_data: Mapping[str, Any]) -> None:
        cols = [("Metric", 70), ("Value", 60)]
        rows = [
            ("Expected TNA", nav_data.get("Expected TNA", nav_data.get("expected_tna", 0.0))),
            ("Custodian TNA", nav_data.get("Custodian TNA", nav_data.get("custodian_tna", 0.0))),
            ("TNA Difference", nav_data.get("TNA Diff ($)", nav_data.get("tna_diff", 0.0))),
            ("Expected NAV", nav_data.get("Expected NAV", nav_data.get("expected_nav", 0.0))),
            ("Custodian NAV", nav_data.get("Custodian NAV", nav_data.get("current_nav", 0.0))),
            ("NAV Difference", nav_data.get("NAV Diff ($)", nav_data.get("difference", 0.0))),
            ("", ""),
            ("Gain/Loss Components", ""),
            ("  Equity G/L", nav_data.get("Equity G/L", nav_data.get("equity_gl", 0.0))),
            ("  Option G/L", nav_data.get("Option G/L", nav_data.get("options_gl", 0.0))),
            ("  Flex Option G/L", nav_data.get("Flex Option G/L", nav_data.get("flex_options_gl", 0.0))),
            ("  Treasury G/L", nav_data.get("Treasury G/L", nav_data.get("treasury_gl", 0.0))),
            ("  Assignment G/L", nav_data.get("Assignment G/L", nav_data.get("assignment_gl", 0.0))),
        ]

        self.pdf.set_font("Arial", "B", 10)
        for label, width in cols:
            self.pdf.cell(width, 8, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=10)

        for metric, value in rows:
            if metric == "":
                self.pdf.ln(3)
                continue
            if metric == "Gain/Loss Components":
                self.pdf.set_font("Arial", "B", 10)
            elif metric.startswith("  "):
                self.pdf.set_font("Arial", size=9)
            else:
                self.pdf.set_font("Arial", size=10)
            self.pdf.cell(cols[0][1], 7, metric, border=1)
            formatted = self._format_value(value)
            if "PASS" in formatted:
                self.pdf.set_fill_color(200, 255, 200)
            elif "FAIL" in formatted:
                self.pdf.set_fill_color(255, 200, 200)
            else:
                self.pdf.set_fill_color(255, 255, 255)
            self.pdf.cell(cols[1][1], 7, formatted, border=1, fill=True, align="R")
            self.pdf.ln()

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
        return str(value)


def build_combined_reconciliation_pdf(
    nav_results: Mapping[str, Any],
    holdings_results: Mapping[str, Any],
    nav_summary: Mapping[str, Iterable[Mapping[str, Any]]],
    holdings_summary: Mapping[str, Iterable[Mapping[str, Any]]],
    date: str,
    output_path: str | Path,
    *,
    compliance_results: Mapping[str, Any] | None = None,
) -> CombinedReconciliationReport:
    """Entry point used by higher level orchestrators."""

    return CombinedReconciliationReport(
        nav_reconciliation_results=nav_results,
        holdings_reconciliation_results=holdings_results,
        nav_recon_summary=nav_summary,
        holdings_recon_summary=holdings_summary,
        date=date,
        output_path=output_path,
        compliance_results=compliance_results,
    )