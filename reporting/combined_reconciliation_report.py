"""Combined NAV and holdings reconciliation PDF report - Updated version."""
from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import pandas as pd

from fpdf import FPDF

from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer
from reporting.compliance_summary_extractor import extract_compliance_summary

logger = logging.getLogger(__name__)


class CombinedReconciliationPDF(FPDF, HoldingsReconciliationRenderer):
    """FPDF subclass with navigation helpers for the combined report."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.alias_nb_pages()
        self.section_links: Dict[str, int] = {}
        self.top_link = self.add_link()
        self.summary_links: Dict[str, int] = {}

    @property
    def pdf(self) -> "CombinedReconciliationPDF":
        return self

    def header(self) -> None:
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Combined Reconciliation Report", 0, 1, "C")

    def footer(self) -> None:
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
        self.pdf.summary_links = self._build_summary_links()

        self._generate_report()
        self.pdf.output(str(self.output_path))
        logger.info("Combined reconciliation report generated at %s", self.output_path)

    def _build_links(self) -> Dict[str, int]:
        funds = sorted(self._get_all_funds())
        links = {fund: self.pdf.add_link() for fund in funds}
        return links

    def _build_summary_links(self) -> Dict[str, int]:
        links = {
            "nav": self.pdf.add_link(),
            "holdings": self.pdf.add_link(),
            "gl": self.pdf.add_link(),
        }
        if self.compliance_results:
            links["compliance"] = self.pdf.add_link()
        return links

    def _get_all_funds(self) -> set[str]:
        funds: set[str] = set()
        for fund_data in self.nav_results.values():
            funds.update(fund_data.keys())
        for fund_data in self.holdings_results.values():
            funds.update(fund_data.keys())
        for summary_list in self.nav_summary.values():
            for summary in summary_list:
                fund_name = summary.get("fund") if isinstance(summary, Mapping) else None
                if fund_name:
                    funds.add(fund_name)
        for summary_list in self.holdings_summary.values():
            for summary in summary_list:
                fund_name = summary.get("fund") if isinstance(summary, Mapping) else None
                if fund_name:
                    funds.add(fund_name)
        return funds

    def _generate_report(self) -> None:
        self._add_title_page()
        self._add_nav_summary_page()
        self._add_holdings_summary_page()
        if self.compliance_results:
            self._add_compliance_summary_page()
        self._add_gl_components_page()
        for fund in sorted(self._get_all_funds()):
            self._add_fund_detail_page(fund)

    def _add_title_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 20)
        self.pdf.cell(0, 18, "Combined Reconciliation Report", ln=True, align="C")
        self.pdf.set_font("Arial", size=14)
        self.pdf.cell(0, 10, f"Date: {self.date}", ln=True, align="C")
        self.pdf.set_font("Arial", "I", 10)
        self.pdf.cell(0, 8, f"Generated: {datetime.date.today():%Y-%m-%d}", ln=True, align="C")
        self.pdf.ln(15)

        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 12, "Table of Contents", ln=True, align="C")
        self.pdf.ln(5)

        # Summary Sections
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Summary Sections", ln=True)
        self.pdf.ln(3)

        summary_sections = [
            ("NAV Reconciliation Summary", self.pdf.summary_links.get("nav")),
            ("Holdings Reconciliation Summary", self.pdf.summary_links.get("holdings")),
            ("Gain/Loss Components", self.pdf.summary_links.get("gl")),
        ]

        if self.compliance_results:
            summary_sections.append(("Compliance Summary", self.pdf.summary_links.get("compliance")))

        for label, link_id in summary_sections:
            if link_id:
                self._add_toc_link(label, link_id)


    def _add_toc_link(self, text: str, link_id: int) -> None:
        self.pdf.set_font("Arial", size=11)
        self.pdf.set_text_color(0, 0, 200)
        x, y = self.pdf.get_x(), self.pdf.get_y()
        self.pdf.cell(160, 6, text, ln=False)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 6, "", ln=True, align="R")
        self.pdf.link(x, y, 160, 6, link_id)

    def _add_toc_sub_link(self, text: str, link_id: int) -> None:
        self.pdf.set_font("Arial", size=10)
        self.pdf.set_text_color(0, 0, 150)
        self.pdf.cell(8)
        x, y = self.pdf.get_x(), self.pdf.get_y()
        self.pdf.cell(152, 5, f"- {text}", ln=False)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 5, "", ln=True, align="R")
        self.pdf.link(x, y, 152, 5, link_id)

    def _add_nav_summary_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_link(self.pdf.summary_links.get("nav", self.pdf.top_link), page=self.pdf.page_no())
        self.pdf.set_link(self.pdf.top_link, page=self.pdf.page_no())

        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, "NAV Reconciliation Summary", ln=True, align="C")
        self.pdf.ln(5)
        self._add_nav_summary_table()

    def _add_nav_summary_table(self) -> None:
        cols = [
            ("Fund", 25),
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

        for fund_name, nav_data in self._iter_nav_fund_data():
            nav_data = nav_data or {}

            # Get NAV difference - check multiple possible field names
            nav_diff = float(nav_data.get("NAV Diff ($)", nav_data.get("difference", 0.0)) or 0.0)

            # Check if NAV is good at 2 and 4 decimal places
            ok2 = bool(nav_data.get("NAV Good (2 Digit)", abs(nav_diff) < 0.01))
            ok4 = bool(nav_data.get("NAV Good (4 Digit)", abs(nav_diff) < 0.0001))

            # Fund name with hyperlink
            width = cols[0][1]
            x, y = self.pdf.get_x(), self.pdf.get_y()
            self.pdf.set_font("Arial", "BU", 8)
            self.pdf.set_text_color(0, 0, 200)
            self.pdf.cell(width, 7, fund_name, border=1, align="C")
            if fund_name in self.fund_links:
                self.pdf.link(x, y, width, 7, self.fund_links[fund_name])
            self.pdf.set_font("Arial", size=8)
            self.pdf.set_text_color(0, 0, 0)

            # Format values
            def _fmt(value: Any, digits: int = 2) -> str:
                try:
                    return f"{float(value):,.{digits}f}"
                except (TypeError, ValueError):
                    return ""

            values = [
                _fmt(nav_data.get("Expected TNA", nav_data.get("expected_tna", 0)), 0),
                _fmt(nav_data.get("Custodian TNA", nav_data.get("custodian_tna", 0)), 0),
                _fmt(nav_data.get("TNA Diff ($)", nav_data.get("tna_diff", 0)), 0),
                _fmt(nav_data.get("Expected NAV", nav_data.get("expected_nav", 0))),
                _fmt(nav_data.get("Custodian NAV", nav_data.get("current_nav", 0))),
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
        self.pdf.set_link(self.pdf.summary_links.get("holdings"), page=self.pdf.page_no())
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
            ("Equity Holdings", 25),
            ("Equity Quantity", 25),
            ("Equity Price", 20),
            ("Option Holdings", 25),
            ("Option Quantity", 25),
            ("Option Price", 20),
            ("Index Holdings", 25),
            ("Index Weight", 25),
        ]

        self.pdf.set_font("Arial", "B", 7)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()

        self.pdf.set_font("Arial", size=7)

        for fund_name in sorted(all_summaries):
            summary = all_summaries[fund_name]
            equity_summary = summary.get("custodian_equity", {})
            option_summary = summary.get("custodian_option", {})
            index_summary = summary.get("index_equity", {})

            if not isinstance(equity_summary, Mapping):
                equity_summary = {}
            if not isinstance(option_summary, Mapping):
                option_summary = {}
            if not isinstance(index_summary, Mapping):
                index_summary = {}

            x, y = self.pdf.get_x(), self.pdf.get_y()
            self.pdf.set_font("Arial", "BU", 7)
            self.pdf.set_text_color(0, 0, 200)
            self.pdf.cell(cols[0][1], 6, fund_name, border=1, align="C")
            if fund_name in self.fund_links:
                self.pdf.link(x, y, cols[0][1], 6, self.fund_links[fund_name])
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.set_font("Arial", size=7)

            values = [
                self._resolve_summary_value(equity_summary, "holdings_discrepancies", "final_recon"),
                self._resolve_summary_value(equity_summary, "raw_recon", "quantity_discrepancies"),
                self._resolve_summary_value(equity_summary, "price_discrepancies", "price_discrepancies_T"),
                self._resolve_summary_value(option_summary, "holdings_discrepancies", "final_recon"),
                self._resolve_summary_value(option_summary, "raw_recon", "quantity_discrepancies"),
                self._resolve_summary_value(option_summary, "price_discrepancies", "price_discrepancies_T"),
                self._resolve_summary_value(index_summary, "holdings_discrepancies"),
                self._resolve_summary_value(index_summary, "significant_diffs", "weight_breaks"),
            ]

            for val, (_, width) in zip(values, cols[1:], strict=False):
                highlight = isinstance(val, (int, float)) and val > 0
                self.pdf.set_fill_color(255, 200, 200) if highlight else self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(width, 6, str(int(val)) if isinstance(val, (int, float)) else str(val), border=1, fill=True, align="C")
            self.pdf.ln()

    def _add_compliance_summary_page(self) -> None:
        self.pdf.add_page()
        link_id = self.pdf.summary_links.get("compliance")
        if link_id:
            self.pdf.set_link(link_id, page=self.pdf.page_no())

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
        self.pdf.cell(0, 5, "Legend: Green = PASS | Red = FAIL | Gray = N/A (Not Applicable)", ln=True)

    def _add_gl_components_page(self) -> None:
        self.pdf.add_page()
        self.pdf.set_link(self.pdf.summary_links.get("gl"), page=self.pdf.page_no())
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Gain/Loss Components by Fund", ln=True, align="C")
        self.pdf.ln(2)

        available_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

        col_defs = [
            {"key": "fund", "label": "Fund", "width": 28, "always": True},
            {
                "key": "beg_tna",
                "label": "Beg. TNA",
                "width": 28,
                "always": True,
                "keys": [
                    "Beginning TNA",
                    "beginning_tna",
                    "beg_tna",
                    "Beginning NAV",
                    "beginning_nav",
                    "prior_nav",
                ],
            },
            {"key": "eqt_gl", "label": "Eqt G/L", "width": 24, "keys": ["Equity G/L", "equity_gl"]},
            {"key": "opt_gl", "label": "Opt G/L", "width": 24, "keys": ["Option G/L", "options_gl"]},
            {
                "key": "flex_gl",
                "label": "Flex G/L",
                "width": 22,
                "keys": ["Flex Option G/L", "flex_options_gl"],
            },
            {
                "key": "tsy_gl",
                "label": "Tsy G/L",
                "width": 22,
                "keys": ["Treasury G/L", "treasury_gl"],
            },
            {
                "key": "assign_gl",
                "label": "Assign G/L",
                "width": 22,
                "keys": ["Assignment G/L", "assignment_gl"],
            },
            {
                "key": "accruals",
                "label": "Accruals",
                "width": 22,
                "keys": ["Accruals", "accruals", "expense_accruals", "Expense Accruals"],
            },
            {
                "key": "distributions",
                "label": "Distributions",
                "width": 22,
                "keys": ["Distributions", "distributions", "distribution"],
            },
            {"key": "other", "label": "Other", "width": 20, "keys": ["Other", "other"]},
            {
                "key": "expected_tna",
                "label": "Expected TNA",
                "width": 28,
                "keys": ["Expected TNA", "expected_tna", "Expected NAV", "expected_nav"],
            },
            {"key": "gl_today", "label": "G/L Today", "width": 26, "always": True},
        ]

        def _resolve_number(nav: Mapping[str, Any], keys: list[str]) -> float:
            for key in keys:
                if key not in nav:
                    continue
                try:
                    return float(nav.get(key) or 0.0)
                except (TypeError, ValueError):
                    continue
            return 0.0

        rows: list[dict[str, float | str]] = []
        totals = {col["key"]: 0.0 for col in col_defs if col["key"] != "fund"}

        for fund_name, nav in self._iter_nav_fund_data():
            nav = nav or {}
            values: dict[str, float | str] = {"fund": fund_name}

            for col in col_defs:
                if col["key"] == "fund":
                    continue
                nav_keys = col.get("keys", [col["label"]])
                values[col["key"]] = _resolve_number(nav, nav_keys)

            values["gl_today"] = (
                values.get("eqt_gl", 0.0)
                + values.get("opt_gl", 0.0)
                + values.get("flex_gl", 0.0)
                + values.get("tsy_gl", 0.0)
                + values.get("assign_gl", 0.0)
                - values.get("accruals", 0.0)
                - abs(values.get("distributions", 0.0))
                + values.get("other", 0.0)
            )

            rows.append(values)
            for key in totals:
                totals[key] += float(values.get(key, 0.0) or 0.0)

        def _has_nonzero(key: str) -> bool:
            return any(abs(float(row.get(key, 0.0) or 0.0)) > 0 for row in rows)

        selected_cols = [
            col
            for col in col_defs
            if col.get("always") or _has_nonzero(col["key"]) or col["key"] == "fund"
        ]

        total_pref_width = sum(col["width"] for col in selected_cols)
        scale = min(1.0, available_width / total_pref_width) if total_pref_width else 1.0

        for col in selected_cols:
            col["render_width"] = max(col["width"] * scale, 16)

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(240, 240, 240)
        for col in selected_cols:
            self.pdf.cell(col["render_width"], 6, col["label"], border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=8)

        for row in rows:
            x, y = self.pdf.get_x(), self.pdf.get_y()
            self.pdf.set_font("Arial", "BU", 8)
            self.pdf.set_text_color(0, 0, 200)
            fund_width = next(col["render_width"] for col in selected_cols if col["key"] == "fund")
            self.pdf.cell(fund_width, 6, str(row["fund"]), border=1, align="C")
            if row["fund"] in self.fund_links:
                self.pdf.link(x, y, fund_width, 6, self.fund_links[row["fund"]])
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.set_font("Arial", size=8)

            for col in selected_cols:
                if col["key"] == "fund":
                    continue
                value = float(row.get(col["key"], 0.0) or 0.0)
                self.pdf.cell(col["render_width"], 6, f"{value:,.0f}", border=1, align="R")
            self.pdf.ln()

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.cell(next(col["render_width"] for col in selected_cols if col["key"] == "fund"), 6, "TOTAL", border=1, fill=True, align="C")
        for col in selected_cols:
            if col["key"] == "fund":
                continue
            value = totals.get(col["key"], 0.0)
            self.pdf.cell(col["render_width"], 6, f"{value:,.0f}", border=1, fill=True, align="R")
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

        funds = sorted(self._get_all_funds())[:5]  # Show first 5 funds
        for fund in funds:
            if fund == current_fund:
                self.pdf.set_font("Arial", "B", 9)
                self.pdf.set_text_color(0, 0, 0)
            else:
                self.pdf.set_font("Arial", "U", 9)
                self.pdf.set_text_color(0, 0, 200)

            fund_x = self.pdf.get_x()
            display_name = fund[:8] if len(fund) > 8 else fund
            self.pdf.cell(25, 8, display_name, ln=False, align="C")

            if fund != current_fund and fund in self.fund_links:
                self.pdf.link(fund_x, start_y, 25, 8, self.fund_links[fund])

        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(10)

    def _add_nav_details(self, nav_data: Mapping[str, Any]) -> None:
        cols = [("Metric", 70), ("Value", 60)]

        # Main NAV metrics table
        rows = [
            ("Expected TNA", nav_data.get("Expected TNA", nav_data.get("expected_tna", 0.0)), 0),
            ("Custodian TNA", nav_data.get("Custodian TNA", nav_data.get("custodian_tna", 0.0)), 0),
            ("TNA Difference", nav_data.get("TNA Diff ($)", nav_data.get("tna_diff", 0.0)), 0),
            ("Expected NAV", nav_data.get("Expected NAV", nav_data.get("expected_nav", 0.0)), 4),
            ("Custodian NAV", nav_data.get("Custodian NAV", nav_data.get("current_nav", 0.0)), 4),
            ("NAV Difference", nav_data.get("NAV Diff ($)", nav_data.get("difference", 0.0)), 4),
        ]

        # Add optional unadjusted options fields if they exist
        if nav_data.get("Expected NAV (Unadj Options)") is not None or nav_data.get("expected_nav_unadjusted") is not None:
            unadj_nav = nav_data.get("Expected NAV (Unadj Options)", nav_data.get("expected_nav_unadjusted"))
            if unadj_nav is not None:
                rows.append(("Expected NAV (Unadj Options)", unadj_nav, 4))

        if nav_data.get("NAV Diff (Adj vs Unadj)") is not None or nav_data.get("nav_diff_adj_vs_unadj") is not None:
            nav_diff_adj = nav_data.get("NAV Diff (Adj vs Unadj)", nav_data.get("nav_diff_adj_vs_unadj"))
            if nav_diff_adj is not None:
                rows.append(("NAV Diff (Adj vs Unadj)", nav_diff_adj, 4))

        self.pdf.set_font("Arial", "B", 10)
        for label, width in cols:
            self.pdf.cell(width, 8, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=10)

        for metric, value, digits in rows:
            self.pdf.cell(cols[0][1], 7, metric, border=1)
            formatted = self._format_currency(value, digits=digits)

            # Highlight unadjusted fields with light yellow background
            if "Unadj" in metric:
                self.pdf.set_fill_color(255, 255, 224)
            else:
                self.pdf.set_fill_color(255, 255, 255)

            self.pdf.cell(cols[1][1], 7, formatted, border=1, fill=True, align="R")
            self.pdf.ln()

        # Gain/Loss Components section
        self.pdf.ln(1)
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 7, "Gain/Loss Components", ln=True)

        gl_rows = [
            ("  Equity G/L", nav_data.get("Equity G/L", nav_data.get("equity_gl", 0.0))),
            ("  Option G/L", nav_data.get("Option G/L", nav_data.get("options_gl", 0.0))),
            ("  Flex Option G/L", nav_data.get("Flex Option G/L", nav_data.get("flex_options_gl", 0.0))),
            ("  Treasury G/L", nav_data.get("Treasury G/L", nav_data.get("treasury_gl", 0.0))),
            ("  Assignment G/L", nav_data.get("Assignment G/L", nav_data.get("assignment_gl", 0.0))),
        ]

        self.pdf.set_font("Arial", size=9)
        for metric, value in gl_rows:
            self.pdf.cell(cols[0][1], 7, metric, border=1)
            self.pdf.cell(cols[1][1], 7, self._format_currency(value, digits=2), border=1, align="R")
            self.pdf.ln()

        accruals_value = nav_data.get("Accruals", nav_data.get("accruals", 0.0))
        distributions_value = nav_data.get("Distributions", nav_data.get("distributions", 0.0))

        for metric, value in (
            ("  Accruals", -abs(accruals_value)),
            ("  Distributions", -abs(distributions_value)),
        ):
            self.pdf.cell(cols[0][1], 7, metric, border=1)
            self.pdf.cell(cols[1][1], 7, self._format_currency(value, digits=2), border=1, align="R")
            self.pdf.ln()

        # NAV Good flags
        self.pdf.ln(3)
        nav_good_rows = [
            ("NAV Good (2 Decimal)", nav_data.get("NAV Good (2 Digit)", nav_data.get("nav_good_two"))),
            ("NAV Good (4 Decimal)", nav_data.get("NAV Good (4 Digit)", nav_data.get("nav_good_four"))),
        ]

        for label, flag in nav_good_rows:
            if flag is None:
                continue
            value = "PASS" if bool(flag) else "FAIL"
            self.pdf.cell(cols[0][1], 7, label, border=1)
            if value == "PASS":
                self.pdf.set_fill_color(200, 255, 200)
            else:
                self.pdf.set_fill_color(255, 200, 200)
            self.pdf.cell(cols[1][1], 7, value, border=1, fill=True, align="C")
            self.pdf.ln()

    def _find_nav_data_for_fund(self, fund_name: str) -> Mapping[str, Any] | None:
        combined = self._collect_nav_data()
        if fund_name in combined:
            return combined[fund_name]
        return None

    def _find_holdings_data_for_fund(self, fund_name: str) -> tuple[Mapping[str, Any] | None, str | None]:
        for date_str, funds_data in self.holdings_results.items():
            if fund_name in funds_data:
                return funds_data[fund_name], date_str
        return None, None

    def _collect_nav_data(self) -> Dict[str, Dict[str, Any]]:
        """Merge NAV details and summary payloads so every fund has data."""
        combined: Dict[str, Dict[str, Any]] = {}

        # First collect from nav_results
        for funds_data in self.nav_results.values():
            for fund_name, nav_data in funds_data.items():
                if not fund_name:
                    continue
                combined.setdefault(fund_name, {}).update(nav_data or {})

        # Then merge in nav_summary data
        for summary_list in self.nav_summary.values():
            for summary in summary_list:
                if not isinstance(summary, Mapping):
                    continue
                fund_name = summary.get("fund")
                if not fund_name:
                    continue
                combined.setdefault(fund_name, {}).update(summary.get("summary", {}) or {})

        return combined

    def _iter_nav_fund_data(self) -> Iterable[tuple[str, Mapping[str, Any]]]:
        for fund_name, nav_data in sorted(self._collect_nav_data().items()):
            yield fund_name, nav_data

    def _resolve_summary_value(self, summary_section: Mapping[str, Any], *keys: str) -> int:
        for key in keys:
            if key not in summary_section:
                continue
            value = summary_section.get(key)
            count = self._coerce_summary_value(value)
            if count is not None:
                return count
        return 0

    def _coerce_summary_value(self, value: Any) -> int | None:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, Mapping) and "count" in value:
            try:
                return int(value["count"])
            except (TypeError, ValueError):
                return 0
        if hasattr(value, "__len__"):
            try:
                return int(len(value))
            except TypeError:
                return 0
        return 0

    def _format_currency(self, value: Any, *, digits: int = 2) -> str:
        if value in (None, ""):
            return ""
        try:
            return f"${float(value):,.{digits}f}"
        except (TypeError, ValueError):
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