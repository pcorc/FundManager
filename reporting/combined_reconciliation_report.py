"""Combined NAV and holdings reconciliation PDF report - Updated version."""
from __future__ import annotations
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from fpdf import FPDF

from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer

logger = logging.getLogger(__name__)

NAV_SUMMARY_COLS = [
    ("Fund", 25),
    ("Expected TNA", 35),
    ("Custodian TNA", 35),
    ("TNA Diff ($)", 30),
    ("Expected NAV", 30),
    ("Custodian NAV", 30),
    ("NAV Diff (2-dec)", 30),
    ("NAV Diff (4-dec)", 30),
    ("Opt Px Diff", 28),
]

# (label, width, summary section, count keys in priority order)
HOLDINGS_SUMMARY_COLS = [
    ("Equity Holdings", 22, "custodian_equity", ("holdings_discrepancies", "final_recon")),
    ("Equity Qty", 20, "custodian_equity", ("raw_recon", "quantity_discrepancies")),
    ("Equity Price", 20, "custodian_equity", ("price_discrepancies", "price_discrepancies_T")),
    ("Option Holdings", 22, "custodian_option", ("holdings_discrepancies", "final_recon")),
    ("Option Qty", 20, "custodian_option", ("raw_recon", "quantity_discrepancies")),
    ("Option Price", 20, "custodian_option", ("price_discrepancies", "price_discrepancies_T")),
    ("Flex Holdings", 20, "custodian_flex_option", ("holdings_discrepancies", "final_recon")),
    ("Flex Price", 18, "custodian_flex_option", ("price_discrepancies", "price_discrepancies_T")),
    ("Index Holdings", 22, "index_equity", ("holdings_discrepancies",)),
    ("Index Weight", 22, "index_equity", ("significant_diffs", "weight_breaks")),
]

# (key, label, preferred width, always shown, NAV payload keys in priority order)
GL_COLS = [
    ("fund", "Fund", 28, True, ()),
    ("beg_tna", "Beg. TNA", 28, True, ("Beginning TNA",)),
    ("eqt_gl", "Eqt G/L", 24, False, ("Equity G/L",)),
    ("opt_gl", "Opt G/L", 24, False, ("Option G/L",)),
    ("flex_gl", "Flex G/L", 22, False, ("Flex Option G/L",)),
    ("tsy_gl", "Tsy G/L", 22, False, ("Treasury G/L",)),
    ("assign_gl", "Assign G/L", 22, False, ("Assignment G/L",)),
    ("accruals", "Accruals", 22, False, ("Accruals",)),
    ("distributions", "Distributions", 22, False, ("Distributions",)),
    ("other", "Other", 20, False, ("Other",)),
    ("expected_tna", "Expected TNA", 28, False, ("Expected TNA", "Expected NAV")),
    ("gl_today", "G/L Today", 26, True, ()),
]


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _format_currency(value: Any, *, digits: int = 2) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"${float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _resolve_number(nav: Mapping[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key not in nav:
            continue
        try:
            return float(nav.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _summary_count(section: Mapping[str, Any], *keys: str) -> int:
    """Count for the first present key: ints/floats, {'count': n} or len(); 0 otherwise."""
    for key in keys:
        if key not in section:
            continue
        value = section.get(key)
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
    return 0


class CombinedReconciliationPDF(FPDF, HoldingsReconciliationRenderer):
    """FPDF subclass with navigation helpers for the combined report."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.alias_nb_pages()
        self.top_link = self.add_link()

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
    """Coordinator that builds the combined reconciliation PDF report.

    ``nav`` maps fund -> NAV legacy dict; ``holdings`` maps fund -> {"summary", "details"}.
    """

    def __init__(
        self,
        nav: Mapping[str, Mapping[str, Any]],
        holdings: Mapping[str, Mapping[str, Any]],
        date: str,
        output_path: str | Path,
    ) -> None:
        self.nav = nav
        self.holdings = holdings
        self.date = str(date)
        self.funds = sorted(set(nav) | set(holdings))
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.pdf = CombinedReconciliationPDF()
        self.fund_links = self._placeholder_links(self.funds)
        self.summary_links = self._placeholder_links(["nav", "holdings", "gl"])

        self._generate_report()
        self.pdf.output(str(self.output_path))
        logger.info("Combined reconciliation report generated at %s", self.output_path)

    def _placeholder_links(self, keys: Iterable[str]) -> Dict[str, int]:
        links: Dict[str, int] = {}
        for key in keys:
            link_id = self.pdf.add_link()
            # Bind to a placeholder page so .link() doesn't error before the
            # real target page is generated; overwritten later by set_link().
            self.pdf.set_link(link_id, page=1)
            links[key] = link_id
        return links

    def _is_cef(self, fund_name: str) -> bool:
        """True if the fund is a closed-end fund."""
        from config.fund_definitions import CLOSED_END_FUNDS
        return fund_name in CLOSED_END_FUNDS

    def _generate_report(self) -> None:
        self._add_title_page()
        self._add_nav_summary_page()
        self._add_holdings_summary_page()
        self._add_gl_components_page()
        for fund in self.funds:
            self._add_fund_detail_page(fund)

    # ------------------------------------------------------------------
    def _fund_cell(self, fund_name: str, width: float, height: float, size: int) -> None:
        """Blue underlined fund name linking to the fund's detail page."""
        x, y = self.pdf.get_x(), self.pdf.get_y()
        self.pdf.set_font("Arial", "BU", size)
        self.pdf.set_text_color(0, 0, 200)
        self.pdf.cell(width, height, fund_name, border=1, align="C")
        if fund_name in self.fund_links:
            self.pdf.link(x, y, width, height, self.fund_links[fund_name])
        self.pdf.set_font("Arial", size=size)
        self.pdf.set_text_color(0, 0, 0)

    def _header_row(self, cols: Sequence[tuple[str, float]], height: float, size: int) -> None:
        self.pdf.set_font("Arial", "B", size)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, height, label, border=1, fill=True, align="C")
        self.pdf.ln()

    def _page_title(self, link_id: int, title: str, size: int, height: float) -> None:
        self.pdf.add_page()
        self.pdf.set_link(link_id, page=self.pdf.page_no())
        self.pdf.set_font("Arial", "B", size)
        self.pdf.cell(0, height, title, ln=True, align="C")

    # ------------------------------------------------------------------
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

        for label, key in (
            ("NAV Reconciliation Summary", "nav"),
            ("Holdings Reconciliation Summary", "holdings"),
            ("Gain/Loss Components", "gl"),
        ):
            self._add_toc_link(label, self.summary_links[key])

    def _add_toc_link(self, text: str, link_id: int) -> None:
        self.pdf.set_font("Arial", size=11)
        self.pdf.set_text_color(0, 0, 200)
        x, y = self.pdf.get_x(), self.pdf.get_y()
        self.pdf.cell(160, 6, text, ln=False)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 6, "", ln=True, align="R")
        self.pdf.link(x, y, 160, 6, link_id)

    def _add_nav_summary_page(self) -> None:
        self._page_title(self.summary_links["nav"], "NAV Reconciliation Summary", 14, 10)
        self.pdf.set_link(self.pdf.top_link, page=self.pdf.page_no())
        self.pdf.ln(5)

        cols = NAV_SUMMARY_COLS
        self._header_row(cols, 7, 8)
        self.pdf.set_font("Arial", size=8)

        for fund_name, nav_data in sorted(self.nav.items()):
            nav_diff = float(nav_data.get("NAV Diff ($)") or 0.0)
            ok2 = bool(nav_data.get("NAV Good (2 Digit)"))
            opt_px_diff = (
                    float(nav_data.get("combined_gl_cust") or 0.0)
                    - float(nav_data.get("combined_gl_vest") or 0.0)
            )

            self._fund_cell(fund_name, cols[0][1], 7, 8)
            values = [
                _fmt(nav_data.get("Expected TNA"), 0),
                _fmt(nav_data.get("Custodian TNA"), 0),
                _fmt(nav_data.get("TNA Diff ($)"), 0),
                _fmt(nav_data.get("Expected NAV")),
                _fmt(nav_data.get("Custodian NAV")),
                _fmt(nav_diff, 2),
                _fmt(nav_diff, 4),
                _fmt(opt_px_diff, 0),
            ]
            for (label, width), value in zip(cols[1:], values, strict=False):
                # The 4-dec cell is always green by design.
                if label == "NAV Diff (2-dec)" and not ok2:
                    self.pdf.set_fill_color(255, 200, 200)
                elif label in ("NAV Diff (2-dec)", "NAV Diff (4-dec)"):
                    self.pdf.set_fill_color(200, 255, 200)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(width, 7, value, border=1, fill=True, align="R")
            self.pdf.ln()

    def _add_holdings_summary_page(self) -> None:
        self._page_title(self.summary_links["holdings"], "Holdings Reconciliation Summary", 14, 10)
        self.pdf.cell(0, 6, f"Date: {self.date}", ln=True, align="C")
        self.pdf.ln(5)

        if not self.holdings:
            self.pdf.cell(0, 8, "No holdings summary data available", ln=True)
            return

        self._header_row([("Fund", 26)] + [(label, width) for label, width, _, _ in HOLDINGS_SUMMARY_COLS], 6, 7)
        self.pdf.set_font("Arial", size=7)

        for fund_name in sorted(self.holdings):
            summary = self.holdings[fund_name].get("summary", {})
            sections = {
                name: summary.get(name, {})
                for name in ("custodian_equity", "custodian_option", "custodian_flex_option", "index_equity")
            }
            if not isinstance(sections["custodian_flex_option"], Mapping):
                sections["custodian_flex_option"] = {}
            values = [_summary_count(sections[section], *keys) for _, _, section, keys in HOLDINGS_SUMMARY_COLS]

            self._fund_cell(fund_name, 26, 6, 7)
            for val, (label, width, _, _) in zip(values, HOLDINGS_SUMMARY_COLS):
                if val > 0 and ("Price" in label or "Weight" in label):
                    self.pdf.set_fill_color(255, 200, 150)  # light orange → price/weight breaks
                elif val > 0:
                    self.pdf.set_fill_color(255, 200, 200)  # red → holdings breaks
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(width, 6, str(val), border=1, fill=True, align="C")
            self.pdf.ln()

    def _add_gl_components_page(self) -> None:
        self._page_title(self.summary_links["gl"], "Gain/Loss Components by Fund", 12, 8)
        self.pdf.ln(2)

        available_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        value_keys = [key for key, *_ in GL_COLS[1:]]

        rows: list[dict[str, float | str]] = []
        totals = dict.fromkeys(value_keys, 0.0)
        for fund_name, nav in sorted(self.nav.items()):
            values: dict[str, float | str] = {"fund": fund_name}
            for key, _, _, _, nav_keys in GL_COLS[1:]:
                values[key] = _resolve_number(nav, nav_keys)
            values["gl_today"] = (
                values["eqt_gl"] + values["opt_gl"] + values["flex_gl"] + values["tsy_gl"] + values["assign_gl"]
                - values["accruals"] - abs(values["distributions"]) + values["other"]
            )
            rows.append(values)
            for key in totals:
                totals[key] += float(values[key] or 0.0)

        selected = [
            (key, label, width)
            for key, label, width, always, _ in GL_COLS
            if always or any(abs(float(row[key] or 0.0)) > 0 for row in rows)
        ]
        total_pref_width = sum(width for _, _, width in selected)
        scale = min(1.0, available_width / total_pref_width) if total_pref_width else 1.0
        widths = {key: max(width * scale, 16) for key, _, width in selected}
        value_cols = [key for key, _, _ in selected if key != "fund"]

        self._header_row([(label, widths[key]) for key, label, _ in selected], 6, 8)
        self.pdf.set_font("Arial", size=8)

        for row in rows:
            self._fund_cell(str(row["fund"]), widths["fund"], 6, 8)
            for key in value_cols:
                self.pdf.cell(widths[key], 6, f"{float(row[key] or 0.0):,.0f}", border=1, align="R")
            self.pdf.ln()

        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.cell(widths["fund"], 6, "TOTAL", border=1, fill=True, align="C")
        for key in value_cols:
            self.pdf.cell(widths[key], 6, f"{totals[key]:,.0f}", border=1, fill=True, align="R")
        self.pdf.ln()

    # ------------------------------------------------------------------
    def _section_banner(self, title: str, size: int, height: float, gap: float) -> None:
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.set_font("Arial", "B", size)
        self.pdf.cell(0, height, title, ln=True, fill=True)
        self.pdf.ln(gap)

    def _add_fund_detail_page(self, fund_name: str) -> None:
        self.pdf.add_page()
        if fund_name in self.fund_links:
            self.pdf.set_link(self.fund_links[fund_name], page=self.pdf.page_no())

        self._add_navigation_bar(fund_name)
        self._section_banner(f"Fund: {fund_name} - Details", 16, 12, 5)

        # ===== NAV Reconciliation section (always rendered) =====
        self._section_banner("NAV Reconciliation", 14, 10, 3)
        nav_data = self.nav.get(fund_name)
        if nav_data:
            try:
                self._add_nav_details(fund_name, nav_data)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to render NAV details for %s", fund_name)
                self.pdf.set_font("Arial", "I", 9)
                self.pdf.cell(0, 7, f"NAV detail rendering error: {exc}", ln=True)
        else:
            self.pdf.set_font("Arial", "I", 10)
            self.pdf.cell(0, 8, f"No NAV reconciliation data available for {fund_name}.", ln=True)

        # ===== Holdings Reconciliation section (always rendered) =====
        if self.pdf.get_y() > 200:
            self.pdf.add_page()
        else:
            self.pdf.ln(10)
        self._section_banner("Holdings Reconciliation", 14, 10, 4)

        holdings_data = self.holdings[fund_name].get("details", {}) if fund_name in self.holdings else None
        if holdings_data:
            try:
                self.pdf.render_fund_holdings_section(fund_name, self.date, holdings_data)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to render holdings for %s", fund_name)
                self.pdf.set_font("Arial", "I", 9)
                self.pdf.cell(0, 7, f"Holdings rendering error: {exc}", ln=True)
        else:
            self.pdf.set_font("Arial", "I", 10)
            self.pdf.cell(0, 8, f"No holdings reconciliation data available for {fund_name}.", ln=True)

    def _add_navigation_bar(self, current_fund: str) -> None:
        """Show every fund in the navigation bar, wrapping to multiple rows so the
        list always fits the page width regardless of fund count."""
        start_x = self.pdf.get_x()
        start_y = self.pdf.get_y()
        available_width = self.pdf.w - 2 * self.pdf.l_margin
        row_height = 7

        funds = self.funds
        summary_cell_w = 30.0
        separator_w = 5.0

        # Reserve width for "Summary | " on the first row; later rows pack as
        # many default-width fund cells as fit.
        first_row_capacity = available_width - summary_cell_w - separator_w
        fund_cell_w = 25.0
        max_first_row = max(int(first_row_capacity // fund_cell_w), 1)
        if len(funds) > max_first_row:
            funds_per_row = max(int(available_width // fund_cell_w), 1)
        else:
            funds_per_row = max_first_row

        # Background strip for the first row.
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.rect(start_x, start_y, available_width, row_height, "F")

        # "Summary" link
        self.pdf.set_font("Arial", "U", 9)
        self.pdf.set_text_color(0, 0, 200)
        self.pdf.cell(summary_cell_w, row_height, "Summary", ln=False, align="C")
        self.pdf.link(start_x, start_y, summary_cell_w, row_height, self.pdf.top_link)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(separator_w, row_height, "|", ln=False, align="C")

        row_index = 0
        col_index = 0
        for fund in funds:
            if col_index >= (max_first_row if row_index == 0 else funds_per_row):
                self.pdf.ln(row_height)
                row_index += 1
                col_index = 0
                self.pdf.set_fill_color(240, 240, 240)
                self.pdf.rect(start_x, self.pdf.get_y(), available_width, row_height, "F")

            if fund == current_fund:
                self.pdf.set_font("Arial", "B", 9)
                self.pdf.set_text_color(0, 0, 0)
            else:
                self.pdf.set_font("Arial", "U", 9)
                self.pdf.set_text_color(0, 0, 200)

            fund_x = self.pdf.get_x()
            fund_y = self.pdf.get_y()
            self.pdf.cell(fund_cell_w, row_height, fund[:8], ln=False, align="C")
            if fund != current_fund and fund in self.fund_links:
                self.pdf.link(fund_x, fund_y, fund_cell_w, row_height, self.fund_links[fund])
            col_index += 1

        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(row_height + 3)

    def _add_nav_details(self, fund_name: str, nav_data: Mapping[str, Any]) -> None:

        log = logging.getLogger("combined_recon")

        def _n(*keys, default=0.0):
            """First present key coerced to float; 0.0 on miss/bad type."""
            for k in keys:
                if k in nav_data and nav_data[k] is not None:
                    try:
                        return float(nav_data[k])
                    except (TypeError, ValueError):
                        continue
            return float(default)

        # Signal if the G/L keys never made it into nav_data (data-path regression).
        if "Equity G/L" not in nav_data:
            log.warning("NAV details for %s missing 'Equity G/L' (keys=%s)",
                        fund_name, sorted(nav_data)[:12])

        label_w, value_w = 70, 60

        # ===== Main NAV metrics table (header inherits the banner's 230 fill) =====
        rows = [
            ("Expected TNA",   _n("Expected TNA"), 0),
            ("Custodian TNA",  _n("Custodian TNA"), 0),
            ("TNA Difference", _n("TNA Diff ($)"), 0),
            ("Option Pricing Diff", _n("combined_gl_cust") - _n("combined_gl_vest"), 0),
            ("Expected NAV",   _n("Expected NAV"), 4),
            ("Custodian NAV",  _n("Custodian NAV"), 4),
            ("NAV Difference", _n("NAV Diff ($)"), 4),
        ]
        self.pdf.set_font("Arial", "B", 10)
        for label, width in (("Metric", label_w), ("Value", value_w)):
            self.pdf.cell(width, 8, label, border=1, fill=True, align="C")
        self.pdf.ln()
        self.pdf.set_font("Arial", size=10)
        for metric, value, digits in rows:
            self.pdf.cell(label_w, 7, metric, border=1)
            self.pdf.set_fill_color(255, 255, 255)
            self.pdf.cell(value_w, 7, _format_currency(value, digits=digits), border=1, fill=True, align="R")
            self.pdf.ln()

        for title, items in (
            ("Gain/Loss Components", (
                ("  Equity G/L",      _n("Equity G/L")),
                ("  Option G/L",      _n("Option G/L")),
                ("  Flex Option G/L", _n("Flex Option G/L")),
                ("  Treasury G/L",    _n("Treasury G/L")),
            )),
            ("Adjustments", (
                ("  Assignment G/L", _n("Assignment G/L")),
                ("  Accruals",       -abs(_n("Accruals"))),
                ("  Distributions",  -abs(_n("Distributions"))),
            )),
        ):
            self.pdf.ln(2)
            self.pdf.set_font("Arial", "B", 10)
            self.pdf.cell(0, 7, title, ln=True)
            self.pdf.set_font("Arial", size=9)
            for metric, val in items:
                self.pdf.cell(label_w, 7, metric, border=1)
                self.pdf.cell(value_w, 7, _format_currency(val, digits=2), border=1, align="R")
                self.pdf.ln()

        # ===== Option Price Sensitivity matrix (CEF only) =====
        if self._is_cef(fund_name) and (_n("combined_gl_vest") or _n("combined_gl_cust")):
            self.pdf.ln(4)
            self.pdf.set_font("Arial", "B", 10)
            self.pdf.cell(0, 7, "Option Price Sensitivity (Vest vs Custodian Pricing)", ln=True)

            col_w = [55, 35, 35, 25]
            self._header_row(list(zip(("", "Vest Prices", "Custodian Prices", "Impact ($)"), col_w)), 6, 8)

            self.pdf.set_font("Arial", size=8)
            for label, kv, kc in (
                ("Listed Option G/L",   "listed_gl_vest",   "listed_gl_cust"),
                ("Flex Option G/L",     "flex_gl_vest",     "flex_gl_cust"),
                ("Combined Option G/L", "combined_gl_vest", "combined_gl_cust"),
            ):
                v, c = _n(kv), _n(kc)
                self.pdf.cell(col_w[0], 6, label, border=1)
                for w, amount in zip(col_w[1:], (v, c, c - v)):
                    self.pdf.cell(w, 6, _format_currency(amount, digits=0), border=1, align="R")
                self.pdf.ln()

            ev, ec = _n("exp_nav_vest"), _n("exp_nav_cust")
            self.pdf.set_font("Arial", "B", 8)
            self.pdf.cell(col_w[0], 6, "Expected NAV", border=1)
            self.pdf.set_font("Arial", size=8)
            for w, amount in zip(col_w[1:], (ev, ec, ec - ev)):
                self.pdf.cell(w, 6, f"{amount:,.4f}", border=1, align="R")
            self.pdf.ln()

            self.pdf.cell(col_w[0], 6, "Custodian NAV", border=1)
            self.pdf.cell(col_w[1] + col_w[2] + col_w[3], 6, f"{_n('cust_nav'):,.4f}", border=1, align="R")
            self.pdf.ln()

        # ===== NAV Difference ($ and %) =====
        self.pdf.ln(3)
        nav_diff = _n("NAV Diff ($)")
        cust_nav = _n("Custodian NAV")
        nav_diff_pct = (nav_diff / cust_nav * 100.0) if cust_nav else 0.0
        self.pdf.set_font("Arial", "B", 9)
        self.pdf.cell(label_w, 7, "NAV Difference ($ / %)", border=1)
        self.pdf.set_font("Arial", size=9)
        self.pdf.cell(value_w, 7, f"{nav_diff:.4f}   ({nav_diff_pct:.4f}%)", border=1, align="R")
        self.pdf.ln()

        # ===== NAV Good flags — transposed compact row; 4-Dec has a white fill =====
        flags = [
            (lbl, f)
            for lbl, f in (("2-Dec", nav_data.get("NAV Good (2 Digit)")), ("4-Dec", nav_data.get("NAV Good (4 Digit)")))
            if f is not None
        ]
        if flags:
            self.pdf.ln(1)
            cell_w = (label_w + value_w) / len(flags)
            self._header_row([(f"NAV Good {lbl}", cell_w) for lbl, _ in flags], 6, 8)
            self.pdf.set_font("Arial", "B", 9)
            for lbl, f in flags:
                passed = bool(f)
                if lbl == "2-Dec":
                    self.pdf.set_fill_color(200, 255, 200) if passed else self.pdf.set_fill_color(255, 200, 200)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(cell_w, 7, "PASS" if passed else "FAIL", border=1, fill=True, align="C")
            self.pdf.ln()
