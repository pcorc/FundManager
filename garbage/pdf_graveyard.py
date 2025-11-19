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
        self.add_title("NAV Reconciliation Summary")
        self.add_subtitle(f"As of {self.report_date}")

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
            self._add_component_sections(payload.get("details", {}))

        self.output()


    # ------------------------------------------------------------------
    def _add_component_sections(self, details: Mapping[str, Any]) -> None:
        component_titles = [
            ("equity", "Equity Holdings"),
            ("options", "Option Holdings"),
            ("flex_options", "Flex Option Holdings"),
            ("treasury", "Treasury Holdings"),
        ]

        for key, title in component_titles:
            self.add_subtitle(title)
            dataframe = ensure_dataframe(details.get(key))
            if dataframe.empty:
                self.add_paragraph("No data available.", font_size=8)
                continue

            headers, rows = self._build_component_table(dataframe)
            self.add_table(
                headers,
                rows,
                column_widths=[35, 22, 22, 28, 28, 32, 32],
                align=["L", "R", "R", "R", "R", "R", "R"],  # Changed from 'alignments'
            )


    # ------------------------------------------------------------------
    def _build_component_table(self, dataframe: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
        columns = self._resolve_component_columns(dataframe)
        headers = [
            "Ticker",
            "Qty T-1",
            "Qty T",
            "Price T-1",
            "Price T",
            "Raw G/L",
            "Adjusted G/L",
        ]

        rows: list[list[str]] = []
        raw_totals: list[float] = []
        adjusted_totals: list[float] = []

        for _, row in dataframe.iterrows():
            ticker = row.get(columns["ticker"]) if columns["ticker"] else ""
            qty_t1 = self._to_float(row, columns["qty_t1"])
            qty_t = self._to_float(row, columns["qty_t"])
            price_t1 = self._to_float(row, columns["price_t1"])
            price_t = self._to_float(row, columns["price_t"])
            raw_gl = self._to_float(row, columns["raw_gl"])
            adj_gl = self._to_float(row, columns["adjusted_gl"])

            raw_totals.append(raw_gl)
            adjusted_totals.append(adj_gl)

            rows.append(
                [
                    ticker or "",
                    format_number(qty_t1, 0),
                    format_number(qty_t, 0),
                    format_number(price_t1, 4),
                    format_number(price_t, 4),
                    format_number(raw_gl, 2),
                    format_number(adj_gl, 2),
                ]
            )

        totals_row = [
            "TOTAL",
            "",
            "",
            "",
            "",
            format_number(sum(raw_totals), 2),
            format_number(sum(adjusted_totals), 2),
        ]
        rows.append(totals_row)

        return headers, rows

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_component_columns(dataframe: pd.DataFrame) -> Dict[str, Optional[str]]:
        lowered = {str(column).lower(): column for column in dataframe.columns}

        def pick(*candidates: str) -> Optional[str]:
            for candidate in candidates:
                if candidate in lowered:
                    return lowered[candidate]
            return None

        return {
            "ticker": pick("ticker", "symbol", "security", "identifier"),
            "qty_t1": pick("quantity_t1", "qty_t1", "prior_quantity", "quantity_prev"),
            "qty_t": pick("quantity_t", "qty_t", "quantity", "current_quantity"),
            "price_t1": pick("price_t1", "prior_price", "previous_price"),
            "price_t": pick("price_t", "price", "current_price"),
            "raw_gl": pick("raw_gl", "gl_raw", "gain_loss_raw", "raw_gain"),
            "adjusted_gl": pick("adjusted_gl", "gl_adj", "gain_loss_adjusted", "adjusted_gain"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(row: pd.Series, column: Optional[str]) -> float:
        if not column:
            return 0.0
        value = row.get(column)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

class ReconciliationReportPDF(BaseReportPDF, HoldingsReconciliationRenderer):
    """Generate the enhanced holdings reconciliation PDF report."""

    def __init__(
        self,
        reconciliation_results: Mapping[str, Mapping[str, Any]] | None,
        recon_summary: Iterable[Dict[str, Any]] | None,
        date: str | None,
        file_path_pdf: str | Path | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> None:
        output_file = file_path_pdf or output_path or f"reconciliation_report_{date}.pdf"
        super().__init__(str(output_file))
        self.reconciliation_results = reconciliation_results or {}
        self.recon_summary = list(recon_summary or [])
        self.date = str(date) if date else ""

        try:
            self.flattened_results = self._flatten_reconciliation_results()
            self._generate_pdf()
        except Exception as exc:  # pragma: no cover - defensive
            self._generate_error_report(str(exc))

    def _flatten_reconciliation_results(self) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        flattened: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for date_str, fund_data in self.reconciliation_results.items():
            for fund_name, recon_dict in (fund_data or {}).items():
                flattened[(fund_name, date_str)] = recon_dict
        return flattened

    def _generate_error_report(self, error_message: str) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Reconciliation Report - ERROR", ln=True, align="C")
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, f"An error occurred while generating the report: {error_message}")
        self.output()

    def _generate_pdf(self) -> None:
        self.pdf.add_page()
        self._add_header("Reconciliation Report", f"Report Date: {self.date}")
        self._add_consolidated_summary()
        if not self.flattened_results:
            self.pdf.set_font("Arial", "B", 12)
            self.pdf.cell(0, 10, "No reconciliation results available", ln=True, align="C")
        else:
            for (fund_name, date_str), recon_data in sorted(self.flattened_results.items()):
                self.render_fund_holdings_section(fund_name, date_str, recon_data)
        self.output()

    def _add_header(self, title: str, subtitle: str) -> None:
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, title, ln=True, align="C")
        self.pdf.set_font("Arial", "", 11)
        self.pdf.cell(0, 8, subtitle, ln=True, align="C")
        self.pdf.ln(6)

    def _add_consolidated_summary(self) -> None:
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Reconciliation Breaks Summary", ln=True)
        self.pdf.ln(2)

        all_summaries: Dict[str, Mapping[str, Any]] = {}
        for summary in self.recon_summary:
            if isinstance(summary, Mapping) and "fund" in summary:
                all_summaries[summary["fund"]] = summary.get("summary", {})

        if not all_summaries:
            return

        cols = [
            ("Fund", 30),
            ("Index\nHoldings", 18),
            ("Index\nWgt Diff", 18),
            ("Cust Eq\nHold Brk", 18),
            ("Cust Eq\nPrice T", 18),
            ("Cust Opt\nHold Brk", 18),
            ("Cust Opt\nPrice T", 18),
            ("Cust Tsy\nHold Brk", 18),
            ("Cust Tsy\nPrice T", 18),
        ]

        self.pdf.set_font("Arial", "B", 6)
        self.pdf.set_fill_color(240, 240, 240)
        max_lines = max(len(header.split("\n")) for header, _ in cols)
        start_y = self.pdf.get_y()
        line_height = 3.5

        x_pos = self.pdf.l_margin
        for header, width in cols:
            lines = header.split("\n")
            for i, line in enumerate(lines):
                self.pdf.set_xy(x_pos, start_y + (i * line_height))
                self.pdf.cell(width, line_height, line, border=1, fill=True, align="C")
            x_pos += width
        self.pdf.set_y(start_y + (max_lines * line_height))

        self.pdf.set_font("Arial", size=7)
        for fund in sorted(all_summaries):
            summary = all_summaries[fund]
            y_pos = self.pdf.get_y()
            self.pdf.set_font("Arial", "B", 7)
            self.pdf.set_xy(self.pdf.l_margin, y_pos)
            self.pdf.cell(cols[0][1], 6, fund, border=1, align="C")

            values = [
                summary.get("index_equity", {}).get("holdings_discrepancies", 0),
                summary.get("index_equity", {}).get("significant_diffs", 0),
                summary.get("custodian_equity", {}).get("final_recon", 0),
                summary.get("custodian_equity", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_option", {}).get("final_recon", 0),
                summary.get("custodian_option", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_treasury", {}).get("final_recon", 0),
                summary.get("custodian_treasury", {}).get("price_discrepancies_T", 0),
            ]

            self.pdf.set_font("Arial", size=7)
            x_pos = self.pdf.l_margin + cols[0][1]
            for val, (_, width) in zip(values, cols[1:]):
                if isinstance(val, (int, float)) and val > 0:
                    self.pdf.set_fill_color(255, 200, 200)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                self.pdf.set_xy(x_pos, y_pos)
                self.pdf.cell(width, 6, str(int(val)), border=1, fill=True, align="C")
                x_pos += width
            self.pdf.set_y(y_pos + 6)

        self.pdf.ln(4)
        self.pdf.set_font("Arial", "I", 7)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 4, "Red highlight indicates breaks found | Hold Brk = Holdings Breaks", ln=1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(6)

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
        # FIX: Use separate calls for title and subtitle
        self.add_title("Daily Oversight Summary")
        self.add_subtitle(f"As of {self.report_date}")

        if self.compliance_results:
            self._render_compliance_overview()

        if self.reconciliation_results:
            self._render_reconciliation_summary()

        if self.nav_results:
            self._render_nav_summary()

        self.output()

    def _metric_table_widths(self, column_count: int) -> list[float]:
        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        if column_count <= 1:
            return [usable_width]
        metric_width = usable_width * 0.3
        remaining = usable_width - metric_width
        fund_width = remaining / (column_count - 1)
        return [metric_width] + [fund_width] * (column_count - 1)

    def _render_compliance_overview(self) -> None:
        self.add_section_heading("Compliance Overview")
        compliance_summary = summarise_compliance_status(self.compliance_results)
        rows = [
            ("Funds Processed", compliance_summary["funds"]),
            ("Funds with Breaches", compliance_summary["funds_in_breach"]),
            ("Checks Evaluated", compliance_summary["total_checks"]),
            ("Checks Failed", compliance_summary["failed_checks"]),
        ]
        self.add_key_value_table(rows, header=("Metric", "Value"))

        funds = sorted(self.compliance_results)
        if not funds:
            return
        breach_row = ["Breached?"]
        failed_row = ["Failed Checks"]
        for fund in funds:
            fund_results = self.compliance_results.get(fund, {})
            failed_count = 0
            breached = False
            for payload in fund_results.values():
                if not isinstance(payload, Mapping):
                    continue
                status = payload.get("is_compliant")
                if status is False or str(status).upper() == "FAIL":
                    breached = True
                    failed_count += 1
            breach_row.append("Yes" if breached else "No")
            failed_row.append(str(failed_count))

        headers = ["Metric"] + funds
        align = ["L"] + ["C"] * len(funds)
        self.add_table(headers, [breach_row, failed_row], column_widths=self._metric_table_widths(len(headers)), align=align)

    def _render_reconciliation_summary(self) -> None:
        self.add_section_heading("Holdings Reconciliation")
        funds = sorted(self.reconciliation_results)
        metrics = []
        for payload in self.reconciliation_results.values():
            summary = payload.get("summary", {}) or {}
            metrics.extend(summary.keys())
        metric_names = sorted(set(metrics))
        if not metric_names or not funds:
            self.add_paragraph("No reconciliation data available.")
            return

        rows = []
        for metric in metric_names:
            row = [metric.replace("_", " ").title()]
            for fund in funds:
                summary = self.reconciliation_results.get(fund, {}).get("summary", {}) or {}
                values = summary.get(metric, {}) or {}
                if isinstance(values, Mapping):
                    total_breaks = sum(
                        int(value)
                        for value in values.values()
                        if isinstance(value, (int, float))
                    )
                elif isinstance(values, (int, float)):
                    total_breaks = int(values)
                else:
                    total_breaks = 0
                row.append(str(total_breaks))
            rows.append(row)

        headers = ["Reconciliation"] + funds
        align = ["L"] + ["R"] * len(funds)
        self.add_table(headers, rows, column_widths=self._metric_table_widths(len(headers)), align=align)

    def _render_nav_summary(self) -> None:
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

        funds = sorted(self.nav_results)
        if not funds:
            return
        metric_rows = []
        metric_map = [
            ("Expected NAV", "expected_nav"),
            ("Custodian NAV", "current_nav"),
            ("NAV Difference", "difference"),
        ]
        for label, key in metric_map:
            row = [label]
            for fund in funds:
                summary = self.nav_results.get(fund, {}).get("summary", {}) or {}
                row.append(format_number(summary.get(key), 4))
            metric_rows.append(row)

        headers = ["Metric"] + funds
        align = ["L"] + ["R"] * len(funds)
        self.add_table(headers, metric_rows, column_widths=self._metric_table_widths(len(headers)), align=align)

