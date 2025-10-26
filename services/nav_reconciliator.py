class NAVReconciliator:
    """Lightweight coordinator for NAV reconciliation."""

    def __init__(self, fund: Fund, analysis_date: str, prior_date: str) -> None:
        self.fund = fund
        self.analysis_date = analysis_date
        self.prior_date = prior_date
        self.results: Dict[str, Dict] = {}

    def run_nav_reconciliation(self) -> Dict[str, Dict]:
        """Compute expected NAV using fund-level data and return the breakdown."""
        equity_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, "equity")
        options_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, "options")
        flex_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, "flex_options")
        treasury_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, "treasury")

        dividends = self.fund.get_dividends(self.analysis_date)
        expenses = self.fund.get_expenses(self.analysis_date)
        distributions = self.fund.get_distributions(self.analysis_date)
        flows_adjustment = self.fund.get_flows_adjustment(self.analysis_date, self.prior_date)

        summary = self._calculate_expected_nav(
            equity_gl,
            options_gl,
            flex_gl,
            treasury_gl,
            dividends,
            expenses,
            distributions,
            flows_adjustment,
        )

        self.results = {
            "summary": summary,
            "detailed_calculations": {
                "equity": equity_gl.details,
                "options": options_gl.details,
                "flex_options": flex_gl.details,
                "treasury": treasury_gl.details,
            },
        }
        return self.results

    def _calculate_expected_nav(
        self,
        equity_gl: GainLossResult,
        options_gl: GainLossResult,
        flex_gl: GainLossResult,
        treasury_gl: GainLossResult,
        dividends: float,
        expenses: float,
        distributions: float,
        flows_adjustment: float,
    ) -> Dict[str, float]:
        prior_nav = getattr(self.fund.data.previous, "nav", 0.0) or 0.0
        current_nav = getattr(self.fund.data.current, "nav", 0.0) or 0.0

        net_gain = (
            equity_gl.adjusted_gl
            + options_gl.adjusted_gl
            + flex_gl.adjusted_gl
            + treasury_gl.adjusted_gl
        )

        expected_nav = prior_nav + net_gain + dividends - expenses - distributions + flows_adjustment
        difference = current_nav - expected_nav

        return {
            "prior_nav": prior_nav,
            "current_nav": current_nav,
            "net_gain": net_gain,
            "dividends": dividends,
            "expenses": expenses,
            "distributions": distributions,
            "flows_adjustment": flows_adjustment,
            "expected_nav": expected_nav,
            "difference": difference,
        }