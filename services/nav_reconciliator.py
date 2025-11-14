"""NAV reconciliation service built on top of the Fund domain object."""
from __future__ import annotations
from typing import Dict
from domain.fund import Fund, GainLossResult
import logging
# from utilities.logger import setup_logger
# import datetime
from config.fund_definitions import DIVERSIFIED_FUNDS, PRIVATE_FUNDS, CLOSED_END_FUNDS


class NAVReconciliator:
    def __init__(self, session, fund_name: str, fund_data: dict, analysis_date, prior_date,
                 analysis_type=None, fund=None, socgen_custodian=None):
        self.session = session
        self.fund_name = fund_name
        self.fund_data = fund_data
        self.holdings_price_breaks = self.fund_data.get("holdings_price_breaks", {})
        self.analysis_date = analysis_date
        self.prior_date = prior_date
        self.analysis_type = analysis_type
        # Fix: Initialize logger properly
        self.logger = logging.getLogger(f"NAVReconciliator_{fund_name}")
        self.results = {}
        self.summary = {}
        self.fund = fund
        self.socgen_custodian = socgen_custodian

    def run_nav_reconciliation(self):
        """Main reconciliation orchestrator - with special handling for assignment/expiration days"""
        self.logger.info(f"Starting NAV reconciliation for {self.fund_name}")

        # 2. Check if this is an option settlement/assignment date FIRST
        is_assignment_day = self._is_option_settlement_date(self.analysis_date)

        # 3. Calculate G/L by asset type
        equity_gl_raw, equity_gl_adj = self._calculate_equity_gl()

        # 4. Handle options differently based on whether it's an assignment day
        assignment_gl = 0.0
        option_gl_raw = 0.0
        option_gl_adj = 0.0

        if is_assignment_day:
            # Calculate assignment P&L for expired options
            assignment_gl = self.process_assignments(self.prior_date)
            # Calculate gain/loss on new options rolled into
            rolled_option_gl = self._calculate_rolled_option_gl()
            option_gl_raw = rolled_option_gl
            option_gl_adj = option_gl_raw
        else:
            # Normal day - calculate option G/L the standard way
            option_gl_raw, option_gl_adj = self._calculate_option_gl()

        # 5. Calculate other asset types
        flex_gl_raw, flex_gl_adj = self._calculate_flex_option_gl()
        treasury_gl_raw, treasury_gl_adj = self._calculate_treasury_gl()

        # 6. Calculate other components
        dividends = self._calculate_dividends()
        expenses = self._calculate_expenses()
        adjusted_beg_tna = self._calculate_t1_flows_adjustment()
        distributions = self._calculate_distributions()

        # 7. Calculate expected vs actual NAV
        results = self._calculate_nav_comparison(
            equity_gl_adj, option_gl_adj, flex_gl_adj,
            treasury_gl_adj, assignment_gl,
            dividends, expenses, distributions, adjusted_beg_tna
        )

        # 8. CRITICAL: Add detailed calculations to results
        results['detailed_calculations'] = {
            'equity_details': getattr(self, 'equity_details', pd.DataFrame()),
            'option_details': getattr(self, 'option_details', pd.DataFrame()),
            'flex_details': getattr(self, 'flex_details', pd.DataFrame()),
            'treasury_details': getattr(self, 'treasury_details', pd.DataFrame()),
        }

        # Add raw data for debugging
        results['raw_equity'] = self.fund_data.get('equity_holdings', pd.DataFrame())
        results['raw_option'] = self.fund_data.get('options_holdings', pd.DataFrame())

        # 9. Store results
        self.results = results
        self.summary = self._build_summary(results)

        self._log_completion(results)
        return self.results, self.summary

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