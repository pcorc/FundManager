"""NAV reconciliation service built on top of the Fund domain object."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import date, datetime
import logging

import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

from processing.fund import Fund
from config.fund_definitions import INDEX_FLEX_FUNDS
from services.nav_recon_dataclasses import (
    AssetClassGainLoss,
    TickerGainLoss,
    NAVComponents,
    NAVSummary,
    NAVReconciliationResults,
)


class NAVReconciliator:
    """
    Calculate daily NAV reconciliation metrics for a fund.

    Refactored to use Fund object and return structured dataclasses.
    """

    def __init__(
            self,
            fund: Fund,
            analysis_date: date | str,
            prior_date: date | str,
    ) -> None:
        """
        Initialize NAV reconciliator with Fund object.

        Args:
            fund: Fund object with loaded current and prior snapshots
            analysis_date: Current analysis date
            prior_date: Prior business day date
        """
        self.fund = fund
        self.fund_name = fund.name
        self.analysis_date = str(analysis_date)
        self.prior_date = str(prior_date)
        self.logger = logging.getLogger(f"NAVReconciliator_{fund.name}")

        # Get fund properties
        self.has_flex_option = fund.has_flex_option
        self.flex_option_pattern = fund.flex_option_pattern
        self.flex_option_type = fund.flex_option_type

        # Legacy index flex check
        self.uses_index_flex = (
                fund.name in INDEX_FLEX_FUNDS or self.flex_option_type == "index"
        )

        # Results containers
        self.results: Dict[str, object] = {}
        self.summary: Dict[str, float] = {}

        self.DETAIL_COLUMNS = [
            "ticker",
            "quantity_t1",
            "quantity_t",
            "price_t1_raw",  # Vest price at T-1
            "price_t_raw",  # Vest price at T
            "price_t1_adj",  # Custodian/adjusted price at T-1
            "price_t_adj",  # Custodian/adjusted price at T
            "gl_raw",  # Raw gain/loss
            "gl_adjusted"  # Adjusted gain/loss
        ]

        # Detail containers
        self.equity_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.option_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.flex_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.treasury_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)

    def run_nav_reconciliation(self) -> NAVReconciliationResults:
        """
        Main reconciliation orchestrator.

        Returns:
            Structured NAVReconciliationResults object
        """
        self.logger.info("Starting NAV reconciliation for %s", self.fund.name)

        # Check if this is an option settlement/assignment day
        is_assignment_day = self.fund.is_option_settlement_date(self.analysis_date)

        # Calculate G/L for each asset class
        equity_gl = self._calculate_equity_gl()

        if is_assignment_day:
            assignment_gl = self._process_assignments()
            # On assignment days, options are "rolled" - just use current value
            option_gl = self._calculate_rolled_option_gl()
        else:
            option_gl = self._calculate_option_gl()
            assignment_gl = 0.0

        flex_gl = self._calculate_flex_option_gl()
        treasury_gl = self._calculate_treasury_gl()

        # Calculate other components
        dividends = self._calculate_dividends()
        expenses = self._calculate_expenses()
        flows_adjustment = self._calculate_flows_adjustment()
        distributions = self._calculate_distributions()
        other = self._calculate_other()

        # Build components
        components = NAVComponents(
            equity=equity_gl,
            options=option_gl,
            flex_options=flex_gl,
            treasury=treasury_gl,
            assignment_gl=assignment_gl,
            dividends=dividends,
            expenses=expenses,
            distributions=distributions,
            flows_adjustment=flows_adjustment,
            other=other,
        )

        # Calculate NAV summary
        summary = self._calculate_nav_summary(components)

        # Build final results
        results = NAVReconciliationResults(
            fund_name=self.fund.name,
            analysis_date=self.analysis_date,
            prior_date=self.prior_date,
            components=components,
            summary=summary,
        )

        self._log_completion(results)
        return results

    # ========================================================================
    # ASSET CLASS GAIN/LOSS CALCULATIONS
    # ========================================================================

    def _calculate_equity_gl(self) -> AssetClassGainLoss:
        """
        Calculate equity G/L with ticker-level details.

        Compares:
        - fund.data.current.vest.equity vs fund.data.prior.vest.equity
        - Applies custodian price adjustments from price_breaks
        """
        # Get all unique tickers across snapshots
        all_tickers = self.fund.gather_all_tickers('equity', include_prior=True)

        if not all_tickers:
            return AssetClassGainLoss(
                asset_class='equity',
                raw_gl=0.0,
                adjusted_gl=0.0,
            )

        # Get holdings DataFrames
        vest_current = self.fund.data.current.vest.equity
        vest_prior = self.fund.data.prior.vest.equity

        # Get price breaks for adjustments
        price_breaks = self.fund.get_price_breaks('equity')

        # Calculate ticker-level details
        ticker_details = []
        for ticker in sorted(all_tickers):
            detail = self._calculate_equity_ticker_gl(
                ticker,
                vest_current,
                vest_prior,
                price_breaks,
            )
            if detail:
                ticker_details.append(detail)

        # Sum totals
        total_raw = sum(t.gl_raw for t in ticker_details)
        total_adjusted = sum(t.gl_adjusted for t in ticker_details)

        return AssetClassGainLoss(
            asset_class='equity',
            raw_gl=total_raw,
            adjusted_gl=total_adjusted,
            ticker_details=ticker_details,
        )

    def _calculate_equity_ticker_gl(
            self,
            ticker: str,
            vest_current: pd.DataFrame,
            vest_prior: pd.DataFrame,
            price_breaks: pd.DataFrame,
    ) -> Optional[TickerGainLoss]:
        """
        Calculate G/L for a single equity ticker.

        Formula:
        - Raw G/L = (price_t_vest - price_t1_vest) * qty_t
        - Adjusted G/L = (price_t_custodian - price_t1_custodian) * qty_t
        """
        # Extract current position
        qty_t = 0
        price_t_vest = 0
        if not vest_current.empty and 'eqyticker' in vest_current.columns:
            ticker_data = vest_current[vest_current['eqyticker'] == ticker]
            if not ticker_data.empty:
                qty_t = ticker_data.iloc[0].get('nav_shares', 0)
                price_t_vest = ticker_data.iloc[0].get('price', 0)

        # Extract prior position
        qty_t1 = 0
        price_t1_vest = 0
        if not vest_prior.empty and 'eqyticker' in vest_prior.columns:
            ticker_data = vest_prior[vest_prior['eqyticker'] == ticker]
            if not ticker_data.empty:
                qty_t1 = ticker_data.iloc[0].get('nav_shares', 0)
                price_t1_vest = ticker_data.iloc[0].get('price', 0)

        # Start with vest prices
        price_t_custodian = price_t_vest
        price_t1_custodian = price_t1_vest

        # Apply custodian price adjustments if available
        if not price_breaks.empty and ticker in price_breaks.index:
            adj_data = price_breaks.loc[ticker]

            # Check for adjusted prices
            if 'price_t_adj' in adj_data:
                price_t_custodian = adj_data['price_t_adj']

            # Check for custodian override price
            if 'price_cust' in adj_data:
                cust_override = adj_data['price_cust']
                if cust_override is not None and pd.notna(cust_override):
                    price_t1_custodian = cust_override
                    price_t_custodian = cust_override

        # Calculate G/L
        gl_raw = (price_t_vest - price_t1_custodian) * qty_t
        gl_adjusted = (price_t_custodian - price_t1_custodian) * qty_t

        # Only return if position exists
        if qty_t != 0 or qty_t1 != 0:
            return TickerGainLoss(
                ticker=ticker,
                quantity_t1=qty_t1,
                quantity_t=qty_t,
                price_t1_vest=price_t1_vest,
                price_t_vest=price_t_vest,
                price_t1_custodian=price_t1_custodian,
                price_t_custodian=price_t_custodian,
                gl_raw=gl_raw,
                gl_adjusted=gl_adjusted,
            )
        return None

    def _calculate_option_gl(self) -> AssetClassGainLoss:
        """
        Calculate regular (non-flex) option G/L with ticker-level details.

        Compares:
        - fund.data.current.vest.options vs fund.data.prior.vest.options
        - Excludes flex options if fund has_flex_option=True
        - Applies 100x multiplier for option contracts
        """
        # Get regular option tickers (excludes flex)
        all_tickers = self.fund.gather_option_tickers(
            include_flex=False,
            include_prior=True,
        )

        if not all_tickers:
            return AssetClassGainLoss(
                asset_class='options',
                raw_gl=0.0,
                adjusted_gl=0.0,
            )

        # Get holdings DataFrames
        vest_current = self.fund.data.current.vest.options
        vest_prior = self.fund.data.prior.vest.options

        # Get price breaks
        price_breaks = self.fund.get_price_breaks('option')

        # Calculate ticker-level details
        ticker_details = []
        for ticker in sorted(all_tickers):
            detail = self._calculate_option_ticker_gl(
                ticker,
                vest_current,
                vest_prior,
                price_breaks,
                multiplier=100.0,  # Option contract multiplier
            )
            if detail:
                ticker_details.append(detail)

        # Sum totals
        total_raw = sum(t.gl_raw for t in ticker_details)
        total_adjusted = sum(t.gl_adjusted for t in ticker_details)

        return AssetClassGainLoss(
            asset_class='options',
            raw_gl=total_raw,
            adjusted_gl=total_adjusted,
            ticker_details=ticker_details,
        )

    def _calculate_flex_option_gl(self) -> AssetClassGainLoss:
        """
        Calculate flex option G/L with ticker-level details.

        Only runs if fund.properties.has_flex_option is True.
        Uses fund.properties.flex_option_pattern to identify flex options.
        """
        if not self.has_flex_option:
            return AssetClassGainLoss(
                asset_class='flex_options',
                raw_gl=0.0,
                adjusted_gl=0.0,
            )

        # Get flex option tickers
        all_tickers = self.fund.gather_option_tickers(
            include_flex=True,
            include_prior=True,
        )

        if not all_tickers:
            return AssetClassGainLoss(
                asset_class='flex_options',
                raw_gl=0.0,
                adjusted_gl=0.0,
            )

        # Get holdings DataFrames
        vest_current = self.fund.data.current.vest.options
        vest_prior = self.fund.data.prior.vest.options

        # Get price breaks
        price_breaks = self.fund.get_price_breaks('option')

        # Calculate ticker-level details
        ticker_details = []
        for ticker in sorted(all_tickers):
            detail = self._calculate_option_ticker_gl(
                ticker,
                vest_current,
                vest_prior,
                price_breaks,
                multiplier=100.0,
            )
            if detail:
                ticker_details.append(detail)

        # Sum totals
        total_raw = sum(t.gl_raw for t in ticker_details)
        total_adjusted = sum(t.gl_adjusted for t in ticker_details)

        return AssetClassGainLoss(
            asset_class='flex_options',
            raw_gl=total_raw,
            adjusted_gl=total_adjusted,
            ticker_details=ticker_details,
        )

    def _calculate_option_ticker_gl(
            self,
            ticker: str,
            vest_current: pd.DataFrame,
            vest_prior: pd.DataFrame,
            price_breaks: pd.DataFrame,
            multiplier: float = 100.0,
    ) -> Optional[TickerGainLoss]:
        """
        Calculate G/L for a single option ticker.

        Similar to equity but with option-specific column names and multiplier.
        """
        # Extract current position
        qty_t = 0
        price_t_vest = 0
        if not vest_current.empty and 'optticker' in vest_current.columns:
            ticker_data = vest_current[vest_current['optticker'] == ticker]
            if not ticker_data.empty:
                qty_t = ticker_data.iloc[0].get('nav_shares', 0)
                price_t_vest = ticker_data.iloc[0].get('price', 0)

        # Extract prior position
        qty_t1 = 0
        price_t1_vest = 0
        if not vest_prior.empty and 'optticker' in vest_prior.columns:
            ticker_data = vest_prior[vest_prior['optticker'] == ticker]
            if not ticker_data.empty:
                qty_t1 = ticker_data.iloc[0].get('nav_shares', 0)
                price_t1_vest = ticker_data.iloc[0].get('price', 0)

        # Start with vest prices
        price_t_custodian = price_t_vest
        price_t1_custodian = price_t1_vest

        # Apply custodian price adjustments
        if not price_breaks.empty and ticker in price_breaks.index:
            adj_data = price_breaks.loc[ticker]

            if 'price_t_adj' in adj_data:
                price_t_custodian = adj_data['price_t_adj']

            if 'price_cust' in adj_data:
                cust_override = adj_data['price_cust']
                if cust_override is not None and pd.notna(cust_override):
                    price_t1_custodian = cust_override
                    price_t_custodian = cust_override

        # Calculate G/L with multiplier
        gl_raw = (price_t_vest - price_t1_custodian) * qty_t * multiplier
        gl_adjusted = (price_t_custodian - price_t1_custodian) * qty_t * multiplier

        if qty_t != 0 or qty_t1 != 0:
            return TickerGainLoss(
                ticker=ticker,
                quantity_t1=qty_t1,
                quantity_t=qty_t,
                price_t1_vest=price_t1_vest,
                price_t_vest=price_t_vest,
                price_t1_custodian=price_t1_custodian,
                price_t_custodian=price_t_custodian,
                gl_raw=gl_raw,
                gl_adjusted=gl_adjusted,
            )
        return None

    def _calculate_treasury_gl(self) -> AssetClassGainLoss:
        """
        Calculate treasury G/L.

        Simple comparison of current vs prior treasury values.
        Usually no ticker-level detail for treasuries.
        """
        current_value = self.fund.data.current.total_treasury_value
        prior_value = self.fund.data.previous.total_treasury_value

        gl = current_value - prior_value

        return AssetClassGainLoss(
            asset_class='treasury',
            raw_gl=gl,
            adjusted_gl=gl,
        )

    def _calculate_rolled_option_gl(self) -> AssetClassGainLoss:
        """
        On assignment days, option G/L is just the current value.
        Options were closed/rolled, so we don't compare to prior.
        """
        current_value = self.fund.data.current.total_option_value

        return AssetClassGainLoss(
            asset_class='options',
            raw_gl=current_value,
            adjusted_gl=current_value,
        )

    # ========================================================================
    # OTHER COMPONENT CALCULATIONS
    # ========================================================================

    def _process_assignments(self) -> float:
        """
        Process option assignments on settlement days.

        Extracts assignment G/L from fund.data.assignments.
        """
        assignments = self.fund.get_assignments(
            filter_date=self.analysis_date
        )

        if assignments.empty:
            return 0.0

        # Try to find P&L columns
        value_cols = [
            col for col in assignments.columns
            if any(keyword in str(col).lower()
                   for keyword in ('pnl', 'gl', 'gain', 'loss', 'amount'))
        ]

        for col in value_cols:
            series = pd.to_numeric(assignments[col], errors='coerce').dropna()
            if not series.empty:
                return float(series.sum())

        # Fallback: calculate from quantity * price
        quantity_cols = [
            col for col in assignments.columns
            if 'quantity' in str(col).lower() or 'contracts' in str(col).lower()
        ]
        price_cols = [
            col for col in assignments.columns
            if 'price' in str(col).lower() or 'premium' in str(col).lower()
        ]

        if quantity_cols and price_cols:
            qty = pd.to_numeric(assignments[quantity_cols[0]], errors='coerce').fillna(0)
            price = pd.to_numeric(assignments[price_cols[0]], errors='coerce').fillna(0)
            return float((qty * price).sum())

        return 0.0

    def _calculate_dividends(self) -> float:
        """Calculate dividend income from equity holdings."""
        return self.fund.get_equity_dividends()

    def _calculate_expenses(self) -> float:
        """
        Calculate expense accruals using fund's expense ratio.

        Formula: TNA * expense_ratio * (days / 365)
        """
        expense_ratio = self.fund.expense_ratio
        tna = self.fund.get_nav_metric('total_net_assets', 'current')

        # Check if Friday (3-day accrual)
        analysis_dt = datetime.strptime(self.analysis_date, "%Y-%m-%d").date()
        days = 3 if analysis_dt.weekday() == 4 else 1

        return tna * expense_ratio * (days / 365)

    def _calculate_distributions(self) -> float:
        """
        Calculate distributions going ex on analysis date.

        Returns distribution amount if ex_date matches analysis_date.
        """
        if not hasattr(self.fund.data, 'distributions'):
            return 0.0

        distributions = self.fund.data.distributions
        if distributions.empty:
            return 0.0

        if 'ex_date' not in distributions.columns:
            return 0.0

        # Convert dates and filter
        distributions = distributions.copy()
        distributions['ex_date'] = pd.to_datetime(
            distributions['ex_date'], errors='coerce'
        ).dt.date

        matching = distributions[
            (distributions['fund'] == self.fund.name) &
            (distributions['ex_date'] == self.analysis_date)
            ]

        if matching.empty or 'distro_amt' not in matching.columns:
            return 0.0

        amount = pd.to_numeric(
            matching['distro_amt'], errors='coerce'
        ).fillna(0).sum()

        if amount:
            self.logger.info(
                "Distribution going ex on %s: $%s",
                self.analysis_date,
                f"{amount:,.2f}",
            )

        return float(amount)

    def _calculate_flows_adjustment(self) -> float:
        """
        Calculate T-1 flows adjustment for creations/redemptions.

        Returns adjusted beginning TNA that accounts for flows.
        """
        if not hasattr(self.fund.data, 'flows'):
            return 0.0

        flows = self.fund.data.flows
        if flows.empty:
            return 0.0

        # Get beginning TNA and shares
        beg_tna = self.fund.get_nav_metric('total_net_assets', 'prior')
        beg_shares = self.fund.get_nav_metric('shares_outstanding', 'prior')

        if beg_shares == 0:
            return 0.0

        # Extract net units from flows
        net_units = 0
        if 'net_units' in flows.columns:
            net_units = pd.to_numeric(
                flows['net_units'], errors='coerce'
            ).fillna(0).iloc[0] if not flows.empty else 0

        # Standard creation unit size
        shares_per_cu = 50000

        # Calculate adjustment
        flows_adjustment_per_share = net_units * (beg_tna / beg_shares)
        tna_adjustment = flows_adjustment_per_share * shares_per_cu

        return beg_tna + tna_adjustment

    def _calculate_other(self) -> float:
        """Extract 'other' impact from fund data if present."""
        if hasattr(self.fund.data, 'other'):
            return float(self.fund.data.other or 0.0)
        return 0.0

    # ========================================================================
    # NAV SUMMARY CALCULATION
    # ========================================================================

    def _calculate_nav_summary(
            self,
            components: NAVComponents,
    ) -> NAVSummary:
        """
        Calculate NAV summary metrics from components.

        Combines all G/L components to calculate expected vs actual NAV.
        """
        # Get TNA metrics
        beginning_tna = self.fund.get_nav_metric('total_net_assets', 'prior')
        adjusted_beg_tna = beginning_tna + components.flows_adjustment

        # Calculate expected TNA
        expected_tna = (
                adjusted_beg_tna +
                components.total_gl_adjusted +
                components.dividends -
                abs(components.expenses) -
                abs(components.distributions)
        )

        # Get custodian TNA
        custodian_tna = self.fund.get_nav_metric('total_net_assets', 'current')
        tna_difference = custodian_tna - expected_tna

        # Get shares
        shares = self.fund.get_nav_metric('shares_outstanding', 'current')

        # Calculate NAV
        expected_nav = expected_tna / shares if shares > 0 else 0.0
        custodian_nav = self.fund.get_nav_metric('nav', 'current')
        nav_difference = custodian_nav - expected_nav

        # Calculate validation flags
        rounded_expected = round(expected_nav, 2)
        diff_pct_4 = abs(custodian_tna / expected_tna - 1) if expected_tna else 0
        diff_pct_2 = abs(custodian_nav / rounded_expected - 1) if rounded_expected else 0

        nav_good_2 = diff_pct_2 <= 0.000055
        nav_good_4 = diff_pct_4 <= 0.000055

        return NAVSummary(
            fund_name=self.fund.name,
            analysis_date=self.analysis_date,
            prior_date=self.prior_date,
            beginning_tna=beginning_tna,
            adjusted_beginning_tna=adjusted_beg_tna,
            expected_tna=expected_tna,
            custodian_tna=custodian_tna,
            tna_difference=tna_difference,
            shares_outstanding=shares,
            expected_nav=expected_nav,
            custodian_nav=custodian_nav,
            nav_difference=nav_difference,
            nav_good_2_decimal=nav_good_2,
            nav_good_4_decimal=nav_good_4,
            diff_pct_4_decimal=diff_pct_4,
            diff_pct_2_decimal=diff_pct_2,
        )

    def _log_completion(self, results: NAVReconciliationResults) -> None:
        """Log completion with summary info."""
        self.logger.info(
            "Completed NAV reconciliation for %s (NAV diff: %.6f)",
            self.fund.name,
            results.summary.nav_difference,
        )