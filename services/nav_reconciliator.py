"""NAV reconciliation service built on top of the Fund domain object."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import date, datetime
import logging

import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

from processing.fund import Fund, FundHoldings
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
        price_sensitivity = self._calculate_price_sensitivity(summary)
        results = NAVReconciliationResults(
            fund_name=self.fund.name,
            analysis_date=self.analysis_date,
            prior_date=self.prior_date,
            components=components,
            summary=summary,
            option_price_impact=price_sensitivity,   # carries the whole matrix dict
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
        - fund.data.current.vest.equity vs fund.data.previous.vest.equity
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
        vest_prior = self.fund.data.previous.vest.equity

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
        def _f(value, default=0.0):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return float(default)
            return float(default) if pd.isna(num) else num

        # Extract current position
        qty_t = 0.0
        price_t_vest = 0.0
        if not vest_current.empty and 'eqyticker' in vest_current.columns:
            ticker_data = vest_current[vest_current['eqyticker'] == ticker]
            if not ticker_data.empty:
                qty_t = _f(ticker_data.iloc[0].get('nav_shares', 0))
                price_t_vest = _f(ticker_data.iloc[0].get('price', 0))

        # Extract prior position
        qty_t1 = 0.0
        price_t1_vest = 0.0
        if not vest_prior.empty and 'eqyticker' in vest_prior.columns:
            ticker_data = vest_prior[vest_prior['eqyticker'] == ticker]
            if not ticker_data.empty:
                qty_t1 = _f(ticker_data.iloc[0].get('nav_shares', 0))
                price_t1_vest = _f(ticker_data.iloc[0].get('price', 0))

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
        - fund.data.current.vest.options vs fund.data.previous.vest.options
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
        vest_prior = self.fund.data.previous.vest.options

        # Roll days should source T-1 prices from execution data
        use_roll_trade_prices = self.fund.is_option_settlement_date(self.analysis_date)
        trades_t1 = self.fund.data.previous.option_trades

        price_breaks = self.fund.get_price_breaks('option')

        ticker_details = []
        for ticker in sorted(all_tickers):
            detail = self._calculate_option_ticker_gl(
                ticker,
                vest_current,
                vest_prior,
                price_breaks,
                multiplier=100.0,  # Option contract multiplier
                trades_t1=trades_t1,
                use_trade_prices=use_roll_trade_prices,
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
        vest_current = self.fund.data.current.vest.flex_options
        vest_prior = self.fund.data.previous.vest.flex_options

        use_roll_trade_prices = self.fund.is_option_settlement_date(self.analysis_date)
        trades_t1 = self.fund.data.previous.flex_option_trades

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
                trades_t1=trades_t1,
                use_trade_prices=use_roll_trade_prices,
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
            trades_t1: Optional[pd.DataFrame] = None,
            use_trade_prices: bool = False,
    ) -> Optional[TickerGainLoss]:
        """
        Calculate G/L for a single option ticker.

        Similar to equity but with option-specific column names and multiplier.
        """

        def _f(value, default=0.0):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return float(default)
            return float(default) if pd.isna(num) else num

        # Extract current position
        qty_t = 0.0
        price_t_vest = 0.0
        if not vest_current.empty and 'optticker' in vest_current.columns:
            ticker_data = vest_current[vest_current['optticker'] == ticker]
            if not ticker_data.empty:
                qty_t = _f(ticker_data.iloc[0].get('nav_shares', 0))
                price_t_vest = _f(ticker_data.iloc[0].get('price', 0))

        # Extract prior position
        qty_t1 = 0.0
        price_t1_vest = 0.0
        if not vest_prior.empty and 'optticker' in vest_prior.columns:
            ticker_data = vest_prior[vest_prior['optticker'] == ticker]
            if not ticker_data.empty:
                qty_t1 = _f(ticker_data.iloc[0].get('nav_shares', 0))
                price_t1_vest = _f(ticker_data.iloc[0].get('price', 0))

        # Start with vest prices
        price_t_custodian = price_t_vest
        price_t1_custodian = price_t1_vest

        # On option roll dates, use execution prices from the T-1 trades feed
        if use_trade_prices:
            trade_price = self._get_trade_execution_price(trades_t1, ticker)
            if trade_price is not None:
                price_t1_vest = _f(trade_price)
                price_t1_custodian = _f(trade_price)

        # Apply custodian price adjustments
        if not price_breaks.empty and ticker in price_breaks.index:
            adj_data = price_breaks.loc[ticker]

            if 'price_t_adj' in adj_data:
                price_t_custodian = _f(adj_data['price_t_adj'])

            if 'price_cust' in adj_data:
                cust_override = adj_data['price_cust']
                if cust_override is not None and pd.notna(cust_override):
                    price_t1_custodian = _f(cust_override)
                    price_t_custodian = _f(cust_override)

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

    def _calculate_flex_price_impact(self) -> dict:
        """Flex option G/L two ways — vest prices vs custodian prices.
        True G/L = (price_T - price_T-1) * qty * 100, matched on flex contract code."""
        if not self.has_flex_option:
            return {}

        def _prep(src) -> pd.DataFrame:
            df = src.copy()
            if df.empty:
                return pd.DataFrame(columns=["flex_code", "price", "qty"])
            code = df["equity_underlying_ticker"] if "equity_underlying_ticker" in df.columns else df.get("cusip")
            return pd.DataFrame({
                "flex_code": code,
                "price": pd.to_numeric(df.get("price"), errors="coerce"),
                "qty": pd.to_numeric(df.get("nav_shares", df.get("shares_cust")), errors="coerce").fillna(0.0),
            })

        vt  = _prep(self.fund.data.current.vest.flex_options)
        vt1 = _prep(self.fund.data.previous.vest.flex_options)
        ct  = _prep(self.fund.data.current.custodian.flex_options)
        ct1 = _prep(self.fund.data.previous.custodian.flex_options)
        if vt.empty:
            return {}

        base = vt.rename(columns={"price": "pv_t"})
        base = base.merge(vt1.rename(columns={"price": "pv_t1"})[["flex_code", "pv_t1"]], on="flex_code", how="left")
        base = base.merge(ct.rename(columns={"price": "pc_t"})[["flex_code", "pc_t"]], on="flex_code", how="left")
        base = base.merge(ct1.rename(columns={"price": "pc_t1"})[["flex_code", "pc_t1"]], on="flex_code", how="left")

        gl_vest = float(((base["pv_t"] - base["pv_t1"]) * base["qty"] * 100).sum())
        gl_cust = float(((base["pc_t"] - base["pc_t1"]) * base["qty"] * 100).sum())
        return {
            "flex_gl_vest_prices": gl_vest,
            "flex_gl_custodian_prices": gl_cust,
            "flex_gl_price_impact": gl_cust - gl_vest,
        }

    def _calculate_price_sensitivity(self, summary) -> dict:
        """Option price sensitivity: listed + flex option G/L computed with vest
        vs custodian prices, and Expected NAV under each pricing source.
        True G/L = (price_T - price_T-1) * qty * 100."""

        def _prep(src, flex: bool) -> pd.DataFrame:
            df = src.copy()
            if df.empty:
                return pd.DataFrame(columns=["k", "price", "qty"])
            if flex:
                code = df["equity_underlying_ticker"] if "equity_underlying_ticker" in df.columns else df.get("cusip")
            else:
                code = df.get("optticker")
            return pd.DataFrame({
                "k": code,
                "price": pd.to_numeric(df.get("price"), errors="coerce"),
                "qty": pd.to_numeric(df.get("nav_shares", df.get("shares_cust")), errors="coerce").fillna(0.0),
            })

        def _gl_two_ways(vc, vp, cc, cp, flex: bool) -> tuple[float, float]:
            vt, vt1 = _prep(vc, flex), _prep(vp, flex)
            ct, ct1 = _prep(cc, flex), _prep(cp, flex)
            if vt.empty:
                return 0.0, 0.0
            b = vt.rename(columns={"price": "pv_t"})
            b = b.merge(vt1.rename(columns={"price": "pv_t1"})[["k", "pv_t1"]], on="k", how="left")
            b = b.merge(ct.rename(columns={"price": "pc_t"})[["k", "pc_t"]], on="k", how="left")
            b = b.merge(ct1.rename(columns={"price": "pc_t1"})[["k", "pc_t1"]], on="k", how="left")
            gl_v = float(((b["pv_t"] - b["pv_t1"]) * b["qty"] * 100).sum())
            gl_c = float(((b["pc_t"] - b["pc_t1"]) * b["qty"] * 100).sum())
            return gl_v, gl_c

        cur, prev = self.fund.data.current, self.fund.data.previous
        listed_v, listed_c = _gl_two_ways(
            cur.vest.options, prev.vest.options,
            cur.custodian.options, prev.custodian.options, flex=False,
        )
        flex_v, flex_c = (0.0, 0.0)
        if self.has_flex_option:
            flex_v, flex_c = _gl_two_ways(
                cur.vest.flex_options, prev.vest.flex_options,
                cur.custodian.flex_options, prev.custodian.flex_options, flex=True,
            )

        combined_v = listed_v + flex_v
        combined_c = listed_c + flex_c

        shares = summary.shares_outstanding or 0
        exp_tna_vest = summary.expected_tna
        exp_tna_cust = exp_tna_vest - combined_v + combined_c   # swap vest option G/L for custodian
        exp_nav_vest = summary.expected_nav
        exp_nav_cust = (exp_tna_cust / shares) if shares else exp_nav_vest
        cust_nav = summary.custodian_nav

        return {
            "listed_gl_vest": listed_v, "listed_gl_cust": listed_c,
            "flex_gl_vest": flex_v, "flex_gl_cust": flex_c,
            "combined_gl_vest": combined_v, "combined_gl_cust": combined_c,
            "exp_nav_vest": exp_nav_vest, "exp_nav_cust": exp_nav_cust,
            "cust_nav": cust_nav,
            "nav_good_vest": bool(abs(round(cust_nav - exp_nav_vest, 4)) <= 0.0001),
            "nav_good_cust": bool(abs(round(cust_nav - exp_nav_cust, 4)) <= 0.0001),
        }


    @staticmethod
    def _get_trade_execution_price(
            trades: Optional[pd.DataFrame], ticker: str
    ) -> Optional[float]:
        """Return a weighted execution price for a ticker from the trades DataFrame."""
        if trades.empty or 'optticker' not in trades.columns:
            return None

        ticker_trades = trades[trades['optticker'] == ticker]
        if ticker_trades.empty:
            return None

        # Determine quantity column for weighting
        qty_col = None
        for candidate in ('quantity', 'qty', 'contracts', 'shares'):
            if candidate in ticker_trades.columns:
                qty_col = candidate
                break

        price_series = pd.to_numeric(ticker_trades.get('price'), errors='coerce')

        if qty_col:
            qty_series = pd.to_numeric(
                ticker_trades[qty_col], errors='coerce'
            ).abs()
            mask = (qty_series > 0) & price_series.notna()
            if mask.any():
                weighted_price = (price_series[mask] * qty_series[mask]).sum() / qty_series[mask].sum()
                return float(weighted_price)

        # Fallback to the last valid price if no quantities
        valid_prices = price_series.dropna()
        if not valid_prices.empty:
            return float(valid_prices.iloc[-1])

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
        """Calculate assignment P&L using T-1 holdings and custodian prices."""
        try:
            expiration_dt = pd.Timestamp(self.analysis_date).date()
        except Exception:
            return 0.0

        t1_options = self.fund.data.previous.vest.options
        t1_cust_options = self.fund.data.previous.custodian.options

        if t1_cust_options.empty:
            return 0.0

        # Merge T-1 custodian options with internal data for strike/equity_price info
        if not t1_options.empty:
            merged_options = t1_cust_options.merge(
                t1_options[[col for col in ['optticker', 'strike', 'equity_underlying_price', 'maturity']
                            if col in t1_options.columns]],
                on='optticker',
                how='left',
                suffixes=('', '_internal')
            )
        else:
            merged_options = t1_cust_options.copy()

        maturity_column = None
        for candidate in ('maturity', 'maturity_date'):
            if candidate in merged_options.columns:
                maturity_column = candidate
                break

        if maturity_column is None:
            return 0.0

        merged_options['maturity'] = pd.to_datetime(
            merged_options[maturity_column], errors='coerce'
        ).dt.date
        expiring_options = merged_options[merged_options['maturity'] == expiration_dt].copy()

        if expiring_options.empty:
            return 0.0

        # Apply price overrides from price breaks (enables T-1 price changes)
        price_breaks = self.fund.get_price_breaks('option')
        if not price_breaks.empty:
            if 'price_cust' in price_breaks.columns:
                expiring_options['price'] = expiring_options['optticker'].map(
                    price_breaks['price_cust']
                ).fillna(expiring_options.get('price', 0))

        expiring_options['equity_price'] = expiring_options.get('equity_price', 0).fillna(0)
        expiring_options['strike'] = expiring_options.get('strike', 0).fillna(0)

        expiring_options['last_part'] = expiring_options['optticker'].astype(str).str.split().str[-1]
        expiring_options['option_type'] = expiring_options['last_part'].str.extract(r'([PC])', expand=False)
        expiring_options['option_type'] = expiring_options['option_type'].fillna('C').str.upper()
        expiring_options.drop('last_part', axis=1, inplace=True)

        expiring_options['price'] = expiring_options.get('price', 0).fillna(0)
        expiring_options['shares_cust'] = expiring_options.get('shares_cust', 0).fillna(0)

        is_call = expiring_options['option_type'] == 'C'
        is_put = ~is_call

        expired_calls = is_call & (expiring_options['equity_price'] < expiring_options['strike'])
        expired_puts = is_put & (expiring_options['equity_price'] > expiring_options['strike'])
        expired_mask = expired_calls | expired_puts

        assigned_mask = ~expired_mask

        expiring_options['intrinsic_value'] = 0.0

        expiring_options.loc[is_call & assigned_mask, 'intrinsic_value'] = (
            expiring_options.loc[is_call & assigned_mask, 'equity_price'] -
            expiring_options.loc[is_call & assigned_mask, 'strike']
        )

        expiring_options.loc[is_put & assigned_mask, 'intrinsic_value'] = (
            expiring_options.loc[is_put & assigned_mask, 'strike'] -
            expiring_options.loc[is_put & assigned_mask, 'equity_price']
        )

        expiring_options['price_difference'] = expiring_options['price'] - expiring_options['intrinsic_value']
        expiring_options['pnl'] = expiring_options['price_difference'] * expiring_options['shares_cust'].abs() * 100

        expired_pnl = expiring_options.loc[expired_mask, 'pnl'].sum()
        assigned_pnl = expiring_options.loc[~expired_mask, 'pnl'].sum()

        net_expiration_activity = assigned_pnl + expired_pnl
        return float(net_expiration_activity)


    def _calculate_dividends(self) -> float:
        """Calculate dividend income from equity holdings."""
        return self.fund.get_equity_dividends()

    def _calculate_expenses(self) -> float:
        """
        Calculate expense accruals using fund's expense ratio.

        Formula: TNA * expense_ratio * (days / 365)
        """
        expense_ratio = self.fund.expense_ratio
        tna = self.fund.data.current.tna

        # Check if Friday (3-day accrual)
        analysis_dt = datetime.strptime(self.analysis_date, "%Y-%m-%d").date()
        days = 3 if analysis_dt.weekday() == 4 else 1

        return tna * expense_ratio * (days / 365)

    def _calculate_option_price_impact(self, summary) -> dict:
        """CEF option-price sensitivity. Counts only option price breaks > $1,
        and computes Expected NAV under vest vs custodian option prices."""
        cur = self.fund.data.current
        oms = cur.vest.options
        cust = cur.custodian.options
        if oms.empty or cust.empty or 'optticker' not in oms.columns or 'optticker' not in cust.columns:
            return {}

        cust_px = cust[['optticker', 'price']].rename(columns={'price': 'price_cust'})
        merged = oms.merge(cust_px, on='optticker', how='inner')
        merged['price_vest'] = pd.to_numeric(merged.get('price'), errors='coerce')
        merged['price_cust'] = pd.to_numeric(merged['price_cust'], errors='coerce')
        merged['qty'] = pd.to_numeric(merged.get('nav_shares'), errors='coerce').fillna(0.0)
        merged['abs_diff'] = (merged['price_vest'] - merged['price_cust']).abs()

        material = merged[merged['abs_diff'] > 1.0]
        mv_vest = float((merged['price_vest'] * merged['qty'] * 100).sum())
        mv_cust = float((merged['price_cust'] * merged['qty'] * 100).sum())
        impact = mv_cust - mv_vest

        shares = summary.shares_outstanding or 0
        exp_nav_vest = summary.expected_nav
        exp_nav_cust = (summary.expected_tna + impact) / shares if shares else exp_nav_vest
        nav_good_cust = abs(round(summary.custodian_nav - exp_nav_cust, 4)) <= 0.0001

        return {
            "option_mv_vest_prices": mv_vest,
            "option_mv_custodian_prices": mv_cust,
            "option_price_impact": impact,
            "expected_nav_vest_prices": exp_nav_vest,
            "expected_nav_custodian_prices": exp_nav_cust,
            "material_break_count": int(len(material)),
            "nav_good_cust_opt": bool(nav_good_cust),
        }


    def _calculate_distributions(self) -> float:
        """
        Calculate distributions going ex on analysis date.

        Returns distribution amount if ex_date matches analysis_date.
        """
        distributions = self.fund.data.distributions
        if distributions is None or distributions.empty or 'ex_date' not in distributions.columns:
            return 0.0

        distributions = distributions.copy()

        try:
            analysis_dt = pd.to_datetime(self.analysis_date).date()
        except Exception:
            analysis_dt = self.analysis_date

        distributions['ex_date'] = pd.to_datetime(
            distributions['ex_date'], errors='coerce'
        ).dt.date

        matching = distributions[
            (distributions.get('fund') == self.fund.name) &
            (distributions['ex_date'] == analysis_dt)
        ] if 'fund' in distributions.columns else distributions[
            distributions['ex_date'] == analysis_dt
        ]

        if matching.empty or 'distro_amt' not in matching.columns:
            return 0.0

        amount = abs(pd.to_numeric(
            matching['distro_amt'], errors='coerce'
        ).fillna(0).sum())

        if amount:
            self.logger.info(
                "Distribution going ex on %s: $%s",
                analysis_dt,
                f"{amount:,.2f}",
            )

        return float(amount)

    def _calculate_flows_adjustment(self) -> float:
        """
        Calculate T-1 flows adjustment for creations/redemptions.

        Returns adjusted beginning TNA that accounts for flows.
        """
        flows = self.fund.data.flows
        if flows is None or flows.empty:
            return 0.0

        # Get beginning TNA and shares
        beg_tna = self.fund.data.previous.tna
        beg_shares = self.fund.data.previous.shares_outstanding

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
        """Extract 'other' impact from fund data."""
        return float(self.fund.data.other or 0.0)

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
        beginning_tna = self.fund.data.previous.tna
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
        custodian_tna = self.fund.data.current.tna
        tna_difference = custodian_tna - expected_tna

        # Get shares
        shares = self.fund.data.current.shares_outstanding

        # Calculate NAV
        expected_nav = expected_tna / shares if shares > 0 else 0.0
        custodian_nav = self.fund.data.current.nav
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