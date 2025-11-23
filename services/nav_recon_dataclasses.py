from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
from datetime import date
import pandas as pd

@dataclass
class TickerGainLoss:
    """Gain/loss details for a single ticker."""
    ticker: str
    quantity_t1: float
    quantity_t: float
    price_t1_vest: float
    price_t_vest: float
    price_t1_custodian: float
    price_t_custodian: float
    gl_raw: float
    gl_adjusted: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "ticker": self.ticker,
            "quantity_t1": self.quantity_t1,
            "quantity_t": self.quantity_t,
            "price_t1_raw": self.price_t1_vest,
            "price_t_raw": self.price_t_vest,
            "price_t1_adj": self.price_t1_custodian,
            "price_t_adj": self.price_t_custodian,
            "gl_raw": self.gl_raw,
            "gl_adjusted": self.gl_adjusted,
        }


@dataclass
class AssetClassGainLoss:
    """Gain/loss for a single asset class with ticker-level detail."""
    asset_class: str  # 'equity', 'options', 'flex_options', 'treasury'
    raw_gl: float
    adjusted_gl: float
    ticker_details: list[TickerGainLoss] = field(default_factory=list)

    @property
    def has_details(self) -> bool:
        """Check if ticker-level details exist."""
        return len(self.ticker_details) > 0

    def to_detail_dataframe(self) -> pd.DataFrame:
        """Convert ticker details to DataFrame."""
        if not self.ticker_details:
            return pd.DataFrame(columns=[
                "ticker", "quantity_t1", "quantity_t",
                "price_t1_raw", "price_t_raw",
                "price_t1_adj", "price_t_adj",
                "gl_raw", "gl_adjusted"
            ])
        return pd.DataFrame([t.to_dict() for t in self.ticker_details])


@dataclass
class NAVComponents:
    """All gain/loss components for NAV calculation."""
    equity: AssetClassGainLoss
    options: AssetClassGainLoss
    flex_options: AssetClassGainLoss
    treasury: AssetClassGainLoss
    assignment_gl: float = 0.0
    dividends: float = 0.0
    expenses: float = 0.0
    distributions: float = 0.0
    flows_adjustment: float = 0.0
    other: float = 0.0

    @property
    def total_gl_raw(self) -> float:
        """Total raw gain/loss across all components."""
        return (
                self.equity.raw_gl +
                self.options.raw_gl +
                self.flex_options.raw_gl +
                self.treasury.raw_gl +
                self.assignment_gl +
                self.other
        )

    @property
    def total_gl_adjusted(self) -> float:
        """Total adjusted gain/loss across all components."""
        return (
                self.equity.adjusted_gl +
                self.options.adjusted_gl +
                self.flex_options.adjusted_gl +
                self.treasury.adjusted_gl +
                self.assignment_gl +
                self.other
        )


@dataclass
class NAVSummary:
    """Summary metrics for NAV reconciliation."""
    fund_name: str
    analysis_date: str
    prior_date: str

    # TNA metrics
    beginning_tna: float
    adjusted_beginning_tna: float
    expected_tna: float
    custodian_tna: float
    tna_difference: float

    # NAV metrics
    shares_outstanding: float
    expected_nav: float
    custodian_nav: float
    nav_difference: float

    # Validation flags
    nav_good_2_decimal: bool
    nav_good_4_decimal: bool

    # Percentages
    diff_pct_4_decimal: float
    diff_pct_2_decimal: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for reporting."""
        return {
            "fund_name": self.fund_name,
            "analysis_date": self.analysis_date,
            "prior_date": self.prior_date,
            "beginning_tna": self.beginning_tna,
            "adjusted_beginning_tna": self.adjusted_beginning_tna,
            "expected_tna": self.expected_tna,
            "custodian_tna": self.custodian_tna,
            "tna_difference": self.tna_difference,
            "shares_outstanding": self.shares_outstanding,
            "expected_nav": self.expected_nav,
            "custodian_nav": self.custodian_nav,
            "nav_difference": self.nav_difference,
            "nav_good_2_decimal": self.nav_good_2_decimal,
            "nav_good_4_decimal": self.nav_good_4_decimal,
            "diff_pct_4_decimal": self.diff_pct_4_decimal,
            "diff_pct_2_decimal": self.diff_pct_2_decimal,
        }


@dataclass
class NAVReconciliationResults:
    """Complete NAV reconciliation output for a single fund."""
    fund_name: str
    analysis_date: str
    prior_date: str
    components: NAVComponents
    summary: NAVSummary

    def to_legacy_dict(self) -> Dict[str, object]:
        """
        Convert to legacy dictionary format for backward compatibility.
        This matches the structure expected by existing reporting code.
        """
        return {
            "Beginning TNA": self.summary.beginning_tna,
            "Adjusted Beginning TNA": self.summary.adjusted_beginning_tna,
            "Equity G/L": self.components.equity.raw_gl,
            "Equity G/L Adj": self.components.equity.adjusted_gl,
            "Option G/L": self.components.options.raw_gl,
            "Option G/L Adj": self.components.options.adjusted_gl,
            "Flex Option G/L": self.components.flex_options.raw_gl,
            "Flex Option G/L Adj": self.components.flex_options.adjusted_gl,
            "Treasury G/L": self.components.treasury.raw_gl,
            "Treasury G/L Adj": self.components.treasury.adjusted_gl,
            "Assignment G/L": self.components.assignment_gl,
            "Accruals": self.components.expenses,
            "Dividends": self.components.dividends,
            "Distributions": self.components.distributions,
            "Flows Adjustment": self.components.flows_adjustment,
            "Other": self.components.other,
            "Expected TNA": self.summary.expected_tna,
            "Custodian TNA": self.summary.custodian_tna,
            "TNA Diff ($)": self.summary.tna_difference,
            "Shares Outstanding": self.summary.shares_outstanding,
            "Expected NAV": self.summary.expected_nav,
            "Custodian NAV": self.summary.custodian_nav,
            "NAV Diff ($)": self.summary.nav_difference,
            "Difference (%) - 4 Digit": self.summary.diff_pct_4_decimal,
            "Difference (%) - 2 Digit": self.summary.diff_pct_2_decimal,
            "NAV Good (2 Digit)": self.summary.nav_good_2_decimal,
            "NAV Good (4 Digit)": self.summary.nav_good_4_decimal,
            "detailed_calculations": {
                "equity_details": self.components.equity.to_detail_dataframe(),
                "option_details": self.components.options.to_detail_dataframe(),
                "flex_details": self.components.flex_options.to_detail_dataframe(),
                "treasury_details": self.components.treasury.to_detail_dataframe(),
            },
            "details": {
                "equity_details": self.components.equity.to_detail_dataframe(),
                "option_details": self.components.options.to_detail_dataframe(),
                "flex_details": self.components.flex_options.to_detail_dataframe(),
                "treasury_details": self.components.treasury.to_detail_dataframe(),
            },
            "summary": self.summary.to_dict(),
        }

