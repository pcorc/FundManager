from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from config.constants import (
    ACT_40_ISSUER_LIMIT,
    ACT_40_OWNERSHIP_LIMIT,
    ACT_40_QUALIFYING_ASSETS_MIN,
    EQUITY_MIN_THRESHOLD,
    ILLIQUID_MAX_THRESHOLD,
    IRC_TOP_1_LIMIT,
    IRC_TOP_2_LIMIT,
    IRC_TOP_3_LIMIT,
    IRC_TOP_4_LIMIT,
    IRS_BOTTOM_50_LIMIT,
    IRS_OWNERSHIP_LIMIT,
    IRS_QUALIFYING_ASSETS_MIN,
    PROSPECTUS_MIN_THRESHOLD,
    PROSPECTUS_OPTIONS_FUNDS,
    RULE_12D1A_OWNERSHIP_LIMIT,
    RULE_12D1A_SINGLE_ASSETS_LIMIT,
    RULE_12D1A_TOTAL_ASSETS_LIMIT,
    RULE_12D2_INSURANCE_LIMIT,
    RULE_12D3_ASSET_LIMIT,
    RULE_12D3_DEBT_LIMIT,
    RULE_12D3_EQUITY_LIMIT,
    VEHICLE_PRIVATE,
    VEHICLE_VIT,
)
from processing.fund import Fund

GICS_CONCENTRATION_THRESHOLD = 0.25
GICS_CLASS_COLUMNS: Tuple[str, ...] = (
    "GICS_SECTOR_NAME",
    "GICS_INDUSTRY_GROUP_NAME",
    "GICS_INDUSTRY_NAME",
)

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Structured result returned from a compliance check."""

    is_compliant: bool
    details: Dict[str, object]
    calculations: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None


def _res(ok, rule: str, calcs=None, error: Optional[str] = None, **details) -> ComplianceResult:
    return ComplianceResult(ok, {"rule": rule, **details}, {} if calcs is None else calcs, error)


def _num(df: pd.DataFrame, column: str):
    # A missing column yields a scalar and .fillna raises, which turns into an error result.
    return pd.to_numeric(df.get(column, 0.0), errors="coerce").fillna(0.0)


def _options_per_underlying(opt_df: pd.DataFrame) -> pd.DataFrame:
    """One row per equity_underlying_ticker (numeric columns summed), so merging options onto
    issuer-level equity never repeats an issuer once per option on it."""
    if opt_df.empty or "equity_underlying_ticker" not in opt_df.columns:
        return opt_df
    numeric_columns = opt_df.select_dtypes(include=[np.number]).columns.tolist()
    aggregation = {column: "sum" for column in numeric_columns}
    aggregation.update(
        {column: "first" for column in opt_df.columns if column not in numeric_columns + ["equity_underlying_ticker"]}
    )
    return opt_df.groupby("equity_underlying_ticker", as_index=False, sort=False, dropna=False).agg(aggregation)


def _ownership_pct(numerator: pd.Series, shares_out: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, shares_out.replace(0, np.nan)).fillna(0.0)


# (result key, method name, display name for error results, error log format), in execution order.
# The order matters: max_15pct_illiquid_sai adds is_illiquid to the shared holdings frames in place.
_RULES: Tuple[Tuple[str, str, str, str], ...] = (
    ("summary_metrics", "calculate_summary_metrics", "summary_metrics", "Error calculating summary metrics for %s: %s"),
    ("gics_compliance", "gics_compliance", "GICS Concentration", "Error in GICS compliance check for %s: %s"),
    ("prospectus_80pct_policy", "prospectus_80pct_policy", "80% Prospectus Policy", "Error in prospectus policy check for %s: %s"),
    ("diversification_IRS_check", "diversification_IRS_check", "IRS Diversification", "Error in IRS diversification check for %s: %s"),
    ("diversification_40act_check", "diversification_40act_check", "40 Act Diversification", "Error in 40 Act diversification check for %s: %s"),
    ("diversification_IRC_check", "diversification_IRC_check", "IRC Diversification", "Error in IRC diversification check for %s: %s"),
    ("max_15pct_illiquid_sai", "max_15pct_illiquid_sai", "15% Illiquid Assets", "Error in 15% illiquid check for %s: %s"),
    ("real_estate_check", "real_estate_check", "Real Estate Exposure", "Error in real estate check for %s: %s"),
    ("commodities_check", "commodities_check", "Commodities Exposure", "Error in commodities check for %s: %s"),
    ("twelve_d1a_other_inv_cos", "twelve_d1a_other_inv_cos", "12d1-1 Other Investment Companies", "Error in 12d1-1 check for %s: %s"),
    ("twelve_d2_insurance_cos", "twelve_d2_insurance_cos", "12d-2 Insurance Companies", "Error in 12d-2 check for %s: %s"),
    ("twelve_d3_sec_biz", "twelve_d3_sec_biz", "12d-3 Securities Businesses", "Error in 12d-3 check for %s: %s"),
)
RULE_KEYS: Tuple[str, ...] = tuple(rule[0] for rule in _RULES)  # public: valid compliance_tests values
RULE_LABELS: Dict[str, str] = {rule[0]: rule[2] for rule in _RULES}


class ComplianceChecker:
    """Runs compliance checks against :class:`Fund` domain objects."""
    def __init__(
        self,
        funds: Optional[Dict[str, Fund]] = None,
        *,
        analysis_type: Optional[str] = None,
    ) -> None:
        self.funds: Dict[str, Fund] = funds or {}
        analysis = (analysis_type or "eod").strip().lower()
        self.analysis_type = analysis if analysis in {"eod", "ex_ante", "ex_post"} else "eod"

    @property
    def _share_col(self) -> str:
        return "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"

    def _held_shares(self, df: pd.DataFrame) -> pd.Series:
        """Shares the fund holds per row (the analysis type's share column, numeric; missing -> 0)."""
        share_col = self._share_col
        if share_col in df.columns:
            held = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0)
        else:
            held = pd.Series(0.0, index=df.index, dtype=float)
        df[share_col] = held
        return held

    def _consolidate_holdings_by_issuer(self, holdings_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate holdings exposure by issuer without conditional adjustments."""
        if holdings_df.empty:
            return holdings_df.copy()

        df = holdings_df.copy()
        df["equity_market_value"] = _num(df, "equity_market_value")
        share_col = self._share_col
        if share_col not in df.columns:
            df[share_col] = 0.0
        else:
            df[share_col] = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0)

        df["eqyticker"] = df["eqyticker"].astype(str)
        google_mask = df["eqyticker"].isin(["GOOG", "GOOGL"])
        if google_mask.any():
            df.loc[google_mask, "eqyticker"] = "GOOGLE"

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        aggregation = {column: "sum" for column in numeric_columns}
        aggregation.update(
            {column: "first" for column in df.columns if column not in numeric_columns + ["eqyticker"]}
        )
        return df.groupby("eqyticker", as_index=False).agg(aggregation).copy()

    def _calculate_index_overlap_adjustments(
        self,
        fund: Fund,
        holdings_df: pd.DataFrame,
        overlap_df: pd.DataFrame,
        total_assets: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply FLEX index overlap adjustments; the caller guarantees index FLEX and a non-empty overlap."""
        overlap_columns = ["security_ticker", "security_weight", "overlap_market_value", "overlap_weight"]
        empty = pd.DataFrame(columns=overlap_columns)

        def _base_result(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = df.copy()
            df["net_market_value"] = _num(df, "equity_market_value") + _num(df, "option_market_value")
            return df, empty

        if holdings_df.empty:
            return holdings_df.copy(), empty

        flex_options = fund.data.current.vest.flex_options
        if flex_options.empty:
            return _base_result(holdings_df)

        overlap = overlap_df.copy()
        overlap["security_weight"] = pd.to_numeric(overlap["security_weight"], errors="coerce").fillna(0.0)
        overlap = overlap.drop_duplicates(subset="security_ticker")

        flex_df = flex_options.copy()
        if "optticker" in flex_df.columns:
            flex_mask = flex_df["optticker"].astype(str).str.upper().str.startswith(
                ("4SPX", "4XSP", "SPX", "XSP", "2SPX", "2XSP")
            )
        elif "ticker_option" in flex_df.columns:
            flex_mask = flex_df["ticker_option"].astype(str).str.upper().isin(["SPX", "XSP"])
        else:
            flex_mask = pd.Series(False, index=flex_df.index)

        flex_positions = flex_df.loc[flex_mask].copy()
        if flex_positions.empty:
            return _base_result(holdings_df)

        # Vest flex frames carry nav_shares (nav_shares_option only exists after an equity/option merge).
        qty_col = "nav_shares_option" if "nav_shares_option" in flex_positions.columns else "nav_shares"
        long_flex = flex_positions.loc[_num(flex_positions, qty_col) > 0].copy()
        if long_flex.empty:
            long_flex = flex_positions.loc[_num(flex_positions, "option_market_value") > 0].copy()

        total_flex_mv = float(_num(long_flex, "option_market_value").sum())
        if total_flex_mv <= 0:
            return _base_result(holdings_df)

        result = holdings_df.copy().reset_index(drop=True)
        if "long_box_market_value_overlap" not in result.columns:
            result["long_box_market_value_overlap"] = 0.0

        merged = result.merge(
            overlap[["security_ticker", "security_weight"]],
            left_on="eqyticker",
            right_on="security_ticker",
            how="left",
        )
        merged["security_weight"] = _num(merged, "security_weight")
        # The listed-option merge repeats an equity once per option on it; spread the box over each ticker once.
        merged.loc[merged["eqyticker"].notna() & merged["eqyticker"].duplicated(), "security_weight"] = 0.0
        merged["overlap_market_value"] = total_flex_mv * merged["security_weight"]
        merged["long_box_market_value_overlap"] = (
            _num(merged, "long_box_market_value_overlap") + merged["overlap_market_value"]
        )
        merged = merged.drop(columns=["security_ticker"], errors="ignore")
        merged["net_market_value"] = (
            _num(merged, "equity_market_value")
            + _num(merged, "option_market_value")
            + merged["long_box_market_value_overlap"]
        )

        overlap_details = merged.loc[
            merged["overlap_market_value"] > 0
        , ["eqyticker", "security_weight", "overlap_market_value"]].copy()
        overlap_details = overlap_details.rename(columns={"eqyticker": "security_ticker"})
        if not overlap_details.empty:
            if total_assets:
                overlap_details["overlap_weight"] = overlap_details["overlap_market_value"] / total_assets
            else:
                overlap_details["overlap_weight"] = 0.0
        else:
            overlap_details = pd.DataFrame(columns=overlap_columns)

        merged = merged.drop(columns=["security_weight", "overlap_market_value"], errors="ignore")
        return merged, overlap_details.reindex(columns=overlap_columns).reset_index(drop=True)

    def _get_current_totals(self, fund: Fund) -> tuple[float, float, float, float]:
        """Return snapshot total assets, total net assets, expenses, and cash."""
        snapshot = fund.data.current
        return snapshot.ta, snapshot.tna, snapshot.expenses, snapshot.cash

    def _extract_long_options(self, opt_df: pd.DataFrame, gics_cols: list) -> pd.DataFrame:
        """Return long-only option rows with GICS columns and option_market_value."""
        if opt_df.empty or "option_market_value" not in opt_df.columns:
            return pd.DataFrame()
        qty = pd.Series(0.0, index=opt_df.index)
        for qty_col in ("nav_shares", "iiv_shares", "quantity", "shares"):
            if qty_col in opt_df.columns:
                qty = pd.to_numeric(opt_df[qty_col], errors="coerce").fillna(0.0)
                break
        long_mask = qty > 0
        if not long_mask.any():
            return pd.DataFrame()
        filtered = opt_df[long_mask].copy()
        opt_gics_cols = [c for c in GICS_CLASS_COLUMNS if c in filtered.columns]
        if not opt_gics_cols:
            return pd.DataFrame()
        opt_frame = filtered[opt_gics_cols].copy()
        opt_frame["_contrib_value"] = pd.to_numeric(
            filtered["option_market_value"], errors="coerce"
        ).fillna(0.0).values
        for col in gics_cols:
            if col not in opt_frame.columns:
                opt_frame[col] = None
        return opt_frame

    def run_compliance_tests(self, test_functions: Optional[list[str]] = None) -> Dict[str, Dict[str, ComplianceResult]]:
        """Execute the requested compliance checks for every fund."""
        rules = _RULES
        if test_functions:
            requested = set(test_functions)
            rules = tuple(rule for rule in _RULES if rule[0] in requested)

        results: Dict[str, Dict[str, ComplianceResult]] = {}
        for fund_name, fund in self.funds.items():
            fund_results: Dict[str, ComplianceResult] = {}
            for test_name, method_name, display, log_msg in rules:
                if test_name == "diversification_IRC_check" and (fund.vehicle or "").lower() != VEHICLE_VIT:
                    continue
                try:
                    fund_results[test_name] = getattr(self, method_name)(fund)
                except Exception as exc:
                    logger.error(log_msg, fund.name, exc)
                    extra = {"fund": fund.name} if test_name == "gics_compliance" else {}
                    fund_results[test_name] = _res(False, display, {}, str(exc), status="error", **extra)
            results[fund_name] = fund_results
        return results

    def calculate_summary_metrics(self, fund: Fund) -> ComplianceResult:
        total_assets, total_net_assets, expenses, cash_value = self._get_current_totals(fund)
        snapshot = fund.data.current
        calculations = {
            "cash_value": cash_value,
            "equity_market_value": snapshot.total_equity_value,
            "option_market_value": snapshot.total_option_value,  # Regular options only
            "flex_option_market_value": snapshot.total_flex_option_value,
            "total_option_market_value": snapshot.total_all_options_value,
            "option_delta_adjusted_notional": snapshot.total_option_delta_adjusted_notional,  # Already includes flex
            "treasury_market_value": snapshot.total_treasury_value,
            "total_assets": total_assets,
            "total_net_assets": total_net_assets,
            "expenses": expenses,
        }
        return _res(True, "summary_metrics", calculations, status="calculated")

    def prospectus_80pct_policy(self, fund: Fund) -> ComplianceResult:
        rule = "80% Prospectus Policy"
        total_assets, total_net_assets, _, total_cash_value = self._get_current_totals(fund)
        current = fund.data.current
        total_equity_market_value = float(current.total_equity_value)
        total_opt_market_value = float(current.total_option_value)
        total_tbill_value = float(current.total_treasury_value)
        total_opt_delta_notional_value = float(current.total_option_delta_adjusted_notional)

        is_closed_end = fund.is_closed_end_fund
        # Case-sensitive on purpose (other rules use the lowercased fund.is_private_fund).
        if fund.vehicle == VEHICLE_PRIVATE:
            skip_reason = "Private funds are excluded from the 80% test"
        elif is_closed_end and fund.name.startswith(("P3", "TR")):
            skip_reason = "TRIF closed-end funds are excluded from the 80% test"
        else:
            skip_reason = None
        if skip_reason:
            calculations = {
                "total_equity_market_value": total_equity_market_value,
                "total_opt_market_value": total_opt_market_value,
                "total_tbill_value": total_tbill_value,
                "total_cash_value": total_cash_value,
                "total_assets": total_assets,
                "total_net_assets": total_net_assets,
            }
            return _res(True, rule, calculations, skipped=True, reason=skip_reason, fund=fund.name)

        is_b3_closed_end = is_closed_end and fund.name.endswith("B3")
        if is_closed_end:
            options_in_scope = fund.has_flex_option or fund.has_listed_option or is_b3_closed_end
            option_primary_amount, option_primary_method = total_opt_market_value, "market_value"
        else:
            options_in_scope = fund.has_listed_option and fund.name in PROSPECTUS_OPTIONS_FUNDS
            option_primary_amount = abs(total_opt_delta_notional_value)
            option_primary_method = "delta_adjusted_notional"

        def ratio(option_amount: float) -> Tuple[float, float, float]:
            numerator = total_equity_market_value + total_tbill_value
            denominator = total_equity_market_value + total_cash_value + total_tbill_value
            if options_in_scope:
                numerator += option_amount
                denominator += option_amount
            return numerator, denominator, numerator / denominator if denominator else 0.0

        numerator, denominator, names_test = ratio(option_primary_amount)
        numerator_mv, denominator_mv, names_test_mv = ratio(total_opt_market_value)
        policy_variant = "closed_end_b3" if is_b3_closed_end else ("closed_end" if is_closed_end else "standard")

        calculations = {
            "total_equity_market_value": total_equity_market_value,
            "total_opt_delta_notional_value": total_opt_delta_notional_value,
            "total_opt_market_value": total_opt_market_value,
            "total_tbill_value": total_tbill_value,
            "total_cash_value": total_cash_value,
            "option_contribution": option_primary_amount if options_in_scope else 0.0,
            "option_contribution_method": option_primary_method,
            "denominator": denominator,
            "numerator": numerator,
            "names_test": names_test,
            "denominator_mv": denominator_mv,
            "numerator_mv": numerator_mv,
            "names_test_mv": names_test_mv,
            "threshold": PROSPECTUS_MIN_THRESHOLD,
            "total_assets": total_assets,
            "total_net_assets": total_net_assets,
            "options_in_scope": options_in_scope,
            "policy_variant": policy_variant,
        }
        return _res(
            names_test >= PROSPECTUS_MIN_THRESHOLD,
            rule,
            calculations,
            options_in_scope=options_in_scope,
            options_valuation_method=option_primary_method,
            policy_variant=policy_variant,
        )

    def diversification_IRS_check(self, fund: Fund) -> ComplianceResult:
        # IRS §851 RIC diversification is for registered investment companies; private funds are not RICs.
        if fund.is_private_fund:
            return _res(
                True, "IRS Diversification", {}, skipped=True,
                reason="Private funds are not RICs under IRS §851", fund_type="Private Fund",
            )

        vest_eqy_holdings = fund.data.current.vest.equity
        vest_opt_holdings = fund.data.current.vest.options
        vest_flex_holdings = fund.data.current.vest.flex_options
        total_tbill_value = float(fund.data.current.total_treasury_value)
        total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)
        overlap_df = fund.data.current.overlap

        vest_eqy_holdings = self._consolidate_holdings_by_issuer(vest_eqy_holdings)

        if fund.has_listed_option:
            holdings_df = pd.merge(
                vest_eqy_holdings,
                _options_per_underlying(vest_opt_holdings),
                left_on="eqyticker",
                right_on="equity_underlying_ticker",
                how="outer",
                suffixes=("", "_option"),
            )
            if "option_market_value" in holdings_df.columns:
                holdings_df["option_market_value"] = holdings_df["option_market_value"].fillna(0)
            else:
                holdings_df["option_market_value"] = 0
        else:
            holdings_df = vest_eqy_holdings

        overlap_details_df = pd.DataFrame(
            columns=["security_ticker", "security_weight", "overlap_market_value", "overlap_weight"]
        )
        has_index_flex = fund.has_flex_option and ((fund.flex_option_type or "").lower() == "index")
        if has_index_flex and not overlap_df.empty:
            holdings_df, overlap_details_df = self._calculate_index_overlap_adjustments(
                fund, holdings_df, overlap_df, total_assets
            )
        else:
            holdings_df["net_market_value"] = holdings_df["equity_market_value"] + holdings_df["option_market_value"]

        if fund.has_flex_option and ((fund.flex_option_type or "").lower() == "single_stock"):
            df_flex_calls = vest_flex_holdings[vest_flex_holdings["option_notional_value"] > 0]
            holdings_df = pd.merge(
                holdings_df,
                _options_per_underlying(df_flex_calls[["equity_underlying_ticker", "option_notional_value"]]),
                left_on="eqyticker",
                right_on="equity_underlying_ticker",
                how="outer",
                suffixes=("", "_flex"),
            )
            holdings_df["net_market_value"] = holdings_df["net_market_value"] + holdings_df["option_notional_value_flex"]

        holdings_df["weight"] = holdings_df["net_market_value"] / total_assets if total_assets else 0.0
        holdings_df["tna_wgt"] = holdings_df["net_market_value"] / total_net_assets if total_net_assets else 0.0

        holdings_df = holdings_df.sort_values("tna_wgt", ascending=False).reset_index(drop=True)
        largest_holding = holdings_df.iloc[0].to_dict() if len(holdings_df) >= 1 else {}
        holdings_df["cumulative_weight"] = holdings_df["tna_wgt"].cumsum()
        bottom_50_mask = holdings_df["cumulative_weight"] > 0.5
        bottom_50_df = holdings_df[bottom_50_mask].copy()
        top_50_df = holdings_df[~bottom_50_mask].copy()

        qualifying_assets_value = float(holdings_df["net_market_value"].sum())
        if qualifying_assets_value > total_assets and total_net_assets:
            condition_IRS_2_a_50 = (total_net_assets - expenses) / total_net_assets >= IRS_QUALIFYING_ASSETS_MIN
        else:
            condition_IRS_2_a_50 = (
                qualifying_assets_value >= IRS_QUALIFYING_ASSETS_MIN * total_net_assets
                if total_net_assets
                else False
            )

        five_pct_gross_assets = total_assets * 0.05
        large_securities = holdings_df[holdings_df["equity_market_value"] >= five_pct_gross_assets]
        sum_large_securities_weights = float(large_securities["tna_wgt"].sum())
        condition_IRS_2_a_5_new = sum_large_securities_weights <= 0.5

        bottom_50_df["exceeds_5_percent"] = bottom_50_df["tna_wgt"] > IRS_BOTTOM_50_LIMIT
        condition_IRS_2_a_5_original = not bool(bottom_50_df["exceeds_5_percent"].any())

        if "EQY_SH_OUT_million" not in bottom_50_df.columns:
            bottom_50_df["EQY_SH_OUT_million"] = 1.0
        denom = bottom_50_df["EQY_SH_OUT_million"].replace(0, 1)
        share_series = self._held_shares(bottom_50_df)
        bottom_50_df["exceeds_10_percent"] = (share_series / denom) > IRS_OWNERSHIP_LIMIT
        condition_IRS_2_a_10 = not bool(bottom_50_df["exceeds_10_percent"].any())

        details = {
            "rule": "IRS Diversification",
            "condition_IRS_1": True,
            "condition_IRS_2_a_50": condition_IRS_2_a_50,
            "condition_IRS_2_a_5": condition_IRS_2_a_5_new,
            "condition_IRS_2_a_5_original": condition_IRS_2_a_5_original,
            "condition_IRS_2_a_10": condition_IRS_2_a_10,
        }
        calculations = {
            "total_assets": total_assets,
            "total_net_assets": total_net_assets,
            "total_tbill_value": total_tbill_value,
            "qualifying_assets_value": qualifying_assets_value,
            "five_pct_gross_assets": five_pct_gross_assets,
            "sum_large_securities_weights": sum_large_securities_weights,
            "large_securities_count": int(len(large_securities)),
            "largest_holding": largest_holding,
            "second_largest_holding": holdings_df.iloc[1].to_dict() if len(holdings_df) >= 2 else {},
            "bottom_50_largest": bottom_50_df.iloc[0].to_dict() if len(bottom_50_df) >= 1 else {},
            "bottom_50_second_largest": bottom_50_df.iloc[1].to_dict() if len(bottom_50_df) >= 2 else {},
            "holdings_df": holdings_df.to_dict("records"),
            "bottom_50_holdings": bottom_50_df.to_dict("records"),
            "top_50_holdings": top_50_df.to_dict("records"),
            "expenses": expenses,
            "bottom_50_exceeds_5_percent": int(bottom_50_df["exceeds_5_percent"].sum()),
            "bottom_50_exceeds_10_percent": int(bottom_50_df["exceeds_10_percent"].sum()),
            "large_securities": large_securities.to_dict("records"),
            "overlap": overlap_details_df.to_dict("records"),
            "overlap_weight_sum": float(overlap_details_df["overlap_weight"].sum()),
            "overlap_market_value_sum": float(overlap_details_df["overlap_market_value"].sum()),
        }
        is_compliant = all([condition_IRS_2_a_50, condition_IRS_2_a_5_new, condition_IRS_2_a_10])
        return ComplianceResult(is_compliant, details, calculations)

    def diversification_40act_check(self, fund: Fund) -> ComplianceResult:
        if fund.is_private_fund:
            return _res(
                True, "40 Act Diversification", {}, skipped=True,
                reason="Private funds are not registered under the 40 Act", fund_type="Private Fund",
            )

        vest_opt_holdings = fund.data.current.vest.options
        vest_eqy_holdings = fund.data.current.vest.equity
        total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)
        vest_eqy_holdings = self._consolidate_holdings_by_issuer(vest_eqy_holdings)
        fund_registration = (fund.diversification_status or "unknown").replace("_", "-").lower()

        long_opts = self._extract_long_options(vest_opt_holdings, [])
        long_flex = self._extract_long_options(fund.data.current.vest.flex_options, [])
        if fund.has_listed_option and fund.listed_option_type == "index":
            holdings_df = vest_eqy_holdings.copy()
            holdings_df["option_market_value"] = 0.0
            holdings_df["net_market_value"] = holdings_df["equity_market_value"]
        else:
            # Long options carry no issuer ticker, so none are attributed to issuers; the merge keeps
            # the original column layout and the KeyError for equity frames without eqyticker.
            holdings_df = pd.merge(
                vest_eqy_holdings,
                pd.DataFrame(columns=["eqyticker", "long_option_mv"]),
                how="left",
                on="eqyticker",
            )
            holdings_df["long_option_mv"] = holdings_df["long_option_mv"].fillna(0.0)
            holdings_df["net_market_value"] = holdings_df["equity_market_value"] + holdings_df["long_option_mv"]

        # Qualifying assets: equity + long options (cash and treasury are not counted)
        long_opt_mv = long_opts["_contrib_value"].sum() if not long_opts.empty else 0.0
        long_flex_mv = long_flex["_contrib_value"].sum() if not long_flex.empty else 0.0
        equity_value = float(vest_eqy_holdings["equity_market_value"].sum()) if not vest_eqy_holdings.empty else 0.0

        qualifying_assets = equity_value + long_opt_mv + long_flex_mv
        qualifying_pct = qualifying_assets / total_assets if total_assets else 0.0
        condition_1_met = qualifying_pct >= ACT_40_QUALIFYING_ASSETS_MIN
        non_qualifying_assets = total_assets - qualifying_assets
        non_qualifying_assets_pct = non_qualifying_assets / total_assets if total_assets else 0.0

        five_pct_threshold = ACT_40_ISSUER_LIMIT * total_assets
        max_excluded_bucket = 0.25 * total_assets

        # Allocate 5%+ issuers (largest first) to the 25% bucket until it is full
        sorted_df = holdings_df.sort_values(by="net_market_value", ascending=False)
        five_pct_plus_issuers = sorted_df[sorted_df["net_market_value"] > five_pct_threshold].copy()
        excluded_securities = []
        excluded_bucket_total = 0.0
        for _, row in five_pct_plus_issuers.iterrows():
            issuer_value = row["net_market_value"]
            if excluded_bucket_total + issuer_value <= max_excluded_bucket:
                excluded_securities.append(row["eqyticker"])
                excluded_bucket_total += issuer_value

        # Remaining securities (< 5% and 5%+ issuers that did not fit) form the 75% bucket
        remaining_securities = sorted_df[~sorted_df["eqyticker"].isin(excluded_securities)].copy()
        issuer_limited = remaining_securities[remaining_securities["net_market_value"] > five_pct_threshold].copy()
        issuer_limited_sum = float(issuer_limited["net_market_value"].sum())
        condition_2a_met = issuer_limited.empty

        occ_weight_mkt_val = abs(float(vest_opt_holdings["option_market_value"].sum()))
        condition_2a_occ_met = occ_weight_mkt_val <= 0.05 * total_net_assets if total_net_assets else False

        if "EQY_SH_OUT_million" not in remaining_securities.columns:
            remaining_securities["EQY_SH_OUT_million"] = 1.0
        remaining_securities["EQY_SH_OUT_million"] = (
            pd.to_numeric(remaining_securities["EQY_SH_OUT_million"], errors="coerce").replace(0, 1)
        )

        share_col = self._share_col
        if share_col not in remaining_securities.columns:
            remaining_securities[share_col] = 0.0
        else:
            remaining_securities[share_col] = pd.to_numeric(remaining_securities[share_col], errors="coerce").fillna(0.0)

        if not remaining_securities.empty:
            ticker_overrides = {"LOGI": 160784460, "RCI": 429073267, "TU": 1550930636}
            for ticker, value in ticker_overrides.items():
                if ticker in remaining_securities["eqyticker"].values:
                    remaining_securities.loc[remaining_securities["eqyticker"] == ticker, "EQY_SH_OUT_million"] = value

        remaining_securities["vest_ownership_of_float"] = (
                remaining_securities[share_col] / remaining_securities["EQY_SH_OUT_million"]
        ).fillna(0.0)
        issuer_compliance_2b = remaining_securities["vest_ownership_of_float"] <= ACT_40_OWNERSHIP_LIMIT
        condition_2b_met = bool(issuer_compliance_2b.all())

        cumulative_weight_excluded = (
            excluded_bucket_total / total_net_assets
            if (excluded_bucket_total and total_net_assets)
            else 0.0
        )
        if not remaining_securities.empty and total_assets:
            remaining_securities["vest_weight"] = remaining_securities["net_market_value"] / total_assets
        else:
            remaining_securities["vest_weight"] = 0.0
        remaining_stocks_details = remaining_securities[
            ["eqyticker", share_col, "net_market_value", "vest_weight", "vest_ownership_of_float"]
        ].to_dict("records")

        actual_diversified = all([condition_1_met, condition_2a_met, condition_2b_met])
        fund_status_today = "diversified" if actual_diversified else "non-diversified"
        if fund_registration in {"diversified", "non-diversified"}:
            expected_diversified = fund_registration == "diversified"
        else:
            expected_diversified = actual_diversified
        expected_status = "diversified" if expected_diversified else "non-diversified"

        calculations = {
            "fund_registration": fund_registration,
            "fund_status_today": fund_status_today,
            "expected_fund_status": expected_status,
            "total_assets": total_assets,
            "net_assets": total_net_assets,
            "expenses": expenses,
            "condition_1_met": condition_1_met,
            "non_qualifying_assets_1_wgt": non_qualifying_assets_pct,
            "qualifying_assets_pct": qualifying_pct,
            "issuer_limited_sum": issuer_limited_sum,
            "issuer_limited_securities": issuer_limited.to_dict("records"),
            "condition_2a_met": condition_2a_met,
            "condition_2b_met": condition_2b_met,
            "issuer_compliance_2b": issuer_compliance_2b.to_dict(),
            "excluded_securities": excluded_securities,
            "cumulative_weight_excluded": cumulative_weight_excluded,
            "cumulative_weight_remaining": 1 - cumulative_weight_excluded,
            "remaining_stocks_details": remaining_stocks_details,
            "share_column_used": share_col,
            "max_ownership_float": float(remaining_securities["vest_ownership_of_float"].max()),
            "condition_2a_occ_met": condition_2a_occ_met,
            "occ_market_value": occ_weight_mkt_val,
            "occ_weight": occ_weight_mkt_val / total_assets if total_assets else 0.0,
        }
        details = {
            "rule": "40 Act Diversification",
            "condition_40act_1": condition_1_met,
            "condition_40act_2a": condition_2a_met,
            "condition_40act_2b": condition_2b_met,
            "condition_40act_2a_occ": condition_2a_occ_met,
            "diversified_since_inception_40act": condition_1_met and condition_2a_met and condition_2b_met,
            "diversification_failure_duration_40act": 0 if actual_diversified else 1,
            "expected_registration_status": expected_status,
            "actual_registration_status": fund_status_today,
        }
        return ComplianceResult(actual_diversified == expected_diversified, details, calculations)

    def diversification_IRC_check(self, fund: Fund) -> ComplianceResult:
        """IRC diversification; the dispatcher only runs it for VIT funds."""
        regular_options = fund.data.current.vest.options
        vest_eqy_holdings = fund.data.current.vest.equity
        total_assets, _, _, _ = self._get_current_totals(fund)
        vest_eqy_holdings = self._consolidate_holdings_by_issuer(vest_eqy_holdings)

        if vest_eqy_holdings.empty:
            raise ValueError("Equity holdings missing")
        if total_assets == 0:
            raise ValueError("Total assets missing")

        holdings_df = pd.merge(
            vest_eqy_holdings,
            regular_options,
            how="left",
            left_on="eqyticker",
            right_on="equity_underlying_ticker",
            suffixes=("", "_option"),
        )
        holdings_df["net_market_value"] = (
                pd.to_numeric(holdings_df["equity_market_value"], errors="coerce").fillna(0.0)
                + pd.to_numeric(holdings_df["option_market_value"], errors="coerce").fillna(0.0)
        )
        holdings_df["weight"] = holdings_df["net_market_value"] / total_assets
        sorted_holdings = holdings_df.sort_values(by="net_market_value", ascending=False)

        top_exposures = [0.0, 0.0, 0.0, 0.0]
        if not sorted_holdings.empty:
            cum_values = sorted_holdings["net_market_value"].cumsum()
            for idx in range(min(4, len(cum_values))):
                top_exposures[idx] = float(cum_values.iloc[idx] / total_assets)

        details = {
            "rule": "IRC Diversification",
            "condition_IRC_55": top_exposures[0] <= IRC_TOP_1_LIMIT,
            "condition_IRC_70": top_exposures[1] <= IRC_TOP_2_LIMIT,
            "condition_IRC_80": top_exposures[2] <= IRC_TOP_3_LIMIT,
            "condition_IRC_90": top_exposures[3] <= IRC_TOP_4_LIMIT,
        }
        calculations = {
            "total_assets": total_assets,
            "top_1": top_exposures[0],
            "top_2": top_exposures[1],
            "top_3": top_exposures[2],
            "top_4": top_exposures[3],
            "top_holdings": sorted_holdings[["eqyticker", "net_market_value"]].head(4).to_dict("records"),
        }
        is_compliant = all(
            details[key] for key in ("condition_IRC_55", "condition_IRC_70", "condition_IRC_80", "condition_IRC_90")
        )
        return ComplianceResult(is_compliant, details, calculations)

    @staticmethod
    def _sector_exposure(fund: Fund, pattern: str) -> Optional[Tuple[float, float]]:
        """Return (sector exposure, total equity exposure), or None when there are no equity holdings."""
        vest_eqy_holdings = fund.data.current.vest.equity
        if vest_eqy_holdings.empty:
            return None
        mask = vest_eqy_holdings["GICS_SECTOR_NAME"].str.contains(pattern, case=False, na=False)
        exposure = float(vest_eqy_holdings.loc[mask, "equity_market_value"].sum())
        return exposure, float(vest_eqy_holdings["equity_market_value"].sum())

    def real_estate_check(self, fund: Fund) -> ComplianceResult:
        rule = "Real Estate Exposure"
        exposures = self._sector_exposure(fund, "Real Estate")
        if exposures is None:
            calculations = {"real_estate_exposure": 0.0, "real_estate_percentage": 0.0, "total_exposure": 0.0}
            return _res(True, rule, calculations, real_estate_check_compliant=True)
        real_estate_exposure, total_exposure = exposures
        calculations = {
            "real_estate_exposure": real_estate_exposure,
            "real_estate_percentage": real_estate_exposure / total_exposure if total_exposure > 0 else 0.0,
            "total_exposure": total_exposure,
        }
        # Same zero-tolerance test as commodities_check.
        is_compliant = real_estate_exposure == 0.0
        return _res(is_compliant, rule, calculations, real_estate_check_compliant=is_compliant)

    def commodities_check(self, fund: Fund) -> ComplianceResult:
        rule = "Commodities Exposure"
        exposures = self._sector_exposure(fund, "Commodities")
        if exposures is None:
            calculations = {"commodities_exposure": 0.0, "commodities_percentage": 0.0, "total_exposure": 0.0}
            return _res(True, rule, calculations, commodities_check_compliant=True)
        commodities_exposure, total_exposure = exposures
        calculations = {
            "commodities_exposure": commodities_exposure,
            "total_exposure": total_exposure,
            "commodities_percentage": commodities_exposure / total_exposure if total_exposure > 0 else 0.0,
        }
        is_compliant = commodities_exposure == 0.0
        return _res(is_compliant, rule, calculations, commodities_check_compliant=is_compliant)

    def twelve_d1a_other_inv_cos(self, fund: Fund) -> ComplianceResult:
        rule = "12d1-1 Other Investment Companies"
        vest_eqy_holdings = fund.data.current.vest.equity
        total_assets, _, _, _ = self._get_current_totals(fund)
        all_pass = {
            "twelve_d1a_other_inv_cos_compliant": True,
            "test_1_pass": True,
            "test_2_pass": True,
            "test_3_pass": True,
        }
        base_calculations = {
            "total_assets": total_assets,
            "investment_companies": [],
            "ownership_pct_max": None,
            "equity_market_value_sum": 0.0,
            "equity_market_value_max": 0.0,
            "ownership_pct_values": [],
        }

        # No equity / no assets (e.g. cash-only private funds) is a skipped PASS rather than an error.
        if total_assets == 0 or vest_eqy_holdings.empty:
            return _res(
                True, rule, base_calculations, skipped=True, reason="No equity holdings or total assets", **all_pass
            )

        thresholds = {
            "test_2_threshold": RULE_12D1A_SINGLE_ASSETS_LIMIT * total_assets,
            "test_3_threshold": RULE_12D1A_TOTAL_ASSETS_LIMIT * total_assets,
        }
        investment_companies = vest_eqy_holdings[~vest_eqy_holdings["REGULATORY_STRUCTURE"].isnull()].copy()
        if investment_companies.empty:
            return _res(True, rule, {**base_calculations, **thresholds}, **all_pass)

        if "EQY_SH_OUT_million" not in investment_companies.columns:
            investment_companies["EQY_SH_OUT_million"] = 0.0
        if "quantity" not in investment_companies.columns:
            investment_companies["quantity"] = 0.0
        investment_companies["ownership_pct"] = _ownership_pct(
            investment_companies["quantity"], investment_companies["EQY_SH_OUT_million"]
        )

        max_ownership_pct = float(investment_companies["ownership_pct"].max())
        max_equity_value = float(investment_companies["equity_market_value"].max())
        sum_equity_value = float(investment_companies["equity_market_value"].sum())
        test_1_pass = max_ownership_pct <= RULE_12D1A_OWNERSHIP_LIMIT
        test_2_pass = max_equity_value <= RULE_12D1A_SINGLE_ASSETS_LIMIT * total_assets
        test_3_pass = sum_equity_value <= RULE_12D1A_TOTAL_ASSETS_LIMIT * total_assets

        calculations = {
            "total_assets": total_assets,
            "investment_companies": investment_companies.to_dict("records"),
            "ownership_pct_max": max_ownership_pct,
            "equity_market_value_sum": sum_equity_value,
            "equity_market_value_max": max_equity_value,
            "ownership_pct_values": investment_companies["ownership_pct"].tolist(),
            **thresholds,
        }
        is_compliant = all([test_1_pass, test_2_pass, test_3_pass])
        return _res(
            is_compliant,
            rule,
            calculations,
            twelve_d1a_other_inv_cos_compliant=is_compliant,
            test_1_pass=test_1_pass,
            test_2_pass=test_2_pass,
            test_3_pass=test_3_pass,
        )

    def twelve_d2_insurance_cos(self, fund: Fund) -> ComplianceResult:
        rule = "12d-2 Insurance Companies"
        vest_eqy_holdings = fund.data.current.vest.equity
        total_assets, _, _, _ = self._get_current_totals(fund)
        insurance_mask = (
            vest_eqy_holdings["GICS_INDUSTRY_GROUP_NAME"].eq("Insurance")
            | vest_eqy_holdings["GICS_INDUSTRY_NAME"].eq("Insurance")
        )
        insurance_holdings = vest_eqy_holdings[insurance_mask].copy()

        if insurance_holdings.empty:
            calculations = {
                "total_assets": total_assets,
                "insurance_holdings": [],
                "insurance_count": 0,
                "max_ownership_pct": 0.0,
                "threshold": RULE_12D2_INSURANCE_LIMIT,
            }
            return _res(True, rule, calculations, twelve_d2_insurance_cos_compliant=True, test_pass=True)

        if "EQY_SH_OUT_million" not in insurance_holdings.columns:
            insurance_holdings["EQY_SH_OUT_million"] = 0.0
        share_series = self._held_shares(insurance_holdings)
        insurance_holdings["ownership_pct"] = _ownership_pct(share_series, insurance_holdings["EQY_SH_OUT_million"])

        max_ownership_pct = float(insurance_holdings["ownership_pct"].max())
        test_pass = max_ownership_pct <= RULE_12D2_INSURANCE_LIMIT
        calculations = {
            "total_assets": total_assets,
            "insurance_holdings": insurance_holdings.to_dict("records"),
            "insurance_count": len(insurance_holdings),
            "max_ownership_pct": max_ownership_pct,
            "threshold": RULE_12D2_INSURANCE_LIMIT,
        }
        return _res(test_pass, rule, calculations, twelve_d2_insurance_cos_compliant=test_pass, test_pass=test_pass)

    def twelve_d3_sec_biz(self, fund: Fund) -> ComplianceResult:
        rule = "12d-3 Securities Businesses"
        vest_eqy_holdings = fund.data.current.vest.equity
        vest_opt_holdings = fund.data.current.vest.options
        total_assets, _, _, _ = self._get_current_totals(fund)
        sec_biz_mask = vest_eqy_holdings["GICS_INDUSTRY_NAME"].isin(["Capital Markets", "Banks"])
        sec_related_businesses = vest_eqy_holdings[sec_biz_mask].copy()

        if sec_related_businesses.empty:
            calculations = {
                "total_assets": total_assets,
                "sec_related_businesses": [],
                "combined_holdings": [],
                "occ_weight_mkt_val": 0.0,
                "occ_weight": 0.0,
                "max_ownership_pct": 0.0,
                "max_weight": 0.0,
            }
            return _res(
                True, rule, calculations,
                twelve_d3_sec_biz_compliant=True, rule_1_pass=True, rule_2_pass=True, rule_3_pass=True,
            )

        sec_related_businesses["EQY_SH_OUT_million"] = sec_related_businesses["EQY_SH_OUT_million"].replace(0, np.nan)
        share_series = self._held_shares(sec_related_businesses)
        sec_related_businesses["ownership_pct"] = (
            share_series / sec_related_businesses["EQY_SH_OUT_million"]
        ).fillna(0.0)
        rule_1_pass = bool((sec_related_businesses["ownership_pct"] <= RULE_12D3_EQUITY_LIMIT).all())

        if "SECURITY_TYP" in vest_eqy_holdings.columns:
            debt_mask = vest_eqy_holdings["SECURITY_TYP"].eq("Debt")
        else:
            debt_mask = pd.Series(False, index=vest_eqy_holdings.index)
        debt_related_businesses = vest_eqy_holdings[debt_mask]
        if debt_related_businesses.empty:
            rule_2_pass = True
        else:
            debt_threshold = RULE_12D3_DEBT_LIMIT * total_assets
            rule_2_pass = bool((debt_related_businesses["equity_market_value"] <= debt_threshold).all())

        combined_holdings = pd.merge(
            sec_related_businesses,
            _options_per_underlying(vest_opt_holdings),
            how="left",
            left_on="eqyticker",
            right_on="equity_underlying_ticker",
            suffixes=("", "_option"),
        )
        combined_holdings["option_market_value"] = _num(combined_holdings, "option_market_value")
        combined_holdings["equity_market_value"] = _num(combined_holdings, "equity_market_value")
        combined_holdings["net_market_value"] = (
            combined_holdings["equity_market_value"] + combined_holdings["option_market_value"]
        )
        combined_holdings["vest_weight"] = (
            combined_holdings["net_market_value"] / total_assets if total_assets else 0.0
        )
        combined_holdings = combined_holdings.loc[:, ~combined_holdings.columns.duplicated()]
        max_weight = float(combined_holdings["vest_weight"].max()) if not combined_holdings.empty else 0.0
        rule_3_pass = max_weight <= RULE_12D3_ASSET_LIMIT

        is_compliant = all([rule_1_pass, rule_2_pass, rule_3_pass])
        calculations = {
            "total_assets": total_assets,
            "sec_related_businesses": sec_related_businesses.to_dict("records"),
            "combined_holdings": combined_holdings.to_dict("records"),
            "max_ownership_pct": float(sec_related_businesses["ownership_pct"].max()),
            "max_weight": max_weight,
        }
        return _res(
            is_compliant, rule, calculations,
            twelve_d3_sec_biz_compliant=is_compliant, rule_1_pass=rule_1_pass,
            rule_2_pass=rule_2_pass, rule_3_pass=rule_3_pass,
        )

    def max_15pct_illiquid_sai(self, fund: Fund) -> ComplianceResult:
        rule = "15% Illiquid Assets"
        vest_opt_holdings = fund.data.current.vest.options
        vest_eqy_holdings = fund.data.current.vest.equity
        total_assets, _, _, _ = self._get_current_totals(fund)

        if total_assets == 0 or vest_eqy_holdings.empty:
            return _res(True, rule, {}, max_15pct_illiquid_sai=True, equity_holdings_85pct_compliant=True)

        # Mutates the shared holdings frames in place, so every later rule also sees is_illiquid.
        if "is_illiquid" not in vest_eqy_holdings.columns:
            vest_eqy_holdings["is_illiquid"] = False
        if "is_illiquid" not in vest_opt_holdings.columns:
            vest_opt_holdings["is_illiquid"] = False

        illiquid_mask = vest_eqy_holdings["is_illiquid"] == True
        illiquid_eqy_value = float(vest_eqy_holdings.loc[illiquid_mask, "equity_market_value"].sum())
        illiquid_opt_value = 0
        if not vest_opt_holdings.empty:
            illiquid_opt_mask = vest_opt_holdings["is_illiquid"] == True
            illiquid_opt_value = float(vest_opt_holdings.loc[illiquid_opt_mask, "option_market_value"].sum())

        total_illiquid_value = illiquid_eqy_value + illiquid_opt_value
        illiquid_percentage = total_illiquid_value / total_assets
        equity_percentage = float(vest_eqy_holdings["equity_market_value"].sum()) / total_assets

        calculations = {
            "total_assets": total_assets,
            "total_illiquid_value": total_illiquid_value,
            "illiquid_percentage": illiquid_percentage,
            "equity_holdings_percentage": equity_percentage,
        }
        return _res(
            illiquid_percentage <= ILLIQUID_MAX_THRESHOLD,
            rule,
            calculations,
            max_15pct_illiquid_sai=True,
            equity_holdings_85pct_compliant=equity_percentage >= EQUITY_MIN_THRESHOLD,
        )

    def gics_compliance(self, fund: Fund) -> ComplianceResult:
        rule = "GICS Concentration"

        def fail(status: str, error: str) -> ComplianceResult:
            return _res(False, rule, {}, error, status=status, fund=fund.name)

        vest_eqy_holdings = fund.data.current.vest.equity
        vest_opt_holdings = fund.data.current.vest.options
        vest_flex_holdings = fund.data.current.vest.flex_options
        total_assets, total_net_assets, _, _ = self._get_current_totals(fund)

        if total_assets == 0.0:
            return fail("missing_total_assets", "Total assets is zero — cannot compute GICS weights")

        equity_values = _num(vest_eqy_holdings, "equity_market_value")
        if equity_values.sum() == 0.0:
            return fail("missing_equity_market_value", "Equity market values are unavailable for weight calculation")

        eqy_gics_cols = [c for c in GICS_CLASS_COLUMNS if c in vest_eqy_holdings.columns]
        eqy_frame = vest_eqy_holdings[eqy_gics_cols].copy()
        eqy_frame["_contrib_value"] = equity_values.values

        long_opt_frame = self._extract_long_options(vest_opt_holdings, eqy_gics_cols)
        long_flex_frame = self._extract_long_options(vest_flex_holdings, eqy_gics_cols)
        frames = [eqy_frame] + [frame for frame in (long_opt_frame, long_flex_frame) if not frame.empty]

        combined_df = pd.concat(frames, ignore_index=True)
        combined_values = pd.to_numeric(combined_df["_contrib_value"], errors="coerce").fillna(0.0)
        fund_weights = combined_values / total_assets
        fund_weights.index = combined_df.index

        gics_summary_fund = self._summarize_gics_exposure(combined_df, fund_weights)
        if not gics_summary_fund:
            return fail("missing_gics_data", "GICS classifications are not available on equity holdings")

        index_equity = fund.data.current.index.equity
        index_df = index_equity.copy() if not index_equity.empty else pd.DataFrame()

        gics_summary_index: Dict[str, Dict[str, float]] = {}
        index_weight_basis: Optional[str] = None
        if not index_df.empty:
            for column in ("weight_index", "index_weight", "benchmark_weight", "weight", "equity_market_value"):
                if column in index_df.columns:
                    candidate = pd.to_numeric(index_df[column], errors="coerce").fillna(0.0)
                    total = float(candidate.sum())
                    if total != 0.0:
                        index_weights = candidate / total
                        index_weights.index = index_df.index
                        index_weight_basis = column
                        gics_summary_index = self._summarize_gics_exposure(index_df, index_weights)
                        break

        compliance_group = self._resolve_gics_compliance_group(fund.name)
        exceeding_fund_gics, exceeding_index_gics, calculations = self._check_exposure(
            gics_summary_fund, gics_summary_index
        )

        # Every group is judged at industry level only.
        fund_over_industries = set(exceeding_fund_gics.get("GICS_INDUSTRY_NAME", {}).keys())
        index_over_industries = set(exceeding_index_gics.get("GICS_INDUSTRY_NAME", {}).keys())
        industry_within_limit = len(fund_over_industries) == 0
        industry_mismatches = fund_over_industries - index_over_industries
        if compliance_group == "dogg":
            # DOGG: no industry may exceed 25%
            failing = fund_over_industries
        elif compliance_group == "tdvi":
            # TDVI: only Information Technology industries may exceed 25%
            industry_sector_map = self._build_industry_to_sector_map(vest_eqy_holdings)
            failing = [
                ind for ind in fund_over_industries
                if industry_sector_map.get(ind) != "Information Technology"
            ]
        else:
            # KNG/FDND and default: an industry may exceed 25% only when the index also exceeds 25% in it
            failing = industry_mismatches
        overall_status = "PASS" if len(failing) == 0 else "FAIL"

        calculations.setdefault("meta", {})
        calculations["meta"].update(
            {
                "total_assets": total_assets,
                "total_net_assets": total_net_assets,
                "fund_weight_basis": "equity_market_value",
                "index_weight_basis": index_weight_basis,
            }
        )
        calculations.setdefault("fund_summary", gics_summary_fund)
        calculations.setdefault("index_summary", gics_summary_index)

        details: Dict[str, object] = {
            "rule": rule,
            "fund": fund.name,
            "overall_gics_compliance": overall_status,
            "industry_exceeds_25": industry_within_limit,
            "exceeding_fund_gics": exceeding_fund_gics,
            "exceeding_index_gics": exceeding_index_gics,
            "fund_exposures": gics_summary_fund,
            "compliance_group": compliance_group,
            "fund_weight_basis": "equity_market_value",
        }
        if index_weight_basis:
            details["index_weight_basis"] = index_weight_basis
        if compliance_group not in {"dogg", "tdvi"} and industry_mismatches:
            details["industry_mismatches"] = sorted(industry_mismatches)

        return ComplianceResult(overall_status == "PASS", details, calculations)

    @staticmethod
    def _resolve_gics_compliance_group(fund_name: str) -> str:
        mapping = {
            "DOGG": "dogg",
            "KNG": "kng_fdnd",
            "KNGIX": "kng_fdnd",
            "FDND": "kng_fdnd",
            "TDVI": "tdvi",
        }
        return mapping.get(fund_name.upper(), "standard")

    def _summarize_gics_exposure(
        self,
        df: pd.DataFrame,
        weight_series: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        if df.empty or weight_series.empty:
            return {}

        weights = weight_series.reindex(df.index, fill_value=0.0)
        working = df.copy()
        working["_normalized_weight"] = weights

        summary: Dict[str, Dict[str, float]] = {}
        for column in GICS_CLASS_COLUMNS:
            if column not in working.columns:
                continue

            categories = working[column]
            if categories.dtype == object:
                categories = categories.astype(str).str.strip()
                categories = categories.replace({"": None})

            mask = categories.notna()
            if not mask.any():
                summary[column] = {}
                continue

            grouped = (
                working.loc[mask]
                .groupby(categories[mask])["_normalized_weight"]
                .sum()
                .sort_values(ascending=False)
            )
            summary[column] = grouped.round(6).to_dict()

        return summary

    def _check_exposure(
        self,
        gics_summary_fund: Mapping[str, Mapping[str, float]],
        gics_summary_index: Mapping[str, Mapping[str, float]],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, object]]]:
        exceeding_fund_gics: Dict[str, Dict[str, float]] = {}
        exceeding_index_gics: Dict[str, Dict[str, float]] = {}
        calculations: Dict[str, Dict[str, object]] = {}

        for gics_class in GICS_CLASS_COLUMNS:
            fund_series = pd.Series(gics_summary_fund.get(gics_class, {}), dtype=float).sort_values(ascending=False)
            index_series = pd.Series(gics_summary_index.get(gics_class, {}), dtype=float).sort_values(ascending=False)
            fund_mask = fund_series > GICS_CONCENTRATION_THRESHOLD
            index_mask = index_series > GICS_CONCENTRATION_THRESHOLD
            exceeding_fund = fund_series[fund_mask].round(6).to_dict()
            exceeding_index = index_series[index_mask].round(6).to_dict()
            exceeding_fund_gics[gics_class] = exceeding_fund
            exceeding_index_gics[gics_class] = exceeding_index
            calculations[gics_class] = {
                "fund_weights": fund_series.round(6).to_dict(),
                "index_weights": index_series.round(6).to_dict(),
                "exceeding_fund": exceeding_fund,
                "exceeding_index": exceeding_index,
                "concentration_threshold": GICS_CONCENTRATION_THRESHOLD,
                "fund_exceeding_count": int(fund_mask.sum()),
                "index_exceeding_count": int(index_mask.sum()),
            }

        return exceeding_fund_gics, exceeding_index_gics, calculations

    def _build_industry_to_sector_map(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map each GICS_INDUSTRY_NAME to the GICS_SECTOR_NAME of its first holding (None if missing)."""
        if df.empty or "GICS_INDUSTRY_NAME" not in df.columns or "GICS_SECTOR_NAME" not in df.columns:
            return {}

        mapping_df = df[["GICS_INDUSTRY_NAME", "GICS_SECTOR_NAME"]].copy()
        for column in ("GICS_INDUSTRY_NAME", "GICS_SECTOR_NAME"):
            if mapping_df[column].dtype == object:
                mapping_df[column] = mapping_df[column].astype(str).str.strip()

        industries = mapping_df["GICS_INDUSTRY_NAME"]
        mapping_df = mapping_df[industries.notna() & (industries != "") & (industries != "nan")]
        mapping_df = mapping_df.drop_duplicates(subset=["GICS_INDUSTRY_NAME"])
        return {
            industry: None if pd.isna(sector) or sector == "" or sector == "nan" else sector
            for industry, sector in zip(mapping_df["GICS_INDUSTRY_NAME"], mapping_df["GICS_SECTOR_NAME"])
        }
