from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from domain.fund import Fund
from utilities.logger import setup_logger
from utilities.ticker_utils import (
    ensure_equity_schema,
    ensure_option_schema,
    ensure_treasury_schema,
)

try:  # pragma: no cover - optional configuration module
    from config import constants as config_constants  # type: ignore
except Exception:  # pragma: no cover - the module is optional
    config_constants = None


def _constant(name: str, default):
    if config_constants is None:
        return default
    return getattr(config_constants, name, default)


PROSPECTUS_MIN_THRESHOLD: float = float(_constant("PROSPECTUS_MIN_THRESHOLD", 0.8))
PROSPECTUS_OPTIONS_FUNDS = set(_constant("PROSPECTUS_OPTIONS_FUNDS", {"KNG", "KNGIX", "DOGG", "FTCSH", "FGSI"}))
IRS_QUALIFYING_ASSETS_MIN: float = float(_constant("IRS_QUALIFYING_ASSETS_MIN", 0.5))
IRS_BOTTOM_50_LIMIT: float = float(_constant("IRS_BOTTOM_50_LIMIT", 0.05))
IRS_OWNERSHIP_LIMIT: float = float(_constant("IRS_OWNERSHIP_LIMIT", 0.1))
CLOSED_END_FUNDS = set(_constant("CLOSED_END_FUNDS", set()))
PRIVATE_FUNDS = set(_constant("PRIVATE_FUNDS", set()))
DIVERSIFIED_FUNDS = set(_constant("DIVERSIFIED_FUNDS", set()))
ACT_40_QUALIFYING_ASSETS_MIN: float = float(_constant("ACT_40_QUALIFYING_ASSETS_MIN", 0.75))
ACT_40_ISSUER_LIMIT: float = float(_constant("ACT_40_ISSUER_LIMIT", 0.05))
ACT_40_OWNERSHIP_LIMIT: float = float(_constant("ACT_40_OWNERSHIP_LIMIT", 0.1))
RULE_12D1A_OWNERSHIP_LIMIT: float = float(_constant("RULE_12D1A_OWNERSHIP_LIMIT", 0.03))
RULE_12D1A_SINGLE_ASSETS_LIMIT: float = float(_constant("RULE_12D1A_SINGLE_ASSETS_LIMIT", 0.05))
RULE_12D1A_TOTAL_ASSETS_LIMIT: float = float(_constant("RULE_12D1A_TOTAL_ASSETS_LIMIT", 0.1))
RULE_12D2_INSURANCE_LIMIT: float = float(_constant("RULE_12D2_INSURANCE_LIMIT", 0.05))
RULE_12D3_EQUITY_LIMIT: float = float(_constant("RULE_12D3_EQUITY_LIMIT", 0.05))
RULE_12D3_DEBT_LIMIT: float = float(_constant("RULE_12D3_DEBT_LIMIT", 0.25))
RULE_12D3_ASSET_LIMIT: float = float(_constant("RULE_12D3_ASSET_LIMIT", 0.1))
IRC_TOP_1_LIMIT: float = float(_constant("IRC_TOP_1_LIMIT", 0.55))
IRC_TOP_2_LIMIT: float = float(_constant("IRC_TOP_2_LIMIT", 0.7))
IRC_TOP_3_LIMIT: float = float(_constant("IRC_TOP_3_LIMIT", 0.8))
IRC_TOP_4_LIMIT: float = float(_constant("IRC_TOP_4_LIMIT", 0.9))
ILLIQUID_MAX_THRESHOLD: float = float(_constant("ILLIQUID_MAX_THRESHOLD", 0.15))
EQUITY_MIN_THRESHOLD: float = float(_constant("EQUITY_MIN_THRESHOLD", 0.85))


logger = setup_logger("compliance_checker", "compliance/logs/compliance_checker.log")


@dataclass
class ComplianceResult:
    """Structured result returned from a compliance check."""

    is_compliant: bool
    details: Dict[str, object]
    calculations: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None


class ComplianceChecker:
    """Runs compliance checks against :class:`Fund` domain objects."""

    _DEFAULT_GICS_LIMIT = 0.25

    def __init__(
        self,
        session,
        funds: Optional[Dict[str, Fund]] = None,
        date=None,
        base_cls=None,
    ) -> None:
        self.session = session
        self.funds: Dict[str, Fund] = funds or {}
        self.date = date
        self.base_cls = base_cls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_compliance_tests(self, test_functions: Optional[list[str]] = None) -> Dict[str, Dict[str, ComplianceResult]]:
        """Execute the requested compliance checks for every fund."""

        available_tests: Dict[str, Callable[[Fund], ComplianceResult]] = {
            "summary_metrics": self.calculate_summary_metrics,
            "prospectus_80pct_policy": self.prospectus_80pct_policy,
            "diversification_IRS_check": self.diversification_IRS_check,
            "diversification_40act_check": self.diversification_40act_check,
            "diversification_IRC_check": self.diversification_IRC_check,
            "max_15pct_illiquid_sai": self.max_15pct_illiquid_sai,
            "real_estate_check": self.real_estate_check,
            "commodities_check": self.commodities_check,
            "twelve_d1a_other_inv_cos": self.twelve_d1a_other_inv_cos,
            "twelve_d2_insurance_cos": self.twelve_d2_insurance_cos,
            "twelve_d3_sec_biz": self.twelve_d3_sec_biz,
            "gics_compliance": self.gics_compliance,
        }

        if test_functions:
            requested = set(test_functions)
            available_tests = {
                name: func for name, func in available_tests.items() if name in requested
            }

        results: Dict[str, Dict[str, ComplianceResult]] = {}

        for fund_name, fund in self.funds.items():
            logger.info("Running compliance tests for %s", fund_name)
            fund_results: Dict[str, ComplianceResult] = {}

            for test_name, test_func in available_tests.items():
                try:
                    fund_results[test_name] = test_func(fund)
                except Exception as exc:  # pragma: no cover - defensive logging path
                    logger.error(
                        "Error executing %s for fund %s: %s",
                        test_name,
                        fund_name,
                        exc,
                        exc_info=True,
                    )
                    fund_results[test_name] = ComplianceResult(
                        is_compliant=False,
                        details={"rule": test_name, "status": "error"},
                        calculations={},
                        error=str(exc),
                    )

            results[fund_name] = fund_results

        return results

    # ------------------------------------------------------------------
    # Individual compliance checks
    # ------------------------------------------------------------------
    def calculate_summary_metrics(self, fund: Fund) -> ComplianceResult:
        equity_df, options_df, treasury_df = self._get_holdings(fund)
        calculations = {
            "cash_value": self._get_cash_value(fund),
            "equity_market_value": float(equity_df["equity_market_value"].sum()),
            "option_delta_adjusted_notional": float(options_df["option_delta_adjusted_notional"].sum()),
            "option_market_value": float(options_df["option_market_value"].sum()),
            "treasury": float(treasury_df["treasury_market_value"].sum()),
            "total_assets": self._get_total_assets(fund)[0],
            "total_net_assets": self._get_total_assets(fund)[1],
        }

        return ComplianceResult(
            is_compliant=True,
            details={"rule": "summary_metrics", "status": "calculated"},
            calculations=calculations,
        )

    def prospectus_80pct_policy(self, fund: Fund) -> ComplianceResult:
        try:
            equity_df, options_df, treasury_df = self._get_holdings(fund)
            total_assets, total_net_assets = self._get_total_assets(fund)
            total_cash_value = self._get_cash_value(fund)

            total_equity_market_value = float(equity_df["equity_market_value"].sum())
            total_opt_market_value = float(options_df["option_market_value"].sum())
            total_tbill_value = float(treasury_df["treasury_market_value"].sum())
            total_opt_delta_notional_value = float(options_df["option_delta_adjusted_notional"].sum())

            options_in_scope = fund.name in PROSPECTUS_OPTIONS_FUNDS

            numerator = total_equity_market_value + total_tbill_value
            denominator = (
                total_equity_market_value + total_cash_value + total_tbill_value
            )

            if options_in_scope:
                numerator += abs(total_opt_delta_notional_value)
                denominator += abs(total_opt_delta_notional_value)

            numerator_mv = total_equity_market_value + total_tbill_value
            denominator_mv = (
                total_equity_market_value + total_cash_value + total_tbill_value
            )

            if options_in_scope:
                numerator_mv += total_opt_market_value
                denominator_mv += total_opt_market_value

            names_test = numerator / denominator if denominator else 0.0
            names_test_mv = numerator_mv / denominator_mv if denominator_mv else 0.0

            calculations = {
                "total_equity_market_value": total_equity_market_value,
                "total_opt_delta_notional_value": total_opt_delta_notional_value,
                "total_opt_market_value": total_opt_market_value,
                "total_tbill_value": total_tbill_value,
                "total_cash_value": total_cash_value,
                "dan": total_opt_delta_notional_value,
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
            }

            return ComplianceResult(
                is_compliant=names_test >= PROSPECTUS_MIN_THRESHOLD,
                details={"rule": "80% Prospectus Policy", "options_in_scope": options_in_scope},
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in prospectus policy check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "80% Prospectus Policy", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def diversification_IRS_check(self, fund: Fund) -> ComplianceResult:
        try:
            vest_eqy_holdings, vest_opt_holdings, vest_treasury_holdings = self._get_holdings(fund)
            total_assets, total_net_assets = self._get_total_assets(fund)
            overlap_df = self._get_overlap(fund)
            expenses = self._get_expenses(fund)

            vest_eqy_holdings = vest_eqy_holdings.copy()
            if total_net_assets:
                vest_eqy_holdings["tna_wgt"] = (
                    vest_eqy_holdings["equity_market_value"] / total_net_assets
                )
            else:
                vest_eqy_holdings["tna_wgt"] = 0.0

            holdings_df = (
                pd.merge(
                    vest_eqy_holdings,
                    vest_opt_holdings,
                    on="equity_ticker",
                    how="outer",
                    suffixes=("", "_option"),
                )
                .fillna(0.0)
                .copy()
            )

            for col in ["equity_market_value", "option_market_value"]:
                if col not in holdings_df.columns:
                    holdings_df[col] = 0.0
                else:
                    holdings_df[col] = pd.to_numeric(holdings_df[col], errors="coerce").fillna(0.0)

            if "equity_ticker" in holdings_df.columns:
                mask_google = holdings_df["equity_ticker"].isin(["GOOG", "GOOGL"])
                if mask_google.any():
                    tmp = holdings_df.copy()
                    tmp["equity_ticker"] = np.where(
                        tmp["equity_ticker"].isin(["GOOG", "GOOGL"]),
                        "GOOGLE",
                        tmp["equity_ticker"],
                    )
                    num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
                    holdings_df = (
                        tmp.groupby("equity_ticker", as_index=False)
                        .agg(
                            {
                                **{
                                    c: "first"
                                    for c in tmp.columns
                                    if c not in num_cols and c != "equity_ticker"
                                },
                                **{c: "sum" for c in num_cols},
                            }
                        )
                        .copy()
                    )

            holdings_df["flex_market_value"] = 0.0

            flex_mask = holdings_df["equity_ticker"].isin(["SPX", "XSP"])
            if "optticker" in holdings_df.columns:
                flex_mask = flex_mask | holdings_df["optticker"].astype(str).str.startswith(("SPX", "XSP"))

            is_closed_end_fund = fund.name in CLOSED_END_FUNDS

            if flex_mask.any() and is_closed_end_fund:
                flex_options = holdings_df.loc[flex_mask].copy()
                qty_col = None
                if "quantity_option" in flex_options.columns:
                    qty_col = "quantity_option"
                elif "quantity" in flex_options.columns:
                    qty_col = "quantity"

                if qty_col:
                    long_flex = flex_options[flex_options[qty_col] > 0]
                else:
                    long_flex = flex_options[flex_options["option_market_value"] > 0]

                total_flex_mv = float(long_flex["option_market_value"].sum())

                if total_flex_mv > 0 and not overlap_df.empty:
                    overlap = overlap_df[["security_ticker", "security_weight"]].drop_duplicates()

                    holdings_df = holdings_df.merge(
                        overlap,
                        left_on="equity_ticker",
                        right_on="security_ticker",
                        how="left",
                    )

                    holdings_df["security_weight"] = (
                        pd.to_numeric(holdings_df["security_weight"], errors="coerce").fillna(0.0)
                    )
                    holdings_df["flex_market_value"] = total_flex_mv * holdings_df["security_weight"]
                    holdings_df = holdings_df.drop(columns=["security_ticker"], errors="ignore")

                    flex_mask_updated = holdings_df["equity_ticker"].isin(["SPX", "XSP"])
                    if "optticker" in holdings_df.columns:
                        flex_mask_updated |= holdings_df["optticker"].astype(str).str.startswith(("SPX", "XSP"))
                    holdings_df = holdings_df.loc[~flex_mask_updated].reset_index(drop=True)

            holdings_df["net_market_value"] = (
                holdings_df["equity_market_value"] + holdings_df["flex_market_value"]
            )
            if total_assets:
                holdings_df["weight"] = holdings_df["net_market_value"] / total_assets
            else:
                holdings_df["weight"] = 0.0

            if total_net_assets:
                holdings_df["tna_wgt"] = holdings_df["equity_market_value"] / total_net_assets
            else:
                holdings_df["tna_wgt"] = 0.0

            holdings_df = holdings_df.sort_values("tna_wgt", ascending=False).reset_index(drop=True)
            largest_holding = holdings_df.iloc[0].to_dict() if len(holdings_df) >= 1 else {}

            holdings_df["cumulative_weight"] = holdings_df["tna_wgt"].cumsum()
            bottom_50_mask = holdings_df["cumulative_weight"] > 0.5
            bottom_50_df = holdings_df[bottom_50_mask].copy()
            top_50_df = holdings_df[~bottom_50_mask].copy()

            qualifying_assets_value = float(holdings_df["net_market_value"].sum())

            if fund.name in {"DOGG"}:
                tmp_df = vest_eqy_holdings.copy()
                tmp_df["net_market_value"] = tmp_df["equity_market_value"]
                if total_net_assets:
                    tmp_df["tna_wgt"] = tmp_df["equity_market_value"] / total_net_assets
                else:
                    tmp_df["tna_wgt"] = 0.0
                condition_IRS_2_a_50 = (
                    (total_net_assets - expenses) / total_net_assets >= IRS_QUALIFYING_ASSETS_MIN
                    if total_net_assets
                    else False
                )
            else:
                if qualifying_assets_value > total_assets and total_net_assets:
                    condition_IRS_2_a_50 = (
                        (total_net_assets - expenses) / total_net_assets >= IRS_QUALIFYING_ASSETS_MIN
                    )
                else:
                    condition_IRS_2_a_50 = (
                        qualifying_assets_value >= IRS_QUALIFYING_ASSETS_MIN * total_net_assets
                        if total_net_assets
                        else False
                    )

            five_pct_gross_assets = total_assets * 0.05
            large_securities = holdings_df[holdings_df["net_market_value"] >= five_pct_gross_assets]
            sum_large_securities_weights = float(large_securities["tna_wgt"].sum())
            condition_IRS_2_a_5_new = sum_large_securities_weights <= 0.5

            if "tna_wgt" not in bottom_50_df.columns:
                bottom_50_df["tna_wgt"] = 0.0
            bottom_50_df["exceeds_5_percent"] = bottom_50_df["tna_wgt"] > IRS_BOTTOM_50_LIMIT
            condition_IRS_2_a_5_original = not bool(bottom_50_df["exceeds_5_percent"].any())

            if "EQY_SH_OUT_million" not in bottom_50_df.columns:
                bottom_50_df["EQY_SH_OUT_million"] = 1.0
            denom = bottom_50_df["EQY_SH_OUT_million"].replace(0, 1)
            if "quantity" not in bottom_50_df.columns:
                bottom_50_df["quantity"] = 0.0
            bottom_50_df["exceeds_10_percent"] = (
                bottom_50_df["quantity"] / denom
            ) > IRS_OWNERSHIP_LIMIT
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
                "bottom_50_exceeds_5_percent": int(
                    bottom_50_df["exceeds_5_percent"].sum()
                )
                if "exceeds_5_percent" in bottom_50_df
                else 0,
                "bottom_50_exceeds_10_percent": int(
                    bottom_50_df["exceeds_10_percent"].sum()
                )
                if "exceeds_10_percent" in bottom_50_df
                else 0,
                "large_securities": large_securities.to_dict("records"),
            }

            is_compliant = all(
                [
                    details["condition_IRS_1"],
                    condition_IRS_2_a_50,
                    condition_IRS_2_a_5_new,
                    condition_IRS_2_a_10,
                ]
            )

            return ComplianceResult(
                is_compliant=is_compliant,
                details=details,
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in IRS diversification check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "IRS Diversification", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def diversification_40act_check(self, fund: Fund) -> ComplianceResult:
        if fund.name in PRIVATE_FUNDS:
            return ComplianceResult(
                is_compliant=True,
                details={
                    "rule": "40 Act Diversification",
                    "skipped": True,
                    "reason": "Private funds are not registered under the 40 Act",
                    "fund_type": "Private Fund",
                },
                calculations={},
            )

        try:
            vest_eqy_holdings, vest_opt_holdings, _ = self._get_holdings(fund)
            total_assets, total_net_assets = self._get_total_assets(fund)
            expenses = self._get_expenses(fund)

            required_opt_cols = [
                "option_notional_value",
                "option_market_value",
                "option_delta_adjusted_notional",
                "option_delta_adjusted_market_value",
            ]

            for col in required_opt_cols:
                if col not in vest_opt_holdings.columns:
                    vest_opt_holdings[col] = 0.0
                else:
                    vest_opt_holdings[col] = pd.to_numeric(
                        vest_opt_holdings[col], errors="coerce"
                    ).fillna(0.0)

            fund_registration = "Diversified" if fund.name in DIVERSIFIED_FUNDS else "Non-diversified"

            if fund.name in {"RDVI", "SDVD", "TDVI", "FTCSH", "FDND", "FGSI"}:
                holdings_df = vest_eqy_holdings.copy()
                holdings_df["net_market_value"] = holdings_df["equity_market_value"]
                net_assets = float(holdings_df["net_market_value"].sum())
            else:
                holdings_df = pd.merge(
                    vest_eqy_holdings,
                    vest_opt_holdings,
                    how="outer",
                    left_on="equity_ticker",
                    right_on="equity_ticker",
                    suffixes=("", "_opt"),
                )
                for col in required_opt_cols:
                    if col not in holdings_df.columns:
                        holdings_df[col] = 0.0
                    else:
                        holdings_df[col] = pd.to_numeric(holdings_df[col], errors="coerce").fillna(0.0)
                if "equity_market_value" not in holdings_df.columns:
                    holdings_df["equity_market_value"] = 0.0
                holdings_df["net_market_value"] = (
                    holdings_df["equity_market_value"] + holdings_df["option_market_value"]
                )
                net_assets = float(holdings_df["net_market_value"].sum())

            non_qualifying_assets = 0.0
            condition_1_met = (
                (total_assets + non_qualifying_assets) / total_assets >= ACT_40_QUALIFYING_ASSETS_MIN
                if total_assets
                else False
            )

            sorted_df = holdings_df.sort_values(by="net_market_value", ascending=False)
            cumulative_sum = sorted_df["net_market_value"].cumsum()
            threshold = 0.25 * total_assets if total_assets > 0 else 0.0
            within_25_percent = sorted_df[cumulative_sum <= threshold].copy()
            if within_25_percent.empty and not sorted_df.empty:
                within_25_percent = sorted_df.head(1).copy()

            excluded_securities = (
                within_25_percent["equity_ticker"].tolist()
                if "equity_ticker" in within_25_percent.columns
                else []
            )

            remaining_securities = (
                sorted_df[~sorted_df["equity_ticker"].isin(excluded_securities)].copy()
                if excluded_securities
                else sorted_df.copy()
            )

            five_pct_threshold = ACT_40_ISSUER_LIMIT * total_assets
            issuer_limited = remaining_securities[
                remaining_securities["net_market_value"] > five_pct_threshold
            ].copy()
            issuer_limited_sum = float(issuer_limited["net_market_value"].sum()) if not issuer_limited.empty else 0.0
            condition_2a_met = issuer_limited_sum == 0.0

            occ_weight_mkt_val = abs(float(vest_opt_holdings["option_market_value"].sum()))
            condition_2a_occ_met = (
                occ_weight_mkt_val <= 0.05 * total_net_assets if total_net_assets else False
            )

            if "EQY_SH_OUT_million" not in remaining_securities.columns:
                remaining_securities["EQY_SH_OUT_million"] = 1.0
            remaining_securities["EQY_SH_OUT_million"] = (
                pd.to_numeric(remaining_securities["EQY_SH_OUT_million"], errors="coerce").replace(0, 1)
            )
            if "quantity" not in remaining_securities.columns:
                remaining_securities["quantity"] = 0.0

            remaining_securities["vest_ownership_of_float"] = (
                remaining_securities["quantity"] / remaining_securities["EQY_SH_OUT_million"]
            ).fillna(0.0)
            issuer_compliance_2b = remaining_securities["vest_ownership_of_float"] <= ACT_40_OWNERSHIP_LIMIT
            condition_2b_met = bool(issuer_compliance_2b.all())

            cumulative_weight_excluded = (
                within_25_percent["net_market_value"].sum() / total_net_assets
                if (not within_25_percent.empty and total_net_assets)
                else 0.0
            )
            cumulative_weight_remaining = 1 - cumulative_weight_excluded

            if not remaining_securities.empty and total_assets:
                remaining_securities["vest_weight"] = (
                    remaining_securities["net_market_value"] / total_assets
                )
            else:
                remaining_securities["vest_weight"] = 0.0

            columns_for_output = [
                col
                for col in [
                    "equity_ticker",
                    "quantity",
                    "net_market_value",
                    "vest_weight",
                    "vest_ownership_of_float",
                ]
                if col in remaining_securities.columns
            ]
            details_df = remaining_securities[columns_for_output].copy() if columns_for_output else pd.DataFrame()
            remaining_stocks_details = details_df.to_dict("records")
            max_ownership_float = (
                float(remaining_securities["vest_ownership_of_float"].max())
                if "vest_ownership_of_float" in remaining_securities.columns
                else 0.0
            )

            calculations = {
                "fund_registration": fund_registration,
                "total_assets": total_assets,
                "net_assets": total_net_assets,
                "expenses": expenses,
                "condition_1_met": condition_1_met,
                "non_qualifying_assets_1_wgt": non_qualifying_assets,
                "issuer_limited_sum": issuer_limited_sum,
                "issuer_limited_securities": issuer_limited.to_dict("records")
                if not issuer_limited.empty
                else [],
                "condition_2a_met": condition_2a_met,
                "condition_2b_met": condition_2b_met,
                "issuer_compliance_2b": issuer_compliance_2b.to_dict()
                if isinstance(issuer_compliance_2b, pd.Series)
                else {"all": issuer_compliance_2b},
                "excluded_securities": excluded_securities,
                "cumulative_weight_excluded": cumulative_weight_excluded,
                "cumulative_weight_remaining": cumulative_weight_remaining,
                "remaining_stocks_details": remaining_stocks_details,
                "max_ownership_float": max_ownership_float,
                "condition_2a_occ_met": condition_2a_occ_met,
                "occ_market_value": occ_weight_mkt_val,
                "occ_weight": occ_weight_mkt_val / total_assets if total_assets else 0.0,
            }

            diversification_failure_duration = None
            if (
                self.session is not None
                and self.base_cls is not None
                and hasattr(self.base_cls, "classes")
                and hasattr(self.base_cls.classes, "compliance_daily_results")
            ):
                try:
                    t_minus_one_date = pd.to_datetime(self.date) - pd.DateOffset(days=1)
                    result_yesterday = (
                        self.session.query(
                            self.base_cls.classes.compliance_daily_results.analysis_type,
                            self.base_cls.classes.compliance_daily_results.diversification_failure_duration_40act,
                        )
                        .filter(
                            self.base_cls.classes.compliance_daily_results.etf == fund.name,
                            self.base_cls.classes.compliance_daily_results.date == t_minus_one_date,
                        )
                        .first()
                    )
                    if result_yesterday:
                        diversification_failure_duration = (
                            result_yesterday.diversification_failure_duration_40act
                        )
                except Exception as exc:  # pragma: no cover - optional path
                    logger.debug("Unable to fetch prior diversification history: %s", exc)

            if diversification_failure_duration is None:
                if all([condition_1_met, condition_2a_met, condition_2b_met]):
                    diversification_failure_duration = 0
                else:
                    diversification_failure_duration = 1

            details = {
                "rule": "40 Act Diversification",
                "condition_40act_1": condition_1_met,
                "condition_40act_2a": condition_2a_met,
                "condition_40act_2b": condition_2b_met,
                "condition_40act_2a_occ": condition_2a_occ_met,
                "diversified_since_inception_40act": condition_1_met and condition_2a_met and condition_2b_met,
                "diversification_failure_duration_40act": diversification_failure_duration,
            }

            is_compliant = condition_1_met and condition_2a_met and condition_2b_met and condition_2a_occ_met

            return ComplianceResult(
                is_compliant=is_compliant,
                details=details,
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in 40 Act diversification check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "40 Act Diversification", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def diversification_IRC_check(self, fund: Fund) -> ComplianceResult:
        try:
            vest_eqy_holdings, vest_opt_holdings, _ = self._get_holdings(fund)
            total_assets, _ = self._get_total_assets(fund)

            if vest_eqy_holdings.empty:
                raise ValueError("Equity holdings missing")
            if total_assets == 0:
                raise ValueError("Total assets missing")

            for df in [vest_eqy_holdings, vest_opt_holdings]:
                if "equity_ticker" not in df.columns and "ticker" in df.columns:
                    df["equity_ticker"] = df["ticker"]

            holdings_df = pd.merge(
                vest_eqy_holdings,
                vest_opt_holdings,
                how="left",
                left_on="equity_ticker",
                right_on="equity_ticker",
                suffixes=("", "_option"),
            ).fillna(0.0)

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

            top_holdings = (
                sorted_holdings[["equity_ticker", "net_market_value"]]
                .head(4)
                .to_dict("records")
            )

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
                "top_holdings": top_holdings,
            }

            is_compliant = all(details.values()) if details else True
            is_compliant = all(
                [
                    details["condition_IRC_55"],
                    details["condition_IRC_70"],
                    details["condition_IRC_80"],
                    details["condition_IRC_90"],
                ]
            )

            return ComplianceResult(
                is_compliant=is_compliant,
                details=details,
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in IRC diversification check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "IRC Diversification", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def real_estate_check(self, fund: Fund) -> ComplianceResult:
        try:
            equity_df, _, _ = self._get_holdings(fund)
            if equity_df.empty:
                raise ValueError("Equity holdings missing")

            real_estate_mask = equity_df["GICS_SECTOR_NAME"].str.contains(
                "Real Estate", case=False, na=False
            )
            real_estate_exposure = float(
                equity_df.loc[real_estate_mask, "equity_market_value"].sum()
            )
            total_exposure = float(equity_df["equity_market_value"].sum())
            real_estate_percentage = (
                real_estate_exposure / total_exposure if total_exposure > 0 else 0.0
            )

            is_compliant = real_estate_percentage == 0.0

            calculations = {
                "real_estate_exposure": real_estate_exposure,
                "real_estate_percentage": real_estate_percentage,
                "total_exposure": total_exposure,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={"rule": "Real Estate Exposure", "real_estate_check_compliant": is_compliant},
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in real estate check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "Real Estate Exposure", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def commodities_check(self, fund: Fund) -> ComplianceResult:
        try:
            equity_df, _, _ = self._get_holdings(fund)
            if equity_df.empty:
                raise ValueError("Equity holdings missing")

            commodities_mask = equity_df["GICS_SECTOR_NAME"].str.contains(
                "Commodities", case=False, na=False
            )
            commodities_exposure = float(
                equity_df.loc[commodities_mask, "equity_market_value"].sum()
            )
            total_exposure = float(equity_df["equity_market_value"].sum())
            commodities_percentage = (
                commodities_exposure / total_exposure if total_exposure > 0 else 0.0
            )

            is_compliant = commodities_exposure == 0.0

            calculations = {
                "commodities_exposure": commodities_exposure,
                "total_exposure": total_exposure,
                "commodities_percentage": commodities_percentage,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={"rule": "Commodities Exposure", "commodities_check_compliant": is_compliant},
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in commodities check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "Commodities Exposure", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def twelve_d1a_other_inv_cos(self, fund: Fund) -> ComplianceResult:
        try:
            equity_df, _, _ = self._get_holdings(fund)
            total_assets, _ = self._get_total_assets(fund)

            if total_assets == 0 or equity_df.empty:
                raise ValueError("Equity holdings or total assets missing")

            inv_co_mask = ~equity_df["REGULATORY_STRUCTURE"].isnull()
            investment_companies = equity_df[inv_co_mask].copy()

            if investment_companies.empty:
                calculations = {
                    "total_assets": total_assets,
                    "investment_companies": [],
                    "ownership_pct_max": None,
                    "equity_market_value_sum": 0.0,
                    "equity_market_value_max": 0.0,
                    "ownership_pct_values": [],
                    "test_2_threshold": RULE_12D1A_SINGLE_ASSETS_LIMIT * total_assets,
                    "test_3_threshold": RULE_12D1A_TOTAL_ASSETS_LIMIT * total_assets,
                }
                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "12d1-1 Other Investment Companies",
                        "twelve_d1a_other_inv_cos_compliant": True,
                        "test_1_pass": True,
                        "test_2_pass": True,
                        "test_3_pass": True,
                    },
                    calculations=calculations,
                )

            if "EQY_SH_OUT_million" not in investment_companies.columns:
                investment_companies["EQY_SH_OUT_million"] = 0.0
            if "quantity" not in investment_companies.columns:
                investment_companies["quantity"] = 0.0

            with np.errstate(divide="ignore", invalid="ignore"):
                investment_companies["ownership_pct"] = np.divide(
                    investment_companies["quantity"],
                    investment_companies["EQY_SH_OUT_million"].replace(0, np.nan),
                ).fillna(0.0)

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
                "test_2_threshold": RULE_12D1A_SINGLE_ASSETS_LIMIT * total_assets,
                "test_3_threshold": RULE_12D1A_TOTAL_ASSETS_LIMIT * total_assets,
            }

            is_compliant = all([test_1_pass, test_2_pass, test_3_pass])

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "12d1-1 Other Investment Companies",
                    "twelve_d1a_other_inv_cos_compliant": is_compliant,
                    "test_1_pass": test_1_pass,
                    "test_2_pass": test_2_pass,
                    "test_3_pass": test_3_pass,
                },
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in 12d1-1 check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "12d1-1 Other Investment Companies", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def twelve_d2_insurance_cos(self, fund: Fund) -> ComplianceResult:
        try:
            vest_eqy_holdings, _, _ = self._get_holdings(fund)
            total_assets, _ = self._get_total_assets(fund)

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
                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "12d-2 Insurance Companies",
                        "twelve_d2_insurance_cos_compliant": True,
                        "test_pass": True,
                    },
                    calculations=calculations,
                )

            if "EQY_SH_OUT_million" not in insurance_holdings.columns:
                insurance_holdings["EQY_SH_OUT_million"] = 0.0
            if "quantity" not in insurance_holdings.columns:
                insurance_holdings["quantity"] = 0.0

            with np.errstate(divide="ignore", invalid="ignore"):
                insurance_holdings["ownership_pct"] = np.divide(
                    insurance_holdings["quantity"],
                    insurance_holdings["EQY_SH_OUT_million"].replace(0, np.nan),
                ).fillna(0.0)

            max_ownership_pct = float(insurance_holdings["ownership_pct"].max())
            test_pass = max_ownership_pct <= RULE_12D2_INSURANCE_LIMIT

            calculations = {
                "total_assets": total_assets,
                "insurance_holdings": insurance_holdings.to_dict("records"),
                "insurance_count": len(insurance_holdings),
                "max_ownership_pct": max_ownership_pct,
                "threshold": RULE_12D2_INSURANCE_LIMIT,
            }

            return ComplianceResult(
                is_compliant=test_pass,
                details={
                    "rule": "12d-2 Insurance Companies",
                    "twelve_d2_insurance_cos_compliant": test_pass,
                    "test_pass": test_pass,
                },
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in 12d-2 check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "12d-2 Insurance Companies", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def twelve_d3_sec_biz(self, fund: Fund) -> ComplianceResult:
        try:
            vest_eqy_holdings, vest_opt_holdings, _ = self._get_holdings(fund)
            total_assets, _ = self._get_total_assets(fund)

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
                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "12d-3 Securities Businesses",
                        "twelve_d3_sec_biz_compliant": True,
                        "rule_1_pass": True,
                        "rule_2_pass": True,
                        "rule_3_pass": True,
                        "rule_3_pass_occ": True,
                    },
                    calculations=calculations,
                )

            sec_related_businesses["EQY_SH_OUT_million"] = sec_related_businesses[
                "EQY_SH_OUT_million"
            ].replace(0, np.nan)
            sec_related_businesses["quantity"] = sec_related_businesses.get("quantity", 0.0)
            sec_related_businesses["ownership_pct"] = (
                sec_related_businesses["quantity"] / sec_related_businesses["EQY_SH_OUT_million"]
            ).fillna(0.0)

            rule_1_pass = (sec_related_businesses["ownership_pct"] <= RULE_12D3_EQUITY_LIMIT).all()

            debt_mask = vest_eqy_holdings["SECURITY_TYP"].eq("Debt") if "SECURITY_TYP" in vest_eqy_holdings.columns else pd.Series(False, index=vest_eqy_holdings.index)
            debt_related_businesses = vest_eqy_holdings[debt_mask]
            if debt_related_businesses.empty:
                rule_2_pass = True
            else:
                debt_threshold = RULE_12D3_DEBT_LIMIT * total_assets
                rule_2_pass = (
                    debt_related_businesses["equity_market_value"] <= debt_threshold
                ).all()

            combined_holdings = pd.merge(
                sec_related_businesses,
                vest_opt_holdings,
                how="left",
                left_on="equity_ticker",
                right_on="equity_ticker",
                suffixes=("", "_option"),
            )
            combined_holdings["option_market_value"] = pd.to_numeric(
                combined_holdings.get("option_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            combined_holdings["equity_market_value"] = pd.to_numeric(
                combined_holdings.get("equity_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            combined_holdings["net_market_value"] = (
                combined_holdings["equity_market_value"] + combined_holdings["option_market_value"]
            )
            combined_holdings["vest_weight"] = (
                combined_holdings["net_market_value"] / total_assets if total_assets else 0.0
            )
            combined_holdings = combined_holdings.loc[:, ~combined_holdings.columns.duplicated()]
            max_weight = float(combined_holdings["vest_weight"].max()) if not combined_holdings.empty else 0.0
            rule_3_pass = max_weight <= RULE_12D3_ASSET_LIMIT

            related_tickers = sec_related_businesses["equity_ticker"].unique()
            ticker_mask = vest_opt_holdings["equity_ticker"].isin(related_tickers)
            occ_exposure = vest_opt_holdings[ticker_mask].copy()
            occ_weight_mkt_val = float(occ_exposure["option_market_value"].sum())
            occ_weight = occ_weight_mkt_val / total_assets if total_assets else 0.0
            rule_3_pass_occ = occ_weight <= 0.05

            is_compliant = all([rule_1_pass, rule_2_pass, rule_3_pass, rule_3_pass_occ])

            calculations = {
                "total_assets": total_assets,
                "sec_related_businesses": sec_related_businesses.to_dict("records"),
                "combined_holdings": combined_holdings.to_dict("records"),
                "occ_weight_mkt_val": occ_weight_mkt_val,
                "occ_weight": occ_weight,
                "max_ownership_pct": float(sec_related_businesses["ownership_pct"].max()),
                "max_weight": max_weight,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "12d-3 Securities Businesses",
                    "twelve_d3_sec_biz_compliant": is_compliant,
                    "rule_1_pass": rule_1_pass,
                    "rule_2_pass": rule_2_pass,
                    "rule_3_pass": rule_3_pass,
                    "rule_3_pass_occ": rule_3_pass_occ,
                },
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in 12d-3 check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "12d-3 Securities Businesses", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def max_15pct_illiquid_sai(self, fund: Fund) -> ComplianceResult:
        try:
            vest_eqy_holdings, vest_opt_holdings, _ = self._get_holdings(fund)
            total_assets, _ = self._get_total_assets(fund)

            if total_assets == 0 or vest_eqy_holdings.empty:
                raise ValueError("Equity holdings or total assets missing")

            if "is_illiquid" not in vest_eqy_holdings.columns:
                vest_eqy_holdings["is_illiquid"] = False
            if "is_illiquid" not in vest_opt_holdings.columns:
                vest_opt_holdings["is_illiquid"] = False

            illiquid_mask = vest_eqy_holdings["is_illiquid"] == True
            illiquid_eqy_value = float(
                vest_eqy_holdings.loc[illiquid_mask, "equity_market_value"].sum()
            )

            required_cols = {"price", "quantity"}
            if required_cols.issubset(vest_opt_holdings.columns):
                opt_illiquid_mask = vest_opt_holdings["is_illiquid"] == True
                illiquid_opt_value = float(
                    (
                        vest_opt_holdings.loc[opt_illiquid_mask, "price"]
                        * vest_opt_holdings.loc[opt_illiquid_mask, "quantity"]
                        * 100
                    ).sum()
                )
            else:
                illiquid_opt_value = 0.0

            total_illiquid_value = illiquid_eqy_value + illiquid_opt_value
            illiquid_percentage = total_illiquid_value / total_assets
            equity_value = float(vest_eqy_holdings["equity_market_value"].sum())
            equity_percentage = equity_value / total_assets

            is_compliant = (
                illiquid_percentage <= ILLIQUID_MAX_THRESHOLD
                and equity_percentage >= EQUITY_MIN_THRESHOLD
            )

            calculations = {
                "total_assets": total_assets,
                "total_illiquid_value": total_illiquid_value,
                "illiquid_percentage": illiquid_percentage,
                "equity_holdings_percentage": equity_percentage,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "SAI Illiquidity",
                    "max_15pct_illiquid_sai": illiquid_percentage <= ILLIQUID_MAX_THRESHOLD,
                    "equity_holdings_85pct_compliant": equity_percentage >= EQUITY_MIN_THRESHOLD,
                },
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in SAI illiquidity check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "SAI Illiquidity", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def gics_compliance(self, fund: Fund) -> ComplianceResult:
        try:
            equity_df = getattr(fund.data.current, "equity", pd.DataFrame())

            if equity_df.empty:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "no_equity_data",
                    },
                    calculations={},
                    error="Equity holdings are unavailable for analysis",
                )

            weight_series, weight_source = self._determine_weight_series(equity_df)
            if weight_series is None or weight_series.sum() == 0:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "missing_weights",
                    },
                    calculations={},
                    error="Unable to determine constituent weights",
                )

            sector_column = self._resolve_gics_sector_column(equity_df)
            if sector_column is None:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "missing_sector_data",
                    },
                    calculations={},
                    error="GICS sector information is not available",
                )

            normalized_weights = weight_series / weight_series.sum()
            exposures = (
                equity_df.assign(_weight=normalized_weights)
                .groupby(sector_column)["_weight"].sum()
                .sort_values(ascending=False)
            )

            exposures = exposures[exposures.index.notna()]
            sector_exposure = exposures.round(6).to_dict()

            gics_limits = {}
            if isinstance(fund.config, dict):
                gics_limits = fund.config.get("gics_limits", {}) or {}

            default_limit = gics_limits.get("_default", self._DEFAULT_GICS_LIMIT)

            breaches = {
                sector: weight
                for sector, weight in sector_exposure.items()
                if weight > gics_limits.get(sector, default_limit) + 1e-9
            }

            return ComplianceResult(
                is_compliant=not breaches,
                details={
                    "rule": "GICS Concentration",
                    "weight_source": weight_source,
                    "limits": gics_limits or {"_default": default_limit},
                    "breaches": breaches,
                },
                calculations={"sector_exposure": sector_exposure},
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in GICS compliance check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "GICS Concentration", "status": "error"},
                calculations={},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_holdings(self, fund: Fund) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        equity = ensure_equity_schema(fund.equity_holdings)
        options = ensure_option_schema(fund.options_holdings)
        treasury = ensure_treasury_schema(fund.treasury_holdings)
        return equity, options, treasury

    def _get_total_assets(self, fund: Fund) -> Tuple[float, float]:
        return float(fund.total_assets or 0.0), float(fund.total_net_assets or 0.0)

    def _get_cash_value(self, fund: Fund) -> float:
        return float(fund.cash_value or 0.0)

    def _get_expenses(self, fund: Fund) -> float:
        try:
            return float(fund.expenses or 0.0)
        except Exception:
            return 0.0

    def _get_overlap(self, fund: Fund) -> pd.DataFrame:
        overlap = getattr(fund.data, "index", pd.DataFrame())
        if not isinstance(overlap, pd.DataFrame) or overlap.empty:
            return pd.DataFrame(columns=["security_ticker", "security_weight"])

        overlap = overlap.copy()
        ticker_column = self._resolve_column(
            overlap,
            ["security_ticker", "ticker", "symbol", "equity_ticker"],
        )
        if ticker_column and ticker_column != "security_ticker":
            overlap["security_ticker"] = overlap[ticker_column]

        weight_column = self._resolve_column(
            overlap,
            ["security_weight", "weight", "index_weight", "wgt"],
        )
        if weight_column and weight_column != "security_weight":
            overlap["security_weight"] = overlap[weight_column]

        overlap["security_weight"] = pd.to_numeric(
            overlap["security_weight"], errors="coerce"
        ).fillna(0.0)
        return overlap[["security_ticker", "security_weight"]]

    def _determine_weight_series(self, equity_df: pd.DataFrame):
        weight_columns = [
            "start_wgt",
            "weight",
            "fund_weight",
            "portfolio_weight",
            "weight_fund",
        ]

        for column in weight_columns:
            if column in equity_df.columns:
                series = pd.to_numeric(equity_df[column], errors="coerce").fillna(0.0)
                return series, column

        if {"price", "quantity"}.issubset(equity_df.columns):
            market_values = (
                pd.to_numeric(equity_df["price"], errors="coerce").fillna(0.0)
                * pd.to_numeric(equity_df["quantity"], errors="coerce").fillna(0.0)
            )
            return market_values, "price*quantity"

        if "market_value" in equity_df.columns:
            market_values = pd.to_numeric(equity_df["market_value"], errors="coerce").fillna(0.0)
            return market_values, "market_value"

        return None, None

    def _resolve_gics_sector_column(self, equity_df: pd.DataFrame) -> Optional[str]:
        for column in [
            "GICS_SECTOR_NAME",
            "gics_sector",
            "sector",
            "sector_name",
        ]:
            if column in equity_df.columns:
                return column
        return None

    def _resolve_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for column in candidates:
            if column in df.columns:
                return column
        return None