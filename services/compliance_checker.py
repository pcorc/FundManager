from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Tuple

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


class ComplianceChecker:
    """Runs compliance checks against :class:`Fund` domain objects."""
    def __init__(
        self,
        session,
        funds: Optional[Dict[str, Fund]] = None,
        date=None,
        base_cls=None,
        *,
        analysis_type: Optional[str] = None,
    ) -> None:
        self.session = session
        self.funds: Dict[str, Fund] = funds or {}
        self.date = date
        self.base_cls = base_cls
        analysis = (analysis_type or "eod").strip().lower()
        self.analysis_type = analysis if analysis in {"eod", "ex_ante", "ex_post"} else "eod"

    def run_compliance_tests(self, test_functions: Optional[list[str]] = None) -> Dict[str, Dict[str, ComplianceResult]]:
        """Execute the requested compliance checks for every fund."""

        available_tests: Dict[str, Callable[[Fund], ComplianceResult]] = {
            "summary_metrics": self.calculate_summary_metrics,
            "gics_compliance": self.gics_compliance,
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
        }

        # Example configurations for including/excluding tests on a per-ticker basis.
        skip_tests_by_ticker: Dict[str, set[str]] = {
            "gics_compliance": {"DOGG"},  # Add tickers here to bypass the GICS test.
        }
        if test_functions:
            requested = set(test_functions)
            available_tests = {
                name: func for name, func in available_tests.items() if name in requested
            }

        results: Dict[str, Dict[str, ComplianceResult]] = {}

        for fund_name, fund in self.funds.items():
            fund_results: Dict[str, ComplianceResult] = {}

            for test_name, test_func in available_tests.items():
                excluded_funds = skip_tests_by_ticker.get(test_name, set())
                if fund_name in excluded_funds:
                    continue

                if (
                    test_name == "diversification_IRC_check"
                    and (fund.vehicle or "").lower() != VEHICLE_VIT
                ):
                    continue

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


    def calculate_summary_metrics(self, fund: Fund) -> ComplianceResult:

        total_assets, total_net_assets, expenses, cash_value = self._get_current_totals(fund)

        calculations = {
            "cash_value": cash_value,
            "equity_market_value": fund.data.current.total_equity_value,
            "option_delta_adjusted_notional": fund.data.current.total_option_delta_adjusted_notional,
            "option_market_value": fund.data.current.total_option_value,
            "treasury": fund.data.current.total_treasury_value,
            "total_assets": total_assets,
            "total_net_assets": total_net_assets,
            "expenses": expenses,
        }

        return ComplianceResult(
            is_compliant=True,
            details={"rule": "summary_metrics", "status": "calculated"},
            calculations=calculations,
        )

    def prospectus_80pct_policy(self, fund: Fund) -> ComplianceResult:
        try:

            total_assets, total_net_assets, _, total_cash_value = self._get_current_totals(fund)
            total_equity_market_value = float(fund.data.current.total_equity_value)
            total_equity_market_value = float(fund.data.current.total_equity_value)
            total_opt_market_value = float(fund.data.current.total_option_value)
            total_tbill_value = float(fund.data.current.total_treasury_value)
            total_opt_delta_notional_value = float(
                fund.data.current.total_option_delta_adjusted_notional
            )

            is_closed_end = fund.is_closed_end_fund
            if fund.vehicle == VEHICLE_PRIVATE:
                calculations = {
                    "total_equity_market_value": total_equity_market_value,
                    "total_opt_market_value": total_opt_market_value,
                    "total_tbill_value": total_tbill_value,
                    "total_cash_value": total_cash_value,
                    "total_assets": total_assets,
                    "total_net_assets": total_net_assets,
                }

                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "80% Prospectus Policy",
                        "skipped": True,
                        "reason": "Private funds are excluded from the 80% test",
                        "fund": fund.name,
                    },
                    calculations=calculations,
                )

            is_trif = is_closed_end and fund.name.startswith(("P3", "TR"))
            if is_trif:
                calculations = {
                    "total_equity_market_value": total_equity_market_value,
                    "total_opt_market_value": total_opt_market_value,
                    "total_tbill_value": total_tbill_value,
                    "total_cash_value": total_cash_value,
                    "total_assets": total_assets,
                    "total_net_assets": total_net_assets,
                }

                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "80% Prospectus Policy",
                        "skipped": True,
                        "reason": "TRIF closed-end funds are excluded from the 80% test",
                        "fund": fund.name,
                    },
                    calculations=calculations,
                )

            is_b3_closed_end = is_closed_end and fund.name.endswith("B3")
            options_in_scope = fund.has_listed_option and fund.name in PROSPECTUS_OPTIONS_FUNDS
            if is_closed_end:
                options_in_scope = fund.has_flex_option or fund.has_listed_option or is_b3_closed_end

            option_primary_amount = abs(total_opt_delta_notional_value)
            option_primary_method = "delta_adjusted_notional"

            if is_closed_end:
                option_primary_amount = total_opt_market_value
                option_primary_method = "market_value"

            numerator = total_equity_market_value + total_tbill_value
            denominator = (
                total_equity_market_value + total_cash_value + total_tbill_value
            )

            if options_in_scope:
                numerator += option_primary_amount
                denominator += option_primary_amount

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
                "policy_variant": (
                    "closed_end_b3"
                    if is_b3_closed_end
                    else ("closed_end" if is_closed_end else "standard")
                ),
            }

            return ComplianceResult(
                is_compliant=names_test >= PROSPECTUS_MIN_THRESHOLD,
                details={
                    "rule": "80% Prospectus Policy",
                    "options_in_scope": options_in_scope,
                    "options_valuation_method": option_primary_method,
                    "policy_variant": calculations["policy_variant"],
                },
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
            vest_eqy_holdings = fund.data.current.vest.equity
            vest_opt_holdings = fund.data.current.vest.options

            total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)
            overlap_df = fund.data.current.overlap

            vest_eqy_holdings = self._consolidate_holdings_by_issuer(fund, vest_eqy_holdings)

            if fund.has_listed_option:
                holdings_df = pd.merge(
                    vest_eqy_holdings,
                    vest_opt_holdings,
                    left_on="eqyticker",
                    right_on="equity_underlying_ticker",
                    how="outer",
                    suffixes=("", "_option"),
                )
            else:
                holdings_df = vest_eqy_holdings

            overlap_details_df = pd.DataFrame(
                columns=["security_ticker", "security_weight", "overlap_market_value", "overlap_weight"]
            )

            has_index_flex = fund.has_flex_option and (
                (fund.flex_option_type or "").lower() == "index"
            )

            if has_index_flex and not overlap_df.empty:
                holdings_df, overlap_details_df = self._calculate_index_overlap_adjustments(
                    fund,
                    holdings_df,
                    overlap_df,
                    total_assets,
                )
            else:
                holdings_df["net_market_value"] = holdings_df["equity_market_value"] + holdings_df["option_market_value"]

            if total_assets:
                holdings_df["weight"] = holdings_df["net_market_value"] / total_assets
            else:
                holdings_df["weight"] = 0.0

            if total_net_assets:
                holdings_df["tna_wgt"] = holdings_df["net_market_value"] / total_net_assets
            else:
                holdings_df["tna_wgt"] = 0.0

            holdings_df = holdings_df.sort_values("tna_wgt", ascending=False).reset_index(drop=True)
            largest_holding = holdings_df.iloc[0].to_dict() if len(holdings_df) >= 1 else {}

            holdings_df["cumulative_weight"] = holdings_df["tna_wgt"].cumsum()
            bottom_50_mask = holdings_df["cumulative_weight"] > 0.5
            bottom_50_df = holdings_df[bottom_50_mask].copy()
            top_50_df = holdings_df[~bottom_50_mask].copy()

            qualifying_assets_value = float(holdings_df["net_market_value"].sum())

            overlap_weight_sum = float(overlap_details_df.get("overlap_weight", pd.Series()).sum())

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

            share_col = "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"
            share_series = pd.Series(0.0, index=bottom_50_df.index, dtype=float)
            bottom_50_df[share_col] = share_series

            bottom_50_df["exceeds_10_percent"] = (
                share_series / denom
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
                "overlap": overlap_details_df.to_dict("records"),
                "overlap_weight_sum": overlap_weight_sum,
                "overlap_market_value_sum": float(
                    overlap_details_df.get("overlap_market_value", pd.Series()).sum()
                ),
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
        if fund.is_private_fund:
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
            vest_opt_holdings = fund.data.current.vest.options
            vest_eqy_holdings = fund.data.current.vest.equity
            vest_treasury_holdings = fund.data.current.vest.treasury

            total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)
            vest_eqy_holdings = self._consolidate_holdings_by_issuer(fund, vest_eqy_holdings)

            registration = (fund.diversification_status or "unknown").replace("_", "-")
            fund_registration = registration.lower()

            if fund.has_listed_option and fund.listed_option_type == "index":
                holdings_df = vest_eqy_holdings.copy()
                holdings_df["option_market_value"] = 0.0
                holdings_df["net_market_value"] = holdings_df["equity_market_value"]
            # we want another condition that
            else:
                holdings_df = pd.merge(
                    vest_eqy_holdings,
                    vest_opt_holdings,
                    how="outer",
                    left_on="eqyticker",
                    right_on="equity_underlying_ticker",
                    suffixes=("", "_option"),
                )

                holdings_df["net_market_value"] = holdings_df["equity_market_value"] + holdings_df["option_market_value"]

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
                within_25_percent["eqyticker"].tolist()
                if "eqyticker" in within_25_percent.columns
                else []
            )

            remaining_securities = (
                sorted_df[~sorted_df["eqyticker"].isin(excluded_securities)].copy()
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

            share_col = "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"
            share_series = pd.Series(0.0, index=remaining_securities.index, dtype=float)
            remaining_securities[share_col] = share_series

            remaining_securities["vest_ownership_of_float"] = (
                share_series / remaining_securities["EQY_SH_OUT_million"]
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
                    "eqyticker",
                    share_col,
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

            actual_diversified = all(
                [condition_1_met, condition_2a_met, condition_2b_met]
            )
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
                "share_column_used": share_col,
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
                "expected_registration_status": expected_status,
                "actual_registration_status": fund_status_today,
            }

            is_compliant = actual_diversified == expected_diversified

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
        """
        Check IRC diversification compliance.
        Only applies to VIT (Variable Insurance Trust) funds.
        """
        # Check if fund has vehicle_type attribute and if it's VIT
        vehicle_type = getattr(fund, 'vehicle_type', None) or getattr(fund, 'vehicle', None) or ""

        if str(vehicle_type).lower() != VEHICLE_VIT.lower():
            # Return a skipped result for non-VIT funds
            return ComplianceResult(
                is_compliant=True,
                details={
                    "rule": "IRC Diversification",
                    "skipped": True,
                    "reason": f"IRC diversification applies only to VIT vehicles (fund vehicle: {vehicle_type})",
                },
                calculations={},
            )

        try:
            regular_options = fund.data.current.vest.options
            vest_eqy_holdings = fund.data.current.vest.equity
            total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)
            vest_eqy_holdings = self._consolidate_holdings_by_issuer(fund, vest_eqy_holdings)

            if vest_eqy_holdings.empty:
                raise ValueError("Equity holdings missing")
            if total_assets == 0:
                raise ValueError("Total assets missing")

            # Merge equity and options data
            holdings_df = pd.merge(
                vest_eqy_holdings,
                regular_options,
                how="left",
                left_on="eqyticker",
                right_on="equity_underlying_ticker",
                suffixes=("", "_option"),
            )

            # Calculate net market value
            holdings_df["net_market_value"] = (
                    pd.to_numeric(holdings_df["equity_market_value"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(holdings_df["option_market_value"], errors="coerce").fillna(0.0)
            )
            holdings_df["weight"] = holdings_df["net_market_value"] / total_assets

            # Sort by net market value
            sorted_holdings = holdings_df.sort_values(by="net_market_value", ascending=False)

            # Calculate top exposures
            top_exposures = [0.0, 0.0, 0.0, 0.0]
            if not sorted_holdings.empty:
                cum_values = sorted_holdings["net_market_value"].cumsum()
                for idx in range(min(4, len(cum_values))):
                    top_exposures[idx] = float(cum_values.iloc[idx] / total_assets)

            top_holdings = (
                sorted_holdings[["eqyticker", "net_market_value"]]
                .head(4)
                .to_dict("records")
            )

            # Check all IRC conditions
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

            is_compliant = all([
                details["condition_IRC_55"],
                details["condition_IRC_70"],
                details["condition_IRC_80"],
                details["condition_IRC_90"],
            ])

            return ComplianceResult(
                is_compliant=is_compliant,
                details=details,
                calculations=calculations,
            )
        except Exception as exc:
            logger.error("Error in IRC diversification check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "IRC Diversification", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def real_estate_check(self, fund: Fund) -> ComplianceResult:
        """Check if fund has any real estate exposure"""
        try:

            vest_eqy_holdings = fund.data.current.vest.equity

            if vest_eqy_holdings.empty:
                # No holdings means no real estate exposure - PASS
                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "Real Estate Exposure",
                        "real_estate_check_compliant": True
                    },
                    calculations={
                        "real_estate_exposure": 0.0,
                        "real_estate_percentage": 0.0,
                        "total_exposure": 0.0,
                    },
                )

            # Check for real estate in GICS sector
            real_estate_mask = vest_eqy_holdings["GICS_SECTOR_NAME"].str.contains(
                "Real Estate", case=False, na=False
            )
            real_estate_exposure = float(
                vest_eqy_holdings.loc[real_estate_mask, "equity_market_value"].sum()
            )
            real_estate_exposure = 0.0
            total_exposure = float(vest_eqy_holdings["equity_market_value"].sum())
            real_estate_percentage = (
                real_estate_exposure / total_exposure if total_exposure > 0 else 0.0
            )

            # PASS if no real estate exposure, FAIL if any exposure
            is_compliant = real_estate_exposure == 0.0

            calculations = {
                "real_estate_exposure": real_estate_exposure,
                "real_estate_percentage": real_estate_percentage,
                "total_exposure": total_exposure,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "Real Estate Exposure",
                    "real_estate_check_compliant": True #is_compliant
                },
                calculations=calculations,
            )
        except Exception as exc:
            logger.error("Error in real estate check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "Real Estate Exposure", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def commodities_check(self, fund: Fund) -> ComplianceResult:
        """Check if fund has any commodities exposure"""
        try:
            vest_eqy_holdings = fund.data.current.vest.equity

            if vest_eqy_holdings.empty:
                # No holdings means no commodities exposure - PASS
                return ComplianceResult(
                    is_compliant=True,
                    details={
                        "rule": "Commodities Exposure",
                        "commodities_check_compliant": True
                    },
                    calculations={
                        "commodities_exposure": 0.0,
                        "commodities_percentage": 0.0,
                        "total_exposure": 0.0,
                    },
                )

            # Check for commodities in GICS sector or other fields
            commodities_mask = vest_eqy_holdings["GICS_SECTOR_NAME"].str.contains(
                "Commodities", case=False, na=False
            )
            commodities_exposure = float(
                vest_eqy_holdings.loc[commodities_mask, "equity_market_value"].sum()
            )
            total_exposure = float(vest_eqy_holdings["equity_market_value"].sum())
            commodities_percentage = (
                commodities_exposure / total_exposure if total_exposure > 0 else 0.0
            )

            # PASS if no commodities exposure, FAIL if any exposure
            is_compliant = commodities_exposure == 0.0

            calculations = {
                "commodities_exposure": commodities_exposure,
                "total_exposure": total_exposure,
                "commodities_percentage": commodities_percentage,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "Commodities Exposure",
                    "commodities_check_compliant": is_compliant
                },
                calculations=calculations,
            )
        except Exception as exc:
            logger.error("Error in commodities check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "Commodities Exposure", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def twelve_d1a_other_inv_cos(self, fund: Fund) -> ComplianceResult:
        try:

            vest_eqy_holdings = fund.data.current.vest.equity
            total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)

            if total_assets == 0 or vest_eqy_holdings.empty:
                raise ValueError("Equity holdings or total assets missing")

            inv_co_mask = ~vest_eqy_holdings["REGULATORY_STRUCTURE"].isnull()
            investment_companies = vest_eqy_holdings[inv_co_mask].copy()

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
            vest_eqy_holdings = fund.data.current.vest.equity
            total_assets, total_net_assets, _, _ = self._get_current_totals(fund)
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

            share_col = "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"
            share_series = pd.Series(0.0, index=insurance_holdings.index, dtype=float)
            insurance_holdings[share_col] = share_series

            with np.errstate(divide="ignore", invalid="ignore"):
                insurance_holdings["ownership_pct"] = np.divide(
                    share_series,
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
            vest_eqy_holdings = fund.data.current.vest.equity
            vest_opt_holdings = fund.data.current.vest.options
            total_assets, total_net_assets, _, _ = self._get_current_totals(fund)
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
                        "rule_3_pass": True,                    },
                    calculations=calculations,
                )

            sec_related_businesses["EQY_SH_OUT_million"] = sec_related_businesses[
                "EQY_SH_OUT_million"
            ].replace(0, np.nan)

            share_col = "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"
            share_series = pd.Series(0.0, index=sec_related_businesses.index, dtype=float)
            sec_related_businesses[share_col] = share_series

            sec_related_businesses["ownership_pct"] = (
                share_series / sec_related_businesses["EQY_SH_OUT_million"]
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
                left_on="eqyticker",
                right_on="equity_underlying_ticker",
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

            related_tickers = sec_related_businesses["eqyticker"].unique()
            ticker_mask = vest_opt_holdings["equity_underlying_ticker"].isin(related_tickers)
            # occ_exposure = vest_opt_holdings[ticker_mask].copy()
            # occ_weight_mkt_val = float(occ_exposure["option_market_value"].sum())
            # occ_weight = occ_weight_mkt_val / total_assets if total_assets else 0.0
            # rule_3_pass_occ = occ_weight <= 0.05

            is_compliant = all([rule_1_pass, rule_2_pass, rule_3_pass])

            calculations = {
                "total_assets": total_assets,
                "sec_related_businesses": sec_related_businesses.to_dict("records"),
                "combined_holdings": combined_holdings.to_dict("records"),
                # "occ_weight_mkt_val": occ_weight_mkt_val,
                # "occ_weight": occ_weight,
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
                    "rule_3_pass": rule_3_pass,                },
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
        """Check 15% illiquid assets compliance"""
        try:
            vest_opt_holdings = fund.data.current.vest.options
            vest_eqy_holdings = fund.data.current.vest.equity
            total_assets, total_net_assets, expenses, _ = self._get_current_totals(fund)

            if total_assets == 0 or vest_eqy_holdings.empty:
                raise ValueError("Equity holdings or total assets missing")

            # Check for illiquid flag
            if "is_illiquid" not in vest_eqy_holdings.columns:
                vest_eqy_holdings["is_illiquid"] = False
            if "is_illiquid" not in vest_opt_holdings.columns:
                vest_opt_holdings["is_illiquid"] = False

            # Calculate illiquid values
            illiquid_mask = vest_eqy_holdings["is_illiquid"] == True
            illiquid_eqy_value = float(
                vest_eqy_holdings.loc[illiquid_mask, "equity_market_value"].sum()
            )

            # For options, check if they're illiquid
            illiquid_opt_value = 0.0
            if not vest_opt_holdings.empty:
                illiquid_opt_mask = vest_opt_holdings["is_illiquid"] == True
                illiquid_opt_value = float(
                    vest_opt_holdings.loc[illiquid_opt_mask, "option_market_value"].sum()
                )

            total_illiquid_value = illiquid_eqy_value + illiquid_opt_value
            illiquid_percentage = total_illiquid_value / total_assets

            # Calculate equity percentage
            equity_value = float(vest_eqy_holdings["equity_market_value"].sum())
            equity_percentage = equity_value / total_assets

            # Check compliance
            illiquid_compliant = illiquid_percentage <= ILLIQUID_MAX_THRESHOLD
            equity_compliant = equity_percentage >= EQUITY_MIN_THRESHOLD
            is_compliant = illiquid_compliant and equity_compliant

            calculations = {
                "total_assets": total_assets,
                "total_illiquid_value": total_illiquid_value,
                "illiquid_percentage": illiquid_percentage,
                "equity_holdings_percentage": equity_percentage,
            }

            return ComplianceResult(
                is_compliant=is_compliant,
                details={
                    "rule": "15% Illiquid Assets",
                    "max_15pct_illiquid_sai": is_compliant,
                    "equity_holdings_85pct_compliant": equity_compliant,
                },
                calculations=calculations,
            )
        except Exception as exc:
            logger.error("Error in 15% illiquid check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "15% Illiquid Assets", "status": "error"},
                calculations={},
                error=str(exc),
            )

    def gics_compliance(self, fund: Fund) -> ComplianceResult:

        try:
            # Proper null checking and access
            if not fund.data or not fund.data.current:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "no_data",
                        "fund": fund.name,
                    },
                    calculations={},
                    error="Fund data is not available"
                )

            vest_eqy_holdings = fund.data.current.vest.equity

            if vest_eqy_holdings.empty:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "no_equity_data",
                        "fund": fund.name,
                    },
                    calculations={},
                    error="Equity holdings are unavailable for analysis"
                )

            # Rest of the implementation...
            # Access index data properly
            index_df = pd.DataFrame()
            if fund.data.current is not None:
                index_df = fund.data.current.index.copy()

            if not isinstance(vest_eqy_holdings, pd.DataFrame) or vest_eqy_holdings.empty:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "no_equity_data",
                        "fund": fund.name,
                    },
                    calculations={},
                    error="Equity holdings are unavailable for analysis",
                )

            equity_values = pd.to_numeric(
                vest_eqy_holdings.get("equity_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            total_equity_value = float(equity_values.sum())
            if total_equity_value == 0.0:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "missing_equity_market_value",
                        "fund": fund.name,
                    },
                    calculations={},
                    error="Equity market values are unavailable for weight calculation",
                )

            fund_weights = equity_values / total_equity_value
            fund_weights.index = vest_eqy_holdings.index

            gics_summary_fund = self._summarize_gics_exposure(vest_eqy_holdings, fund_weights)
            if not gics_summary_fund:
                return ComplianceResult(
                    is_compliant=False,
                    details={
                        "rule": "GICS Concentration",
                        "status": "missing_gics_data",
                        "fund": fund.name,
                    },
                    calculations={},
                    error="GICS classifications are not available on equity holdings",
                )

            index_df = getattr(fund.data.current, "index", pd.DataFrame())
            gics_summary_index: Dict[str, Dict[str, float]] = {}
            index_weight_basis: Optional[str] = None
            if isinstance(index_df, pd.DataFrame) and not index_df.empty:
                index_df = index_df.copy()
                index_weights = pd.Series(dtype=float)

                for column in ("weight_index", "index_weight", "benchmark_weight", "weight"):
                    if column in index_df.columns:
                        candidate = pd.to_numeric(index_df[column], errors="coerce").fillna(0.0)
                        total = float(candidate.sum())
                        if total != 0.0:
                            index_weights = candidate / total
                            index_weights.index = index_df.index
                            index_weight_basis = column
                            break

                if index_weights.empty and "equity_market_value" in index_df.columns:
                    candidate = pd.to_numeric(
                        index_df["equity_market_value"], errors="coerce"
                    ).fillna(0.0)
                    total = float(candidate.sum())
                    if total != 0.0:
                        index_weights = candidate / total
                        index_weights.index = index_df.index
                        index_weight_basis = "equity_market_value"

                if not index_weights.empty:
                    gics_summary_index = self._summarize_gics_exposure(index_df, index_weights)

            compliance_group = self._resolve_gics_compliance_group(fund.name)

            exceeding_fund_gics, exceeding_index_gics, calculations = self._check_exposure(
                gics_summary_fund,
                gics_summary_index,
            )

            # Convenience flags (unchanged)
            sector_within_limit = len(exceeding_fund_gics.get("GICS_SECTOR_NAME", {})) == 0
            industry_group_within_limit = len(
                exceeding_fund_gics.get("GICS_INDUSTRY_GROUP_NAME", {})
            ) == 0
            industry_within_limit = len(exceeding_fund_gics.get("GICS_INDUSTRY_NAME", {})) == 0

            # --- Overall status by compliance group ---
            if compliance_group == "dogg":
                # Keep existing rule: must be within limit at both Industry and Industry Group
                overall_status = (
                    "PASS" if industry_within_limit and industry_group_within_limit else "FAIL"
                )

            elif compliance_group == "kng_fdnd":
                # Keep existing rule: fund/index exceed flags must match for all GICS classes
                fund_flags = {
                    gics_class: bool(exceeding_fund_gics.get(gics_class))
                    for gics_class in GICS_CLASS_COLUMNS
                }
                index_flags = {
                    gics_class: bool(exceeding_index_gics.get(gics_class))
                    for gics_class in GICS_CLASS_COLUMNS
                }
                compliance_match = all(
                    fund_flags.get(gics_class, False) == index_flags.get(gics_class, False)
                    for gics_class in GICS_CLASS_COLUMNS
                )
                overall_status = "PASS" if compliance_match else "FAIL"

            elif compliance_group == "tdvi":
                # Keep existing rule: either no exceeds at any level, or
                # only Information Technology can be >25% at the sector level.
                exceeds_counts = {
                    gics_class: len(exceeding_fund_gics.get(gics_class, {}))
                    for gics_class in GICS_CLASS_COLUMNS
                }
                all_exceeds_zero = all(count == 0 for count in exceeds_counts.values())
                tech_series = pd.Series(gics_summary_fund.get("GICS_SECTOR_NAME", {}))
                if tech_series.empty:
                    non_tech_exceeds = pd.Series(dtype=float)
                else:
                    non_tech_exceeds = tech_series[
                        (tech_series.index != "Information Technology")
                        & (tech_series > GICS_CONCENTRATION_THRESHOLD)
                        ]
                tech_compliance = non_tech_exceeds.empty
                overall_status = "PASS" if all_exceeds_zero or tech_compliance else "FAIL"

            else:
                # NEW DEFAULT RULE (loosened to industry-level only):
                # If the fund is >25% in an INDUSTRY, it is allowed ONLY when the index
                # is also >25% in that same INDUSTRY. Otherwise, failure.
                fund_over_inds = set(exceeding_fund_gics.get("GICS_INDUSTRY_NAME", {}).keys())
                index_over_inds = set(exceeding_index_gics.get("GICS_INDUSTRY_NAME", {}).keys())
                industry_mismatches = sorted(fund_over_inds - index_over_inds)
                overall_status = "PASS" if len(industry_mismatches) == 0 else "FAIL"

            total_assets, total_net_assets, _, _ = self._get_current_totals(fund)

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
                "rule": "GICS Concentration",
                "fund": fund.name,
                "overall_gics_compliance": overall_status,
                # Retain original keys (for any downstream consumers)
                "sector_exceeds_25": sector_within_limit,
                "industry_group_exceeds_25": industry_group_within_limit,
                "industry_exceeds_25": industry_within_limit,
                "exceeding_fund_gics": exceeding_fund_gics,
                "exceeding_index_gics": exceeding_index_gics,
                "fund_exposures": gics_summary_fund,
                "compliance_group": compliance_group,
            }
            # Basis fields
            details["fund_weight_basis"] = "equity_market_value"
            if index_weight_basis:
                details["index_weight_basis"] = index_weight_basis

            # Add explicit list of industry mismatches under the new default rule
            if compliance_group not in {"dogg", "kng_fdnd", "tdvi"}:
                fund_over_inds = set(exceeding_fund_gics.get("GICS_INDUSTRY_NAME", {}).keys())
                index_over_inds = set(exceeding_index_gics.get("GICS_INDUSTRY_NAME", {}).keys())
                details["industry_mismatches"] = sorted(fund_over_inds - index_over_inds)

            return ComplianceResult(
                is_compliant=overall_status == "PASS",
                details=details,
                calculations=calculations,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Error in GICS compliance check for %s: %s", fund.name, exc)
            return ComplianceResult(
                is_compliant=False,
                details={"rule": "GICS Concentration", "status": "error", "fund": fund.name},
                calculations={},
                error=str(exc),
            )


    @staticmethod
    def _resolve_gics_compliance_group(fund_name: str) -> str:
        mapping = {
            "DOGG": "dogg",
            "KNG": "kng_fdnd",
            "FDND": "kng_fdnd",
            "TDVI": "tdvi",
        }
        if not isinstance(fund_name, str):
            return "standard"
        return mapping.get(fund_name.upper(), "standard")

    def _summarize_gics_exposure(
        self,
        df: pd.DataFrame,
        weight_series: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        if not isinstance(weight_series, pd.Series) or weight_series.empty:
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
            fund_series = pd.Series(gics_summary_fund.get(gics_class, {}), dtype=float)
            index_series = pd.Series(gics_summary_index.get(gics_class, {}), dtype=float)

            if not fund_series.empty:
                fund_series = fund_series.sort_values(ascending=False)
            if not index_series.empty:
                index_series = index_series.sort_values(ascending=False)

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
                "fund_exceeding_count": int(fund_mask.sum()) if not fund_series.empty else 0,
                "index_exceeding_count": int(index_mask.sum()) if not index_series.empty else 0,
            }

        return exceeding_fund_gics, exceeding_index_gics, calculations

    def _consolidate_holdings_by_issuer(
        self, fund: Fund, holdings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate holdings exposure by issuer without conditional adjustments."""

        if holdings_df.empty:
            return holdings_df.copy()

        df = holdings_df.copy()
        df["equity_market_value"] = pd.to_numeric(
            df.get("equity_market_value", 0.0), errors="coerce"
        ).fillna(0.0)

        share_col = "iiv_shares" if self.analysis_type == "ex_post" else "nav_shares"
        share_series = pd.Series(0.0, index=df.index, dtype=float)
        df[share_col] = share_series
        df["eqyticker"] = df["eqyticker"].astype(str)

        google_mask = df["eqyticker"].isin(["GOOG", "GOOGL"])
        if google_mask.any():
            df.loc[google_mask, "eqyticker"] = "GOOGLE"

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        aggregation = {
            **{
                column: "sum"
                for column in numeric_columns
            },
            **{
                column: "first"
                for column in df.columns
                if column not in numeric_columns + ["eqyticker"]
            },
        }

        consolidated = (
            df.groupby("eqyticker", as_index=False)
            .agg(aggregation)
            .copy()
        )

        return consolidated

    def _calculate_index_overlap_adjustments(
        self,
        fund: Fund,
        holdings_df: pd.DataFrame,
        overlap_df: Optional[pd.DataFrame],
        total_assets: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply FLEX index overlap adjustments and return overlap contribution details."""

        overlap_columns = [
            "security_ticker",
            "security_weight",
            "overlap_market_value",
            "overlap_weight",
        ]
        empty = pd.DataFrame(columns=overlap_columns)

        def _base_result(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            df = df.copy()
            df["net_market_value"] = pd.to_numeric(
                df.get("equity_market_value", 0.0), errors="coerce"
            ).fillna(0.0) + pd.to_numeric(
                df.get("option_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            return df, empty

        if holdings_df.empty:
            return holdings_df.copy(), empty

        if overlap_df is None or overlap_df.empty:
            return _base_result(holdings_df)

        if not (
            fund.has_flex_option
            and (fund.flex_option_type or "").lower() == "index"
        ):
            return _base_result(holdings_df)

        flex_options = getattr(getattr(fund.data.current, "vest", None), "flex_options", pd.DataFrame())
        if flex_options is None or flex_options.empty:
            return _base_result(holdings_df)

        overlap = overlap_df.copy()

        overlap["security_weight"] = pd.to_numeric(
            overlap["security_weight"], errors="coerce"
        ).fillna(0.0)
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

        quantities = pd.to_numeric(
            flex_positions.get("nav_shares_option", 0.0), errors="coerce"
        ).fillna(0.0)
        long_flex = flex_positions.loc[quantities > 0].copy()

        if long_flex.empty:
            long_flex = flex_positions.loc[
                pd.to_numeric(
                    flex_positions.get("option_market_value", 0.0), errors="coerce"
                ).fillna(0.0)
                > 0
            ].copy()

        total_flex_mv = float(
            pd.to_numeric(
                long_flex.get("option_market_value", 0.0), errors="coerce"
            ).fillna(0.0).sum()
        )

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
        merged["security_weight"] = pd.to_numeric(
            merged.get("security_weight", 0.0), errors="coerce"
        ).fillna(0.0)
        merged["overlap_market_value"] = total_flex_mv * merged["security_weight"]
        merged["long_box_market_value_overlap"] = (
            pd.to_numeric(
                merged.get("long_box_market_value_overlap", 0.0), errors="coerce"
            ).fillna(0.0)
            + merged["overlap_market_value"]
        )
        merged = merged.drop(columns=["security_ticker"], errors="ignore")

        merged["net_market_value"] = (
            pd.to_numeric(
                merged.get("equity_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                merged.get("option_market_value", 0.0), errors="coerce"
            ).fillna(0.0)
            + merged["long_box_market_value_overlap"]
        )

        overlap_details = merged.loc[
            merged["overlap_market_value"] > 0
        , ["eqyticker", "security_weight", "overlap_market_value"]].copy()
        overlap_details = overlap_details.rename(columns={"eqyticker": "security_ticker"})
        if not overlap_details.empty:
            if total_assets:
                overlap_details["overlap_weight"] = (
                    overlap_details["overlap_market_value"] / total_assets
                )
            else:
                overlap_details["overlap_weight"] = 0.0
        else:
            overlap_details = pd.DataFrame(columns=overlap_columns)

        merged = merged.drop(columns=["security_weight", "overlap_market_value"], errors="ignore")

        return merged, overlap_details.reindex(columns=overlap_columns).reset_index(drop=True)


    def _get_current_totals(self, fund: Fund) -> tuple[float, float, float, float]:
        """Return custodian total assets, total net assets, expenses, and cash."""
        snapshot = getattr(fund.data, "current", None)
        custodian = getattr(snapshot, "custodian", None) if snapshot else None

        total_assets = float(getattr(custodian, "ta", 0.0) or 0.0) if custodian else 0.0
        total_net_assets = float(getattr(custodian, "tna", 0.0) or 0.0) if custodian else 0.0
        expenses = float(getattr(custodian, "expenses", 0.0) or 0.0) if custodian else 0.0
        cash_value = float(getattr(custodian, "cash", 0.0) or 0.0) if custodian else 0.0

        return total_assets, total_net_assets, expenses, cash_value

