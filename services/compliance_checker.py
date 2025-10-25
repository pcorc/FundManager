import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from config.constants import *  # Your existing constants
from config.fund_classifications import DIVERSIFIED_FUNDS, NON_DIVERSIFIED_FUNDS, PRIVATE_FUNDS, CLOSED_END_FUNDS
from domain.fund import Fund  # ← ADD THIS IMPORT
from utilities.logger import setup_logger
logger = setup_logger("compliance_checker", "compliance/logs/compliance_checker.log")


@dataclass
class ComplianceResult:
    """Structured result for compliance checks"""
    is_compliant: bool
    details: Dict
    calculations: Dict = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.calculations is None:
            self.calculations = {}


class ComplianceChecker:
    def __init__(self, session, funds: Dict[str, 'Fund'] = None, date=None, base_cls=None):
        """
        Updated ComplianceChecker that works with Fund objects instead of raw data.

        Args:
            session: SQLAlchemy session
            funds: Dictionary of Fund objects (name -> Fund instance)
            date: Analysis date
            base_cls: Base SQLAlchemy class
        """
        self.session = session
        self.funds = funds or {}
        self.date = date
        self.base_cls = base_cls

    def run_compliance_tests(self, test_functions=None):
        """
        Run compliance tests using Fund objects instead of raw data dictionaries.
        """
        results = {}

        available_tests = {
            "gics_compliance": self.gics_compliance,
            "prospectus_80pct_policy": self.prospectus_80pct_policy,
            "diversification_40act_check": self.diversification_40act_check,
            "diversification_IRS_check": self.diversification_IRS_check,
            "diversification_IRC_check": self.diversification_IRC_check,
            "max_15pct_illiquid_sai": self.max_15pct_illiquid_sai,
            "real_estate_check": self.real_estate_check,
            "commodities_check": self.commodities_check,
            "twelve_d1a_other_inv_cos": self.twelve_d1a_other_inv_cos,
            "twelve_d2_insurance_cos": self.twelve_d2_insurance_cos,
            "twelve_d3_sec_biz": self.twelve_d3_sec_biz,
        }

        # Filter tests if specific ones requested
        if test_functions:
            test_functions_set = set(test_functions)
            available_tests = {name: func for name, func in available_tests.items()
                               if name in test_functions_set}

        for fund_name, fund in self.funds.items():
            logger.info(f"Running compliance tests for {fund_name}")
            results[fund_name] = {}

            try:
                # Calculate summary metrics using Fund properties
                summary_metrics = self.calculate_summary_metrics(fund)
                results[fund_name]["summary_metrics"] = summary_metrics
                results[fund_name]["cash_data"] = summary_metrics.get("cash_value", 0)
            except Exception as e:
                logger.error(f"Error calculating summary metrics for {fund_name}: {e}", exc_info=True)
                results[fund_name]["summary_metrics"] = self._get_empty_summary_metrics()
                results[fund_name]["cash_data"] = 0

            # Execute compliance checks
            for test_name, test_func in available_tests.items():
                try:
                    # Pass Fund object instead of fund name
                    result = test_func(fund)
                    results[fund_name][test_name] = result
                except Exception as e:
                    logger.error(f"Error executing {test_name} for fund {fund_name}: {e}", exc_info=True)
                    results[fund_name][test_name] = ComplianceResult(
                        is_compliant=False,
                        details={},
                        error=str(e)
                    )

        return results

    def calculate_summary_metrics(self, fund: 'Fund') -> Dict:
        """
        Calculate summary metrics using Fund object properties.
        """
        return {
            'cash_value': fund.cash_value,
            'equity_market_value': fund.total_equity_value,
            'option_delta_adjusted_notional': fund.total_option_delta_adjusted_notional,
            'option_market_value': fund.total_option_value,
            'treasury': fund.total_treasury_value,
            'total_assets': fund.total_assets,
            'total_net_assets': fund.total_net_assets
        }

    def _get_empty_summary_metrics(self) -> Dict:
        """Return empty summary metrics for error cases"""
        return {
            "cash_value": 0,
            "equity_market_value": 0,
            "option_delta_adjusted_notional": 0,
            "option_market_value": 0,
            "treasury": 0,
            "total_assets": 0,
            "total_net_assets": 0
        }

    # Updated compliance methods to use Fund objects
    def prospectus_80pct_policy(self, fund: 'Fund') -> ComplianceResult:
        """
        Check prospectus 80% policy using Fund object.
        """
        try:
            total_assets = fund.total_assets
            total_net_assets = fund.total_net_assets
            cash_value = fund.cash_value
            treasury_value = fund.total_treasury_value
            equity_value = fund.total_equity_value
            option_value = fund.total_option_value
            option_delta_notional = fund.total_option_delta_adjusted_notional

            # Identify if options count toward numerator
            options_in_scope = fund.name in {"KNG", "KNGIX", "DOGG", "FTCSH", "FGSI"}

            if options_in_scope:
                numerator = equity_value + abs(option_delta_notional) + treasury_value
            else:
                numerator = equity_value + treasury_value

            denominator = equity_value + abs(option_delta_notional) + cash_value + treasury_value
            names_test = numerator / denominator if denominator > 0 else 0

            # Market-value-based variant
            numerator_mv = equity_value + (option_value if options_in_scope else 0) + treasury_value
            denominator_mv = equity_value + option_value + cash_value + treasury_value
            names_test_mv = numerator_mv / denominator_mv if denominator_mv > 0 else 0

            return ComplianceResult(
                is_compliant=names_test >= PROSPECTUS_MIN_THRESHOLD,
                details={
                    "policy_type": "80% Prospectus Policy",
                    "options_in_scope": options_in_scope
                },
                calculations={
                    "total_equity_market_value": equity_value,
                    "total_opt_delta_notional_value": option_delta_notional,
                    "total_opt_market_value": option_value,
                    "total_treasury_value": treasury_value,
                    "total_cash_value": cash_value,
                    "denominator": denominator,
                    "numerator": numerator,
                    "names_test": names_test,
                    "denominator_mv": denominator_mv,
                    "numerator_mv": numerator_mv,
                    "names_test_mv": names_test_mv,
                    "threshold": PROSPECTUS_MIN_THRESHOLD
                }
            )

        except Exception as e:
            return ComplianceResult(
                is_compliant=False,
                details={},
                error=f"Error in prospectus policy check: {str(e)}"
            )

    def diversification_IRS_check(self, fund: 'Fund') -> ComplianceResult:
        """
        IRS diversification check using Fund object.
        """
        try:
            # Use Fund methods to get processed holdings data
            holdings_df = fund.get_irs_holdings_data()  # This would be a new method on Fund
            total_assets = fund.total_assets
            total_net_assets = fund.total_net_assets
            expenses = fund.expenses

            # Your existing IRS check logic, but using the pre-processed data from Fund
            # ... (rest of your IRS logic using holdings_df from fund)

            return ComplianceResult(
                is_compliant=condition_IRS_2_a_50 and condition_IRS_2_a_5_new and condition_IRS_2_a_10,
                details={
                    "rule": "IRS Diversification",
                    "conditions_met": {
                        "50%_qualifying_assets": condition_IRS_2_a_50,
                        "5%_large_securities": condition_IRS_2_a_5_new,
                        "10%_ownership": condition_IRS_2_a_10
                    }
                },
                calculations=calculations
            )

        except Exception as e:
            return ComplianceResult(
                is_compliant=False,
                details={},
                error=f"Error in IRS diversification check: {str(e)}"
            )

    def diversification_40act_check(self, fund: 'Fund') -> ComplianceResult:
        """
        40 Act diversification check using Fund object.
        """
        # Skip private funds
        if fund.is_private_fund:
            return ComplianceResult(
                is_compliant=True,
                details={
                    "skipped": True,
                    "reason": "Private funds are not registered under the 40 Act",
                    "fund_type": "Private Fund"
                }
            )

        try:
            # Use Fund properties and methods
            holdings_df = fund.get_40act_holdings_data()
            total_assets = fund.total_assets
            total_net_assets = fund.total_net_assets

            # Your existing 40 Act logic using Fund data
            # ... (rest of your 40 Act logic)

            return ComplianceResult(
                is_compliant=condition_1_met and condition_2a_met and condition_2b_met,
                details={
                    "rule": "40 Act Diversification",
                    "fund_registration": fund.registration_type,
                    "conditions_met": {
                        "75%_qualifying_assets": condition_1_met,
                        "5%_issuer_limit": condition_2a_met,
                        "10%_ownership_limit": condition_2b_met
                    }
                },
                calculations=calculations
            )

        except Exception as e:
            return ComplianceResult(
                is_compliant=False,
                details={},
                error=f"Error in 40 Act diversification check: {str(e)}"
            )

    def gics_compliance(self, fund: 'Fund') -> ComplianceResult:
        """
        GICS compliance check using Fund object.
        """
        try:
            # Use Fund methods to get GICS data
            gics_data = fund.get_gics_exposure()
            total_assets = fund.total_assets

            # Your existing GICS logic using Fund data
            # ... (rest of your GICS logic)

            return ComplianceResult(
                is_compliant=overall_gics_compliance == "PASS",
                details={
                    "rule": "GICS Concentration",
                    "overall_compliance": overall_gics_compliance
                },
                calculations=calculations
            )

        except Exception as e:
            return ComplianceResult(
                is_compliant=False,
                details={},
                error=f"Error in GICS compliance check: {str(e)}"
            )

    # Update other compliance methods similarly...
    def diversification_IRC_check(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings, fund.total_assets, etc.
        pass

    def real_estate_check(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings with GICS data
        pass

    def commodities_check(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings with GICS data
        pass

    def twelve_d1a_other_inv_cos(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings with regulatory structure data
        pass

    def twelve_d2_insurance_cos(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings with GICS data
        pass

    def twelve_d3_sec_biz(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.equity_holdings and fund.options_holdings
        pass

    def max_15pct_illiquid_sai(self, fund: 'Fund') -> ComplianceResult:
        # Use fund.illiquid_holdings and fund.total_assets
        pass

    def gics_compliance(self, etf: str):
        """GICS compliance using weight analyzer"""
        fund_data = self.fund_results.get(etf, {})

        # Use weight analyzer for all calculations
        weight_analysis = self.weight_analyzer.analyze_weights(etf, fund_data)

        # Your existing GICS compliance logic, but using weight_analysis
        fund_exposures = weight_analysis.gics_fund
        index_exposures = weight_analysis.gics_index

        # Rest of your GICS compliance logic remains similar
        # but now uses the structured weight_analysis data

        return self._evaluate_gics_compliance(etf, fund_exposures, index_exposures)

    def _evaluate_gics_compliance(self, etf: str, fund_exposures: Dict, index_exposures: Dict):
        """Your existing GICS evaluation logic"""
        # This becomes much cleaner with pre-calculated weights
        etf_compliance_groups = {
            'DOGG': 'dogg_compliance',
            'KNG': 'kng_fdnd_compliance',
            'FDND': 'kng_fdnd_compliance',
            'TDVI': 'tdvi_compliance'
        }

        compliance_group = etf_compliance_groups.get(etf, 'standard_compliance')

        if compliance_group == 'dogg_compliance':
            return self._check_dogg_compliance(fund_exposures)
        elif compliance_group == 'kng_fdnd_compliance':
            return self._check_kng_compliance(fund_exposures, index_exposures)