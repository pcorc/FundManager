

import pandas as pd
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class TradingComplianceAnalyzer:
    """
    Analyzes compliance differences between ex_ante and ex_post for funds with trading activity.
    Compares high-level PASS/FAIL status for each compliance check.
    """

    def __init__(self, results_ex_ante: Dict, results_ex_post: Dict, date, traded_funds_info: Dict):
        """
        Initialize the analyzer.

        Args:
            results_ex_ante: Compliance results from ex_ante analysis
            results_ex_post: Compliance results from ex_post analysis
            date: Analysis date
            traded_funds_info: Dict with trading activity by fund
                {fund_name: {'total_traded': X, 'equity': Y, 'options': Z, 'treasury': W}}
        """
        self.results_ex_ante = results_ex_ante
        self.results_ex_post = results_ex_post
        self.date = date
        self.traded_funds_info = traded_funds_info

    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis function that compares ex_ante vs ex_post compliance for traded funds.

        Returns:
            dict: {
                'date': date,
                'summary': {...},
                'funds': {
                    'fund_name': {
                        'trade_info': {...},
                        'checks': {...}
                    }
                }
            }
        """
        traded_fund_names = list(self.traded_funds_info.keys())

        comparison_data = {
            'date': self.date,
            'summary': {
                'total_funds_analyzed': len(traded_fund_names),
                'funds_with_compliance_changes': 0,
                'total_checks_changed': 0,
            },
            'funds': {}
        }

        for fund_name in traded_fund_names:
            fund_comparison = self._compare_fund_compliance(fund_name)
            comparison_data['funds'][fund_name] = fund_comparison

            # Update summary statistics
            if fund_comparison['has_changes']:
                comparison_data['summary']['funds_with_compliance_changes'] += 1

            comparison_data['summary']['total_checks_changed'] += fund_comparison['num_changes']

        return comparison_data

    def _compare_fund_compliance(self, fund_name: str) -> Dict[str, Any]:
        """
        Compares compliance results for a single fund between ex_ante and ex_post.
        """
        fund_data = {
            'fund_name': fund_name,
            'trade_info': self.traded_funds_info[fund_name],
            'checks': {},
            'has_changes': False,
            'num_changes': 0
        }

        # Get compliance results for this fund
        ante_results = self.results_ex_ante.get(fund_name, {})
        post_results = self.results_ex_post.get(fund_name, {})

        # Get all compliance check names (exclude non-check keys)
        excluded_keys = {'summary_metrics', 'cash_data', 'fund_object'}
        all_checks = set(list(ante_results.keys()) + list(post_results.keys())) - excluded_keys

        for check_name in all_checks:
            ante_check = ante_results.get(check_name, {})
            post_check = post_results.get(check_name, {})

            # Get overall PASS/FAIL status for each check
            ante_status = self._get_overall_check_status(check_name, ante_check)
            post_status = self._get_overall_check_status(check_name, post_check)

            changed = ante_status != post_status

            fund_data['checks'][check_name] = {
                'status_before': ante_status,
                'status_after': post_status,
                'changed': changed
            }

            if changed:
                fund_data['has_changes'] = True
                fund_data['num_changes'] += 1

        return fund_data

    def _get_overall_check_status(self, check_name: str, check_result: Dict) -> str:
        """
        Determines the overall PASS/FAIL status for a compliance check.
        A check FAILS if any boolean condition is False or any string status is "FAIL".

        Args:
            check_name: Name of the compliance check
            check_result: Dictionary containing check results

        Returns:
            'PASS', 'FAIL', or 'UNKNOWN'
        """
        if not check_result or not isinstance(check_result, dict):
            return 'UNKNOWN'

        # Skip nested calculation details
        keys_to_skip = {'calculations', 'details_before', 'details_after', 'error'}

        # Collect all boolean and string compliance indicators
        compliance_indicators = []

        for key, value in check_result.items():
            if key in keys_to_skip:
                continue

            # Check boolean values
            if isinstance(value, bool):
                compliance_indicators.append(value)

            # Check string values for PASS/FAIL
            elif isinstance(value, str):
                if value.upper() == 'PASS':
                    compliance_indicators.append(True)
                elif value.upper() == 'FAIL':
                    compliance_indicators.append(False)

        # If we found any indicators, check if all passed
        if compliance_indicators:
            return 'PASS' if all(compliance_indicators) else 'FAIL'

        return 'UNKNOWN'

    @staticmethod
    def get_traded_funds_info(fund_manager) -> Dict[str, Dict[str, float]]:
        """
        Helper method to extract trading activity from fund manager results.

        Args:
            fund_manager: FundManager instance with loaded data

        Returns:
            Dict of {fund_name: {'total_traded': X, 'equity': Y, 'options': Z, 'treasury': W}}
        """
        traded_funds_info = {}

        for fund_name, fund_data in fund_manager.results.items():
            # Get holdings dataframes
            equity_df = fund_data.get("equity_holdings", pd.DataFrame())
            options_df = fund_data.get("options_holdings", pd.DataFrame())
            treasury_df = fund_data.get("treasury_holdings", pd.DataFrame())

            # Sum trade_rebal from each asset type
            equity_trade = 0.0
            if not equity_df.empty and 'trade_rebal' in equity_df.columns:
                equity_trade = float(equity_df['trade_rebal'].sum())

            options_trade = 0.0
            if not options_df.empty and 'trade_rebal' in options_df.columns:
                options_trade = float(options_df['trade_rebal'].sum())

            treasury_trade = 0.0
            if not treasury_df.empty and 'trade_rebal' in treasury_df.columns:
                treasury_trade = float(treasury_df['trade_rebal'].sum())

            # Calculate total trade_rebal
            total_trade = equity_trade + options_trade + treasury_trade

            # If any trading activity detected, add to traded_funds
            if total_trade != 0:
                traded_funds_info[fund_name] = {
                    'total_traded': total_trade,
                    'equity': equity_trade,
                    'options': options_trade,
                    'treasury': treasury_trade
                }
                logger.info(f"Fund {fund_name} traded: equity={equity_trade:.2f}, "
                            f"options={options_trade:.2f}, treasury={treasury_trade:.2f}")

        return traded_funds_info