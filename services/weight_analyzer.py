# services/weight_analyzer.py
import pandas as pd
import logging
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class WeightAnalysis:
    fund_weights: Dict
    index_weights: Dict
    gics_fund: Dict
    gics_index: Dict
    comparison: Dict


class WeightAnalyzer:
    """Handles all weight calculations for funds and benchmarks"""

    def __init__(self, gics_mapping: pd.DataFrame):
        self.gics_mapping = gics_mapping
        self.logger = logging.getLogger(__name__)

    def analyze_weights(self, fund_name: str, fund_data: Dict) -> WeightAnalysis:
        """Complete weight analysis for a fund"""
        # Calculate fund weights
        fund_weights, gics_fund = self._calculate_fund_weights(fund_data)

        # Calculate index weights
        index_weights, gics_index = self._calculate_index_weights(fund_name, fund_data)

        # Compare weights
        comparison = self._compare_weights(fund_weights, index_weights, gics_fund, gics_index)

        return WeightAnalysis(
            fund_weights=fund_weights,
            index_weights=index_weights,
            gics_fund=gics_fund,
            gics_index=gics_index,
            comparison=comparison
        )

    def _calculate_fund_weights(self, fund_data: Dict) -> Tuple[Dict, Dict]:
        """Calculate fund constituent weights and GICS summaries"""
        vest_eqy = fund_data.get('equity_holdings', pd.DataFrame()).copy()
        vest_opt = fund_data.get('options_holdings', pd.DataFrame()).copy()

        # Calculate market values
        if not vest_eqy.empty and {'price', 'quantity'}.issubset(vest_eqy.columns):
            vest_eqy['market_value'] = vest_eqy['price'] * vest_eqy['quantity']

        # Calculate constituent weights
        constituent_weights = vest_eqy.set_index('equity_ticker')['start_wgt'].to_dict()

        # Calculate GICS summaries
        gics_summary = {
            col: vest_eqy.groupby(col)['start_wgt'].sum()
            for col in self.gics_mapping.columns if col in vest_eqy.columns
        }

        return constituent_weights, gics_summary

    def _calculate_index_weights(self, fund_name: str, fund_data: Dict) -> Tuple[Dict, Dict]:
        """Calculate index constituent weights and GICS summaries"""
        if fund_name == "DOGG":
            # DOGG uses its own holdings as benchmark
            index_data = fund_data.get('equity_holdings', pd.DataFrame()).copy()
            index_data.rename(columns={'start_wgt': 'weight_index'}, inplace=True)
        else:
            index_data = fund_data.get('index_holdings', pd.DataFrame()).copy()

        if index_data.empty:
            self.logger.warning(f"{fund_name}: No index data for weight calculation")
            return {}, {}

        # Calculate constituent weights
        constituent_weights = index_data.set_index('equity_ticker')['weight_index'].to_dict()

        # Calculate GICS summaries
        gics_summary = {
            col: index_data.groupby(col)['weight_index'].sum()
            for col in self.gics_mapping.columns if col in index_data.columns
        }

        return constituent_weights, gics_summary

    def _compare_weights(self, fund_weights: Dict, index_weights: Dict,
                         gics_fund: Dict, gics_index: Dict) -> Dict:
        """Compare fund weights vs index weights"""
        comparison = {}

        # Constituent weight differences
        all_tickers = set(fund_weights.keys()) | set(index_weights.keys())
        weight_differences = {
            ticker: fund_weights.get(ticker, 0) - index_weights.get(ticker, 0)
            for ticker in all_tickers
        }

        # GICS exposure differences
        gics_differences = {}
        for gics_class in ['GICS_SECTOR_NAME', 'GICS_INDUSTRY_NAME', 'GICS_INDUSTRY_GROUP_NAME']:
            if gics_class in gics_fund and gics_class in gics_index:
                fund_series = pd.Series(gics_fund[gics_class])
                index_series = pd.Series(gics_index[gics_class])
                gics_differences[gics_class] = (fund_series - index_series).to_dict()

        comparison['constituent_differences'] = weight_differences
        comparison['gics_differences'] = gics_differences

        return comparison