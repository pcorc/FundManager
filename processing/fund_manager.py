# management/fund_manager.py
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import logging

# Import your existing services
from services.compliance_checker import ComplianceChecker
from services.nav_reconciliator import NAVReconciliator
from services.reconciliator import Reconciliator

# Import your domain classes
from domain.fund import Fund, FundData, FundMetrics
from processing.bulk_data_loader import BulkDataStore


@dataclass
class FundResult:
    """Results from processing a single fund"""
    fund_name: str
    compliance_results: Dict[str, Any] = None
    reconciliation_results: Dict[str, Any] = None
    nav_results: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ProcessingResults:
    """Container for the outcome of the daily operations."""

    fund_results: Dict[str, FundResult]
    summary: Dict[str, Any]


class FundManager:
    """Manages processing for ALL funds using bulk-loaded data"""

    def __init__(self, fund_registry, data_store: BulkDataStore, analysis_type: str = "eod"):
        self.fund_registry = fund_registry
        self.data_store = data_store
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(__name__)
        self.available_funds = list(data_store.loaded_funds)

    def run_daily_operations(self, operations: List[str]) -> ProcessingResults:
        """Run specified operations for ALL funds using cached data"""
        results: Dict[str, FundResult] = {}
        summary = {
            "requested_operations": operations,
            "processed_funds": 0,
            "funds_with_errors": 0,
            "errors": [],
        }

        for fund_name in self.available_funds:
            self.logger.info(f"Processing {fund_name}...")
            fund_result = FundResult(fund_name=fund_name)

            try:
                # Get fund configuration from registry
                fund_config = self.fund_registry.get_fund(fund_name)
                if not fund_config:
                    fund_result.errors.append(f"Fund not found in registry")
                    continue

                # Create Fund instance with bulk data
                fund = self._create_fund_from_bulk_data(fund_name, fund_config)

                # Run requested operations
                if 'compliance' in operations:
                    fund_result.compliance_results = self._run_compliance(fund)

                if 'reconciliation' in operations:
                    fund_result.reconciliation_results = self._run_reconciliation(fund)

                if 'nav_reconciliation' in operations:
                    fund_result.nav_results = self._run_nav_reconciliation(fund)

                summary["processed_funds"] += 1

            except Exception as e:
                error_msg = f"Error processing {fund_name}: {str(e)}"
                fund_result.errors.append(error_msg)
                self.logger.error(error_msg)
                summary["errors"].append(error_msg)
                summary["funds_with_errors"] += 1

            results[fund_name] = fund_result

        summary["total_funds"] = len(results)
        return ProcessingResults(fund_results=results, summary=summary)

    def _create_fund_from_bulk_data(self, fund_name: str, fund_config) -> Fund:
        fund_data_dict = self.data_store.fund_data.get(fund_name, {})

        fund_data = FundData()

        # POPULATE WITH CUSTODIAN DATA
        fund_data.current = FundMetrics(
            equity=fund_data_dict.get('custodian_equity', pd.DataFrame()),
            options=fund_data_dict.get('custodian_option', pd.DataFrame()),
            treasury=fund_data_dict.get('custodian_treasury', pd.DataFrame()),
            cash=self._extract_cash_value(fund_data_dict.get('cash', pd.DataFrame())),
            # EXTRACT CUSTODIAN-PROVIDED VALUES
            custodian_total_assets=self._extract_custodian_total_assets(fund_data_dict),
            custodian_total_net_assets=self._extract_custodian_total_net_assets(fund_data_dict)
        )

        fund_data.expense_ratio = fund_config.expense_ratio

        fund = Fund(fund_name, fund_config.mapping_data)
        fund.data = fund_data
        return fund

    def _extract_custodian_total_assets(self, fund_data_dict: Dict) -> float:
        """Extract total assets from custodian NAV data"""
        nav_data = fund_data_dict.get('nav', pd.DataFrame())
        if nav_data.empty:
            return 0.0

        # Look for custodian total assets
        asset_columns = ['total_assets', 'gross_assets', 'assets', 'total_assets_value']
        for col in asset_columns:
            if col in nav_data.columns:
                return nav_data[col].sum()

        self.logger.warning("No custodian total assets found")
        return 0.0

    def _extract_cash_value(self, cash_data: pd.DataFrame) -> float:
        """Extract cash value from cash data"""
        if cash_data.empty:
            return 0.0

        cash_columns = ['cash_value', 'cash', 'amount', 'value']
        for col in cash_columns:
            if col in cash_data.columns:
                return cash_data[col].sum()

        self.logger.warning("No cash value column found")
        return 0.0

    def _extract_custodian_total_net_assets(self, fund_data_dict: Dict) -> float:
        """Extract total net assets from custodian NAV data"""
        nav_data = fund_data_dict.get('nav', pd.DataFrame())
        if nav_data.empty:
            return 0.0

        # Look for custodian net assets
        net_columns = ['net_assets', 'total_net_assets', 'nav', 'net_asset_value']
        for col in net_columns:
            if col in nav_data.columns:
                return nav_data[col].sum()

        self.logger.warning("No custodian total net assets found")
        return 0.0

    def _run_nav_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        """Run NAV reconciliation using your existing NAVReconciliator"""
        try:
            # Determine prior date
            prior_date = self._get_prior_date(self.data_store.date)

            # Your NAVReconciliator expects Fund instance and dates
            nav_reconciliator = NAVReconciliator(
                fund=fund,
                analysis_date=self.data_store.date,
                prior_date=prior_date
            )

            return nav_reconciliator.run_nav_reconciliation()

        except Exception as e:
            self.logger.error(f"NAV reconciliation error for {fund.name}: {e}")
            return {'errors': [str(e)], 'differences': []}

    def _run_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        """Run reconciliation"""
        try:
            reconciliator = Reconciliator(
                fund=fund,
                analysis_type=self.analysis_type
            )
            reconciliator.run_all_reconciliations()
            return reconciliator.get_summary()
        except Exception as e:
            self.logger.error(f"Reconciliation error for {fund.name}: {e}")
            return {'errors': [str(e)], 'breaks': []}

    def _get_prior_date(self, current_date: Any) -> str:
        """Get prior business date - you might want to improve this"""
        from datetime import datetime, timedelta, date as date_cls

        if isinstance(current_date, datetime):
            current = current_date
        elif isinstance(current_date, date_cls):
            current = datetime.combine(current_date, datetime.min.time())
        else:
            current = datetime.strptime(str(current_date), '%Y-%m-%d')

        prior = current - timedelta(days=1)
        return prior.strftime('%Y-%m-%d')