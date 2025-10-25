# management/fund_manager.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import date
import logging

# Import your existing services
from services.compliance_checker import ComplianceChecker
from services.nav_reconciliator import NAVReconciliator
from services.reconciliator import Reconciliator

# Import your domain classes
from domain.fund import Fund
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


class FundManager:
    """Manages processing for ALL funds using bulk-loaded data"""

    def __init__(self, fund_registry, data_store: BulkDataStore, analysis_type: str = "eod"):
        self.fund_registry = fund_registry
        self.data_store = data_store
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(__name__)
        self.available_funds = list(data_store.loaded_funds)

    def run_daily_operations(self, operations: List[str]) -> Dict[str, FundResult]:
        """Run specified operations for ALL funds using cached data"""
        results = {}

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

            except Exception as e:
                error_msg = f"Error processing {fund_name}: {str(e)}"
                fund_result.errors.append(error_msg)
                self.logger.error(error_msg)

            results[fund_name] = fund_result

        return results

    def _create_fund_from_bulk_data(self, fund_name: str, fund_config) -> Fund:
        """Create a Fund instance populated with bulk data"""
        # Get all data for this fund from bulk store
        fund_data_dict = self.data_store.fund_data.get(fund_name, {})

        # Create FundData structure
        fund_data = FundData()

        # Populate current holdings
        fund_data.current = FundHoldings(
            equity=fund_data_dict.get('custodian_equity', pd.DataFrame()),
            options=fund_data_dict.get('custodian_option', pd.DataFrame()),
            treasury=fund_data_dict.get('custodian_treasury', pd.DataFrame()),
            cash=self._extract_cash_value(fund_data_dict.get('cash', pd.DataFrame())),
            nav=self._extract_nav_value(fund_data_dict.get('nav', pd.DataFrame()))
        )

        # Populate previous holdings (you might need to load T-1 data separately)
        fund_data.previous = FundHoldings(
            equity=pd.DataFrame(),  # Placeholder - you'll need to load T-1 data
            options=pd.DataFrame(),
            treasury=pd.DataFrame(),
            cash=0.0,
            nav=0.0
        )

        # Populate additional data
        fund_data.flows = 0.0  # Calculate from flows data if available
        fund_data.expense_ratio = fund_config.expense_ratio
        fund_data.basket = fund_data_dict.get('basket', pd.DataFrame())
        fund_data.index = fund_data_dict.get('index', pd.DataFrame())

        # Create Fund instance
        fund = Fund(fund_name, fund_config.mapping_data)
        fund.data = fund_data

        return fund

    def _extract_cash_value(self, cash_data: pd.DataFrame) -> float:
        """Extract cash value from cash data DataFrame"""
        if cash_data.empty:
            return 0.0
        # Adjust this based on your cash table structure
        return cash_data.get('cash_value', 0).sum() if 'cash_value' in cash_data.columns else 0.0

    def _extract_nav_value(self, nav_data: pd.DataFrame) -> float:
        """Extract NAV value from nav data DataFrame"""
        if nav_data.empty:
            return 0.0
        # Adjust this based on your NAV table structure
        return nav_data.get('nav_value', 0).sum() if 'nav_value' in nav_data.columns else 0.0

    def _run_compliance(self, fund: Fund) -> Dict[str, Any]:
        """Run compliance checks using your existing ComplianceChecker"""
        try:
            # Your ComplianceChecker expects a dict of funds
            compliance_checker = ComplianceChecker(
                session=None,  # No session needed - data is already loaded
                funds={fund.name: fund},  # Pass as single-item dict
                date=self.data_store.date,
                base_cls=None
            )

            # Run all compliance tests
            results = compliance_checker.run_compliance_tests()
            return results.get(fund.name, {})

        except Exception as e:
            self.logger.error(f"Compliance error for {fund.name}: {e}")
            return {'errors': [str(e)], 'violations': []}

    def _run_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        """Run reconciliation using your existing Reconciliator"""
        try:
            # Your Reconciliator expects a Fund instance
            reconciliator = Reconciliator(
                fund=fund,
                analysis_type=self.analysis_type
            )

            # Run all reconciliations
            reconciliator.run_all_reconciliations()
            return reconciliator.get_summary()

        except Exception as e:
            self.logger.error(f"Reconciliation error for {fund.name}: {e}")
            return {'errors': [str(e)], 'breaks': []}

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

    def _get_prior_date(self, current_date: str) -> str:
        """Get prior business date - you might want to improve this"""
        from datetime import datetime, timedelta
        current = datetime.strptime(current_date, '%Y-%m-%d')
        prior = current - timedelta(days=1)
        return prior.strftime('%Y-%m-%d')