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
from domain.fund import Fund, FundData, FundSnapshot
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
        """Create a Fund instance populated with bulk data"""
        # Get all data for this fund from bulk store
        fund_data_dict = self.data_store.fund_data.get(fund_name, {})

        # Create FundData structure
        fund_data = FundData()

        # Populate current holdings with Vest (OMS) data when available
        fund_data.current = FundSnapshot(
            equity=fund_data_dict.get('vest_equity', pd.DataFrame()),
            options=fund_data_dict.get('vest_option', pd.DataFrame()),
            treasury=fund_data_dict.get('vest_treasury', pd.DataFrame()),
            cash=self._extract_cash_value(fund_data_dict.get('cash', pd.DataFrame())),
            nav=self._extract_nav_per_share(fund_data_dict.get('nav', pd.DataFrame())),
        )

        # Populate previous holdings (T-1) if available
        fund_data.previous = FundSnapshot(
            equity=pd.DataFrame(),
            options=pd.DataFrame(),
            treasury=pd.DataFrame(),
            cash=0.0,
            nav=0.0,
        )

        # Store custodian data separately for reconciliation
        fund_data.custodian_equity = fund_data_dict.get('custodian_equity', pd.DataFrame())
        fund_data.custodian_option = fund_data_dict.get('custodian_option', pd.DataFrame())
        fund_data.custodian_treasury = fund_data_dict.get('custodian_treasury', pd.DataFrame())
        fund_data.vest_equity = fund_data_dict.get('vest_equity', pd.DataFrame())
        fund_data.vest_option = fund_data_dict.get('vest_option', pd.DataFrame())
        fund_data.vest_treasury = fund_data_dict.get('vest_treasury', pd.DataFrame())

        # Populate additional data
        fund_data.flows = 0.0  # Calculate from flows data if available
        fund_data.expense_ratio = fund_config.expense_ratio
        fund_data.basket = fund_data_dict.get('basket', pd.DataFrame())
        fund_data.index = fund_data_dict.get('index', pd.DataFrame())
        fund_data.equity_trades = fund_data_dict.get('equity_trades', pd.DataFrame())
        fund_data.cr_rd_data = fund_data_dict.get('cr_rd', pd.DataFrame())

        # Create Fund instance
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

        cash_columns = ['cash_value', 'cash', 'amount', 'value', 'end_balance']
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

    def _extract_custodian_nav_per_share(self, fund_data_dict: dict) -> float:
        """Extract NAV per share using known field name from your custodian"""
        nav_data = fund_data_dict.get('nav_per_share', 0.0)

        try:
            if hasattr(nav_data, 'iloc'):  # Handle DataFrame/Series
                return float(nav_data.iloc[0]) if len(nav_data) > 0 else 0.0
            return float(nav_data)
        except (ValueError, TypeError):
            return 0.0

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

    def _run_compliance(self, fund: Fund) -> Dict[str, Any]:
        """Run compliance checks"""
        try:
            compliance_checker = ComplianceChecker(
                session=None,
                funds={fund.name: fund},
                date=self.data_store.date,
                base_cls=None
            )
            results = compliance_checker.run_compliance_tests()
            return results.get(fund.name, {})
        except Exception as e:
            self.logger.error(f"Compliance error for {fund.name}: {e}")
            return {'errors': [str(e)], 'violations': []}

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


    def _extract_nav_per_share(self, nav_source, custodian_type=None):
        """Extract NAV per share from either a mapping or direct DataFrame."""
        if isinstance(nav_source, dict):
            nav_data = nav_source.get('custodian_nav', pd.DataFrame())
        else:
            nav_data = nav_source

        if not isinstance(nav_data, pd.DataFrame) or nav_data.empty:
            return 0.0

        row = nav_data.iloc[0]

        # Custodian-specific column mapping
        custodian_columns = {
            'bny': 'nav',
            'umb': 'cntnetvalue',
            'socgen': 'nav_per_share'  # if you have SocGen
        }

        # If we know the custodian, try their specific column first
        if custodian_type and custodian_type in custodian_columns:
            custodian_col = custodian_columns[custodian_type]
            if custodian_col in row and pd.notna(row[custodian_col]):
                try:
                    return float(row[custodian_col])
                except (ValueError, TypeError):
                    pass

        # Fallback: try all common column names
        for col in ['nav_per_share', 'nav', 'cntnetvalue', 'navps']:
            if col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    continue

        return 0.0