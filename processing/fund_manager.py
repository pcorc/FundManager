# management/fund_manager.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import date
import logging

from processing.bulk_data_loader import BulkDataStore, DataStoreAccess
from domain.fund import Fund
from processing.compliance_engine import ComplianceEngine
from processing.reconciliation_engine import ReconciliationEngine
from processing.nav_engine import NAVEngine


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
class DailyResults:
    """Results from processing all funds for a day"""
    date: str
    fund_results: Dict[str, FundResult]
    summary: Dict[str, Any] = None

    def __post_init__(self):
        if self.summary is None:
            self.summary = {}
        if self.fund_results is None:
            self.fund_results = {}


class FundManager:
    """Manages processing for ALL funds using bulk-loaded data"""

    def __init__(self, fund_registry, data_store: BulkDataStore):
        self.fund_registry = fund_registry
        self.data_store = data_store
        self.data_access = DataStoreAccess(data_store)
        self.logger = logging.getLogger(__name__)

        # Initialize engines (they work with cached data)
        self.compliance_engine = ComplianceEngine(self.data_access)
        self.reconciliation_engine = ReconciliationEngine(self.data_access)
        self.nav_engine = NAVEngine(self.data_access)

        # Track which funds we're processing
        self.available_funds = self.data_access.get_all_funds_with_data()

    def run_daily_operations(self, operations: List[str]) -> DailyResults:
        """Run specified operations for ALL funds using cached data"""
        daily_results = DailyResults(date=self.data_store.date)

        for fund_name in self.available_funds:
            self.logger.info(f"Processing {fund_name}...")

            fund_result = FundResult(fund_name=fund_name)

            try:
                # Get fund domain object
                fund = self.fund_registry.get_fund(fund_name)
                if not fund:
                    fund_result.errors.append(f"Fund not found in registry")
                    continue

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

            daily_results.fund_results[fund_name] = fund_result

        # Generate summary statistics
        daily_results.summary = self._generate_summary(daily_results)

        return daily_results

    def _run_compliance(self, fund: Fund) -> Dict[str, Any]:
        """Run compliance checks using cached data"""
        try:
            # Get all necessary data from cache
            fund_data = {
                'custodian_equity': self.data_access.get_fund_data(fund.name, 'custodian_equity'),
                'custodian_options': self.data_access.get_fund_data(fund.name, 'custodian_option'),
                'custodian_treasury': self.data_access.get_fund_data(fund.name, 'custodian_treasury'),
                'index': self.data_access.get_fund_data(fund.name, 'index'),
                'nav': self.data_access.get_fund_data(fund.name, 'nav')
            }

            return self.compliance_engine.run_checks(fund, fund_data)

        except Exception as e:
            self.logger.error(f"Compliance error for {fund.name}: {e}")
            return {'errors': [str(e)], 'violations': []}

    def _run_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        """Run holdings reconciliation using cached data"""
        try:
            # Get reconciliation data from cache
            reconciliation_data = {
                'custodian_equity': self.data_access.get_fund_data(fund.name, 'custodian_equity'),
                'custodian_options': self.data_access.get_fund_data(fund.name, 'custodian_option'),
                'custodian_treasury': self.data_access.get_fund_data(fund.name, 'custodian_treasury'),
                'vest_equity': self.data_access.get_fund_data(fund.name, 'vest_equity'),
                'vest_options': self.data_access.get_fund_data(fund.name, 'vest_option'),
                'vest_treasury': self.data_access.get_fund_data(fund.name, 'vest_treasury')
            }

            return self.reconciliation_engine.reconcile(fund, reconciliation_data)

        except Exception as e:
            self.logger.error(f"Reconciliation error for {fund.name}: {e}")
            return {'errors': [str(e)], 'breaks': []}

    def _run_nav_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        """Run NAV reconciliation using cached data"""
        try:
            # Get NAV data from cache
            nav_data = {
                'custodian_nav': self.data_access.get_fund_data(fund.name, 'nav'),
                'custodian_cash': self.data_access.get_fund_data(fund.name, 'cash'),
                'custodian_equity': self.data_access.get_fund_data(fund.name, 'custodian_equity'),
                'custodian_options': self.data_access.get_fund_data(fund.name, 'custodian_option'),
                'custodian_treasury': self.data_access.get_fund_data(fund.name, 'custodian_treasury')
            }

            return self.nav_engine.reconcile(fund, nav_data)

        except Exception as e:
            self.logger.error(f"NAV reconciliation error for {fund.name}: {e}")
            return {'errors': [str(e)], 'differences': []}

    def _generate_summary(self, daily_results: DailyResults) -> Dict[str, Any]:
        """Generate summary statistics for the day"""
        total_funds = len(daily_results.fund_results)
        funds_with_errors = sum(1 for r in daily_results.fund_results.values() if r.errors)

        # Compliance summary
        compliance_violations = 0
        for result in daily_results.fund_results.values():
            if result.compliance_results and 'violations' in result.compliance_results:
                compliance_violations += len(result.compliance_results['violations'])

        # Reconciliation summary
        reconciliation_breaks = 0
        for result in daily_results.fund_results.values():
            if result.reconciliation_results and 'breaks' in result.reconciliation_results:
                reconciliation_breaks += len(result.reconciliation_results['breaks'])

        return {
            'total_funds_processed': total_funds,
            'funds_with_errors': funds_with_errors,
            'compliance_violations': compliance_violations,
            'reconciliation_breaks': reconciliation_breaks,
            'success_rate': ((total_funds - funds_with_errors) / total_funds * 100) if total_funds > 0 else 0
        }

    def get_fund_data_availability(self) -> Dict[str, List[str]]:
        """Check what data is available for each fund"""
        availability = {}
        for fund_name in self.available_funds:
            availability[fund_name] = self.data_access.get_available_data_types(fund_name)
        return availability

    def validate_data_completeness(self) -> Dict[str, List[str]]:
        """Validate that we have all required data for each fund"""
        issues = {}

        for fund_name in self.available_funds:
            fund = self.fund_registry.get_fund(fund_name)
            if not fund:
                continue

            missing_data = []
            available_types = self.data_access.get_available_data_types(fund_name)

            # Check required data types based on fund type
            required_data = fund.get_required_data_types()
            for data_type in required_data:
                if data_type not in available_types:
                    missing_data.append(data_type)

            if missing_data:
                issues[fund_name] = missing_data

        return issues