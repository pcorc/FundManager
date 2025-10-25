# management/fund_manager.py
class FundManager:
    """Orchestrates operations across multiple funds"""

    def __init__(self, data_loader: FundDataLoader, custodian_registry: CustodianRegistry):
        self.data_loader = data_loader
        self.custodian_registry = custodian_registry
        self.funds: Dict[str, Fund] = {}

    def register_fund(self, fund: Fund):
        """Register a fund with the manager"""
        # Inject dependencies
        fund._data_loader = self.data_loader
        fund._custodian_provider = self.custodian_registry.get_provider(fund)
        self.funds[fund.name] = fund

    def run_daily_operations(self, date: date, operations: List[str]) -> Dict[str, Any]:
        """Run specified operations for all funds"""
        results = {}

        for fund_name, fund in self.funds.items():
            fund_results = {}

            if 'compliance' in operations:
                fund_results['compliance'] = fund.check_compliance(date)

            if 'reconciliation' in operations:
                fund_results['reconciliation'] = fund.reconcile_holdings(date)

            if 'nav_reconciliation' in operations:
                fund_results['nav_reconciliation'] = fund.reconcile_nav(date)

            results[fund_name] = fund_results

        return results

    def get_fund(self, name: str) -> Fund:
        """Get fund by name"""
        return self.funds[name]


