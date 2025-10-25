# services/baserrr.py
class BusinessService(ABC):
    """Base class for all business services"""

    def __init__(self, fund: Fund):
        self.fund = fund


# services/reconciliation.py
class HoldingsReconciler(BusinessService):
    """Handles holdings reconciliation business logic"""

    def reconcile(self, date: date) -> 'ReconciliationResult':
        current_snapshot = self.fund.get_snapshot(date)
        prior_snapshot = self.fund.get_prior_snapshot(date)

        # Compare Vest vs Custodian vs Index
        vest_data = current_snapshot.holdings
        custodian_data = self._load_custodian_data(date)
        index_data = self._load_index_data(date)

        discrepancies = self._find_discrepancies(vest_data, custodian_data)
        price_breaks = self._find_price_breaks(vest_data, custodian_data)

        return ReconciliationResult(discrepancies, price_breaks)


# services/compliance.py
class ComplianceChecker(BusinessService):
    """Handles compliance checking business logic"""

    def check(self, date: date) -> 'ComplianceResult':
        snapshot = self.fund.get_snapshot(date)

        results = {}
        results['diversification'] = self._check_diversification(snapshot)
        results['concentration'] = self._check_concentration(snapshot)
        results['liquidity'] = self._check_liquidity(snapshot)

        return ComplianceResult(results)


# services/nav_reconciliation.py
class NAVReconciler(BusinessService):
    """Handles NAV reconciliation business logic"""

    def reconcile(self, date: date) -> 'NAVReconciliationResult':
        current = self.fund.get_snapshot(date)
        prior = self.fund.get_prior_snapshot(date)

        # Calculate components using Fund's methods
        gain_loss = current.calculate_gain_loss(prior)
        dividends = self._calculate_dividends(current, prior)
        expenses = self._calculate_expenses(current)
        flows = self._calculate_flows(current, prior)

        expected_nav = self._calculate_expected_nav(prior, gain_loss, dividends, expenses, flows)
        actual_nav = current.nav

        return NAVReconciliationResult(expected_nav, actual_nav, gain_loss)


class GainLossCalculator:
    """Pure calculator class - no state"""

    @staticmethod
    def calculate(current: FundSnapshot, prior: FundSnapshot) -> 'GainLossResult':
        """Calculate gain/loss across all asset classes"""
        equity_gl = GainLossCalculator._calculate_equity_gl(current, prior)
        option_gl = GainLossCalculator._calculate_option_gl(current, prior)
        treasury_gl = GainLossCalculator._calculate_treasury_gl(current, prior)

        return GainLossResult(equity_gl, option_gl, treasury_gl)