from domain.fund import Fund  # ← ADD THIS IMPORT


# services/nav_reconciliator.py
class NAVReconciliator:
    def __init__(self, fund: Fund, analysis_date: str, prior_date: str):
        self.fund = fund
        self.analysis_date = analysis_date
        self.prior_date = prior_date
        self.results = {}

    def run_nav_reconciliation(self):
        """Much simpler - uses Fund's G/L calculations"""
        # Calculate G/L for all asset types using Fund methods
        equity_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, 'equity')
        options_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, 'options')
        flex_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, 'flex_options')
        treasury_gl = self.fund.calculate_gain_loss(self.analysis_date, self.prior_date, 'treasury')

        # Get other components
        dividends = self.fund.get_dividends(self.analysis_date)
        expenses = self.fund.get_expenses(self.analysis_date)
        distributions = self.fund.get_distributions(self.analysis_date)
        flows_adjustment = self.fund.get_flows_adjustment(self.analysis_date, self.prior_date)

        # Calculate expected NAV
        results = self._calculate_expected_nav(
            equity_gl.adjusted_gl,
            options_gl.adjusted_gl,
            flex_gl.adjusted_gl,
            treasury_gl.adjusted_gl,
            dividends, expenses, distributions, flows_adjustment
        )

        # Store detailed calculations from Fund
        self.results = {
            'summary': results,
            'detailed_calculations': {
                'equity': equity_gl.details,
                'options': options_gl.details,
                'flex_options': flex_gl.details,
                'treasury': treasury_gl.details
            }
        }

        return self.results

    def run_nav_reconciliation_v2(self):
        # Compare custodian vs your calculation
        custodian_assets = self.fund.custodian_total_assets
        calculated_assets = self.fund.calculated_total_assets
        assets_difference = custodian_assets - calculated_assets

        custodian_net_assets = self.fund.custodian_total_net_assets
        calculated_net_assets = self.fund.calculated_total_net_assets
        net_assets_difference = custodian_net_assets - calculated_net_assets

        return {
            'assets_reconciliation': {
                'custodian': custodian_assets,
                'calculated': calculated_assets,
                'difference': assets_difference
            },
            'net_assets_reconciliation': {
                'custodian': custodian_net_assets,
                'calculated': calculated_net_assets,
                'difference': net_assets_difference
            }
        }