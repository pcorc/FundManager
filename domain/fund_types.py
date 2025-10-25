# domain/fund_types.py - SPECIALIZATIONS
class ETF(Fund):
    """ETF-specific business logic"""

    def calculate_creation_units(self):
        pass


class MutualFund(Fund):
    """Mutual fund specific logic"""

    def calculate_share_class_nav(self):
        pass


class HedgeFund(Fund):
    """Hedge fund specific logic"""

    def calculate_management_fees(self):
        pass