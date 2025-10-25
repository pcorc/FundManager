# providers/custodians/umb.py
from providers.base import CustodianProvider


class UMBProvider(CustodianProvider):
    """UMB data provider"""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls

    def get_holdings(self, request: DataRequest) -> pd.DataFrame:
        """UMB-specific holdings logic"""
        # Your existing UMB logic
        pass

    def get_nav(self, request: DataRequest) -> pd.DataFrame:
        """UMB-specific NAV logic"""
        pass

    def get_cash(self, request: DataRequest) -> pd.DataFrame:
        """UMB-specific cash logic"""
        pass