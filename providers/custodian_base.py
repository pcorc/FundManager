# custodians/base.py
class CustodianProvider(ABC):
    """Abstract custodian provider"""

    @abstractmethod
    def get_holdings(self, fund: Fund, date: date) -> Holdings:
        pass

    @abstractmethod
    def get_nav(self, fund: Fund, date: date) -> float:
        pass

    @abstractmethod
    def get_cash(self, fund: Fund, date: date) -> float:
        pass



# custodians/registry.py
class CustodianRegistry:
    """Manages custodian instances"""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls
        self._providers = {}

    def get_provider(self, fund: Fund) -> CustodianProvider:
        """Get appropriate custodian for fund"""
        custodian_type = self._map_fund_to_custodian(fund.name)

        if custodian_type not in self._providers:
            self._providers[custodian_type] = self._create_provider(custodian_type)

        return self._providers[custodian_type]

    def _create_provider(self, custodian_type: str) -> CustodianProvider:
        """Factory method for custodian providers"""
        providers = {
            'BNY': BNYCustodian,
            'UMB': UMBCustodian,
            'SocGen': SocGenCustodian
        }
        provider_class = providers.get(custodian_type)
        return provider_class(self.session, self.base_cls)