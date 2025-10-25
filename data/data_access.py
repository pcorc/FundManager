# data/data_access.py
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract data source (Vest, Custodian, Index)"""

    @abstractmethod
    def get_holdings(self, fund: Fund, date: date) -> Holdings:
        pass

    @abstractmethod
    def get_nav(self, fund: Fund, date: date) -> float:
        pass


class VestDataSource(DataSource):
    """Internal Vest OMS data"""

    def get_holdings(self, fund: Fund, date: date) -> Holdings:
        # Your existing tif_oms_* table queries
        pass


class CustodianDataSource(DataSource):
    """Custodian data with provider abstraction"""

    def __init__(self, provider: 'CustodianProvider'):
        self.provider = provider

    def get_holdings(self, fund: Fund, date: date) -> Holdings:
        return self.provider.get_holdings(fund, date)


class IndexDataSource(DataSource):
    """Index provider data"""

    def get_holdings(self, fund: Fund, date: date) -> Holdings:
        # Index constituent data
        pass


class FundDataLoader:
    """Orchestrates data loading from multiple sources"""

    def __init__(self, sources: Dict[str, DataSource]):
        self.sources = sources

    def load_fund_data(self, fund: Fund, date: date) -> Dict[str, Holdings]:
        """Load data from all sources for reconciliation"""
        return {
            'vest': self.sources['vest'].get_holdings(fund, date),
            'custodian': self.sources['custodian'].get_holdings(fund, date),
            'index': self.sources['index'].get_holdings(fund, date)
        }