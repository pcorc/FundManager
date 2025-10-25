# providers/registry.py
from typing import Dict, Type
from providers.base import DataProvider, CustodianProvider, IndexProvider
from providers.custodians.bny import BNYProvider
from providers.custodians.umb import UMBProvider
from providers.custodians.socgen import SocGenProvider
from providers.index_providers.sp import SPIndexProvider
from providers.index_providers.nasdaq import NasdaqIndexProvider


class ProviderRegistry:
    """Manages all data providers - custodians, index providers, etc."""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls
        self._custodian_providers: Dict[str, CustodianProvider] = {}
        self._index_providers: Dict[str, IndexProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all provider instances"""
        # Custodian providers
        custodian_classes = {
            'BNY': BNYProvider,
            'UMB': UMBProvider,
            'SocGen': SocGenProvider
        }

        for name, provider_class in custodian_classes.items():
            self._custodian_providers[name] = provider_class(self.session, self.base_cls)

        # Index providers
        index_classes = {
            'sp_holdings': SPIndexProvider,
            'nasdaq_holdings': NasdaqIndexProvider
        }

        for table_name, provider_class in index_classes.items():
            self._index_providers[table_name] = provider_class(self.session, self.base_cls)

    def get_custodian_provider(self, custodian_type: str) -> CustodianProvider:
        """Get custodian provider by type"""
        return self._custodian_providers.get(custodian_type)

    def get_index_provider(self, index_table: str) -> IndexProvider:
        """Get index provider by table name"""
        return self._index_providers.get(index_table)

    def get_provider_for_request(self, request: DataRequest) -> DataProvider:
        """Get appropriate provider for any data request"""
        if request.data_type == 'index_weights':
            return self.get_index_provider(request.table_name)
        else:
            # For holdings/NAV/cash, determine custodian from fund
            custodian_type = self._map_fund_to_custodian(request.fund_name)
            return self.get_custodian_provider(custodian_type)

    def _map_fund_to_custodian(self, fund_name: str) -> str:
        """Map fund to custodian type"""
        custodian_map = {
            'DOGG': 'BNY',
            'KNG': 'BNY',
            'RDVI': 'BNY',
            'HE3B1': 'UMB',
            'P20127': 'UMB'
            # ... your mapping logic
        }
        return custodian_map.get(fund_name, 'BNY')  # Default to BNY