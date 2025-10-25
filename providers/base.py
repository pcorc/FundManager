# providers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataRequest:
    """Standardized data request"""
    fund_name: str
    data_type: str  # 'holdings', 'nav', 'cash', 'index_weights'
    date: str
    asset_class: Optional[str] = None
    table_name: Optional[str] = None  # Specific table if needed


class DataProvider(ABC):
    """Base interface for ALL data providers"""

    @abstractmethod
    def get_data(self, request: DataRequest) -> pd.DataFrame:
        """Get data for any request - unified interface"""
        pass

    @abstractmethod
    def supports_data_type(self, data_type: str) -> bool:
        """Check if provider supports this data type"""
        pass


class CustodianProvider(DataProvider):
    """Base for custodian providers (BNY, UMB, SocGen)"""

    @abstractmethod
    def get_holdings(self, request: DataRequest) -> pd.DataFrame:
        """Get holdings data - custodian-specific implementation"""
        pass

    @abstractmethod
    def get_nav(self, request: DataRequest) -> pd.DataFrame:
        """Get NAV data"""
        pass

    @abstractmethod
    def get_cash(self, request: DataRequest) -> pd.DataFrame:
        """Get cash data"""
        pass

    def get_data(self, request: DataRequest) -> pd.DataFrame:
        """Route to appropriate method based on data_type"""
        handlers = {
            'holdings': self.get_holdings,
            'nav': self.get_nav,
            'cash': self.get_cash
        }

        handler = handlers.get(request.data_type)
        if handler:
            return handler(request)
        return pd.DataFrame()

    def supports_data_type(self, data_type: str) -> bool:
        return data_type in ['holdings', 'nav', 'cash']


class IndexProvider(DataProvider):
    """Base for index providers (S&P, NASDAQ, CBOE)"""

    @abstractmethod
    def get_index_weights(self, request: DataRequest) -> pd.DataFrame:
        """Get index constituent weights"""
        pass

    def get_data(self, request: DataRequest) -> pd.DataFrame:
        if request.data_type == 'index_weights':
            return self.get_index_weights(request)
        return pd.DataFrame()

    def supports_data_type(self, data_type: str) -> bool:
        return data_type == 'index_weights'