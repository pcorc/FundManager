# providers/custodians/bny.py
from providers.base import CustodianProvider
import pandas as pd


class BNYProvider(CustodianProvider):
    """BNY Mellon data provider"""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls

    def get_holdings(self, request: DataRequest) -> pd.DataFrame:
        """Get BNY holdings - your existing logic"""
        table = getattr(self.base_cls.classes, request.table_name or 'bny_us_holdings_v2')

        # Your existing BNY query logic
        query = self.session.query(table).filter(
            table.date == request.date,
            table.fund == request.fund_name
        )

        df = pd.read_sql(query.statement, self.session.bind)
        return self._standardize_output(df, request.asset_class)

    def get_nav(self, request: DataRequest) -> pd.DataFrame:
        """Get BNY NAV data"""
        table = getattr(self.base_cls.classes, request.table_name or 'bny_us_nav_v2')

        query = self.session.query(table).filter(
            table.date == request.date
        )

        return pd.read_sql(query.statement, self.session.bind)

    def get_cash(self, request: DataRequest) -> pd.DataFrame:
        """Get BNY cash data"""
        # Your existing BNY cash logic
        pass

    def _standardize_output(self, df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
        """Standardize column names - your existing mapping logic"""
        column_mapping = {
            'equity': {
                'security_sedol': 'sedol',
                'sharespar': 'shares_cust',
                'price_base': 'price',
                'traded_market_value_base': 'market_value'
            },
            'options': {
                'security_description_long_1': 'optticker',
                'sharespar': 'shares_cust',
                'price_base': 'price'
            }
        }

        mapping = column_mapping.get(asset_class, {})
        return df.rename(columns=mapping)