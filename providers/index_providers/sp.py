# providers/index_providers/sp.py
from providers.base import IndexProvider


class SPIndexProvider(IndexProvider):
    """S&P index data provider"""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls

    def get_index_weights(self, request: DataRequest) -> pd.DataFrame:
        """Get S&P index weights"""
        table = getattr(self.base_cls.classes, 'sp_holdings')

        query = self.session.query(table).filter(
            table.date == request.date
        )

        return pd.read_sql(query.statement, self.session.bind)