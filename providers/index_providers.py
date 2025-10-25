# utilities/index_manager.py - IMPROVED
import pandas as pd
from sqlalchemy import literal
import logging


class IndexManager:
    def __init__(self, session, base_cls=None):
        self.session = session
        self.base_cls = base_cls
        self.logger = logging.getLogger(__name__)

    def get_index_data(self, index_table, date, fund_name, account_number_index=None):
        """
        Retrieve index data for any index provider.
        """
        if not index_table:
            self.logger.warning(f"No index table specified for fund {fund_name}")
            return pd.DataFrame()

        try:
            # Map table names to methods
            provider_map = {
                'nasdaq_holdings': self._get_nasdaq_holdings,
                'sp_holdings': self._get_sp_holdings,
                'cboe_holdings': self._get_cboe_holdings,
                'dogg_index': self._get_dogg_index
            }

            if index_table in provider_map:
                return provider_map[index_table](date, account_number_index)
            else:
                self.logger.error(f"Unknown index table: {index_table}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error retrieving {index_table} data: {e}")
            return pd.DataFrame()


    def _get_nasdaq_holdings(self, date, account_number_index):
        """Get NASDAQ index holdings"""
        query = (self.session.query(
            self.base_cls.classes.nasdaq_holdings.date.label('date'),
            self.base_cls.classes.nasdaq_holdings.fund.label('fund'),
            self.base_cls.classes.nasdaq_holdings.ticker.label('equity_ticker'),
            self.base_cls.classes.nasdaq_holdings.index_weight.label('weight_index'),
            self.base_cls.classes.nasdaq_holdings.price.label('price_index'),
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_SECTOR_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_GROUP_NAME
        ).join(
            self.base_cls.classes.bbg_equity_flds_blotter, self.base_cls.classes.nasdaq_holdings.ticker == self.base_cls.classes.bbg_equity_flds_blotter.TICKER
        ).filter(
            self.base_cls.classes.nasdaq_holdings.date == date,
            self.base_cls.classes.nasdaq_holdings.fund == account_number_index,
            self.base_cls.classes.nasdaq_holdings.time_of_day == 'EOD'
        ).group_by(
            self.base_cls.classes.nasdaq_holdings.ticker
        ))

        x=pd.read_sql(query.statement, self.session.bind)

        return pd.read_sql(query.statement, self.session.bind)

    def _get_sp_holdings(self, date, account_number_index):
        """Get S&P index holdings"""
        query = self.session.query(
            self.base_cls.classes.sp_holdings.EFFECTIVE_DATE.label('date'),
            self.base_cls.classes.sp_holdings.INDEX_CODE.label('fund'),
            self.base_cls.classes.sp_holdings.TICKER.label('equity_ticker'),
            self.base_cls.classes.sp_holdings.INDEX_WEIGHT.label('weight_index'),
            self.base_cls.classes.sp_holdings.LOCAL_PRICE.label('price_index'),
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_SECTOR_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_GROUP_NAME
        ).join(
            self.base_cls.classes.bbg_equity_flds_blotter, self.base_cls.classes.sp_holdings.TICKER == self.base_cls.classes.bbg_equity_flds_blotter.TICKER
        ).filter(
            self.base_cls.classes.sp_holdings.EFFECTIVE_DATE == date,
            self.base_cls.classes.sp_holdings.INDEX_CODE == account_number_index
        )
        x = pd.read_sql(query.statement, self.session.bind)
        return pd.read_sql(query.statement, self.session.bind)

    def _get_cboe_holdings(self, date):
        """Get CBOE index holdings"""
        query = self.session.query(
            self.base_cls.classes.cboe_holdings.date.label('date'),
            self.base_cls.classes.cboe_holdings.index_name.label('fund'),
            self.base_cls.classes.cboe_holdings.ticker.label('equity_ticker'),
            self.base_cls.classes.cboe_holdings.stock_weight.label('weight_index'),
            self.base_cls.classes.cboe_holdings.price.label('price_index'),
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_SECTOR_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_GROUP_NAME
        ).join(
            self.base_cls.classes.bbg_equity_flds_blotter, self.base_cls.classes.cboe_holdings.ticker == self.base_cls.classes.bbg_equity_flds_blotter.TICKER
        ).filter(
            self.base_cls.classes.cboe_holdings.date == date,
            self.base_cls.classes.cboe_holdings.index_name == 'SPATI'
        )
        return pd.read_sql(query.statement, self.session.bind)

    def _get_dogg_index(self, date):
        """Get DOGG index holdings"""
        query = self.session.query(
            self.base_cls.classes.dogg_index.DATE.label('date'),
            self.base_cls.classes.dogg_index.TICKER.label('equity_ticker'),
            literal(0.10).label('weight_index'),
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_SECTOR_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_NAME,
            self.base_cls.classes.bbg_equity_flds_blotter.GICS_INDUSTRY_GROUP_NAME
        ).join(
            self.base_cls.classes.bbg_equity_flds_blotter, self.base_cls.classes.dogg_index.TICKER == self.base_cls.classes.bbg_equity_flds_blotter.TICKER
        ).filter(
            self.base_cls.classes.dogg_index.DATE == date
        )
        return pd.read_sql(query.statement, self.session.bind)