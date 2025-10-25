# config/database.py
from sqlalchemy import create_engine, MetaData, Table, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
import os

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "mysql+pymysql://trevorlack:muo1kw1eke77xijp@db-mysql-nyc1-91761-do-user-8924672-0.b.db.ondigitalocean.com:25060/first_trust_usa")


# Create the SQLAlchemy engine
engine = create_engine(DB_CONNECTION_STRING, echo=False)

# Reflect metadata
metadata = MetaData()
Base = automap_base(metadata=metadata)

# Table schemas to reflect
schemas = {
    'accounts_mapping': ['account_numbers', 'funds', 'fund_service_providers', 'fund_recon_mappings'],
    'first_trust_vit': ['bny_vit_holdings', 'bny_vit_nav', 'bny_vit_cash'],
    'first_trust_usa': ['bny_us_holdings_v2', 'bny_us_nav_v2', 'socgen_holdings_statement', 'tif_frwd_data',
                        'master_accounts', 'etf_flows', 'socgen_equity_statement', 'bny_us_cash'],
    'ftcm': ['umb_cef_px', 'umb_cef_nav', 'umb_ftmix_px', 'umb_ftmix_nav', 'umb_cef_cash', 'overlap_test'],
    'mf_fund_data': ['ccva_cash', 'ccva_holdings', 'ccva_nav'],
    'bloomberg_emsx': ['bbg_equity_flds_blotter', 'bbg_options_flds_blotter', 'emsx_equity_route_sub',
                       'emsx_equity_order_sub', 'bbg_feed_equity_closes', 'daily_bbg_flex_pricing'],
    'compliance': ['compliance_daily_results', 'gics_mapper'],
    'pricing_data': ['cboe_holdings', 'sp_cls', 'sp_holdings', 'nasdaq_holdings', 'dogg_index',
                     'tif_index_mappings', 'tif_iiv_index_def'],
    'reconciliation': ['tif_oms_option_holdings', 'tif_oms_equity_holdings', 'tif_oms_treasury_holdings'],
    'calendar': ['settlement_holidays', 'distributions']
}

# Views with synthetic primary keys
schema_views = {
    "accounts_mapping": [
        {
            "name": "v_fund_properties",
            "primary_keys": [{"name": "fund_tickers", "type": String}]
        }
    ]
}

# Reflect tables
def reflect_tables(engine, metadata, schemas):
    for schema, tables in schemas.items():
        metadata.reflect(engine, schema=schema, only=tables)

# Reflect views with manual PKs
def reflect_views(engine, metadata, schema_views):
    for schema, views in schema_views.items():
        for view in views:
            view_name = view["name"]
            pk_defs = [
                Column(pk["name"], pk["type"], primary_key=True)
                for pk in view.get("primary_keys", [])
            ]
            Table(view_name, metadata, *pk_defs, autoload_with=engine, schema=schema)

# Run reflections
reflect_tables(engine, metadata, schemas)
reflect_views(engine, metadata, schema_views)

# Prepare automap base
Base.prepare()

# Session maker
Session = sessionmaker(bind=engine)

def init_session():
    """Initialize and return a new SQLAlchemy session."""
    return Session()
