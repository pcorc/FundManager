# config/database.py
"""Database access. The connection string comes from the environment
(``DB_CONNECTION_STRING``), optionally via a gitignored ``.env`` file in the
repo root; see ``.env.example``. Nothing connects at import time: the engine
and reflected metadata are built on the first ``initialize_database()`` call.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Column, MetaData, String, Table, create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Table schemas to reflect
schemas = {
    'accounts_mapping': ['account_numbers', 'funds', 'fund_service_providers'], #fund_recon_mappings
    'first_trust_vit': ['bny_vit_holdings', 'bny_vit_nav', 'bny_vit_cash'],
    'first_trust_usa': ['bny_us_holdings_v2', 'bny_us_nav_v2', 'socgen_holdings_statement', 'tif_frwd_data',
                        'master_accounts', 'etf_flows', 'socgen_equity_statement', 'bny_us_cash'],
    'ftcm': ['umb_cef_px', 'umb_cef_nav', 'umb_ftmix_px', 'umb_ftmix_nav', 'umb_cef_cash', 'overlap_test'],
    'mf_fund_data': ['ccva_cash', 'ccva_holdings', 'ccva_nav'],
    'bloomberg_emsx': ['bbg_equity_flds_blotter', 'bbg_options_flds_blotter', 'emsx_equity_route_sub',
                       'emsx_equity_order_sub', 'bbg_feed_equity_closes', 'daily_bbg_flex_pricing'],
    'compliance': ['compliance_daily_results', 'gics_mapper'],
    'pricing_data': ['cboe_holdings', 'sp_cls', 'sp_holdings', 'nasdaq_holdings', 'nasdaq_pro', 'dogg_index',
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


def reflect_tables(engine, metadata, schemas):
    for schema, tables in schemas.items():
        metadata.reflect(engine, schema=schema, only=tables)


def reflect_views(engine, metadata, schema_views):
    for schema, views in schema_views.items():
        for view in views:
            pk_defs = [Column(pk["name"], pk["type"], primary_key=True) for pk in view.get("primary_keys", [])]
            Table(view["name"], metadata, *pk_defs, autoload_with=engine, schema=schema)


@lru_cache(maxsize=None)
def _session_factory():
    url = os.getenv("DB_CONNECTION_STRING")
    if not url:
        raise RuntimeError("DB_CONNECTION_STRING is not set; copy .env.example to .env and fill it in")
    engine = create_engine(url, echo=False)
    metadata = MetaData()
    base = automap_base(metadata=metadata)
    reflect_tables(engine, metadata, schemas)
    reflect_views(engine, metadata, schema_views)
    base.prepare()
    return sessionmaker(bind=engine), base


def initialize_database():
    """Return a new SQLAlchemy session and the automap Base."""
    session_cls, base = _session_factory()
    return session_cls(), base
