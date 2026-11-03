# config/database.py
"""Database access. The connection string comes from the environment
(``DB_CONNECTION_STRING``), optionally via a gitignored ``.env`` file in the
repo root; see ``.env.example``. Nothing connects at import time: the engine
is built, and connected once, on the first ``initialize_database()`` call.
No tables are reflected; every query is plain SQL against the reconciliation.*
views.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@lru_cache(maxsize=None)
def _session_factory():
    url = os.getenv("DB_CONNECTION_STRING")
    if not url:
        raise RuntimeError("DB_CONNECTION_STRING is not set; copy .env.example to .env and fill it in")
    engine = create_engine(url, echo=False)
    engine.connect().close()  # connect up front, as the former reflection did, so an unreachable DB raises here
    return sessionmaker(bind=engine)


def initialize_database():
    """Return a new SQLAlchemy session and ``None`` (the former automap Base; kept so callers can unpack a pair)."""
    return _session_factory()(), None
