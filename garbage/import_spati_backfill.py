import os
import pandas as pd
from sqlalchemy import text
from config.database import engine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spati_backfill.csv")
TARGET_TABLE = "cboe_holdings"
TARGET_SCHEMA = "pricing_data"

# ⚠️  DANGER: set to True only if you intend to delete existing rows before insert.
#    Rows between DELETE_START and DELETE_END will be permanently removed.
ENABLE_DELETE = False
DELETE_START = "2025-10-20"
DELETE_END = "2026-04-10"


def import_spati_backfill(csv_path: str = CSV_PATH) -> None:
    # --- Load CSV -------------------------------------------------------
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    # Drop unused columns if present
    df = df.drop(columns=[c for c in ["options_PO", "IndexValue"] if c in df.columns])

    # Normalize column names
    df = df.rename(columns={"TICKER": "ticker"})

    # Normalize date columns to datetime.date for MySQL compatibility
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date

    # strike is NOT NULL in schema — 0 used as placeholder for index rows with no strike
    if "strike" not in df.columns:
        df["strike"] = 0

    # Confirm expected columns are present
    expected = {"date", "expiration_date", "ticker", "stock_weight", "put_call", "index_name", "price", "strike"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    print(f"  date range in file : {df['date'].min()} -> {df['date'].max()}")
    print(f"  unique dates       : {df['date'].nunique()}")
    print(f"  unique tickers     : {df['ticker'].nunique()}")

    with engine.begin() as conn:
        # --- Delete existing rows in range (disabled by default) --------
        if ENABLE_DELETE:
            delete_sql = text(f"""
                DELETE FROM {TARGET_SCHEMA}.{TARGET_TABLE}
                WHERE date BETWEEN :start AND :end
            """)
            result = conn.execute(delete_sql, {"start": DELETE_START, "end": DELETE_END})
            print(f"  ⚠️  Deleted {result.rowcount:,} existing rows from {TARGET_SCHEMA}.{TARGET_TABLE}")
        else:
            print("  Delete skipped (ENABLE_DELETE=False)")

        # --- Insert -----------------------------------------------------
        df.to_sql(
            name=TARGET_TABLE,
            con=conn,
            schema=TARGET_SCHEMA,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        print(f"  Inserted {len(df):,} rows into {TARGET_SCHEMA}.{TARGET_TABLE}")

    print("Done.")


if __name__ == "__main__":
    import_spati_backfill()
