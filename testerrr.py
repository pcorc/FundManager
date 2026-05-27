"""Dump reconciliation views to CSV for a fund / date window / mode."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sqlalchemy import text, bindparam

from config.database import initialize_database


# ---- knobs ----------------------------------------------------------------
FUND = "P2726"
DATES = ("2026-05-14", "2026-05-13")
MODE = "eod"          # "eod" | "compliance" | "reconciliation" | "nav" | "trading_compliance"
# ---------------------------------------------------------------------------

# Which OMS analysis_type each mode reconciles against.
ANALYSIS_TYPE_BY_MODE = {
    "eod":                "eod",
    "compliance":         "eod",
    "reconciliation":     "eod",
    "nav":                "eod",
    "trading_compliance": "ex_post",
}

ANALYSIS_TYPE = ANALYSIS_TYPE_BY_MODE.get(MODE, "eod")
OUT_DIR = Path("./query_dumps") / f"{FUND}_{MODE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (filename, sql, needs_analysis_type)
QUERIES = {
    "fund_metadata":
        ("SELECT * FROM reconciliation.v_fund_metadata WHERE fund_ticker = :fund", False),

    "custodian_equity":
        ("SELECT * FROM reconciliation.v_custodian_equity "
         "WHERE fund_ticker = :fund AND `date` IN :dates ORDER BY `date`, eqyticker", False),

    "custodian_option":
        ("SELECT * FROM reconciliation.v_custodian_option "
         "WHERE fund_ticker = :fund AND `date` IN :dates ORDER BY `date`, optticker", False),

    "vest_equity":
        ("SELECT * FROM reconciliation.v_vest_equity "
         "WHERE fund_ticker = :fund AND `date` IN :dates AND analysis_type = :atype "
         "ORDER BY `date`, eqyticker", True),

    "vest_option":
        ("SELECT * FROM reconciliation.v_vest_option "
         "WHERE fund_ticker = :fund AND `date` IN :dates AND analysis_type = :atype "
         "ORDER BY `date`, optticker", True),

    "custodian_nav":
        ("SELECT * FROM reconciliation.v_custodian_nav "
         "WHERE fund_ticker = :fund AND `date` IN :dates ORDER BY `date`", False),

    "custodian_cash":
        ("SELECT * FROM reconciliation.v_custodian_cash "
         "WHERE fund_ticker = :fund AND `date` IN :dates ORDER BY `date`", False),

    "recon_equity":
        ("SELECT * FROM reconciliation.v_recon_custodian_equity "
         "WHERE fund_ticker = :fund AND `date` IN :dates "
         "ORDER BY `date`, discrepancy_type, eqyticker", False),

    "recon_option":
        ("SELECT * FROM reconciliation.v_recon_custodian_option "
         "WHERE fund_ticker = :fund AND `date` IN :dates "
         "ORDER BY `date`, discrepancy_type, optticker", False),
}


def main() -> None:
    session, _ = initialize_database()
    print(f"Fund={FUND}  mode={MODE}  analysis_type={ANALYSIS_TYPE}  dates={DATES}")
    try:
        for name, (sql, needs_atype) in QUERIES.items():
            params = {"fund": FUND, "dates": DATES}
            if needs_atype:
                params["atype"] = ANALYSIS_TYPE
            stmt = text(sql)
            if ":dates" in sql:
                stmt = stmt.bindparams(bindparam("dates", expanding=True))
            df = pd.read_sql(stmt, session.bind, params=params)
            out_path = OUT_DIR / f"{FUND}_{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {name:20s} -> {out_path}  ({len(df)} rows)")
    finally:
        session.close()


if __name__ == "__main__":
    main()