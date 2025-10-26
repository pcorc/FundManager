# main.py - SIMPLIFIED
from datetime import datetime
from config.database import initialize_database
from config.fund_registry import FundRegistry
from processing.bulk_data_loader import BulkDataLoader
from processing.fund_manager import FundManager
from services.weight_analyzer import WeightAnalyzer
import os

def main():
    # 🎛️ CONFIGURATION - TOGGLE HERE!
    TARGET_DATE = datetime(2025, 10, 24).date()  # ⚡ CHANGE DATE HERE
    TARGET_FUNDS = ['DOGG']  # ⚡ CHANGE FUNDS HERE (empty = all)
    OPERATIONS = ['compliance', 'reconciliation', 'nav_reconciliation']  # ⚡ CHANGE OPS HERE
    OPERATIONS = ['compliance']

    OUTPUT_DIR = './reports'
    OUTPUT_DIR = os.getenv("EXPORT_PATH", "G:/Shared drives/Portfolio Management/Funds/Archive/Daily_Compliance")

    # STEP 1: Initialize database
    session, Base = initialize_database()

    try:
        # STEP 2: Load fund configurations
        registry = FundRegistry.from_database(session, Base)

        # STEP 3: Filter funds if specific ones requested
        if TARGET_FUNDS:
            filtered_funds = {k: v for k, v in registry.funds.items() if k in TARGET_FUNDS}
            registry.funds = filtered_funds
            print(f"🔍 Filtered to {len(registry.funds)} funds")

        # STEP 4: Bulk load data
        bulk_loader = BulkDataLoader(session, Base, registry)
        data_store = bulk_loader.load_all_data_for_date(TARGET_DATE)

    finally:
        session.close()  # Close database - done with SQL!

    # STEP 5: Process everything from memory
    fund_manager = FundManager(registry, data_store)
    results = fund_manager.run_daily_operations(OPERATIONS)

    # STEP 6: Generate reports
    report_orchestrator = ReportOrchestrator(OUTPUT_DIR)
    # report_orchestrator.generate_reports(results, TARGET_DATE)

    print(f"✅ Done! Processed {len(results.fund_results)} funds")
    print(f"📊 Summary: {results.summary}")


if __name__ == "__main__":
    main()