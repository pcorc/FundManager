# main.py - simplified usage
# Choose mode
# from config.operation_config import OperationMode, OperationConfig
# config = OperationConfig.create(OperationMode.EOD)

# Process
# from processing.fund_processor import FundProcessor
# processor = FundProcessor(session, Base, funds_config)
# result = processor.process_date(date, config)

# Report
# from reporting.report_orchestrator import ReportOrcA
# orchestrator = ReportOrchestrator(output_dir, session, Base, gics_dict)
# orchestrator.generate_reports(result, config)


# main.py
import datetime
from config.database import init_session, Base
from config.fund_registry import FundRegistry
from data.unified_data_access import UnifiedDataAccess
from processing.data_loader import DataLoader
from services.compliance_checker import ComplianceChecker
from services.weight_analyzer import WeightAnalyzer


def run_compliance(fund_name: str, date: datetime.date, analysis_type: str = "eod"):
    """
    Run compliance for a single fund on a specific date.

    Args:
        fund_name: Fund ticker (e.g., "DOGG")
        date: Analysis date
        analysis_type: "eod" or "ex_post"
    """
    print(f"🚀 Running compliance for {fund_name} on {date} ({analysis_type})")

    # 1. Initialize database
    session = init_session()

    try:
        # 2. Load fund configuration
        fund_registry = FundRegistry(session, Base)
        fund_config = fund_registry.get_config(fund_name)

        if not fund_config:
            print(f"❌ Fund {fund_name} not found in configuration")
            return None

        # 3. Initialize data access
        data_access = UnifiedDataAccess(session, Base, fund_registry)

        # 4. Load all required data
        print("📊 Loading fund data...")
        data_loader = DataLoader(data_access, fund_registry)
        fund_data = data_loader.load_fund_data(fund_name, date, analysis_type)

        if fund_data.is_empty():
            print(f"❌ No data found for {fund_name} on {date}")
            return None

        # 5. Run weight analysis (for GICS compliance)
        print("⚖️ Calculating weights...")
        weight_analyzer = WeightAnalyzer()
        weight_analysis = weight_analyzer.analyze_weights(fund_name, fund_data)
        fund_data.weight_analysis = weight_analysis

        # 6. Run compliance checks
        print("🔍 Running compliance checks...")
        compliance_checker = ComplianceChecker()
        compliance_results = compliance_checker.run_compliance(fund_name, fund_data, analysis_type)

        # 7. Display results
        print("\n📋 COMPLIANCE RESULTS:")
        print(f"Fund: {fund_name}")
        print(f"Date: {date}")
        print(f"Analysis Type: {analysis_type}")
        print("-" * 50)

        for test_name, result in compliance_results.items():
            status = "✅ PASS" if result.get('compliant', False) else "❌ FAIL"
            print(f"{test_name}: {status}")

            # Show details for failures
            if not result.get('compliant', False):
                if 'calculations' in result:
                    print(f"   Details: {result['calculations']}")

        return compliance_results

    except Exception as e:
        print(f"❌ Error running compliance: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        session.close()


if __name__ == "__main__":
    # Example usage - modify these parameters
    FUND_NAME = "DOGG"  # Change this to test different funds
    ANALYSIS_DATE = datetime.date(2025, 10, 24)  # Use a date with data
    ANALYSIS_TYPE = "eod"  # or "ex_post"

    results = run_compliance(FUND_NAME, ANALYSIS_DATE, ANALYSIS_TYPE)
    print("done")