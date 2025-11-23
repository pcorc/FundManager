# management/fund_manager.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from datetime import date, datetime, timedelta
import pandas as pd
import logging

# Import your existing services
from services.compliance_checker import ComplianceChecker
from services.nav_reconciliator import NAVReconciliator
from services.reconciliator import Reconciliator

# Import your domain classes
from domain.fund import Fund, FundData, FundSnapshot, FundHoldings
from processing.bulk_data_loader import BulkDataStore


@dataclass
class FundResult:
    """Results from processing a single fund"""
    fund_name: str
    compliance_results: Dict[str, Any] = None
    reconciliation_results: Dict[str, Any] = None
    nav_results: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ProcessingResults:
    """Container for the outcome of the daily operations."""
    fund_results: Dict[str, FundResult]
    summary: Dict[str, Any]


class FundManager:
    """Manages processing for ALL funds using bulk-loaded data"""

    def __init__(self, fund_registry, data_store: BulkDataStore, analysis_type: str = "eod"):
        self.fund_registry = fund_registry
        self.data_store = data_store
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(__name__)
        self.available_funds = list(data_store.loaded_funds)

    def run_daily_operations(
            self, operations: List[str], *, compliance_tests: Optional[Sequence[str]] = None
    ) -> ProcessingResults:
        """Run specified operations for ALL funds using cached data"""
        results: Dict[str, FundResult] = {}
        summary = {
            "requested_operations": operations,
            "processed_funds": 0,
            "funds_with_errors": 0,
            "errors": [],
        }

        tests = [test for test in compliance_tests or [] if test]

        for fund_name in self.available_funds:
            self.logger.info(f"Processing {fund_name}...")
            fund_result = FundResult(fund_name=fund_name)

            try:
                # Get fund configuration from registry
                fund_config = self.fund_registry.get_fund(fund_name)
                if not fund_config:
                    fund_result.errors.append(f"Fund not found in registry")
                    continue

                # Create Fund instance with bulk data
                fund = self._create_fund_from_bulk_data(fund_name, fund_config)

                # Run requested operations
                if 'compliance' in operations:
                    fund_result.compliance_results = self._run_compliance(fund, tests)

                if 'reconciliation' in operations:
                    fund_result.reconciliation_results = self._run_reconciliation(fund, fund_name)

                if 'nav_reconciliation' in operations:
                    fund_result.nav_results = self._run_nav_reconciliation(fund, fund_name)

                summary["processed_funds"] += 1

            except Exception as e:
                error_msg = f"Error processing {fund_name}: {str(e)}"
                fund_result.errors.append(error_msg)
                self.logger.error(error_msg)
                summary["errors"].append(error_msg)
                summary["funds_with_errors"] += 1

            results[fund_name] = fund_result

        summary["total_funds"] = len(results)
        return ProcessingResults(fund_results=results, summary=summary)

    def _run_compliance(self, fund: Fund, tests: List[str]) -> Dict[str, Any]:
        """Run compliance checks"""
        try:
            checker = ComplianceChecker(
                getattr(self.data_store, "session", None),
                funds={fund.name: fund},
                date=getattr(self.data_store, "date", None),
                base_cls=getattr(self.data_store, "base_cls", None),
                analysis_type=self.analysis_type,
            )

            requested_tests = [test for test in tests if test]
            checker_results = checker.run_compliance_tests(
                test_functions=requested_tests or None
            )

            fund_results = dict(checker_results.get(fund.name, {}))
            fund_results["fund_object"] = fund
            return fund_results

        except Exception as e:
            self.logger.error(f"Compliance error for {fund.name}: {e}")
            return {'errors': [str(e)], 'violations': []}

    def _run_nav_reconciliation(self, fund: Fund, fund_name: str) -> Dict[str, Any]:
        """Run NAV reconciliation using your existing NAVReconciliator"""
        try:
            # Determine prior date
            prior_date = self._get_prior_date(self.data_store.date)

            # Get fund data from data store
            fund_data_dict = self.data_store.fund_data.get(fund_name, {})

            # NAVReconciliator expects positional arguments: session, fund_name, fund_data, analysis_date, prior_date
            nav_reconciliator = NAVReconciliator(
                self.data_store.session,  # session
                fund_name,  # fund_name
                fund_data_dict,  # fund_data dictionary
                self.data_store.date,  # analysis_date
                prior_date,  # prior_date
                analysis_type=self.analysis_type,  # optional kwargs
                fund=fund,  # optional kwargs
                socgen_custodian=getattr(self.data_store, 'socgen_custodian', None)  # optional kwargs
            )

            return nav_reconciliator.run_nav_reconciliation()

        except Exception as e:
            self.logger.error(f"NAV reconciliation error for {fund_name}: {e}")
            return {'errors': [str(e)], 'differences': []}

    def _run_reconciliation(self, fund: Fund, fund_name: str) -> Dict[str, Any]:
        """Run reconciliation"""
        try:
            # Get fund data from data store
            fund_data_dict = self.data_store.fund_data.get(fund_name, {})

            # Reconciliator expects positional arguments: fund_name, fund_data, analysis_type
            reconciliator = Reconciliator(
                fund_name,  # fund_name
                fund_data_dict,  # fund_data
                self.analysis_type,  # analysis_type (optional)
                fund = fund
            )

            reconciliator.run_all_reconciliations()
            summary = reconciliator.get_summary()

            detail_payload: Dict[str, Dict[str, Any]] = {}
            for name, result in reconciliator.results.items():
                sections: Dict[str, Any] = {}
                for attr in [
                    "raw_recon",
                    "final_recon",
                    "price_discrepancies_T",
                    "price_discrepancies_T1",
                    "merged_data",
                    "regular_options",
                    "flex_options",
                ]:
                    value = getattr(result, attr, None)
                    if value is not None:
                        sections[attr] = value
                detail_payload[name] = sections

            return {
                "summary": summary,
                "details": detail_payload,
            }
        except Exception as e:
            self.logger.error(f"Reconciliation error for {fund_name}: {e}")
            return {'errors': [str(e)], 'breaks': []}

    def _get_prior_date(self, current_date):
        """Get the prior business date"""

        if isinstance(current_date, pd.Timestamp):
            current = current_date.to_pydatetime().date()
        elif isinstance(current_date, datetime):
            current = current_date.date()
        elif isinstance(current_date, date):
            current = current_date
        else:
            try:
                current = pd.Timestamp(current_date).date()
            except Exception:
                current = datetime.fromisoformat(str(current_date)).date()

        prior = current - timedelta(days=1)
        # Skip weekends
        while prior.weekday() >= 5:  # Saturday = 5, Sunday = 6
            prior = prior - timedelta(days=1)
        return prior

    def _create_fund_from_bulk_data(self, fund_name: str, fund_config) -> Fund:
        """Create a Fund instance populated with bulk data"""
        # Get all data for this fund from bulk store
        fund_data_dict = self.data_store.fund_data.get(fund_name, {})

        # Get fund configuration for flex options
        has_flex = fund_config.config.get('has_flex_option', False)
        flex_option_type = fund_config.config.get('flex_option_type', '')

        # Determine flex pattern based on flex_option_type
        if has_flex:
            if flex_option_type == 'index':
                flex_pattern = r'SPX|XSP'  # Index flex options
            elif flex_option_type == 'single_stock':
                flex_pattern = r'^2'  # Options starting with "2"
            else:
                flex_pattern = None
        else:
            flex_pattern = None

        # Helper function to split options
        def split_options(options_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Split options into regular and flex based on pattern"""
            if not has_flex or flex_pattern is None or options_df.empty:
                return options_df, pd.DataFrame()

            if 'optticker' not in options_df.columns:
                return options_df, pd.DataFrame()

            mask = options_df['optticker'].str.contains(flex_pattern, na=False, regex=True)
            flex_options = options_df[mask].copy()
            regular_options = options_df[~mask].copy()

            return regular_options, flex_options

        # Split all option DataFrames
        vest_options, vest_flex = split_options(fund_data_dict.get('vest_option', pd.DataFrame()))
        vest_options_t1, vest_flex_t1 = split_options(fund_data_dict.get('vest_option_t1', pd.DataFrame()))
        cust_options, cust_flex = split_options(fund_data_dict.get('custodian_option', pd.DataFrame()))
        cust_options_t1, cust_flex_t1 = split_options(fund_data_dict.get('custodian_option_t1', pd.DataFrame()))

        # Create FundData structure
        fund_data = FundData()
        df_nav_t = fund_data_dict.get('nav', pd.DataFrame())
        df_nav_t1 = fund_data_dict.get('nav_t1', pd.DataFrame())

        # Populate current holdings with Vest (OMS) data when available
        fund_data.current = FundSnapshot(
            vest=FundHoldings(
                equity=fund_data_dict.get('vest_equity', pd.DataFrame()),
                options=vest_options,  # Regular options only
                flex_options=vest_flex,
                treasury=fund_data_dict.get('vest_treasury', pd.DataFrame()),
            ),
            custodian=FundHoldings(
                equity = fund_data_dict.get('custodian_equity', pd.DataFrame()),
                options = cust_options,  # Regular options only
                flex_options = cust_flex,
                treasury = fund_data_dict.get('custodian_treasury', pd.DataFrame()),
                cash = self._extract_cash_value(fund_data_dict.get('cash', pd.DataFrame())),
                nav = self._extract_nav_per_share(df_nav_t),
                total_assetsa = self._extract_total_assets(df_nav_t),
                total_net_assets = self._extract_total_net_assets(df_nav_t),
                expenses = self._extract_expense_value(df_nav_t),
                shares_outstanding = self._extract_shares_outstanding(df_nav_t),
        ),
        index = FundHoldings(
            equity=fund_data_dict.get('index', pd.DataFrame()),
            options=pd.DataFrame(),
            flex_options=pd.DataFrame(),
            treasury=pd.DataFrame(),
        ),

        # Add other values for T
        equity_trades = fund_data_dict.get('equity_trades', pd.DataFrame()),
        cr_rd_data = fund_data_dict.get('cr_rd', pd.DataFrame()),
        flows = self._extract_flow_value(fund_data_dict.get('flows', pd.DataFrame())),
        fund_name = fund_name,
        )

        # Populate T-1 snapshot
        fund_data.previous = FundSnapshot(
            vest=FundHoldings(
                equity=fund_data_dict.get('vest_equity_t1', pd.DataFrame()),
                options=vest_options_t1,  # Regular options only
                flex_options=vest_flex_t1,
                treasury=fund_data_dict.get('vest_treasury_t1', pd.DataFrame()),
            ),
            custodian=FundHoldings(
                equity=fund_data_dict.get('custodian_equity_t1', pd.DataFrame()),
                options=cust_options_t1,  # Regular options only
                flex_options=cust_flex_t1,
                treasury=fund_data_dict.get('custodian_treasury_t1', pd.DataFrame()),
                cash=self._extract_cash_value(fund_data_dict.get('cash_t1', pd.DataFrame())),
                nav=self._extract_nav_per_share(df_nav_t1),
                ta=self._extract_total_assets(df_nav_t1),
                tna=self._extract_total_net_assets(df_nav_t1),
                expenses=self._extract_expense_value(df_nav_t1),
                shares_outstanding=self._extract_shares_outstanding(df_nav_t1),
            ),
            index=FundHoldings(
                equity=fund_data_dict.get('index', pd.DataFrame()),
                options=pd.DataFrame(),
                flex_options=pd.DataFrame(),
                treasury=pd.DataFrame(),
            ),

            # Add other values for T-1
            equity_trades=pd.DataFrame(),
            cr_rd_data=pd.DataFrame(),
            flows=self._extract_flow_value(fund_data_dict.get('flows_t1', pd.DataFrame())),
            fund_name=fund_name,
        )

        # Create Fund instance
        fund = Fund(
            name=fund_name,
            config=fund_config.config,  # Pass the full config dictionary
            base_cls=getattr(self.data_store, 'base_cls', None)
        )
        fund.data = fund_data

        return fund

    def _extract_total_net_assets(self, nav_df):
        """Extract Total Net Assets from NAV data"""
        if isinstance(nav_df, pd.DataFrame) and not nav_df.empty:

            for col in ['total_net_assets', 'tna', 'net_assets', 'nav_total']:
                if col in nav_df.columns:
                    series = pd.to_numeric(nav_df[col], errors='coerce').dropna()
                    if not series.empty:
                        return float(series.iloc[0])
        return 0.0

    def _extract_total_assets(self, nav_df):
        """Extract Total Assets (gross) from NAV data"""
        if isinstance(nav_df, pd.DataFrame) and not nav_df.empty:

            for col in ['total_assets', 'gross_assets', 'gross_value']:
                if col in nav_df.columns:
                    series = pd.to_numeric(nav_df[col], errors='coerce').dropna()
                    if not series.empty:
                        return float(series.iloc[0])

        return 0.0

    def _extract_cash_value(self, cash_data: pd.DataFrame) -> float:
        """Extract cash value from cash data"""
        if cash_data.empty:
            return 0.0

        cash_columns = ['cash_value']
        for col in cash_columns:
            if col in cash_data.columns:
                return cash_data[col].sum()

        self.logger.warning("No cash value column found")
        return 0.0

    def _extract_nav_per_share(self, nav_source):
        """Extract NAV per share from either a mapping or direct DataFrame."""
        # Implementation depends on your data structure
        if isinstance(nav_source, pd.DataFrame) and not nav_source.empty:
            nav_columns = ['nav_per_share', 'nav', 'net_asset_value']
            for col in nav_columns:
                if col in nav_source.columns:
                    return nav_source[col].iloc[0] if len(nav_source) > 0 else 0.0
        return 0.0


    def _extract_expense_value(self, nav_df):
        """Extract expense value"""
        if isinstance(nav_df, pd.DataFrame) and not nav_df.empty:
            for col in ['expense_amount', 'expenses']:
                if col in nav_df.columns:
                    series = pd.to_numeric(nav_df[col], errors='coerce').dropna()
                    if not series.empty:
                        return float(series.iloc[0])

        return 0.0

    def _extract_flow_value(self, flows_data: pd.DataFrame) -> float:
        """Extract net flows value from the provided DataFrame."""
        if flows_data.empty:
            return 0.0

        flow_columns = ['net_flows', 'flows', 'amount', 'value']
        for col in flow_columns:
            if col in flows_data.columns:
                return flows_data[col].sum()
        return 0.0

    def _extract_shares_outstanding(self, nav_df):
        """Extract shares outstanding from NAV DataFrame"""
        if isinstance(nav_df, pd.DataFrame) and not nav_df.empty:

            for col in ['shares_outstanding', 'shares', 'total_shares']:
                if col in nav_df.columns:
                    series = pd.to_numeric(nav_df[col], errors='coerce').dropna()
                    if not series.empty:
                        return float(series.iloc[0])

        return 0.0