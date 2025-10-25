# processing/fund_processor.py
from dataclasses import dataclass
from typing import Dict, Any, List
import logging


@dataclass
class ProcessingResult:
    date: str
    compliance_results: Dict[str, Any] = None
    reconciliation_results: Dict[str, Any] = None
    nav_reconciliation_results: Dict[str, Any] = None
    traded_funds_info: Dict[str, Any] = None
    analysis_type: str = None


class FundProcessor:
    """Efficiently processes funds for different operation modes"""

    def __init__(self, session, base_cls, funds_config: Dict):
        self.session = session
        self.base_cls = base_cls
        self.funds_config = funds_config
        self.logger = logging.getLogger(__name__)

    def process_date(self, date: str, config: OperationConfig) -> ProcessingResult:
        """Process a single date with given configuration"""
        result = ProcessingResult(date=date, analysis_type=config.analysis_type)

        if config.mode == OperationMode.TRADING_COMPLIANCE:
            return self._process_trading_compliance(date, config)
        else:
            return self._process_regular_operations(date, config)

    def _process_regular_operations(self, date: str, config: OperationConfig) -> ProcessingResult:
        """Process regular EOD or single analysis type operations"""
        result = ProcessingResult(date=date, analysis_type=config.analysis_type)

        # Single fund manager for all operations
        fund_manager = FundManager(
            session=self.session,
            funds=self.funds_config,
            date=date,
            analysis_type=config.analysis_type,
            base_cls=self.base_cls
        )

        # Run requested operations
        if config.run_compliance:
            result.compliance_results = fund_manager.run_daily_compliance(
                test_functions=config.test_functions
            )

        if config.run_reconciliation:
            recon_results, recon_summaries = fund_manager.run_daily_reconciliation()
            result.reconciliation_results = recon_results

            if config.run_nav_reconciliation:
                nav_results, nav_summaries = fund_manager.run_nav_recon()
                result.nav_reconciliation_results = nav_results

        return result

    def _process_trading_compliance(self, date: str, config: OperationConfig) -> ProcessingResult:
        """Process trading compliance (ex_ante vs ex_post comparison)"""
        result = ProcessingResult(date=date)

        # Run ex_ante compliance
        fund_manager_ante = FundManager(
            session=self.session,
            funds=self.funds_config,
            date=date,
            analysis_type='ex_ante',
            base_cls=self.base_cls
        )
        results_ex_ante = fund_manager_ante.run_daily_compliance(
            test_functions=config.test_functions
        )

        # Run ex_post compliance
        fund_manager_post = FundManager(
            session=self.session,
            funds=self.funds_config,
            date=date,
            analysis_type='ex_post',
            base_cls=self.base_cls
        )
        results_ex_post = fund_manager_post.run_daily_compliance(
            test_functions=config.test_functions
        )

        # Identify traded funds
        result.traded_funds_info = TradingComplianceAnalyzer.get_traded_funds_info(fund_manager_post)

        if result.traded_funds_info:
            # Only store results for traded funds
            result.compliance_results = {
                'ex_ante': {k: v for k, v in results_ex_ante.items() if k in result.traded_funds_info},
                'ex_post': {k: v for k, v in results_ex_post.items() if k in result.traded_funds_info},
                'comparison': TradingComplianceAnalyzer(
                    results_ex_ante=results_ex_ante,
                    results_ex_post=results_ex_post,
                    date=date,
                    traded_funds_info=result.traded_funds_info
                ).analyze()
            }

        return result