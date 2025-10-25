# config/operation_config.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class OperationMode(Enum):
    EOD = "eod"  # End of Day - run all 3 operations
    TRADING_COMPLIANCE = "trading_compliance"  # ex_ante vs ex_post comparison
    COMPLIANCE_ONLY = "compliance_only"
    RECONCILIATION_ONLY = "reconciliation_only"


@dataclass
class OperationConfig:
    mode: OperationMode
    analysis_type: str  # 'eod', 'ex_ante', 'ex_post'
    run_compliance: bool
    run_reconciliation: bool
    run_nav_reconciliation: bool
    test_functions: List[str]

    @classmethod
    def create(cls, mode: OperationMode, analysis_type: str = None) -> 'OperationConfig':
        """Factory method for different operation modes"""
        configs = {
            OperationMode.EOD: OperationConfig(
                mode=OperationMode.EOD,
                analysis_type='eod',
                run_compliance=True,
                run_reconciliation=True,
                run_nav_reconciliation=True,
                test_functions=[...]  # Your full test list
            ),
            OperationMode.TRADING_COMPLIANCE: OperationConfig(
                mode=OperationMode.TRADING_COMPLIANCE,
                analysis_type=None,  # Will use both ex_ante and ex_post
                run_compliance=True,
                run_reconciliation=False,
                run_nav_reconciliation=False,
                test_functions=[...]  # Compliance tests only
            ),
            OperationMode.COMPLIANCE_ONLY: OperationConfig(
                mode=OperationMode.COMPLIANCE_ONLY,
                analysis_type='eod',
                run_compliance=True,
                run_reconciliation=False,
                run_nav_reconciliation=False,
                test_functions=[...]
            )
        }
        return configs[mode]