# config/operation_config.py
from dataclasses import dataclass
from enum import Enum


class OperationMode(Enum):
    EOD = "eod"  # End of Day - run all 3 operations
    TRADING_COMPLIANCE = "trading_compliance"  # ex_ante vs ex_post comparison
    COMPLIANCE_ONLY = "compliance_only"
    RECONCILIATION_ONLY = "reconciliation_only"


@dataclass
class OperationConfig:
    """Which pipeline produced an email (selects recipients, subject and body)."""

    mode: OperationMode

    @classmethod
    def create(cls, mode: OperationMode) -> "OperationConfig":
        return cls(mode=mode)
