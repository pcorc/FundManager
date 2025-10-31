"""Registry of statically-defined fund configurations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config.fund_definitions import FUND_DEFINITIONS


@dataclass
class FundClass:
    """Simple wrapper around the raw configuration mapping."""

    name: str
    config: Dict[str, Any]

    @property
    def mapping_data(self) -> Dict[str, Any]:
        return self.config

    @property
    def expense_ratio(self) -> float:
        return float(self.config.get("expense_ratio", 0.0) or 0.0)

    @property
    def vehicle(self) -> Optional[str]:
        return self.config.get("vehicle_wrapper")

    @property
    def custodian_type(self) -> Optional[str]:
        for key in (
            "custodian_equity_holdings",
            "custodian_navs",
            "custodian_option_holdings",
        ):
            table = self.config.get(key)
            if isinstance(table, str) and table:
                if "bny" in table.lower():
                    return "bny"
                if "umb" in table.lower():
                    return "umb"
                if "socgen" in table.lower() or "sg" in table.lower():
                    return "socgen"
        return None

    def get_required_tables(self) -> List[str]:
        """Return a list of non-empty table names referenced by this fund."""

        keys: Iterable[str] = (
            "custodian_equity_holdings",
            "custodian_option_holdings",
            "custodian_treasury_holdings",
            "custodian_navs",
            "cash_table",
            "vest_equity_holdings",
            "vest_options_holdings",
            "vest_treasury_holdings",
            "basket",
            "flows",
            "sg_custodian_holdings",
            "index_holdings",
            "option_custodian_assignment",
            "overlap_table",
        )

        tables: List[str] = []
        for key in keys:
            table = self.config.get(key)
            if isinstance(table, str) and table and table.upper() != "NULL":
                tables.append(table)
        return tables


class FundRegistry:
    """Central access point for fund metadata."""

    def __init__(self, definitions: Optional[Dict[str, Dict[str, Any]]] = None):
        self._definitions = definitions or FUND_DEFINITIONS
        self.funds: Dict[str, FundClass] = {}

    @classmethod
    def from_database(cls, session, base_cls) -> "FundRegistry":
        registry = cls()
        registry.reload(session=session, base_cls=base_cls)
        return registry

    def reload(self, *, session=None, base_cls=None) -> None:
        """Refresh the registry from the static dictionary, enriching with accounts."""

        self.funds.clear()
        account_numbers: Dict[str, Dict[str, Any]] = {}
        if session is not None and base_cls is not None:
            account_numbers = self._load_account_numbers(session, base_cls)

        for fund_name, payload in self._definitions.items():
            config = dict(payload)
            config.setdefault("fund", fund_name)

            numbers = account_numbers.get(fund_name)
            if numbers:
                config["account_numbers"] = numbers
                config.setdefault(
                    "account_number_custodian",
                    self._derive_custodian_account_number(numbers),
                )

            self.funds[fund_name] = FundClass(name=fund_name, config=config)

    def get_fund(self, fund_name: str) -> Optional[FundClass]:
        return self.funds.get(fund_name)

    def get_all_funds(self) -> Dict[str, FundClass]:
        return dict(self.funds)

    def get_funds_by_custodian(self, custodian_type: str) -> List[FundClass]:
        custodian_type = (custodian_type or "").lower()
        return [
            fund
            for fund in self.funds.values()
            if (fund.custodian_type or "").lower() == custodian_type
        ]

    def get_required_tables(self) -> List[str]:
        tables: set[str] = set()
        for fund in self.funds.values():
            tables.update(fund.get_required_tables())
        return sorted(tables)

    def _load_account_numbers(self, session, base_cls) -> Dict[str, Dict[str, Any]]:
        account_numbers_tbl = getattr(base_cls.classes, "account_numbers", None)
        if account_numbers_tbl is None:
            return {}

        query = session.query(
            account_numbers_tbl.fund,
            account_numbers_tbl.account_type,
            account_numbers_tbl.service_provider,
            account_numbers_tbl.account_number,
        )

        df_accounts = pd.read_sql(query.statement, session.bind)
        account_mapping: Dict[str, Dict[str, Any]] = {}

        for _, account_row in df_accounts.iterrows():
            fund = account_row.get("fund")
            if not isinstance(fund, str):
                continue
            fund_key = fund.strip()
            if not fund_key:
                continue

            account_number = account_row.get("account_number")
            if pd.isna(account_number):
                continue
            account_number_str = str(account_number).strip()
            if not account_number_str:
                continue

            account_type = account_row.get("account_type")
            service_provider = account_row.get("service_provider")
            account_type_key = (
                str(account_type).strip().lower() if isinstance(account_type, str) else None
            )
            provider_key = (
                str(service_provider).strip().lower()
                if isinstance(service_provider, str)
                else None
            )

            fund_numbers = account_mapping.setdefault(fund_key, {})
            if provider_key == "sg" and account_type_key != "collateral":
                accounts = fund_numbers.setdefault("sg", [])
                if account_number_str not in accounts:
                    accounts.append(account_number_str)
                continue

            key = account_type_key or provider_key or "other"
            if key in fund_numbers and isinstance(fund_numbers[key], list):
                if account_number_str not in fund_numbers[key]:
                    fund_numbers[key].append(account_number_str)
            elif key in fund_numbers and fund_numbers[key] != account_number_str:
                existing = fund_numbers[key]
                values = existing if isinstance(existing, list) else [existing]
                if account_number_str not in values:
                    values.append(account_number_str)
                fund_numbers[key] = values
            else:
                fund_numbers[key] = account_number_str

        return account_mapping

    @staticmethod
    def _derive_custodian_account_number(numbers: Dict[str, Any]) -> Optional[str]:
        if not numbers:
            return None

        priority_keys = [
            "account_number_custodian",
            "custodian",
            "primary",
            "account",
        ]

        for key in priority_keys:
            value = numbers.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return value[0]

        sg_accounts = numbers.get("sg")
        if isinstance(sg_accounts, list) and sg_accounts:
            return sg_accounts[0]

        for key, value in numbers.items():
            if key == "collateral":
                continue
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return value[0]
        return None