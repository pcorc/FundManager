"""Fund model / loader behaviour that needs no DB: metadata-derived fund attributes and fund order.

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import fund_definitions  # noqa: E402
from processing.bulk_data_loader import BulkDataLoader  # noqa: E402
from processing.fund import build_fund_registry  # noqa: E402

DAY, PREV = date(2026, 10, 30), date(2026, 10, 29)


class FundMetadataAttributeTests(unittest.TestCase):
    """vehicle / is_* / index_ticker_join follow fund.config, which the loader fills after construction."""

    def setUp(self):
        self.registry = build_fund_registry()

    def test_attributes_empty_before_metadata(self):
        fund = self.registry["PF227"]
        self.assertIsNone(fund.vehicle)
        self.assertIsNone(fund.index_ticker_join)
        self.assertFalse(fund.is_private_fund or fund.is_closed_end_fund or fund.is_etf)

    def test_attributes_follow_merged_metadata(self):
        cases = {"PF227": ("PF", "private_fund"), "HE3B1": ("CEF", "closed_end_fund"),
                 "KNG": ("TIF", "etf"), "FTCSH": ("VIT", "vit")}
        for name, (strategy, vehicle) in cases.items():
            fund = self.registry[name]
            fund.config.update({"eod_report_strategy": strategy, "index_ticker_join": "SPXFIOUP"})
            self.assertEqual(fund.vehicle, vehicle, name)
            self.assertEqual(fund.is_private_fund, vehicle == "private_fund", name)
            self.assertEqual(fund.is_closed_end_fund, vehicle == "closed_end_fund", name)
            self.assertEqual(fund.is_etf, vehicle == "etf", name)
            self.assertEqual(fund.index_ticker_join, "SPXFIOUP", name)

    def test_config_replaced_in_place(self):  # what the golden replay does with fund.config
        fund = self.registry["KNG"]
        fund.config.clear()
        fund.config.update({"eod_report_strategy": "CEF", "vehicle_wrapper": "etf"})
        self.assertEqual(fund.vehicle, "etf")  # an explicit vehicle_wrapper wins
        fund.config.update({"vehicle_wrapper": None, "index_ticker_join": " "})
        self.assertEqual(fund.vehicle, "closed_end_fund")
        self.assertIsNone(fund.index_ticker_join)


class VehicleWrapperTests(unittest.TestCase):
    def test_reads_loaded_cohorts(self):
        cohorts = {"TIF": {"KNG"}, "CEF": {"HE3B1"}, "PF": {"PF227"}, "VIT": {"FTCSH"}}
        with mock.patch.dict(fund_definitions._STRATEGY_TO_COHORT, cohorts):
            self.assertEqual(fund_definitions.vehicle_wrapper("KNG"), "etf")
            self.assertEqual(fund_definitions.vehicle_wrapper("HE3B1"), "closed_end_fund")
            self.assertEqual(fund_definitions.vehicle_wrapper("PF227"), "private_fund")
            self.assertEqual(fund_definitions.vehicle_wrapper("FTCSH"), "vit")
            self.assertEqual(fund_definitions.vehicle_wrapper("NOPE"), "")


class LoaderMetadataOrderTests(unittest.TestCase):
    def test_metadata_merged_before_index_load(self):
        registry = {n: f for n, f in build_fund_registry().items() if n in ("KNG", "PF227")}
        loader = BulkDataLoader(session=None, fund_registry=registry)
        metadata = {"KNG": {"fund_ticker": "KNG", "eod_report_strategy": "TIF", "index_ticker_join": "SPATI",
                            "expense_ratio": 0.0075},
                    "PF227": {"fund_ticker": "PF227", "eod_report_strategy": "PF", "index_ticker_join": None}}
        seen = {}

        def fake_sql(sql, **params):
            if "v_fund_metadata" in sql:
                return pd.DataFrame(list(metadata.values()))
            if "v_index_holdings" in sql:
                seen["codes"] = params["codes"]
                seen["vehicles"] = {n: f.vehicle for n, f in registry.items()}
            return pd.DataFrame()

        with mock.patch.object(loader, "_query_sql", side_effect=fake_sql):
            loader.load_all_data_for_date(DAY, previous_date=PREV, analysis_type="eod")
        self.assertEqual(seen["codes"], ["SPATI"])
        self.assertEqual(seen["vehicles"], {"KNG": "etf", "PF227": "private_fund"})
        self.assertEqual(registry["KNG"].expense_ratio, 0.0075)

    def test_index_rows_sliced_per_fund_ticker(self):
        registry = {n: f for n, f in build_fund_registry().items() if n in ("HE3B1", "HE3B2")}
        loader = BulkDataLoader(session=None, fund_registry=registry)
        metadata = [{"fund_ticker": n, "eod_report_strategy": "CEF", "index_ticker_join": "SPXFIOUP"}
                    for n in registry]
        index_rows = pd.DataFrame([
            {"source": "cboe", "date": day, "fund": "SPXFIOUP", "fund_ticker": owner, "eqyticker": tkr,
             "weight_index": 0.5}
            for day in (DAY, PREV) for owner in ("HE3B1", "HE3B2", "PF227") for tkr in ("AAPL", "MSFT")
        ])

        def fake_sql(sql, **params):
            if "v_fund_metadata" in sql:
                return pd.DataFrame(metadata)
            if "v_index_holdings" in sql:
                return index_rows
            return pd.DataFrame()

        with mock.patch.object(loader, "_query_sql", side_effect=fake_sql):
            store = loader.load_all_data_for_date(DAY, previous_date=PREV, analysis_type="eod")
        for name in registry:
            for key, day in (("index", DAY), ("index_t1", PREV)):
                frame = store.fund_data[name][key]
                self.assertEqual(sorted(frame["eqyticker"]), ["AAPL", "MSFT"], (name, key))
                self.assertEqual(set(frame["fund_ticker"]), {name})
                self.assertEqual(set(frame["date"]), {day})


if __name__ == "__main__":
    unittest.main()
