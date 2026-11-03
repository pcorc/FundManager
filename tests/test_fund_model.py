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
from processing.bulk_data_loader import BulkDataLoader, BulkDataStore  # noqa: E402
from processing.fund import build_fund_registry  # noqa: E402
from processing.fund_manager import FundManager  # noqa: E402

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


class SnapshotBuildTests(unittest.TestCase):
    def test_previous_snapshot_uses_t1_index_and_both_carry_overlap(self):
        registry = build_fund_registry()
        frames = {
            "index": pd.DataFrame({"eqyticker": ["AAPL"], "weight_index": [0.6]}),
            "index_t1": pd.DataFrame({"eqyticker": ["MSFT"], "weight_index": [0.4]}),
            "overlap": pd.DataFrame({"security_ticker": ["AAPL"], "security_weight": [0.07]}),
            "overlap_t1": pd.DataFrame({"security_ticker": ["MSFT"], "security_weight": [0.06]}),
        }
        store = BulkDataStore(date=str(DAY), fund_data={"HE3B1": frames}, loaded_funds={"HE3B1"})
        fund = registry["HE3B1"]
        FundManager(registry, store)._populate_fund_data(fund)
        self.assertEqual(list(fund.data.current.index.equity["eqyticker"]), ["AAPL"])
        self.assertEqual(list(fund.data.previous.index.equity["eqyticker"]), ["MSFT"])
        self.assertEqual(list(fund.data.current.overlap["security_ticker"]), ["AAPL"])
        self.assertEqual(list(fund.data.previous.overlap["security_ticker"]), ["MSFT"])


class FundOrderTests(unittest.TestCase):
    def test_available_funds_follow_registry_order_not_set_order(self):
        registry = {n: f for n, f in build_fund_registry().items() if n in ("DOGG", "KNG", "RDVI", "TDVI")}
        store = BulkDataStore(date=str(DAY), loaded_funds={"TDVI", "KNG", "DOGG", "RDVI", "ZZZ", "AAA"})
        manager = FundManager(registry, store)
        # registry order first; loaded funds missing from the registry last (sorted) so they still error out
        self.assertEqual(manager.available_funds, ["DOGG", "KNG", "RDVI", "TDVI", "AAA", "ZZZ"])


class IndexFlexOverlapTests(unittest.TestCase):
    def test_overlap_spreads_long_flex_value_over_constituents(self):
        from processing.fund import FundData, FundHoldings, FundSnapshot
        from services.compliance_checker import ComplianceChecker

        fund = build_fund_registry()["HE3B1"]  # index FLEX fund
        flex = pd.DataFrame({
            "optticker": ["4SPX US 04/13/28 C6000.01", "4SPX US 04/13/28 P6000.01"],
            "nav_shares": [19, -19],  # Vest flex frames carry nav_shares, not nav_shares_option
            "option_market_value": [4_000_000.0, -300_000.0],
        })
        fund.data = FundData(current=FundSnapshot(vest=FundHoldings(flex_options=flex)))
        holdings = pd.DataFrame({"eqyticker": ["AAPL", "MSFT"], "equity_market_value": [100.0, 50.0],
                                 "option_market_value": [0.0, 0.0]})
        overlap = pd.DataFrame({"security_ticker": ["AAPL", "MSFT"], "security_weight": [0.075, 0.06]})

        checker = ComplianceChecker(funds={fund.name: fund}, analysis_type="eod")
        merged, details = checker._calculate_index_overlap_adjustments(fund, holdings, overlap, 10_000_000.0)
        self.assertEqual(list(details["security_ticker"]), ["AAPL", "MSFT"])
        self.assertAlmostEqual(details["overlap_market_value"].iloc[0], 4_000_000.0 * 0.075)
        self.assertAlmostEqual(details["overlap_weight"].iloc[1], 4_000_000.0 * 0.06 / 10_000_000.0)
        self.assertAlmostEqual(merged["net_market_value"].iloc[0], 100.0 + 300_000.0)

    def test_overlap_counted_once_per_ticker_when_options_repeat_the_equity(self):
        from processing.fund import FundData, FundHoldings, FundSnapshot
        from services.compliance_checker import ComplianceChecker

        fund = build_fund_registry()["HE3B1"]
        flex = pd.DataFrame({"optticker": ["4SPX US 04/13/28 C6000.01"], "nav_shares": [19],
                             "option_market_value": [4_000_000.0]})
        fund.data = FundData(current=FundSnapshot(vest=FundHoldings(flex_options=flex)))
        # the listed-option outer merge yields one AAPL row per AAPL option
        holdings = pd.DataFrame({"eqyticker": ["AAPL", "AAPL", "MSFT"], "equity_market_value": [100.0] * 3,
                                 "option_market_value": [-1.0, -2.0, 0.0]})
        overlap = pd.DataFrame({"security_ticker": ["AAPL", "MSFT"], "security_weight": [0.075, 0.06]})

        checker = ComplianceChecker(funds={fund.name: fund}, analysis_type="eod")
        merged, details = checker._calculate_index_overlap_adjustments(fund, holdings, overlap, 10_000_000.0)
        self.assertEqual(list(details["security_ticker"]), ["AAPL", "MSFT"])
        self.assertAlmostEqual(merged["long_box_market_value_overlap"].sum(), 4_000_000.0 * (0.075 + 0.06))
        self.assertEqual(list(merged["net_market_value"]), [100.0 - 1.0 + 300_000.0, 100.0 - 2.0, 100.0 + 240_000.0])


class OptionPriceBreakWeightTests(unittest.TestCase):
    def test_closed_end_top5_ranks_short_call_book_by_gross_exposure(self):
        from processing.fund import FundData, FundHoldings, FundSnapshot
        from services.reconciliator import Reconciliator

        fund = build_fund_registry()["HE3B1"]
        fund.config["eod_report_strategy"] = "CEF"  # closed-end: listed option price breaks keep the top 5
        tickers = [f"T{i} US 10/30/26 C100" for i in range(7)]
        contracts = [-1, -70, -2, -50, -30, -3, -40]  # all short calls: the net option MV is negative
        vest = pd.DataFrame({"optticker": tickers, "nav_shares": contracts, "price": [1.0] * 7})
        cust = pd.DataFrame({"optticker": tickers, "quantity": contracts, "price": [1.5] * 7})
        fund.data = FundData(current=FundSnapshot(vest=FundHoldings(options=vest),
                                                  custodian=FundHoldings(options=cust)))

        recon = Reconciliator(fund)
        recon.reconcile_custodian_option()
        breaks = recon.results["custodian_option"].price_discrepancies_T
        self.assertEqual(sorted(breaks["optticker"]), sorted(tickers[i] for i in (1, 3, 4, 6, 5)))
        self.assertAlmostEqual(breaks["option_weight"].max(), 70 / 196)


if __name__ == "__main__":
    unittest.main()
