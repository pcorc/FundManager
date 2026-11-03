"""NAVReconciliator summary arithmetic on hand-built snapshots (no DB).

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from processing.fund import FundData, FundSnapshot, build_fund_registry  # noqa: E402
from services.nav_recon_dataclasses import AssetClassGainLoss, NAVComponents  # noqa: E402
from services.nav_reconciliator import NAVReconciliator  # noqa: E402


def _reconciliator(*, net_units, beg_tna=1_000_000.0, beg_shares=50_000.0, shares=100_000.0, nav=20.0):
    fund = build_fund_registry()["KNG"]
    fund.data = FundData(
        current=FundSnapshot(tna=shares * nav, nav=nav, shares_outstanding=shares, fund_name="KNG"),
        previous=FundSnapshot(tna=beg_tna, shares_outstanding=beg_shares, fund_name="KNG"),
    )
    fund.data.flows = pd.DataFrame({"net_units": [net_units]})
    return NAVReconciliator(fund, "2026-10-30", "2026-10-29")


def _components(flows_adjustment):
    zero = [AssetClassGainLoss(asset_class=a, raw_gl=0.0, adjusted_gl=0.0)
            for a in ("equity", "options", "flex_options", "treasury")]
    return NAVComponents(*zero, flows_adjustment=flows_adjustment)


class FlowsAdjustmentTests(unittest.TestCase):
    """D3: the flows adjustment is the TNA change only; beginning TNA is added once."""

    def test_adjustment_is_units_times_creation_unit_value(self):
        # One creation unit of 50,000 shares at beginning NAV 20.00 adds 1,000,000 of TNA.
        recon = _reconciliator(net_units=1)
        self.assertAlmostEqual(recon._calculate_flows_adjustment(), 1_000_000.0)

    def test_no_flows_leaves_beginning_tna_unchanged(self):
        recon = _reconciliator(net_units=0)
        adj = recon._calculate_flows_adjustment()
        self.assertEqual(adj, 0.0)
        summary = recon._calculate_nav_summary(_components(adj))
        self.assertAlmostEqual(summary.adjusted_beginning_tna, 1_000_000.0)
        self.assertAlmostEqual(summary.expected_tna, 1_000_000.0)

    def test_creation_matches_custodian(self):
        # T-1: 50,000 shares, TNA 1.0M; one unit created; T: 100,000 shares at the same NAV.
        recon = _reconciliator(net_units=1)
        summary = recon._calculate_nav_summary(_components(recon._calculate_flows_adjustment()))
        self.assertAlmostEqual(summary.adjusted_beginning_tna, 2_000_000.0)
        self.assertAlmostEqual(summary.expected_tna, summary.custodian_tna)
        self.assertAlmostEqual(summary.expected_nav, 20.0)
        self.assertTrue(summary.nav_good_4_decimal)


if __name__ == "__main__":
    unittest.main()
