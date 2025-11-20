"""NAV reconciliation service built on top of the Fund domain object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import logging
import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

from domain.fund import Fund, GainLossResult
from config.fund_definitions import FUND_DEFINITIONS, INDEX_FLEX_FUNDS


@dataclass
class _ComponentGL:
    """Container for raw/adjusted gain-loss values and optional detail frame."""

    raw: float = 0.0
    adjusted: float = 0.0
    details: pd.DataFrame = field(default_factory=pd.DataFrame)


class NAVReconciliator:
    """Calculate daily NAV reconciliation metrics for a fund with detailed ticker-level breakdowns."""

    DETAIL_COLUMNS = [
        "ticker",
        "quantity_t1",
        "quantity_t",
        "price_t1_raw",
        "price_t_raw",
        "price_t1_adj",
        "price_t_adj",
        "gl_raw",
        "gl_adjusted",
    ]

    def __init__(
        self,
        session,
        fund_name: str,
        fund_data: Mapping[str, object],
        analysis_date,
        prior_date,
        analysis_type=None,
        fund: Fund | None = None,
        socgen_custodian=None,
    ) -> None:
        self.session = session
        self.fund_name = fund_name
        self.fund_data = dict(fund_data or {})
        self.holdings_price_breaks = self.fund_data.get("holdings_price_breaks", {})
        self.analysis_date = analysis_date
        self.prior_date = prior_date
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(f"NAVReconciliator_{fund_name}")
        self.results: Dict[str, object] = {}
        self.summary: Dict[str, float] = {}
        self.fund: Fund | None = fund
        self.socgen_custodian = socgen_custodian
        self.fund_definition = FUND_DEFINITIONS.get(fund_name, {})
        metadata_flex = bool(self.fund_definition.get("has_flex_option"))
        fund_flex = bool(getattr(self.fund, "has_flex_option", False))
        self.has_flex_option = metadata_flex or fund_flex
        flex_type = (
            getattr(self.fund, "flex_option_type", None)
            or self.fund_definition.get("flex_option_type")
            or ""
        )
        self.flex_option_type = flex_type.lower() or None
        self.uses_index_flex = (
            fund_name in INDEX_FLEX_FUNDS or self.flex_option_type == "index"
        )

        self._component_key_map: dict[str, tuple[str | None, str | None]] = {
            "equity": ("vest_equity", "vest_equity_t1"),
            "options": ("vest_option", "vest_option_t1"),
            "flex_options": ("vest_option", "vest_option_t1"),
            "treasury": ("vest_treasury", "vest_treasury_t1"),
        }

        # Containers that hold per component detail frames
        self.equity_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.option_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.flex_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)
        self.treasury_details = pd.DataFrame(columns=self.DETAIL_COLUMNS)

    # ------------------------------------------------------------------
    def run_nav_reconciliation(self) -> Dict[str, object]:
        """Main reconciliation orchestrator with detailed ticker-level data."""

        self.logger.info("Starting NAV reconciliation for %s", self.fund_name)

        is_assignment_day = self._is_option_settlement_date(self.analysis_date)

        # Calculate component G/Ls with details
        equity_gl = self._calculate_equity_gl_with_details()

        assignment_gl = 0.0
        if is_assignment_day:
            assignment_gl = self.process_assignments(self.prior_date)
            rolled_gl = self._calculate_rolled_option_gl()
            option_gl = _ComponentGL(raw=rolled_gl, adjusted=rolled_gl)
        else:
            option_gl = self._calculate_option_gl_with_details()

        flex_gl = self._calculate_flex_option_gl_with_details()
        treasury_gl = self._calculate_treasury_gl()

        dividends = self._calculate_dividends()
        expenses = self._calculate_expenses()
        flows_adjustment = self._calculate_t1_flows_adjustment()
        distributions = self._calculate_distributions()
        other_impact = self._extract_numeric_from_data("other")

        results = self._calculate_nav_comparison(
            equity_gl,
            option_gl,
            flex_gl,
            treasury_gl,
            assignment_gl,
            dividends,
            expenses,
            distributions,
            flows_adjustment,
            other_impact,
        )

        # Expose detailed calculations for downstream reporting
        results["detailed_calculations"] = {
            "equity_details": self.equity_details,
            "option_details": self.option_details,
            "flex_details": self.flex_details,
            "treasury_details": self.treasury_details,
        }

        results["details"] = results["detailed_calculations"]
        results["raw_equity"] = self._payload_frame("vest_equity", "equity_holdings")
        results["raw_option"] = self._payload_frame("vest_option", "options_holdings")

        self.results = results
        self.summary = results.get("summary", {})  # type: ignore[assignment]

        self._log_completion(results)
        return self.results

    # ------------------------------------------------------------------
    def _calculate_equity_gl_with_details(self) -> _ComponentGL:
        """Calculate equity G/L with ticker-level details."""

        equity_t = self._payload_frame("vest_equity", "equity_holdings")
        equity_t1 = self._payload_frame("vest_equity_t1", "equity_holdings_t1")

        if equity_t.empty and equity_t1.empty:
            return self._component_gain_loss("equity")

        details_list = []
        total_gl_raw = 0.0
        total_gl_adj = 0.0

        price_breaks = self.holdings_price_breaks.get("equity", pd.DataFrame())

        all_tickers: set[str] = set()
        if not equity_t.empty and "eqyticker" in equity_t.columns:
            all_tickers.update(equity_t["eqyticker"].unique())
        if not equity_t1.empty and "eqyticker" in equity_t1.columns:
            all_tickers.update(equity_t1["eqyticker"].unique())

        for ticker in sorted(all_tickers):
            qty_t = 0
            qty_t1 = 0
            price_t_raw = 0
            price_t1_raw = 0

            if not equity_t.empty:
                ticker_data = equity_t[equity_t["eqyticker"] == ticker]
                if not ticker_data.empty:
                    qty_t = ticker_data.iloc[0].get("nav_shares", 0)
                    price_t_raw = ticker_data.iloc[0].get("price", 0)

            if not equity_t1.empty:
                ticker_data_t1 = equity_t1[equity_t1["eqyticker"] == ticker]
                if not ticker_data_t1.empty:
                    qty_t1 = ticker_data_t1.iloc[0].get("nav_shares", 0)
                    price_t1_raw = ticker_data_t1.iloc[0].get("price", 0)

            price_t_adj = price_t_raw
            price_t1_adj = price_t1_raw

            if not price_breaks.empty and ticker in price_breaks.index:
                adj_data = price_breaks.loc[ticker]
                if "price_t_adj" in adj_data:
                    price_t_adj = adj_data["price_t_adj"]
                if "price_t1_adj" in adj_data:
                    price_t1_adj = adj_data["price_t1_adj"]

            gl_raw = (price_t_raw - price_t1_raw) * qty_t
            gl_adj = (price_t_adj - price_t1_adj) * qty_t

            if qty_t != 0 or qty_t1 != 0:
                details_list.append(
                    {
                        "ticker": ticker,
                        "quantity_t1": qty_t1,
                        "quantity_t": qty_t,
                        "price_t1_raw": price_t1_raw,
                        "price_t_raw": price_t_raw,
                        "price_t1_adj": price_t1_adj,
                        "price_t_adj": price_t_adj,
                        "gl_raw": gl_raw,
                        "gl_adjusted": gl_adj,
                    }
                )

                total_gl_raw += gl_raw
                total_gl_adj += gl_adj

        self.equity_details = pd.DataFrame(details_list, columns=self.DETAIL_COLUMNS)
        return _ComponentGL(raw=total_gl_raw, adjusted=total_gl_adj, details=self.equity_details)

    # ------------------------------------------------------------------
    def _calculate_option_gl_with_details(self) -> _ComponentGL:
        """Calculate option G/L with ticker-level details (non-flex)."""

        option_t = self._payload_frame("vest_option", "option_holdings")
        option_t1 = self._payload_frame("vest_option_t1", "option_holdings_t1")

        if option_t.empty and option_t1.empty:
            return self._component_gain_loss("options")

        details_list = []
        total_gl_raw = 0.0
        total_gl_adj = 0.0

        price_breaks = self.holdings_price_breaks.get("option", pd.DataFrame())

        all_tickers: set[str] = set()
        if not option_t.empty and "optticker" in option_t.columns:
            non_flex = option_t[~option_t["optticker"].str.contains("SPX|XSP", na=False)]
            all_tickers.update(non_flex["optticker"].unique())
        if not option_t1.empty and "optticker" in option_t1.columns:
            non_flex_t1 = option_t1[~option_t1["optticker"].str.contains("SPX|XSP", na=False)]
            all_tickers.update(non_flex_t1["optticker"].unique())

        for ticker in sorted(all_tickers):
            qty_t = 0
            qty_t1 = 0
            price_t_raw = 0
            price_t1_raw = 0

            if not option_t.empty:
                ticker_data = option_t[option_t["optticker"] == ticker]
                if not ticker_data.empty:
                    qty_t = ticker_data.iloc[0].get("nav_shares", 0)
                    price_t_raw = ticker_data.iloc[0].get("price", 0)

            if not option_t1.empty:
                ticker_data_t1 = option_t1[option_t1["optticker"] == ticker]
                if not ticker_data_t1.empty:
                    qty_t1 = ticker_data_t1.iloc[0].get("nav_shares", 0)
                    price_t1_raw = ticker_data_t1.iloc[0].get("price", 0)

            price_t_adj = price_t_raw
            price_t1_adj = price_t1_raw

            if not price_breaks.empty and ticker in price_breaks.index:
                adj_data = price_breaks.loc[ticker]
                if "price_t_adj" in adj_data:
                    price_t_adj = adj_data["price_t_adj"]
                if "price_t1_adj" in adj_data:
                    price_t1_adj = adj_data["price_t1_adj"]

            gl_raw = (price_t_raw - price_t1_raw) * qty_t * 100
            gl_adj = (price_t_adj - price_t1_adj) * qty_t * 100

            if qty_t != 0 or qty_t1 != 0:
                details_list.append(
                    {
                        "ticker": ticker,
                        "quantity_t1": qty_t1,
                        "quantity_t": qty_t,
                        "price_t1_raw": price_t1_raw,
                        "price_t_raw": price_t_raw,
                        "price_t1_adj": price_t1_adj,
                        "price_t_adj": price_t_adj,
                        "gl_raw": gl_raw,
                        "gl_adjusted": gl_adj,
                    }
                )

                total_gl_raw += gl_raw
                total_gl_adj += gl_adj

        self.option_details = pd.DataFrame(details_list, columns=self.DETAIL_COLUMNS)
        return _ComponentGL(raw=total_gl_raw, adjusted=total_gl_adj, details=self.option_details)

    # ------------------------------------------------------------------
    def _calculate_flex_option_gl_with_details(self) -> _ComponentGL:
        """Calculate flex option G/L with ticker-level details."""

        if not self.has_flex_option:
            return _ComponentGL()

        option_t = self._payload_frame("vest_option", "option_holdings")
        option_t1 = self._payload_frame("vest_option_t1", "option_holdings_t1")

        if option_t.empty and option_t1.empty:
            return self._component_gain_loss("flex_options")

        details_list = []
        total_gl_raw = 0.0
        total_gl_adj = 0.0

        price_breaks = self.holdings_price_breaks.get("option", pd.DataFrame())

        all_tickers: set[str] = set()
        if not option_t.empty and "optticker" in option_t.columns:
            flex_only = option_t[option_t["optticker"].str.contains("SPX|XSP", na=False)]
            all_tickers.update(flex_only["optticker"].unique())
        if not option_t1.empty and "optticker" in option_t1.columns:
            flex_only_t1 = option_t1[option_t1["optticker"].str.contains("SPX|XSP", na=False)]
            all_tickers.update(flex_only_t1["optticker"].unique())

        for ticker in sorted(all_tickers):
            qty_t = 0
            qty_t1 = 0
            price_t_raw = 0
            price_t1_raw = 0

            if not option_t.empty:
                ticker_data = option_t[option_t["optticker"] == ticker]
                if not ticker_data.empty:
                    qty_t = ticker_data.iloc[0].get("nav_shares", 0)
                    price_t_raw = ticker_data.iloc[0].get("price", 0)

            if not option_t1.empty:
                ticker_data_t1 = option_t1[option_t1["optticker"] == ticker]
                if not ticker_data_t1.empty:
                    qty_t1 = ticker_data_t1.iloc[0].get("nav_shares", 0)
                    price_t1_raw = ticker_data_t1.iloc[0].get("price", 0)

            price_t_adj = price_t_raw
            price_t1_adj = price_t1_raw

            if not price_breaks.empty and ticker in price_breaks.index:
                adj_data = price_breaks.loc[ticker]
                if "price_t_adj" in adj_data:
                    price_t_adj = adj_data["price_t_adj"]
                if "price_t1_adj" in adj_data:
                    price_t1_adj = adj_data["price_t1_adj"]

            gl_raw = (price_t_raw - price_t1_raw) * qty_t * 100
            gl_adj = (price_t_adj - price_t1_adj) * qty_t * 100

            if qty_t != 0 or qty_t1 != 0:
                details_list.append(
                    {
                        "ticker": ticker,
                        "quantity_t1": qty_t1,
                        "quantity_t": qty_t,
                        "price_t1_raw": price_t1_raw,
                        "price_t_raw": price_t_raw,
                        "price_t1_adj": price_t1_adj,
                        "price_t_adj": price_t_adj,
                        "gl_raw": gl_raw,
                        "gl_adjusted": gl_adj,
                    }
                )

                total_gl_raw += gl_raw
                total_gl_adj += gl_adj

        self.flex_details = pd.DataFrame(details_list, columns=self.DETAIL_COLUMNS)
        return _ComponentGL(raw=total_gl_raw, adjusted=total_gl_adj, details=self.flex_details)

    # ------------------------------------------------------------------
    def _payload_frame(self, *keys: str) -> pd.DataFrame:
        """Return the first DataFrame within fund_data matching keys."""
        for key in keys:
            value = self.fund_data.get(key)
            if isinstance(value, pd.DataFrame):
                return value
        return pd.DataFrame()

    def _snapshot_key(self, component: str, snapshot: str) -> str | None:
        mapping = self._component_key_map.get(component, (None, None))
        return mapping[0] if snapshot == "current" else mapping[1]

    def _extract_from_nav_data(self, snapshot: str, columns: Sequence[str]) -> float:
        nav_key = "nav" if snapshot == "current" else "nav_t1"
        nav_df = self.fund_data.get(nav_key)
        if isinstance(nav_df, pd.DataFrame) and not nav_df.empty:
            for column in columns:
                if column in nav_df.columns:
                    series = pd.to_numeric(nav_df[column], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[0])
        return 0.0

    def _calculate_treasury_gl(self) -> _ComponentGL:
        result = self._component_gain_loss("treasury")
        self.treasury_details = self._build_component_detail_frame(result)
        return result

    def _component_gain_loss(self, asset_class: str) -> _ComponentGL:
        current_value = self._fetch_component_value(asset_class, snapshot="current")
        prior_value = self._fetch_component_value(asset_class, snapshot="previous")
        difference = current_value - prior_value

        if difference != 0.0:
            return _ComponentGL(raw=difference, adjusted=difference)

        if isinstance(self.fund, Fund):
            gain_loss = self.fund.calculate_gain_loss(
                str(self.analysis_date),
                str(self.prior_date),
                asset_class,
            )
            return _ComponentGL(
                raw=float(gain_loss.raw_gl or 0.0),
                adjusted=float(gain_loss.adjusted_gl or gain_loss.raw_gl or 0.0),
            )

        return _ComponentGL()

    def _build_component_detail_frame(self, component: _ComponentGL) -> pd.DataFrame:
        if component.raw == 0.0 and component.adjusted == 0.0:
            return pd.DataFrame(columns=self.DETAIL_COLUMNS)

        return pd.DataFrame(
            [
                {
                    "ticker": "TOTAL",
                    "quantity_t1": 0.0,
                    "quantity_t": 0.0,
                    "price_t1_raw": 0.0,
                    "price_t_raw": 0.0,
                    "price_t1_adj": 0.0,
                    "price_t_adj": 0.0,
                    "gl_raw": component.raw,
                    "gl_adjusted": component.adjusted,
                }
            ],
            columns=self.DETAIL_COLUMNS,
        )

    def _calculate_dividends(self) -> float:
        if isinstance(self.fund, Fund):
            return float(self.fund.get_dividends(str(self.analysis_date)) or 0.0)
        return self._extract_numeric_from_data("dividends")

    def _calculate_expenses(self) -> float:
        if isinstance(self.fund, Fund):
            return float(self.fund.get_expenses(str(self.analysis_date)) or 0.0)
        return self._extract_numeric_from_data("expenses")

    def _calculate_distributions(self) -> float:
        if isinstance(self.fund, Fund):
            return float(self.fund.get_distributions(str(self.analysis_date)) or 0.0)
        return self._extract_numeric_from_data("distributions")

    def _calculate_t1_flows_adjustment(self) -> float:
        if isinstance(self.fund, Fund):
            return float(
                self.fund.get_flows_adjustment(str(self.analysis_date), str(self.prior_date))
                or 0.0
            )
        return self._extract_numeric_from_data("flows_adjustment")

    def _calculate_rolled_option_gl(self) -> float:
        current_value = self._fetch_component_value("options", snapshot="current")
        return current_value

    def process_assignments(self, prior_date) -> float:
        assignments = self.fund_data.get("option_assignments") or self.fund_data.get("assignments")
        if not isinstance(assignments, pd.DataFrame) or assignments.empty:
            return 0.0

        df = assignments.copy()
        date_cols = [col for col in df.columns if "date" in str(col).lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                target = pd.Timestamp(self.analysis_date).normalize()
                df = df[df[date_cols[0]].dt.normalize() == target]
            except Exception:
                pass

        if df.empty:
            return 0.0

        value_cols = [
            col
            for col in df.columns
            if any(keyword in str(col).lower() for keyword in ("pnl", "gl", "gain", "loss", "amount"))
        ]
        for col in value_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.sum())

        quantity_cols = [col for col in df.columns if "quantity" in str(col).lower() or "contracts" in str(col).lower()]
        price_cols = [col for col in df.columns if "price" in str(col).lower() or "premium" in str(col).lower()]
        if quantity_cols and price_cols:
            qty = pd.to_numeric(df[quantity_cols[0]], errors="coerce").fillna(0.0)
            price = pd.to_numeric(df[price_cols[0]], errors="coerce").fillna(0.0)
            return float((qty * price).sum())

        return 0.0

    def _calculate_nav_comparison(
        self,
        equity_gl: _ComponentGL,
        option_gl: _ComponentGL,
        flex_gl: _ComponentGL,
        treasury_gl: _ComponentGL,
        assignment_gl: float,
        dividends: float,
        expenses: float,
        distributions: float,
        flows_adjustment: float,
        other_impact: float,
    ) -> Dict[str, object]:
        begin_tna = self._extract_from_nav_data("previous", ["tna", "total_net_assets", "net_assets"]) or self._safe_float(
            getattr(getattr(self.fund, "data", None), "previous", None), "total_net_assets"
        )
        adjusted_begin_tna = begin_tna + flows_adjustment

        total_gain = (
            equity_gl.adjusted
            + option_gl.adjusted
            + flex_gl.adjusted
            + treasury_gl.adjusted
            + assignment_gl
            + other_impact
        )

        expected_tna = adjusted_begin_tna + total_gain + dividends - expenses - distributions
        custodian_tna = self._extract_from_nav_data("current", ["tna", "total_net_assets", "net_assets"]) or self._safe_float(
            getattr(getattr(self.fund, "data", None), "current", None), "total_net_assets"
        )
        tna_diff = custodian_tna - expected_tna

        equity_gl_result = GainLossResult(equity_gl.raw, equity_gl.adjusted)
        option_gl_result = GainLossResult(option_gl.raw, option_gl.adjusted)
        flex_gl_result = GainLossResult(flex_gl.raw, flex_gl.adjusted)
        treasury_gl_result = GainLossResult(treasury_gl.raw, treasury_gl.adjusted)

        nav_summary = self._calculate_expected_nav(
            equity_gl_result,
            option_gl_result,
            flex_gl_result,
            treasury_gl_result,
            dividends,
            expenses,
            distributions,
            flows_adjustment,
            assignment_gl,
            other_impact,
        )

        shares_outstanding = self._extract_shares_outstanding()
        expected_nav = nav_summary.get("expected_nav", 0.0)
        custodian_nav = nav_summary.get("current_nav", 0.0)
        nav_diff = nav_summary.get("difference", 0.0)

        results: Dict[str, object] = {
            "Beginning TNA": begin_tna,
            "Adjusted Beginning TNA": adjusted_begin_tna,
            "Equity G/L": equity_gl.raw,
            "Equity G/L Adj": equity_gl.adjusted,
            "Option G/L": option_gl.raw,
            "Option G/L Adj": option_gl.adjusted,
            "Flex Option G/L": flex_gl.raw,
            "Flex Option G/L Adj": flex_gl.adjusted,
            "Treasury G/L": treasury_gl.raw,
            "Treasury G/L Adj": treasury_gl.adjusted,
            "Assignment G/L": assignment_gl,
            "Accruals": expenses,
            "Dividends": dividends,
            "Distributions": distributions,
            "Flows Adjustment": flows_adjustment,
            "Other": other_impact,
            "Expected TNA": expected_tna,
            "Custodian TNA": custodian_tna,
            "TNA Diff ($)": tna_diff,
            "Shares Outstanding": shares_outstanding,
            "Expected NAV": expected_nav,
            "Custodian NAV": custodian_nav,
            "NAV Diff ($)": nav_diff,
            "NAV Good (2 Digit)": abs(nav_diff) < 0.01,
            "NAV Good (4 Digit)": abs(nav_diff) < 0.0001,
            "summary": nav_summary,
        }

        return results

    def _calculate_expected_nav(
        self,
        equity_gl: GainLossResult,
        options_gl: GainLossResult,
        flex_gl: GainLossResult,
        treasury_gl: GainLossResult,
        dividends: float,
        expenses: float,
        distributions: float,
        flows_adjustment: float,
        assignment_gl: float = 0.0,
        other_impact: float = 0.0,
    ) -> Dict[str, float]:
        prior_nav = self._extract_from_nav_data("previous", ["nav", "nav_price", "nav_value"]) or self._safe_float(
            getattr(getattr(self.fund, "data", None), "previous", None), "nav"
        )
        current_nav = self._extract_from_nav_data("current", ["nav", "nav_price", "nav_value"]) or self._safe_float(
            getattr(getattr(self.fund, "data", None), "current", None), "nav"
        )

        net_gain = (
            equity_gl.adjusted_gl
            + options_gl.adjusted_gl
            + flex_gl.adjusted_gl
            + treasury_gl.adjusted_gl
            + assignment_gl
            + other_impact
        )

        expected_nav = prior_nav + net_gain + dividends - expenses - distributions + flows_adjustment
        difference = current_nav - expected_nav

        return {
            "prior_nav": prior_nav,
            "current_nav": current_nav,
            "net_gain": net_gain,
            "dividends": dividends,
            "expenses": expenses,
            "distributions": distributions,
            "flows_adjustment": flows_adjustment,
            "expected_nav": expected_nav,
            "difference": difference,
        }

    def _build_summary(self, results: Mapping[str, object]) -> Dict[str, float]:
        summary = results.get("summary") if isinstance(results, Mapping) else {}
        if isinstance(summary, Mapping):
            return dict(summary)  # type: ignore[return-value]
        return {}

    def _log_completion(self, results: Mapping[str, object]) -> None:
        summary = results.get("summary") if isinstance(results, Mapping) else {}
        if isinstance(summary, Mapping):
            diff = summary.get("difference")
            try:
                diff_float = float(diff) if diff is not None else 0.0
                self.logger.info(
                    "Completed NAV reconciliation for %s (NAV diff: %.6f)",
                    self.fund_name,
                    diff_float,
                )
                return
            except (TypeError, ValueError):
                pass
        self.logger.info("Completed NAV reconciliation for %s", self.fund_name)

    def _is_option_settlement_date(self, analysis_date) -> bool:
        if analysis_date is None:
            return False

        timestamp = pd.Timestamp(analysis_date)
        tenor = ""
        if isinstance(self.fund, Fund):
            config = getattr(self.fund, "config", {}) or {}
            if isinstance(config, Mapping):
                tenor = str(config.get("option_roll_tenor", "")).lower()

        if tenor == "weekly":
            return timestamp.weekday() == 4
        if tenor == "monthly":
            wom_third_friday = pd.date_range(
                timestamp.replace(day=1),
                timestamp + MonthEnd(0),
                freq="WOM-3FRI",
            )
            if not wom_third_friday.empty and timestamp.normalize() == wom_third_friday[0].normalize():
                return True
            last_business_day = (timestamp + MonthEnd(0)) - BDay(0)
            return timestamp.normalize() == pd.Timestamp(last_business_day).normalize()
        if tenor == "quarterly":
            if timestamp.month in (3, 6, 9, 12):
                wom_third_friday = pd.date_range(
                    timestamp.replace(day=1),
                    timestamp + MonthEnd(0),
                    freq="WOM-3FRI",
                )
                return any(timestamp.normalize() == value.normalize() for value in wom_third_friday)

        wom_third_friday = pd.date_range(
            timestamp.replace(day=1),
            timestamp + MonthEnd(0),
            freq="WOM-3FRI",
        )
        if any(timestamp.normalize() == value.normalize() for value in wom_third_friday):
            return True

        assignment_data = self.fund_data.get("option_assignments")
        if isinstance(assignment_data, pd.DataFrame) and not assignment_data.empty:
            date_cols = [col for col in assignment_data.columns if "date" in str(col).lower()]
            if date_cols:
                try:
                    dates = pd.to_datetime(assignment_data[date_cols[0]], errors="coerce").dt.normalize()
                    return timestamp.normalize() in dates.values
                except Exception:
                    pass

        return False

    def _fetch_component_value(self, component: str, *, snapshot: str) -> float:
        key = self._snapshot_key(component, snapshot)
        if key:
            value = self._extract_numeric_from_data(key)
            if value:
                return value

        if isinstance(self.fund, Fund):
            data = getattr(getattr(self.fund, "data", None), snapshot, None)
            if data is not None:
                attr = {
                    "equity": "total_equity_value",
                    "options": "total_option_value",
                    "flex_options": "total_option_value",
                    "treasury": "total_treasury_value",
                }.get(component)
                if attr:
                    return self._safe_float(data, attr)
        return 0.0

    def _extract_numeric_from_data(self, key: str) -> float:
        value = self.fund_data.get(key)
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, pd.Series):
                return float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())
            if isinstance(value, pd.DataFrame) and not value.empty:
                numeric_cols = value.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    return float(value[numeric_cols[0]].sum())
        except (TypeError, ValueError):
            return 0.0
        return 0.0

    def _extract_shares_outstanding(self) -> float:
        nav_data = self.fund_data.get("nav")
        if isinstance(nav_data, pd.DataFrame) and not nav_data.empty:
            candidates = [
                "shares_outstanding",
                "shares",
                "units",
            ]
            for column in candidates:
                if column in nav_data.columns:
                    series = pd.to_numeric(nav_data[column], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[0])
        return 0.0

    @staticmethod
    def _safe_float(obj, attribute: str) -> float:
        if obj is None:
            return 0.0
        try:
            value = getattr(obj, attribute, 0.0)
        except AttributeError:
            return 0.0
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0