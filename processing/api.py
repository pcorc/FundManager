"""Single entry point for running FundManager: ``RunRequest`` in, ``RunResult`` out.

Scripts, schedulers and a UI all go through :func:`run`. A request names the
funds (explicit tickers and/or cohort names), the operations, an optional
subset of compliance rules, and either one as-of date or a date range run
per day or stacked into one compliance workbook. :func:`build_run` turns the
preset modes in ``config.run_configurations.MODES`` into requests, and
:func:`catalog` lists what a UI can offer.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from config import fund_definitions
from config.database import initialize_database
from config.run_configurations import COHORTS, FULL_COMPLIANCE_TESTS, MODES
from processing.bulk_data_loader import BulkDataLoader
from processing.fund import FundRegistry, build_fund_registry
from processing.run_modes import (
    EODRunParameters,
    TradingComplianceParameters,
    flatten_eod_paths,
    flatten_trading_paths,
    run_eod_mode,
    run_eod_range_mode,
    run_trading_mode,
)
from utilities.workdays import business_date_range, business_day_offset

EOD_OPERATIONS = ("compliance", "reconciliation", "nav_reconciliation")
TRADING_COMPLIANCE = "trading_compliance"
_EOD_REPORT_ALIASES = {"compliance": "compliance", "holdings": "reconciliation", "reconciliation": "reconciliation",
                       "nav": "nav_reconciliation", "nav_reconciliation": "nav_reconciliation"}


@dataclass(frozen=True)
class RunRequest:
    """Everything a run needs. ``compliance_tests=()`` runs every rule plus summary metrics."""

    operations: Tuple[str, ...]
    start_date: date
    end_date: Optional[date] = None
    funds: Tuple[str, ...] = ()
    cohorts: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    compliance_tests: Tuple[str, ...] = tuple(FULL_COMPLIANCE_TESTS)
    date_mode: str = "per_day"  # "per_day" or "stacked" (one compliance workbook over the range)
    generate_daily_reports: bool = False  # stacked mode only
    create_pdf: bool = True
    send_email: bool = True
    output_dir: str = MODES["eod"]["output_dir"]
    output_tag: Optional[str] = None
    previous_date: Optional[date] = None
    ex_ante_date: Optional[date] = None
    ex_post_date: Optional[date] = None
    custodian_date: Optional[date] = None
    custodian_previous_date: Optional[date] = None


@dataclass
class RunResult:
    exit_code: int = 0
    summaries: Dict[str, Mapping[str, object]] = field(default_factory=dict)
    paths: Dict[str, Dict[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Building requests
# ---------------------------------------------------------------------------

def select_funds(cohorts: Iterable[Set[str]] = (), funds: Iterable[str] = (), exclude: Iterable[str] = ()) -> List[str]:
    """Union the cohorts, intersect with ``funds`` when both are given, drop ``exclude``; sorted."""
    selected: Set[str] = set()
    for cohort in cohorts:
        selected |= set(cohort)
    funds = list(funds)
    if funds:
        selected = (selected & set(funds)) if selected else set(funds)
    return sorted(selected - set(exclude))


def build_run(
    mode: str | List[str],
    *,
    cohorts: Optional[List[Set[str]]] = None,
    funds: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_tag: Optional[str] = None,
    compliance_tests: Optional[List[str]] = None,
    send_email: bool = True,
) -> RunRequest | List[RunRequest]:
    """Build one request from a preset mode (or a list, one per mode, tagged ``<tag>_<mode>``)."""
    if isinstance(mode, list):
        return [
            build_run(m, cohorts=cohorts, funds=funds, start_date=start_date, end_date=end_date,
                      output_tag=f"{output_tag}_{m}" if output_tag else m,
                      compliance_tests=compliance_tests, send_email=send_email)
            for m in mode
        ]
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(MODES)}")
    if start_date is None:
        raise ValueError("start_date is required")
    preset = MODES[mode]
    trading = preset["analysis_type"] == TRADING_COMPLIANCE
    return RunRequest(
        operations=(TRADING_COMPLIANCE,) if trading else _eod_operations(preset["eod_reports"]),
        start_date=date.fromisoformat(str(start_date)),
        end_date=date.fromisoformat(str(end_date or start_date)),
        funds=tuple(select_funds(cohorts or (), funds or ())),
        compliance_tests=tuple(preset["compliance_tests"] if compliance_tests is None else compliance_tests),
        date_mode="stacked" if preset["date_mode"] == "range" else "per_day",
        generate_daily_reports=preset.get("generate_daily_reports", True),
        create_pdf=preset["create_pdf"],
        send_email=send_email,
        output_dir=preset["output_dir"],
        output_tag=output_tag or mode,
    )


def catalog() -> Dict[str, object]:
    """What a UI can offer: cohorts (loaded from the DB), funds, operations and compliance rules."""
    load_cohorts()
    return {
        "cohorts": {name: sorted(members) for name, members in COHORTS.items()},
        "funds": sorted(fund_definitions.FUND_DEFINITIONS),
        "operations": list(EOD_OPERATIONS) + [TRADING_COMPLIANCE],
        "compliance_rules": list(FULL_COMPLIANCE_TESTS),
        "date_modes": ["per_day", "stacked"],
    }


def load_cohorts() -> None:
    """Populate the cohort sets from the DB (needed before cohorts are resolved to tickers)."""
    session, _ = initialize_database()
    try:
        fund_definitions.load_cohorts_from_db(session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def execute_run(request: RunRequest | Sequence[RunRequest]) -> int:
    """Run one request or several in order, stopping at the first non-zero exit code."""
    for req in [request] if isinstance(request, RunRequest) else request:
        code = run(req).exit_code
        if code:
            return code
    return 0


def run(request: RunRequest) -> RunResult:
    if request.cohorts:
        load_cohorts()
        request = replace(request, cohorts=(), funds=tuple(select_funds(
            [COHORTS[name] for name in request.cohorts], request.funds, request.exclude)))
    elif request.exclude:
        request = replace(request, exclude=(), funds=tuple(select_funds(funds=request.funds, exclude=request.exclude)))
    if TRADING_COMPLIANCE in request.operations and len(request.operations) > 1:
        raise ValueError("trading_compliance cannot be combined with EOD operations in one request")

    result = RunResult()
    if request.date_mode == "stacked":
        result.exit_code = _run_stacked(request, result)
        return result
    for trade_date in business_date_range(request.start_date, request.end_date or request.start_date):
        result.exit_code = _run_day(request, trade_date, result)
        if result.exit_code:
            break
    return result


def _run_day(req: RunRequest, trade_date: date, result: RunResult) -> int:
    logger = _logger("fund_manager.main")
    trading = TRADING_COMPLIANCE in req.operations
    funds = _tokens(req.funds, upper=True)
    output_dir = Path(req.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s run for %s (funds: %s)", "trading_compliance" if trading else "eod",
                trade_date.isoformat(), ",".join(funds) if funds else "ALL")

    session, base_cls = initialize_database()
    try:
        fund_definitions.load_cohorts_from_db(session)
        registry = filter_registry(build_fund_registry(session, base_cls), funds)
        if not registry:
            logger.error("No funds available after applying filters")
            return 1
        loader = BulkDataLoader(session, registry)
        if trading:
            params = _trading_parameters(req, trade_date)
            ex_ante = loader.load_all_data_for_date(params.ex_ante_date, previous_date=params.vest_previous_date,
                                                    analysis_type="ex_ante")
            ex_post = loader.load_all_data_for_date(params.ex_post_date, previous_date=params.custodian_previous_date,
                                                    analysis_type="ex_post")
        else:
            params = _eod_parameters(req, trade_date)
            store = loader.load_all_data_for_date(params.trade_date, previous_date=params.previous_trade_date,
                                                  analysis_type="eod")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    day = trade_date.isoformat()
    if trading:
        logger.info(
            "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
            params.ex_ante_date, params.ex_post_date, params.vest_previous_date,
            params.custodian_date, params.custodian_previous_date,
        )
        results_ex_ante, results_ex_post, artefacts = run_trading_mode(
            registry, ex_ante_store=ex_ante, ex_post_store=ex_post, params=params,
            output_dir=output_dir, output_tag=req.output_tag,
        )
        _log_summary(logger, "ex-ante", results_ex_ante.summary)
        _log_summary(logger, "ex-post", results_ex_post.summary)
        result.summaries[f"{day} ex-ante"] = results_ex_ante.summary
        result.summaries[f"{day} ex-post"] = results_ex_post.summary
        result.paths[day] = _log_paths(logger, flatten_trading_paths(artefacts))
    else:
        logger.info("EOD dates — trade date: %s, previous business date: %s",
                    params.trade_date, params.previous_trade_date)
        processing_results, artefacts = run_eod_mode(
            registry, data_store=store, params=params, output_dir=output_dir, output_tag=req.output_tag,
        )
        _log_summary(logger, "eod", processing_results.summary)
        result.summaries[day] = processing_results.summary
        result.paths[day] = _log_paths(logger, flatten_eod_paths(artefacts))
    logger.info("Run completed successfully")
    return 0


def _run_stacked(req: RunRequest, result: RunResult) -> int:
    logger = _logger("fund_manager.main.range")
    if TRADING_COMPLIANCE in req.operations:
        raise ValueError("run_time_series only supports analysis_type='eod'")
    funds = _tokens(req.funds, upper=True)
    start, end = req.start_date, req.end_date or req.start_date
    output_dir = Path(req.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting time-series EOD run for %s funds between %s and %s",
                ",".join(funds) if funds else "ALL", start.isoformat(), end.isoformat())
    operations = _require_operations(req)

    session, base_cls = initialize_database()
    try:
        registry = filter_registry(build_fund_registry(session, base_cls), funds)
        if not registry:
            logger.error("No funds available after applying filters")
            return 1
        range_results = run_eod_range_mode(
            session, registry, start_date=start, end_date=end, operations=operations,
            compliance_tests=_tokens(req.compliance_tests), output_dir=output_dir,
            create_pdf=req.create_pdf, generate_daily_reports=req.generate_daily_reports,
            output_tag=req.output_tag,
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    if not range_results.results_by_date:
        logger.warning("No business days found within the provided range")
        return 0
    for trade_date, results in range_results.results_by_date.items():
        _log_summary(logger, trade_date.isoformat(), results.summary)
        result.summaries[trade_date.isoformat()] = results.summary
    for date_str, artefacts in range_results.daily_artefacts.items():
        result.paths[date_str] = _log_paths(logger, flatten_eod_paths(artefacts))
    if range_results.stacked_compliance:
        result.paths["stacked"] = _log_paths(
            logger, flatten_eod_paths({"compliance": range_results.stacked_compliance}))
    logger.info("Time-series run completed successfully")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eod_parameters(req: RunRequest, trade_date: date) -> EODRunParameters:
    return EODRunParameters(
        trade_date=trade_date,
        previous_trade_date=req.previous_date or business_day_offset(trade_date, -1),
        operations=_require_operations(req),
        compliance_tests=_tokens(req.compliance_tests),
        output_tag=req.output_tag,
        create_pdf=req.create_pdf,
        send_email=req.send_email,
    )


def _trading_parameters(req: RunRequest, trade_date: date) -> TradingComplianceParameters:
    ex_ante = req.ex_ante_date or trade_date
    custodian = req.custodian_date or business_day_offset(ex_ante, -1)
    return TradingComplianceParameters(
        ex_ante_date=ex_ante,
        ex_post_date=req.ex_post_date or ex_ante,
        vest_previous_date=business_day_offset(ex_ante, -1),
        custodian_date=custodian,
        custodian_previous_date=req.custodian_previous_date or business_day_offset(custodian, -1),
        compliance_tests=_tokens(req.compliance_tests),
        create_pdf=req.create_pdf,
        output_tag=req.output_tag,
    )


def _eod_operations(reports: Iterable[str]) -> Tuple[str, ...]:
    tokens = {r.lower() for r in reports}
    if not tokens or "all" in tokens:
        tokens |= {"compliance", "holdings", "nav"}
    if "holdings_nav" in tokens:
        tokens |= {"holdings", "nav"}
    wanted = {_EOD_REPORT_ALIASES[t] for t in tokens if t in _EOD_REPORT_ALIASES}
    return tuple(op for op in EOD_OPERATIONS if op in wanted)


def _require_operations(req: RunRequest) -> List[str]:
    operations = [op for op in EOD_OPERATIONS if op in req.operations]
    if not operations:
        raise ValueError("At least one report family must be selected for an EOD run")
    return operations


def _tokens(values: Iterable[str], *, upper: bool = False) -> List[str]:
    """Split comma/space separated entries and drop 'all'; duplicates are removed within an entry only."""
    out: List[str] = []
    for value in values:
        entry: List[str] = []
        for token in str(value).replace(",", " ").split():
            if token.lower() != "all" and token not in entry:
                entry.append(token)
        out.extend(t.upper() if upper else t for t in entry)
    return out


def filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return only the requested funds (all of them when ``funds`` is empty)."""
    if not funds:
        return registry
    missing = [fund for fund in funds if fund not in registry]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")
    return {fund: registry[fund] for fund in funds}


def _logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def _log_summary(logger: logging.Logger, label: str, summary: Mapping[str, object]) -> None:
    logger.info("Processing summary (%s): %s", label, ", ".join(f"{k}={v}" for k, v in summary.items()))


def _log_paths(logger: logging.Logger, paths: Dict[str, str]) -> Dict[str, str]:
    if not paths:
        logger.warning("No report artefacts were produced")
    for label, path in sorted(paths.items()):
        logger.info("Generated %s -> %s", label, path)
    return paths
