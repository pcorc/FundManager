"""Single entry point for running FundManager: ``RunRequest`` in, ``RunResult`` out.

Scripts, schedulers and a UI all go through :func:`run`. A request names the
funds (explicit tickers and/or cohort names), the operations, an optional
subset of compliance rules, and either one as-of date or a date range run
per day or stacked into one compliance workbook. :func:`build_run` turns the
preset modes in ``config.run_configurations.MODES`` into requests, and
:func:`catalog` lists what a UI can offer. :func:`validate` rejects bad input
up front, ``run(request, on_progress=...)`` reports :class:`ProgressEvent`s,
``RunResult.files`` lists the reports written and :func:`bundle` zips them.
``RunResult.views`` holds each day's results as display tables (``reporting.result_views``).
``processing.worker`` runs a JSON request in its own process.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from config import fund_definitions
from config.database import initialize_database
from config.run_configurations import COHORTS, FULL_COMPLIANCE_TESTS, MODES
from processing.bulk_data_loader import BulkDataLoader
from processing.fund import FundRegistry, build_fund_registry
from processing.fund_manager import ProgressCallback, ProgressEvent, emit_progress, progress_for_day
from processing.run_modes import (
    EODRunParameters,
    TradingComplianceParameters,
    flatten_eod_paths,
    flatten_trading_paths,
    run_eod_mode,
    run_eod_range_mode,
    run_trading_mode,
)
from reporting.result_views import build_day_view, build_trading_view
from services.compliance_checker import RULE_KEYS, RULE_LABELS
from utilities.workdays import business_date_range, business_day_offset

__all__ = ["COMPLIANCE_TEST_KEYS", "DATE_MODES", "InvalidRequest", "OPERATIONS", "ProgressEvent", "ReportFile",
           "RunRequest", "RunResult", "build_run", "bundle", "catalog", "execute_run", "filter_registry",
           "load_cohorts", "request_from_json", "request_to_json", "rule_catalog", "run", "select_funds",
           "validate"]

EOD_OPERATIONS = ("compliance", "reconciliation", "nav_reconciliation")
TRADING_COMPLIANCE = "trading_compliance"
OPERATIONS = EOD_OPERATIONS + (TRADING_COMPLIANCE,)
DATE_MODES = ("per_day", "stacked")
# "summary" is an alias of summary_metrics (the Summary sheet / PDF section), kept for existing callers.
COMPLIANCE_TEST_KEYS = ("summary",) + RULE_KEYS
_STEP_LABELS = {"summary": "Summary sheet (alias of summary_metrics)"}
_TUPLE_FIELDS = ("operations", "funds", "cohorts", "exclude", "compliance_tests")
_DATE_FIELDS = ("start_date", "end_date", "previous_date", "ex_ante_date", "ex_post_date",
                "custodian_date", "custodian_previous_date")
_SCALAR_TYPES = {"date_mode": (str,), "generate_daily_reports": (bool,), "create_pdf": (bool,),
                 "send_email": (bool,), "output_dir": (str,), "output_tag": (str, type(None))}
_EOD_REPORT_ALIASES = {"compliance": "compliance", "holdings": "reconciliation", "reconciliation": "reconciliation",
                       "nav": "nav_reconciliation", "nav_reconciliation": "nav_reconciliation"}


class InvalidRequest(ValueError):
    """A request ``run`` rejects before doing any work (bad choices, or a selection with no funds)."""


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


@dataclass(frozen=True)
class ReportFile:
    """A report a run wrote. ``day`` is the ISO date or "stacked"; ``kind`` is "excel" or "pdf"."""

    day: str
    label: str
    path: str
    kind: str


@dataclass
class RunResult:
    exit_code: int = 0
    summaries: Dict[str, Mapping[str, object]] = field(default_factory=dict)
    paths: Dict[str, Dict[str, str]] = field(default_factory=dict)  # day -> label -> path, as reported
    files: List[ReportFile] = field(default_factory=list)  # only the paths that exist on disk
    errors: List[str] = field(default_factory=list)
    # ISO day -> reporting.result_views DayView (per day, each day of a stacked range, trading days).
    views: Dict[str, Dict[str, object]] = field(default_factory=dict)


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
    if cohorts:
        _require_exact_tickers("funds", funds or ())
    selected = select_funds(cohorts or (), funds or ())
    if cohorts and not selected:
        raise InvalidRequest("Fund selection is empty: the cohorts/funds given share no tickers "
                             "(cohort sets are filled by load_cohorts())")
    return RunRequest(
        operations=(TRADING_COMPLIANCE,) if trading else _eod_operations(preset["eod_reports"]),
        start_date=date.fromisoformat(str(start_date)),
        end_date=date.fromisoformat(str(end_date or start_date)),
        funds=tuple(selected),
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
        "rule_catalog": rule_catalog(),
        "date_modes": list(DATE_MODES),
    }


def rule_catalog() -> List[Dict[str, str]]:
    """Every key ``compliance_tests`` accepts, with its display label, in run order."""
    return [{"key": key, "label": _STEP_LABELS.get(key) or RULE_LABELS[key]} for key in COMPLIANCE_TEST_KEYS]


def load_cohorts() -> None:
    """Populate the cohort sets from the DB (needed before cohorts are resolved to tickers)."""
    session, _ = initialize_database()
    try:
        fund_definitions.load_cohorts_from_db(session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Validation and JSON
# ---------------------------------------------------------------------------

def validate(request: RunRequest) -> RunRequest:
    """Raise ``InvalidRequest`` (listing the valid choices) for anything ``run`` would reject or mis-read."""
    for name in _TUPLE_FIELDS:
        if isinstance(getattr(request, name), str):
            raise InvalidRequest(f"{name} must be a list or tuple of strings, not a string")
    _check_choices("operation", request.operations, OPERATIONS)
    if not request.operations:
        raise InvalidRequest(f"At least one operation must be selected. Valid choices: {', '.join(OPERATIONS)}")
    if TRADING_COMPLIANCE in request.operations and len(request.operations) > 1:
        raise InvalidRequest("trading_compliance cannot be combined with EOD operations in one request")
    if request.date_mode not in DATE_MODES:
        raise InvalidRequest(f"Unknown date_mode {request.date_mode!r}. Valid choices: {', '.join(DATE_MODES)}")
    if request.date_mode == "stacked" and TRADING_COMPLIANCE in request.operations:
        raise InvalidRequest("run_time_series only supports analysis_type='eod'")
    for name in _DATE_FIELDS:
        value = getattr(request, name)
        if not isinstance(value, date) and (value is not None or name == "start_date"):
            raise InvalidRequest(f"{name} must be a date, got {value!r}")
    if request.end_date is not None and request.end_date < request.start_date:
        raise InvalidRequest(f"end_date {request.end_date} is before start_date {request.start_date}")
    _check_choices("compliance rule", _tokens(request.compliance_tests), COMPLIANCE_TEST_KEYS)
    _check_choices("cohort", request.cohorts, COHORTS)
    _check_choices("fund", _tokens(request.funds, upper=True) + _tokens(request.exclude, upper=True),
                   fund_definitions.FUND_DEFINITIONS)
    if request.cohorts or request.exclude:  # these selections match tickers exactly (no upper-casing/splitting)
        _require_exact_tickers("funds", request.funds)
        _require_exact_tickers("exclude", request.exclude)
    if not request.cohorts:  # cohort members come from the DB, so run() checks those selections
        _resolve_selection(request)
    return request


def _check_choices(kind: str, values: Iterable[object], valid: Iterable[str]) -> None:
    valid = list(valid)
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise InvalidRequest(f"Unknown {kind}(s): {', '.join(map(repr, unknown))}. "
                             f"Valid choices: {', '.join(valid)}")


def _require_exact_tickers(name: str, values: Iterable[object]) -> None:
    inexact = [value for value in values if _tokens([value], upper=True) != [value]]
    if inexact:
        raise InvalidRequest(f"{name} must hold exact upper-case tickers, one per entry and no 'all', when cohorts "
                             f"or exclude are given; got {', '.join(map(repr, inexact))}")


def _resolve_selection(request: RunRequest) -> RunRequest:
    """Resolve cohorts/exclude into an explicit sorted fund list; raise if nothing is left to run."""
    funds, exclude = request.funds, request.exclude
    if request.cohorts:
        load_cohorts()
        selected = select_funds([COHORTS[name] for name in request.cohorts], funds, exclude)
    elif exclude:
        selected = select_funds(funds=funds or fund_definitions.FUND_DEFINITIONS, exclude=exclude)
    else:
        return request
    if not selected:
        raise InvalidRequest(f"Fund selection is empty (cohorts={list(request.cohorts)}, funds={list(funds)}, "
                             f"exclude={list(exclude)})")
    return replace(request, cohorts=(), exclude=(), funds=tuple(selected))


def request_to_json(request: RunRequest) -> str:
    """JSON text for ``request`` (ISO dates, lists); :func:`request_from_json` reverses it."""
    data = {f.name: getattr(request, f.name) for f in fields(RunRequest)}
    for name in _TUPLE_FIELDS:
        data[name] = list(data[name])
    for name in _DATE_FIELDS:
        if data[name] is not None:
            data[name] = data[name].isoformat()
    return json.dumps(data, indent=1)


def request_from_json(text: str) -> RunRequest:
    """Parse :func:`request_to_json` output (or any subset of the fields) into a ``RunRequest``."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("A run request must be a JSON object")
    known = [f.name for f in fields(RunRequest)]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown request field(s): {', '.join(unknown)}. Valid fields: {', '.join(known)}")
    missing = [name for name in ("operations", "start_date") if data.get(name) is None]
    if missing:
        raise ValueError(f"Missing required request field(s): {', '.join(missing)}")
    for name in _TUPLE_FIELDS:
        if name in data:
            value = data[name]
            if value is not None and not isinstance(value, (str, list)):
                raise ValueError(f"{name} must be a list of strings, got {value!r}")
            data[name] = (value,) if isinstance(value, str) else tuple(value or ())
            if not all(isinstance(item, str) for item in data[name]):
                raise ValueError(f"{name} must be a list of strings, got {value!r}")
    for name, types in _SCALAR_TYPES.items():
        if name in data and not isinstance(data[name], types):
            raise ValueError(f"{name} must be {' or '.join(t.__name__ for t in types)}, got {data[name]!r}")
    for name in _DATE_FIELDS:
        if data.get(name) is not None:
            data[name] = date.fromisoformat(str(data[name]))
    return RunRequest(**data)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def execute_run(request: RunRequest | Sequence[RunRequest], on_progress: Optional[ProgressCallback] = None) -> int:
    """Run one request or several in order, stopping at the first non-zero exit code."""
    for req in [request] if isinstance(request, RunRequest) else request:
        code = run(req, on_progress).exit_code
        if code:
            return code
    return 0


def run(request: RunRequest, on_progress: Optional[ProgressCallback] = None) -> RunResult:
    """Validate and execute ``request``; ``on_progress`` gets a :class:`ProgressEvent` per step."""
    request = _resolve_selection(validate(request))

    result = RunResult()
    if request.date_mode == "stacked":
        progress = progress_for_day(on_progress, "stacked")
        result.exit_code = _run_stacked(request, result, progress)
        emit_progress(progress, "done", message="failed" if result.exit_code else "completed")
        return result
    days = business_date_range(request.start_date, request.end_date or request.start_date)
    for index, trade_date in enumerate(days, 1):
        progress = progress_for_day(on_progress, trade_date.isoformat())
        emit_progress(progress, "day", index=index, total=len(days))
        result.exit_code = _run_day(request, trade_date, result, progress)
        emit_progress(progress, "done", index=index, total=len(days),
                      message="failed" if result.exit_code else "completed")
        if result.exit_code:
            break
    return result


def _run_day(req: RunRequest, trade_date: date, result: RunResult,
             on_progress: Optional[ProgressCallback] = None) -> int:
    logger = _logger("fund_manager.main")
    day = trade_date.isoformat()
    trading = TRADING_COMPLIANCE in req.operations
    funds = _tokens(req.funds, upper=True)
    output_dir = Path(req.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s run for %s (funds: %s)", "trading_compliance" if trading else "eod",
                day, ",".join(funds) if funds else "ALL")

    emit_progress(on_progress, "db")
    session, base_cls = initialize_database()
    try:
        fund_definitions.load_cohorts_from_db(session)
        registry = filter_registry(build_fund_registry(session, base_cls), funds)
        if not registry:
            logger.error("No funds available after applying filters")
            result.errors.append(f"{day}: No funds available after applying filters")
            return 1
        loader = BulkDataLoader(session, registry)
        if trading:
            params = _trading_parameters(req, trade_date)
            emit_progress(on_progress, "load", total=len(registry), message="ex_ante")
            ex_ante = loader.load_all_data_for_date(params.ex_ante_date, previous_date=params.vest_previous_date,
                                                    analysis_type="ex_ante")
            emit_progress(on_progress, "load", total=len(registry), message="ex_post")
            ex_post = loader.load_all_data_for_date(params.ex_post_date, previous_date=params.custodian_previous_date,
                                                    analysis_type="ex_post")
        else:
            params = _eod_parameters(req, trade_date)
            emit_progress(on_progress, "load", total=len(registry), message="eod")
            store = loader.load_all_data_for_date(params.trade_date, previous_date=params.previous_trade_date,
                                                  analysis_type="eod")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        result.errors.append(f"{day}: Processing failed: {exc}")
        return 1
    finally:
        session.close()

    email_errors: List[str] = []  # failures run_*_mode only logs
    if trading:
        logger.info(
            "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
            params.ex_ante_date, params.ex_post_date, params.vest_previous_date,
            params.custodian_date, params.custodian_previous_date,
        )
        results_ex_ante, results_ex_post, artefacts = run_trading_mode(
            registry, ex_ante_store=ex_ante, ex_post_store=ex_post, params=params,
            output_dir=output_dir, output_tag=req.output_tag, on_progress=on_progress, errors=email_errors,
        )
        _log_summary(logger, "ex-ante", results_ex_ante.summary)
        _log_summary(logger, "ex-post", results_ex_post.summary)
        result.summaries[f"{day} ex-ante"] = results_ex_ante.summary
        result.summaries[f"{day} ex-post"] = results_ex_post.summary
        _record_paths(result, day, _log_paths(logger, flatten_trading_paths(artefacts)))
        _add_view(result, day, logger, lambda: build_trading_view(
            results_ex_ante, results_ex_post, getattr(getattr(artefacts, "report", None), "comparison_data", None),
            day=trade_date, compliance_tests=params.compliance_tests))
    else:
        logger.info("EOD dates — trade date: %s, previous business date: %s",
                    params.trade_date, params.previous_trade_date)
        processing_results, artefacts = run_eod_mode(
            registry, data_store=store, params=params, output_dir=output_dir, output_tag=req.output_tag,
            on_progress=on_progress, errors=email_errors,
        )
        _log_summary(logger, "eod", processing_results.summary)
        result.summaries[day] = processing_results.summary
        _record_paths(result, day, _log_paths(logger, flatten_eod_paths(artefacts)))
        _add_view(result, day, logger, lambda: build_day_view(
            processing_results, day=trade_date, operations=params.operations,
            compliance_tests=params.compliance_tests))
    result.errors.extend(f"{day}: {error}" for error in email_errors)
    logger.info("Run completed successfully")
    return 0


def _run_stacked(req: RunRequest, result: RunResult, on_progress: Optional[ProgressCallback] = None) -> int:
    logger = _logger("fund_manager.main.range")
    funds = _tokens(req.funds, upper=True)
    start, end = req.start_date, req.end_date or req.start_date
    output_dir = Path(req.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting time-series EOD run for %s funds between %s and %s",
                ",".join(funds) if funds else "ALL", start.isoformat(), end.isoformat())
    operations = _require_operations(req)

    emit_progress(on_progress, "db")
    session, base_cls = initialize_database()
    try:
        registry = filter_registry(build_fund_registry(session, base_cls), funds)
        if not registry:
            logger.error("No funds available after applying filters")
            result.errors.append("stacked: No funds available after applying filters")
            return 1
        range_results = run_eod_range_mode(
            session, registry, start_date=start, end_date=end, operations=operations,
            compliance_tests=_tokens(req.compliance_tests), output_dir=output_dir,
            create_pdf=req.create_pdf, generate_daily_reports=req.generate_daily_reports,
            output_tag=req.output_tag, on_progress=on_progress,
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        result.errors.append(f"stacked: Processing failed: {exc}")
        return 1
    finally:
        session.close()

    if not range_results.results_by_date:
        logger.warning("No business days found within the provided range")
        return 0
    for trade_date, results in range_results.results_by_date.items():
        _log_summary(logger, trade_date.isoformat(), results.summary)
        result.summaries[trade_date.isoformat()] = results.summary
        _add_view(result, trade_date.isoformat(), logger, lambda: build_day_view(
            results, day=trade_date, operations=operations, compliance_tests=_tokens(req.compliance_tests)))
    for date_str, artefacts in range_results.daily_artefacts.items():
        _record_paths(result, date_str, _log_paths(logger, flatten_eod_paths(artefacts)))
    if range_results.stacked_compliance:
        _record_paths(result, "stacked", _log_paths(
            logger, flatten_eod_paths({"compliance": range_results.stacked_compliance})))
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
        send_email=req.send_email,
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


def _add_view(result: RunResult, day: str, logger: logging.Logger, build) -> None:
    """Store ``build()`` as the day's view; a failure is logged and listed in errors, never failing the run."""
    try:
        result.views[day] = build()
    except Exception as exc:
        logger.warning("Result views for %s could not be built: %s", day, exc, exc_info=True)
        result.errors.append(f"{day}: Result views could not be built: {type(exc).__name__}: {exc}")


def _record_paths(result: RunResult, day: str, paths: Dict[str, str]) -> None:
    result.paths[day] = paths
    for label, path in sorted(paths.items()):
        if os.path.isfile(path):
            kind = label.rsplit("_", 1)[-1]
            if kind not in ("excel", "pdf"):
                kind = "pdf" if str(path).lower().endswith(".pdf") else "excel"
            result.files.append(ReportFile(day, label, str(path), kind))


def bundle(result: RunResult, zip_path: str | os.PathLike) -> Path:
    """Zip each distinct existing file in ``result.files`` once (the shared NAV/holdings workbook included)."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    names: Set[str] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for report in result.files:
            key = os.path.normcase(os.path.abspath(report.path))
            if key in seen or not os.path.isfile(report.path):
                continue
            seen.add(key)
            name = Path(report.path).name
            if name in names:
                name = f"{report.day}/{name}"
            names.add(name)
            archive.write(report.path, arcname=name)
    return zip_path
