"""Command line entry point for the FundManager reporting suite."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

from config.database import initialize_database
from config.fund_registry import FundRegistry
from processing.run_modes import (
    fetch_data_stores,
    flatten_eod_paths,
    flatten_trading_paths,
    plan_eod_requests,
    plan_trading_requests,
    run_eod_mode,
    run_trading_mode,
)
from utilities.cli_options import (
    parse_arguments,
    resolve_eod_parameters,
    resolve_trading_parameters,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point used by ``python -m`` or direct execution."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fund_manager.main")

    options = parse_arguments(argv)
    output_dir = Path(options.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting %s run for %s (funds: %s)",
        options.analysis_type,
        options.as_of_date.isoformat(),
        ",".join(options.funds) if options.funds else "ALL",
    )

    params = None
    data_stores: Mapping[str, object] = {}

    session, base_cls = initialize_database()
    try:
        registry = FundRegistry.from_database(session, base_cls)
        registry = _filter_registry(registry, options.funds)
        if not registry.funds:
            logger.error("No funds available after applying filters")
            return 1

        if options.analysis_type == "trading_compliance":
            params = resolve_trading_parameters(options)
            requests = plan_trading_requests(params)
        else:
            params = resolve_eod_parameters(options)
            requests = plan_eod_requests(params)

        data_stores = fetch_data_stores(session, base_cls, registry, requests)

    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    if options.analysis_type == "trading_compliance":
        assert params is not None
        trading_params = params
        logger.info(
            "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
            trading_params.ex_ante_date,
            trading_params.ex_post_date,
            trading_params.vest_previous_date,
            trading_params.custodian_date,
            trading_params.custodian_previous_date,
        )
        results_ex_ante, results_ex_post, artefacts = run_trading_mode(
            registry,
            ex_ante_store=data_stores["ex_ante"],
            ex_post_store=data_stores["ex_post"],
            params=trading_params,
            output_dir=output_dir,
        )

        _log_processing_summary(logger, "ex-ante", results_ex_ante.summary)
        _log_processing_summary(logger, "ex-post", results_ex_post.summary)
        _log_generated_paths(logger, flatten_trading_paths(artefacts))
    else:
        assert params is not None
        eod_params = params
        logger.info(
            "EOD dates — trade date: %s, previous business date: %s",
            eod_params.trade_date,
            eod_params.previous_trade_date,
        )
        processing_results, artefacts = run_eod_mode(
            registry,
            data_store=data_stores["eod"],
            params=eod_params,
            output_dir=output_dir,
        )

        _log_processing_summary(logger, "eod", processing_results.summary)
        _log_generated_paths(logger, flatten_eod_paths(artefacts))

    logger.info("Run completed successfully")
    return 0


def _filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return a registry containing only the requested funds (if any)."""

    if not funds:
        return registry

    missing = [fund for fund in funds if fund not in registry.funds]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")

    filtered = FundRegistry()
    filtered.funds = {fund: registry.funds[fund] for fund in funds}
    return filtered


def _log_processing_summary(logger: logging.Logger, label: str, summary: Mapping[str, object]) -> None:
    pretty = ", ".join(f"{key}={value}" for key, value in summary.items())
    logger.info("Processing summary (%s): %s", label, pretty)


def _log_generated_paths(logger: logging.Logger, paths: Mapping[str, str]) -> None:
    if not paths:
        logger.warning("No report artefacts were produced")
        return
    for label, path in sorted(paths.items()):
        logger.info("Generated %s -> %s", label, path)


if __name__ == "__main__":
    raise SystemExit(main())