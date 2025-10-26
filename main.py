"""Minimal command-line entry point wiring FundManager run modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from config.database import initialize_database
from config.fund_registry import FundRegistry
from processing.cli import parse_arguments, resolve_eod_parameters, resolve_trading_parameters
from processing.run_executor import RunExecutor, filter_registry


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=_resolve_log_level(),
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

    session, base_cls = initialize_database()

    try:
        registry = FundRegistry.from_database(session, base_cls)
        registry = filter_registry(registry, options.funds)
        if not registry.funds:
            logger.error("No funds available after applying filters")
            return 1

        executor = RunExecutor(session, base_cls, registry, output_dir, logger)

        if options.analysis_type == "trading_compliance":
            params = resolve_trading_parameters(options)
            logger.info(
                "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
                params.ex_ante_date,
                params.ex_post_date,
                params.vest_previous_date,
                params.custodian_date,
                params.custodian_previous_date,
            )
            outcome = executor.run_trading_compliance(params)
            executor.log_processing_summary("ex-ante", outcome.results_ex_ante.summary)
            executor.log_processing_summary("ex-post", outcome.results_ex_post.summary)
            executor.log_generated_paths(outcome.generated_paths)
        else:
            params = resolve_eod_parameters(options)
            logger.info(
                "EOD dates — trade date: %s, previous business date: %s",
                params.trade_date,
                params.previous_trade_date,
            )
            outcome = executor.run_eod(params)
            executor.log_processing_summary("eod", outcome.results.summary)
            executor.log_generated_paths(outcome.generated_paths)

    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    logger.info("Run completed successfully")
    return 0


def _resolve_log_level() -> str:
    import os

    return os.getenv("LOG_LEVEL", "INFO").upper()


if __name__ == "__main__":
    RUNTIME_OVERRIDES = {
        "analysis_type": "eod",
        "as_of_date": "2025-10-24",
        "funds": ["DOGG"],
        "previous_date": "2025-10-23",
        "compliance_tests": ["limit-checks", "trade-size"],
        "create_pdf": True,
        "output_dir": "./outputs",
    }
    raise SystemExit(main(overrides=RUNTIME_OVERRIDES))