"""Command-line entry point for FundManager.

    python main.py                       # runs the request configured below
    python main.py eod --cohort ETF_FUNDS --start 2026-10-30 --tag etfs --no-email
    python main.py time_series --fund KNG --fund RDVI --start 2026-07-01 --end 2026-09-30

Everything goes through ``processing.api`` (``build_run`` presets -> ``run``);
schedulers and a UI should call that module directly. More examples live in
docs/build_run_examples.txt.
"""
from __future__ import annotations

import argparse
import sys

from config.fund_definitions import ETF_FUNDS
from config.run_configurations import COHORTS, MODES
from processing.api import build_run, execute_run, load_cohorts
from services.compliance_checker import RULE_KEYS


def _parse_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--cohort", action="append", default=[], choices=sorted(COHORTS))
    parser.add_argument("--fund", action="append", default=[])
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (as-of date, or range start)")
    parser.add_argument("--end", help="YYYY-MM-DD range end (default: --start)")
    parser.add_argument("--test", action="append", choices=[*RULE_KEYS, "all"],
                        help="compliance rule to run (repeatable; default: all)")
    parser.add_argument("--tag", help="output tag added to report file names")
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args(argv)
    return build_run(
        args.mode,
        cohorts=[COHORTS[name] for name in args.cohort] or None,
        funds=args.fund or None,
        start_date=args.start,
        end_date=args.end,
        output_tag=args.tag,
        compliance_tests=args.test,
        send_email=not args.no_email,
    )


if __name__ == "__main__":
    load_cohorts()
    if len(sys.argv) > 1:
        request = _parse_cli(sys.argv[1:])
    else:
        request = build_run(
            "eod",
            cohorts=[ETF_FUNDS],
            start_date="2026-10-30",
            end_date="2026-10-30",
            output_tag="etfs",
            send_email=False,
        )
    raise SystemExit(execute_run(request))
