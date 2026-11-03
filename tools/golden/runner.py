"""Run one golden scenario in this process with the DB touchpoints recorded or replayed.

Usage (invoked by golden.py, one subprocess per scenario):
    python runner.py <scenario> <record|replay> <data_dir> <run_root>

The three live-code DB touchpoints are wrapped:
    BulkDataLoader.load_all_data_for_date   (all holdings / NAV / trades views)
    config.fund_definitions.load_cohorts_from_db
    processing.fund._load_account_numbers
Replay is keyed (not sequence-based) so refactored code may call them in a
different order or a different number of times; an unrecorded key raises.
Email is always intercepted at EmailManager._send_email and never sent.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_MISSING = "__golden_missing__"


def _key(target_date, previous_date, analysis_type) -> str:
    return f"{target_date}|{previous_date}|{analysis_type}"


def install_hooks(mode: str, data_dir: Path):
    import config.fund_definitions as fd
    import processing.bulk_data_loader as bdl
    import processing.fund as fund_mod

    data_dir.mkdir(parents=True, exist_ok=True)
    loads_path = data_dir / "loads.pkl"
    static_path = data_dir / "static.pkl"
    loads = pickle.loads(loads_path.read_bytes()) if mode == "replay" else {}
    static = pickle.loads(static_path.read_bytes()) if mode == "replay" else {}

    orig_load = bdl.BulkDataLoader.load_all_data_for_date
    orig_cohorts = fd.load_cohorts_from_db
    orig_accounts = fund_mod._load_account_numbers

    loader_logger = logging.getLogger(bdl.__name__)

    def load(self, target_date, *, previous_date=None, analysis_type=None):
        key = _key(target_date, previous_date, analysis_type)
        if mode == "record":
            seen: list = []
            grab = logging.Handler()
            grab.emit = lambda rec: seen.append((rec.name, rec.levelno, rec.getMessage()))
            loader_logger.addHandler(grab)
            try:
                store = orig_load(self, target_date, previous_date=previous_date, analysis_type=analysis_type)
            finally:
                loader_logger.removeHandler(grab)
            side = {n: (copy.deepcopy(f.config), getattr(f, "expense_ratio", _MISSING))
                    for n, f in self.fund_registry.items()}
            loads[key] = (pickle.dumps(store), side, seen)
            loads_path.write_bytes(pickle.dumps(loads))
            return pickle.loads(loads[key][0])
        if key not in loads:
            raise KeyError(f"golden replay: no recorded load for {key}")
        blob, side, seen = loads[key]
        for name, level, message in seen:  # re-emit the loader's own log lines
            logging.getLogger(name).log(level, "%s", message)
        for name, fund in self.fund_registry.items():
            if name in side:
                cfg, expense = side[name]
                fund.config.clear()
                fund.config.update(copy.deepcopy(cfg))
                if expense != _MISSING:
                    fund.expense_ratio = expense
        return pickle.loads(blob)

    def cohorts(session):
        if mode == "record":
            orig_cohorts(session)
            static["cohorts"] = {id_: sorted(s) for id_, s in fd._STRATEGY_TO_COHORT.items()}
            static_path.write_bytes(pickle.dumps(static))
            return
        for strategy, target in fd._STRATEGY_TO_COHORT.items():
            target.clear()
            target.update(static["cohorts"][strategy])

    def accounts(session, base_cls):
        if mode == "record":
            static["accounts"] = orig_accounts(session, base_cls)
            static_path.write_bytes(pickle.dumps(static))
        return copy.deepcopy(static["accounts"])

    bdl.BulkDataLoader.load_all_data_for_date = load
    fd.load_cohorts_from_db = cohorts
    fund_mod._load_account_numbers = accounts


def install_email_capture(emails: list):
    from notifications.email_manager import EmailManager

    def capture(self, recipients, subject, body_html, attachments):
        emails.append({
            "recipients": list(recipients),
            "subject": subject,
            "body_html": body_html,
            "attachments": [str(a) for a in attachments],
        })

    EmailManager._send_email = capture


def _redirect(cfg, out_dir: Path):
    """Point a run config (legacy dict or RunRequest) at the scenario's output dir."""
    if isinstance(cfg, dict):
        return {**cfg, "output_dir": str(out_dir)}
    from dataclasses import replace
    return replace(cfg, output_dir=str(out_dir))


def main(argv):
    scenario, mode, data_dir, run_root = argv[1], argv[2], Path(argv[3]), Path(argv[4])
    sys.argv = sys.argv[:1]  # main.main() parses sys.argv via argparse
    out_dir, meta_dir = run_root / "out", run_root / "meta"
    for d in (out_dir, meta_dir, run_root / "cwd"):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(run_root / "cwd")  # relative logs/compliance.log lands here, not in the repo

    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.FileHandler(meta_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)

    install_hooks(mode, data_dir)
    emails: list = []
    install_email_capture(emails)

    from tools.golden.scenarios import SCENARIOS
    import config.fund_definitions as fd
    from config.database import initialize_database
    import main as app

    status = {"exit_code": None, "error": None}
    try:
        session, _ = initialize_database()
        try:
            fd.load_cohorts_from_db(session)
        finally:
            session.close()
        spec = dict(SCENARIOS[scenario])
        mode_arg = spec.pop("mode")
        spec["cohorts"] = [getattr(fd, name) for name in spec.get("cohorts", [])] or None
        cfg = app.build_run(mode_arg, **spec)
        if isinstance(cfg, list):
            cfg = [_redirect(c, out_dir) for c in cfg]
        else:
            cfg = _redirect(cfg, out_dir)
        status["exit_code"] = app.execute_run(cfg)
    except BaseException as exc:  # record, don't hide: an exception is part of the output
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
    handler.flush()

    (meta_dir / "emails.json").write_text(json.dumps(emails, indent=1, default=str), encoding="utf-8")
    (meta_dir / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
