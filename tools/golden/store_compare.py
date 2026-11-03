"""Compare the BulkDataStore built by an old commit's loader with the working tree's, against the live DB.

Usage (project interpreter, from the repo root):
    python tools/golden/store_compare.py [--old REV] [--self] [--reuse]

--old REV   commit whose processing/config/utilities/services are exported (default ad3df9b)
--self      compare OLD with a second OLD run (self-consistency baseline)
--reuse     reuse an existing OLD pickle instead of reloading it

Each side runs in its own isolated subprocess (python -I) that imports only that side's code,
loads every LOADS entry for the full fund registry (read-only SELECTs) and pickles plain data
under the gitignored tools/golden/_work/. Every key the NEW store keeps must match exactly
(assert_frame_equal(check_exact=True), dict key order included); keys the NEW side no longer
produces are reported, and any not listed in the REMOVED_* sets fails the comparison.
"""
from __future__ import annotations

import argparse
import copy
import io
import logging
import os
import pickle
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "tools" / "golden" / "_work"
PYTHON = r"C:/Users/PatrickCorcoran/AppData/Local/Programs/Python/Python311/python.exe"
CODE_DIRS = ("processing", "config", "utilities", "services")

LOADS = (  # label, target_date, previous_date, analysis_type
    ("eod_2026-10-30", "2026-10-30", "2026-10-29", "eod"),
    ("eod_2026-10-29", "2026-10-29", "2026-10-28", "eod"),
    ("ex_ante_2026-10-30", "2026-10-30", "2026-10-29", "ex_ante"),
    ("ex_post_2026-10-30", "2026-10-30", "2026-10-29", "ex_post"),
)

# Intentionally dropped by Track C (unread views, unread trade splits, dead enrichment).
REMOVED_DATA_KEYS = {
    "bbg_unresolved", "bbg_unresolved_t1", "assignments", "assignments_t1",
    "block_options", "block_options_t1", "basket", "basket_t1", "sg_holdings", "sg_holdings_t1",
    "equity_trades_t1", "option_trades", "option_trades_t1", "treasury_trades", "treasury_trades_t1",
}
REMOVED_STORE_ATTRS = {"loaded_data_types"}
REMOVED_CONFIG_KEYS = {"account_numbers", "account_number_custodian"}

_SQL_FROM = re.compile(r"\bFROM\s+([\w.`]+)", re.IGNORECASE)


# ----------------------------------------------------------------------
# Child: load with one side's code and pickle plain data
# ----------------------------------------------------------------------
def _dump(code_root: str, out_path: str) -> int:
    root_path = Path(code_root).resolve()
    sys.path.insert(0, str(root_path))
    from datetime import date

    phase = ["init"]
    logs: list = []      # (phase, logger, level, message) for non-SQLAlchemy loggers
    queries: list = []   # (phase, relation) for each SQL statement SQLAlchemy logged

    class Grab(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if rec.name.startswith("sqlalchemy"):
                match = _SQL_FROM.search(msg) if msg.lstrip().upper().startswith("SELECT") else None
                if match:
                    queries.append((phase[0], match.group(1).replace("`", "")))
                return
            logs.append((phase[0], rec.name, rec.levelno, msg))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(Grab())
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)  # SQLAlchemy defaults to WARN; only Grab sees these

    from config.database import initialize_database
    from processing.bulk_data_loader import BulkDataLoader
    from processing.fund import build_fund_registry

    for mod in ("config.database", "processing.bulk_data_loader", "processing.fund"):
        mod_file = Path(sys.modules[mod].__file__).resolve()
        if root_path not in mod_file.parents:
            raise RuntimeError(f"{mod} imported from {mod_file}, not {root_path}")

    out = {"code_root": str(root_path), "loads": {}, "logs": logs, "queries": queries}
    session, base_cls = initialize_database()
    try:
        for label, target, previous, analysis in LOADS:
            phase[0] = label
            registry = build_fund_registry(session, base_cls)
            store = BulkDataLoader(session, registry).load_all_data_for_date(
                date.fromisoformat(target), previous_date=date.fromisoformat(previous), analysis_type=analysis,
            )
            out["loads"][label] = {
                "store": dict(vars(store)),
                "funds": {
                    name: {
                        "config": copy.deepcopy(fund.config),
                        "attrs": {k: v for k, v in vars(fund).items() if k not in ("config", "data")},
                    }
                    for name, fund in registry.items()
                },
            }
    finally:
        session.close()
    Path(out_path).write_bytes(pickle.dumps(out))
    return 0


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def _export(rev: str) -> Path:
    dest = WORK / "store_compare_old" / rev
    if not (dest / "processing").is_dir():
        blob = subprocess.run(["git", "archive", rev, *CODE_DIRS], cwd=REPO, check=True,
                              capture_output=True).stdout
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
            tar.extractall(dest)
    return dest


def _child_env() -> dict:
    env = os.environ.copy()
    if not env.get("DB_CONNECTION_STRING"):
        from dotenv import dotenv_values
        value = dotenv_values(REPO / ".env").get("DB_CONNECTION_STRING")
        if value:
            env["DB_CONNECTION_STRING"] = value
    return env


def _run_side(code_root: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([PYTHON, "-I", str(Path(__file__).resolve()), "_dump", str(code_root), str(out)],
                   cwd=WORK, env=_child_env(), check=True)
    return pickle.loads(out.read_bytes())


def _same(a, b, path: str, errors: list) -> None:
    import pandas as pd

    if type(a) is not type(b):
        errors.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as exc:
            errors.append(f"{path}: {exc}")
    elif isinstance(a, pd.Series):
        try:
            pd.testing.assert_series_equal(a, b, check_exact=True)
        except AssertionError as exc:
            errors.append(f"{path}: {exc}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            errors.append(f"{path}: keys {list(a)} != {list(b)}")
        for key in a:
            if key in b:
                _same(a[key], b[key], f"{path}[{key!r}]", errors)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            errors.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]", errors)
    elif a is b or (a != a and b != b):  # identical, or both NaN/NaT
        pass
    elif not bool(a == b):
        errors.append(f"{path}: {a!r} != {b!r}")


def _split_keys(old: dict, new: dict, path: str, allowed: set, errors: list, removed: set) -> tuple:
    """Return (old restricted to new's keys, new); record removed keys, flag added/unexpected ones."""
    gone = [k for k in old if k not in new]
    added = [k for k in new if k not in old]
    removed.update(gone)
    if added:
        errors.append(f"{path}: keys added {added}")
    unexpected = [k for k in gone if k not in allowed]
    if unexpected:
        errors.append(f"{path}: keys removed unexpectedly {unexpected}")
    return {k: v for k, v in old.items() if k in new}, new


def compare(old: dict, new: dict) -> tuple:
    errors: list = []
    removed = {"store": set(), "data": set(), "config": set(), "attrs": set()}
    if list(old["loads"]) != list(new["loads"]):
        errors.append(f"loads {list(old['loads'])} != {list(new['loads'])}")
    for label in old["loads"]:
        if label not in new["loads"]:
            continue
        o, n = old["loads"][label], new["loads"][label]
        o_store, n_store = _split_keys(o["store"], n["store"], f"{label}.store", REMOVED_STORE_ATTRS,
                                       errors, removed["store"])
        o_data, n_data = o_store.pop("fund_data", {}), n_store.pop("fund_data", {})
        _same(o_store, n_store, f"{label}.store", errors)  # date, fund_metadata, gics_mapping, loaded_funds
        if list(o_data) != list(n_data):
            errors.append(f"{label}.fund_data: funds {list(o_data)} != {list(n_data)}")
        for fund in o_data:
            if fund in n_data:
                ok, nk = _split_keys(o_data[fund], n_data[fund], f"{label}.fund_data[{fund}]",
                                     REMOVED_DATA_KEYS, errors, removed["data"])
                _same(ok, nk, f"{label}.fund_data[{fund}]", errors)
        if list(o["funds"]) != list(n["funds"]):
            errors.append(f"{label}.registry: {list(o['funds'])} != {list(n['funds'])}")
        for fund in o["funds"]:
            if fund in n["funds"]:
                for part, allowed in (("config", REMOVED_CONFIG_KEYS), ("attrs", set())):
                    a, b = _split_keys(o["funds"][fund][part], n["funds"][fund][part], f"{label}.{fund}.{part}",
                                       allowed, errors, removed[part])
                    _same(a, b, f"{label}.{fund}.{part}", errors)
    old_logs = [entry for entry in old["logs"] if entry[0] != "init"]
    new_logs = [entry for entry in new["logs"] if entry[0] != "init"]
    if old_logs != new_logs:
        errors.append(f"non-SQL log lines differ:\n  old={old_logs}\n  new={new_logs}")
    return errors, removed


def _query_diff(old: dict, new: dict) -> list:
    lines = []
    for phase in ["init", *old["loads"]]:
        o = [q for p, q in old["queries"] if p == phase]
        n = [q for p, q in new["queries"] if p == phase]
        dropped = [q for q in o if q not in n]
        added = [q for q in n if q not in o]
        lines.append(f"  {phase}: {len(o)} -> {len(n)} SELECTs"
                     + (f"; no longer queried: {sorted(set(dropped))}" if dropped else "")
                     + (f"; newly queried: {sorted(set(added))}" if added else ""))
    return lines


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "_dump":
        return _dump(argv[2], argv[3])
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="ad3df9b")
    parser.add_argument("--self", dest="self_check", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv[1:])

    old_root = _export(args.old)
    old_pkl = WORK / "store_compare" / f"old_{args.old}.pkl"
    old = pickle.loads(old_pkl.read_bytes()) if args.reuse and old_pkl.exists() else _run_side(old_root, old_pkl)
    if args.self_check:
        new = _run_side(old_root, WORK / "store_compare" / f"old_{args.old}_again.pkl")
        label = f"OLD({args.old}) vs OLD({args.old})"
    else:
        new = _run_side(REPO, WORK / "store_compare" / "new.pkl")
        label = f"OLD({args.old}) vs NEW(worktree)"

    errors, removed = compare(old, new)
    print(f"store_compare {label}: loads={', '.join(old['loads'])}")
    for part, keys in removed.items():
        if keys:
            print(f"  removed {part}: {sorted(keys)}")
    print(f"non-SQL log lines during loads: {sum(e[0] != 'init' for e in old['logs'])} -> "
          f"{sum(e[0] != 'init' for e in new['logs'])}")
    print("SQL statements per phase (old -> new):")
    print("\n".join(_query_diff(old, new)))
    for err in errors:
        print(f"MISMATCH {err}")
    print("RESULT: " + ("OK" if not errors else f"FAIL ({len(errors)} mismatches)"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
