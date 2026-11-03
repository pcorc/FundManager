"""Golden-master driver.

    python tools/golden/golden.py capture [scenario ...]   # live DB -> recorded data -> baseline
    python tools/golden/golden.py check   [scenario ...]   # replay recorded data -> current, diff vs baseline
    python tools/golden/golden.py snapshot NAME            # keep a copy of baseline + data in snapshots/NAME
    python tools/golden/golden.py rebaseline [scenario ...]  # accept current outputs as the new baseline
    python tools/golden/golden.py report OUT.md [--against SNAPSHOT] [--no-run] [scenario ...]
                                                           # markdown before/after review for sign-off

capture runs each scenario twice: once against the live DB (recording) and once
replaying the recording (the baseline), then diffs the two as a harness
self-test. check never touches the DB's business views. Set GOLDEN_WORK to
share one recording/baseline between git worktrees.

rebaseline copies CURRENT/<scenario> (from the last check/report) over the
baseline; equivalent-request scenarios share their base's baseline, so they
are only checked against it. report re-runs the scenarios (unless --no-run)
and diffs baseline (or snapshots/SNAPSHOT) vs current; exit code 0 = written.

Scenario runs use GOLDEN_PYTHON (default: the project's Python 3.11) with
PYTHONHASHSEED=0. The differ needs openpyxl + pypdf in the interpreter that
runs this script.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tools.golden.differ import _norm_text, diff_roots  # noqa: E402
from tools.golden.report import render, review_scenario  # noqa: E402
from tools.golden.scenarios import EQUIVALENT_REQUESTS, SCENARIOS  # noqa: E402

WORK = Path(os.getenv("GOLDEN_WORK", HERE / "_work"))  # recorded data + baseline (shareable across worktrees)
CURRENT = HERE / "_work" / "current"  # this checkout's outputs
PYTHON = os.getenv(
    "GOLDEN_PYTHON", r"C:\Users\PatrickCorcoran\AppData\Local\Programs\Python\Python311\python.exe"
)


def _base(scenario: str) -> str:
    """Scenario whose recorded data and baseline an equivalent request reuses."""
    return EQUIVALENT_REQUESTS[scenario][0] if scenario in EQUIVALENT_REQUESTS else scenario


def _run(scenario: str, mode: str, run_root: Path) -> None:
    shutil.rmtree(run_root, ignore_errors=True)
    env = dict(os.environ, PYTHONHASHSEED="0", FUNDMANAGER_EMAIL_MODE="display")
    subprocess.run(
        [PYTHON, str(HERE / "runner.py"), scenario, mode, str(WORK / "data" / _base(scenario)), str(run_root)],
        env=env, check=True,
    )


def _report(name: str, problems) -> bool:
    print(f"{'OK  ' if not problems else 'DIFF'} {name}")
    for p in problems:
        print(f"       {p}")
    return not problems


def _rehome(text: str, src: Path, dst: Path) -> str:
    for a, b in ((str(src), str(dst)), (str(src).replace("\\", "/"), str(dst).replace("\\", "/"))):
        text = text.replace(a.replace("\\", "\\\\"), b.replace("\\", "\\\\")).replace(a, b)
    return text


def _copy_run(src: Path, dst: Path, home: Path = None) -> None:
    """Copy a run root to dst, re-pointing the run-root paths in its meta/ (logs/emails/status) at home (default dst).

    meta/ is written as UTF-8 by the runner; cwd/ (locale-encoded app logs, never diffed) is copied verbatim.
    """
    home = home or dst
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    for f in (dst / "meta").rglob("*"):
        if f.is_file():
            text = f.read_bytes().decode("utf-8")
            moved = _rehome(text, src, home)
            if _norm_text(moved, home) != _norm_text(text, src):
                raise RuntimeError(f"re-pointing paths changed {f} beyond its run root")
            f.write_bytes(moved.encode("utf-8"))


def _unknown(names) -> bool:
    bad = [n for n in names if n not in SCENARIOS and n not in EQUIVALENT_REQUESTS]
    if bad:
        print(f"unknown scenario(s): {', '.join(bad)}; choose from {', '.join([*SCENARIOS, *EQUIVALENT_REQUESTS])}")
    return bool(bad)


def snapshot(args) -> int:
    if len(args) != 1 or not re.fullmatch(r"[\w-]+", args[0]):
        print("usage: golden.py snapshot NAME   (a plain folder name)")
        return 2
    dst = WORK / "snapshots" / args[0]
    if dst.exists():
        print(f"snapshot already exists: {dst}")
        return 1
    tmp = dst.with_name(dst.name + ".partial")  # renamed into place only once complete
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for src in sorted(p for p in (WORK / "baseline").iterdir() if p.is_dir()):
            _copy_run(src, tmp / src.name, dst / src.name)
        shutil.copytree(WORK / "data", tmp / "data")
        tmp.rename(dst)
    except (OSError, UnicodeDecodeError, RuntimeError) as exc:
        print(f"snapshot not written: {exc}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"snapshot written: {dst}")
    return 0


def rebaseline(names) -> int:
    names = names or [*SCENARIOS, *EQUIVALENT_REQUESTS]
    if _unknown(names):
        return 2
    missing = [n for n in names if not (CURRENT / n / "meta" / "status.json").is_file()]
    if missing:
        print(f"no current output for {', '.join(missing)}; run check or report first")
        return 1
    bases = [n for n in names if n not in EQUIVALENT_REQUESTS]
    staging = WORK / ".rebaseline-staging"  # copy everything first so a failure leaves the baseline untouched
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for name in bases:
            _copy_run(CURRENT / name, staging / name, WORK / "baseline" / name)
    except (OSError, UnicodeDecodeError, RuntimeError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"baseline unchanged, could not copy current outputs: {exc}")
        return 1
    try:
        for name in bases:
            shutil.rmtree(WORK / "baseline" / name, ignore_errors=True)
            (staging / name).rename(WORK / "baseline" / name)
            print(f"REBASELINED {name}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok = True
    for name in (n for n in names if n in EQUIVALENT_REQUESTS):  # checked against their (new) base
        ok &= _report(name, diff_roots(WORK / "baseline" / _base(name), CURRENT / name))
    return 0 if ok else 1


def report(args) -> int:
    parser = argparse.ArgumentParser(prog="golden.py report")
    parser.add_argument("out")
    parser.add_argument("scenarios", nargs="*")
    parser.add_argument("--against", help="snapshot name (default: the baseline)")
    parser.add_argument("--no-run", action="store_true", help="reuse existing current outputs")
    opts = parser.parse_intermixed_args(args)
    names = opts.scenarios or [*SCENARIOS, *EQUIVALENT_REQUESTS]
    if _unknown(names):
        return 2
    ref = WORK / "snapshots" / opts.against if opts.against else WORK / "baseline"
    if not ref.is_dir():
        print(f"reference not found: {ref}")
        return 1
    reviews = []
    for name in names:
        if not opts.no_run:
            _run(name, "replay", CURRENT / name)
        rv = review_scenario(name, ref / _base(name), CURRENT / name)
        print(f"{'DIFF' if rv.changed else 'OK  '} {name}")
        reviews.append(rv)
    label = f"snapshot `{opts.against}`" if opts.against else "baseline"
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(reviews, label), encoding="utf-8")
    print(f"report written: {out.resolve()}")
    return 0


def main(argv) -> int:
    commands = {"snapshot": snapshot, "rebaseline": rebaseline, "report": report}
    if len(argv) >= 2 and argv[1] in commands:
        return commands[argv[1]](argv[2:])
    if len(argv) < 2 or argv[1] not in ("capture", "check"):
        print(__doc__)
        return 2
    command, names = argv[1], argv[2:] or list(SCENARIOS) + (list(EQUIVALENT_REQUESTS) if argv[1] == "check" else [])
    ok = True
    for name in names:
        if command == "capture":
            shutil.rmtree(WORK / "data" / name, ignore_errors=True)
            _run(name, "record", WORK / "record" / name)
            _run(name, "replay", WORK / "baseline" / name)
            ok &= _report(f"{name} (replay fidelity)", diff_roots(WORK / "record" / name, WORK / "baseline" / name))
        else:
            _run(name, "replay", CURRENT / name)
            ok &= _report(name, diff_roots(WORK / "baseline" / _base(name), CURRENT / name))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
