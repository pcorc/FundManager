"""Golden-master driver.

    python tools/golden/golden.py capture [scenario ...]   # live DB -> recorded data -> baseline
    python tools/golden/golden.py check   [scenario ...]   # replay recorded data -> current, diff vs baseline

capture runs each scenario twice: once against the live DB (recording) and once
replaying the recording (the baseline), then diffs the two as a harness
self-test. check never touches the DB's business views.

Scenario runs use GOLDEN_PYTHON (default: the project's Python 3.11) with
PYTHONHASHSEED=0. The differ needs openpyxl + pypdf in the interpreter that
runs this script.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tools.golden.differ import diff_roots  # noqa: E402
from tools.golden.scenarios import EQUIVALENT_REQUESTS, SCENARIOS  # noqa: E402

WORK = HERE / "_work"
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


def main(argv) -> int:
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
            _run(name, "replay", WORK / "current" / name)
            ok &= _report(name, diff_roots(WORK / "baseline" / _base(name), WORK / "current" / name))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
