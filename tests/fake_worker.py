"""Stand-in for ``python -m processing.worker`` in the UI tests (no DB, no imports from the repo).

    python fake_worker.py request.json --result result.json

Behaviour follows the request's ``output_tag``: ``slow`` sleeps between funds
(for lock / cancel / timeout tests), ``crash`` exits 1 without a result,
``invalid`` exits 2 like a rejected request, ``leak`` fails with a DB error that
quotes a connection string (for the redaction tests); anything else succeeds,
writing one small Excel (and PDF when ``create_pdf``) file per day into
``output_dir``. stderr gets a non-ASCII line, like the real worker's log.
"""
import json
import os
import sys
import time
from pathlib import Path


def emit(**event):
    sys.stdout.write(json.dumps({"stage": event.pop("stage"), "day": None, "fund": None, "index": None,
                                 "total": None, "message": "", **event}) + "\n")
    sys.stdout.flush()


def main() -> int:
    request_path, result_path = Path(sys.argv[1]), Path(sys.argv[sys.argv.index("--result") + 1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    tag = request.get("output_tag") or ""
    sys.stderr.write(f"cwd={os.getcwd()}\nPYTHONPATH={os.environ.get('PYTHONPATH', '')}\n"
                     f"PYTHONUNBUFFERED={os.environ.get('PYTHONUNBUFFERED', '')}\n"
                     "EOD dates — trade date: fake\n")
    sys.stderr.flush()
    if tag == "crash":
        sys.stderr.write("Traceback: boom\n")
        return 1
    if tag == "leak":
        secret = "mysql+pymysql://fmuser:Hunter2Secret@db.internal/fm"
        sys.stderr.write(f"OperationalError: cannot connect to {secret}\n")
        result_path.write_text(json.dumps({"exit_code": 1, "summaries": {}, "paths": {}, "files": [],
                                           "errors": [f"ArgumentError: Could not parse URL '{secret}'"]}),
                               encoding="utf-8")
        return 1
    if tag == "invalid":
        result_path.write_text(json.dumps({"exit_code": 2, "summaries": {}, "paths": {}, "files": [],
                                           "errors": ["Invalid request: bad"]}), encoding="utf-8")
        return 2
    funds = request.get("funds") or ["KNG", "RDVI"]
    out = Path(request["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("logs", "compliance.log").write_text("fake log\n", encoding="utf-8")
    day = request["start_date"]
    emit(stage="day", day=day, index=1, total=1)
    emit(stage="db", day=day)
    emit(stage="load", day=day, total=len(funds), message="eod")
    emit(stage="process", day=day, message="eod")
    for index, fund in enumerate(funds, 1):
        emit(stage="fund", day=day, fund=fund, index=index, total=len(funds), message="eod")
        if tag == "slow":
            time.sleep(30)
    emit(stage="reports", day=day)
    files, paths = [], {}
    excel = out / "Compliance" / f"compliance_report_{day}.xlsx"
    excel.parent.mkdir(parents=True, exist_ok=True)
    excel.write_bytes(b"PK fake excel")
    paths["compliance_excel"] = str(excel)
    files.append({"day": day, "label": "compliance_excel", "path": str(excel), "kind": "excel"})
    if request.get("create_pdf"):
        pdf = out / "Compliance" / f"compliance_report_{day}.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        paths["compliance_pdf"] = str(pdf)
        files.append({"day": day, "label": "compliance_pdf", "path": str(pdf), "kind": "pdf"})
    emit(stage="done", day=day, index=1, total=1, message="completed")
    result = {"exit_code": 0, "summaries": {day: {"processed_funds": len(funds), "funds_with_errors": 0}},
              "paths": {day: paths}, "files": files, "errors": []}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
