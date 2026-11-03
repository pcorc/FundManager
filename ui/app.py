"""FundManager web UI (Streamlit): New run, Run history and Reports.

    python -m streamlit run ui/app.py [--server.address 0.0.0.0 --server.port 8501]

The app holds no business logic: it builds a ``processing.api.RunRequest``,
validates it in-process and hands it to ``ui.runner``, which runs
``python -m processing.worker`` in its own process and run folder under
``FUNDMANAGER_UI_OUTPUT_ROOT``. Runs never write to the shared drive;
"Publish" copies a finished run's reports to ``FUNDMANAGER_UI_PUBLISH_ROOT``.

Settings (environment or ``.env``): ``FUNDMANAGER_UI_OUTPUT_ROOT``,
``FUNDMANAGER_UI_PUBLISH_ROOT``, ``FUNDMANAGER_UI_TIMEOUT_MINUTES``,
``FUNDMANAGER_UI_USER_HEADER`` (SSO proxy header with the username),
``FUNDMANAGER_UI_PASSWORD`` (optional shared password; also read from
``st.secrets``) and ``FUNDMANAGER_UI_CATALOG_TTL`` (seconds, default 600).
"""
from __future__ import annotations

import getpass
import hmac
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `streamlit run ui/app.py` only puts ui/ on the path
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from config import fund_definitions  # noqa: E402
from processing import api  # noqa: E402
from ui import history_page, new_run, runner  # noqa: E402

PAGES = {"New run": new_run.render, "Run history": history_page.render_history, "Reports": history_page.render_reports}
_log = logging.getLogger("fundmanager.ui")


def _catalog_ttl() -> int:
    try:
        return max(int(os.getenv("FUNDMANAGER_UI_CATALOG_TTL") or 600), 0)
    except ValueError:
        return 600


@st.cache_data(ttl=_catalog_ttl(), show_spinner="Loading fund groups from the database…")
def load_catalog() -> dict:
    """``processing.api.catalog()`` (reads the cohorts from the DB), cached for the TTL."""
    return api.catalog()


def get_catalog() -> dict:
    """The cached catalog, or a DB-free fallback (no fund groups) when the DB cannot be reached."""
    try:
        return load_catalog()
    except Exception as exc:  # the DB is down or misconfigured: keep the app usable with explicit tickers
        _log.error("Loading the catalog failed: %s", type(exc).__name__)  # no message: it may name the DB
        st.error(f"Could not load the fund groups from the database ({type(exc).__name__}); see the server log. "
                 "You can still pick funds by ticker. Use 'Reload fund groups' to retry.")
        return {"cohorts": {}, "funds": sorted(fund_definitions.FUND_DEFINITIONS),
                "operations": list(api.OPERATIONS), "rule_catalog": api.rule_catalog(),
                "date_modes": list(api.DATE_MODES)}


def _configured_password() -> Optional[str]:
    value = os.getenv("FUNDMANAGER_UI_PASSWORD")
    if not value:
        try:
            value = st.secrets.get("FUNDMANAGER_UI_PASSWORD")
        except Exception:  # no secrets.toml
            value = None
    return str(value) if value else None


def password_gate() -> bool:
    """``True`` when no password is configured or this session entered it (``hmac.compare_digest``)."""
    expected = _configured_password()
    if expected is None or st.session_state.get("authenticated"):
        return True
    st.title("FundManager")
    with st.form("login"):
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.session_state.pop("login_password", None)
            st.rerun()
        st.error("Wrong password.")
    return False


def current_user() -> Optional[str]:
    """The SSO proxy's username header when ``FUNDMANAGER_UI_USER_HEADER`` is set, else the OS user."""
    header = os.getenv("FUNDMANAGER_UI_USER_HEADER", "").strip()
    if header:
        return st.context.headers.get(header) or None
    return getpass.getuser()


def main() -> None:
    st.set_page_config(page_title="FundManager", page_icon=":material/fact_check:", layout="wide")
    if not password_gate():
        return
    user = current_user()
    if user is None:
        st.error("Not signed in: the sign-in header is missing. Open the app through the company SSO link.")
        return
    st.session_state.setdefault("page", "New run")
    with st.sidebar:
        st.markdown("### FundManager")
        page = st.radio("Page", list(PAGES), key="page", label_visibility="collapsed")
        st.caption(f"Signed in as **{user}**")
        st.caption(f"Output: `{runner.output_root()}`")
        publish = runner.publish_root()
        st.caption(f"Publish: `{publish}`" if publish else "Publish: off")
        if st.button("Reload fund groups", help="Re-read the fund groups (cohorts) from the database"):
            load_catalog.clear()
    if page == "New run":
        new_run.render(get_catalog(), user)
    else:
        PAGES[page](user)


main()
