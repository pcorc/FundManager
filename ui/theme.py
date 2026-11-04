"""The UI's look: one soft dark palette and the CSS that styles Streamlit's widgets and the app's HTML.

``.streamlit/config.toml`` sets the same palette as Streamlit's theme (so its
own widgets, focus rings and the progress bar match); :func:`inject` adds the
CSS for the custom HTML pieces built in :mod:`ui.components` and for the
widget groups the pages key by prefix:

- ``nrf_<ticker>`` fund checkboxes, ``nrr_<rule>`` compliance-check checkboxes and
  ``nr_op_<operation>`` operation checkboxes render as selectable tiles that light
  up when ticked;
- ``nr_settings`` is the New run settings panel; ``fm_nav`` the top navigation.

Call :func:`inject` once at the top of every script run (it emits a style-only
``st.html``, which takes no space on the page).
"""
from __future__ import annotations

import streamlit as st

# Soft dark palette: no pure black, no neon. Keep in step with .streamlit/config.toml.
BG = "#1F2430"
SURFACE = "#262D3B"
SURFACE_2 = "#2C3446"
SURFACE_3 = "#333C50"
BORDER = "#363F52"
TEXT = "#E3E7EF"
# Text tiers: TEXT for values, MUTED for secondary text people read (labels, footnotes, reasons), FAINT for
# hints and meta lines. Each clears the 4.5:1 body-text minimum on BG, SURFACE and SURFACE_2.
MUTED = "#B0B9CA"
FAINT = "#929CB0"
ACCENT = "#7C9CF5"
ACCENT_TEXT = "#A9BDF8"
BAND = "#3A4A75"  # rule-card header band (a soft blue)
PASS = "#7FCB9B"
FAIL = "#EE8E8E"
WARN = "#E6C07B"
NA = "#A3ACBD"

FONT_STACK = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = f"""
:root {{
  --fm-bg: {BG}; --fm-surface: {SURFACE}; --fm-surface-2: {SURFACE_2}; --fm-surface-3: {SURFACE_3};
  --fm-border: {BORDER}; --fm-text: {TEXT}; --fm-muted: {MUTED}; --fm-faint: {FAINT};
  --fm-accent: {ACCENT}; --fm-accent-text: {ACCENT_TEXT}; --fm-band: {BAND};
  --fm-accent-soft: rgba(124, 156, 245, 0.13); --fm-accent-line: rgba(124, 156, 245, 0.55);
  --fm-pass: {PASS}; --fm-fail: {FAIL}; --fm-warn: {WARN}; --fm-na: {NA};
  --fm-pass-soft: rgba(127, 203, 155, 0.13); --fm-fail-soft: rgba(238, 142, 142, 0.13);
  --fm-warn-soft: rgba(230, 192, 123, 0.14); --fm-na-soft: rgba(163, 172, 189, 0.12);
  --fm-radius: 12px; --fm-shadow: 0 1px 2px rgba(10, 12, 20, 0.25), 0 8px 24px rgba(10, 12, 20, 0.18);
}}

/* ---- page frame -------------------------------------------------------------------------------------- */
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(124,156,245,0.07), transparent 60%), var(--fm-bg); }}
header[data-testid="stHeader"] {{ background: transparent; height: 2.5rem; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1680px; }}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT_STACK}; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
[data-testid="stHeading"] h1 {{ font-size: 1.55rem; font-weight: 650; padding: 0.2rem 0 0; }}
[data-testid="stCaptionContainer"] {{ color: var(--fm-muted); }}
hr {{ border-color: var(--fm-border) !important; opacity: .7; }}

/* ---- top bar ----------------------------------------------------------------------------------------- */
.fm-brand {{ display: flex; align-items: center; gap: .65rem; height: 2.5rem; }}
.fm-brand-mark {{ width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(135deg, #8EA9F7, #6C86D9); color: #1F2430; font-weight: 800; font-size: .95rem; }}
.fm-brand-name {{ font-weight: 650; font-size: 1.08rem; color: var(--fm-text); }}
.fm-brand-sub {{ color: var(--fm-faint); font-size: .8rem; margin-left: .15rem; }}
.st-key-fm_nav [data-testid="stButtonGroup"] button {{ min-width: 7.5rem; }}
@media (max-width: 1440px) {{ .fm-brand-sub {{ display: none; }} .st-key-fm_nav [data-testid="stButtonGroup"] button {{ min-width: 6rem; }} }}
.fm-page-head {{ margin: .4rem 0 1.1rem; }}
.fm-page-head h1 {{ font-size: 1.5rem; font-weight: 650; margin: 0; color: var(--fm-text); }}
.fm-page-head p {{ margin: .2rem 0 0; color: var(--fm-muted); font-size: .92rem; }}

/* ---- Streamlit widgets: softer surfaces -------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][class*="st-key-"][style*="border"] {{
  border-radius: var(--fm-radius); }}
.st-key-nr_settings {{ background: var(--fm-surface); border: 1px solid var(--fm-border) !important;
  border-radius: 16px; padding: 1.1rem 1.15rem 0; box-shadow: var(--fm-shadow); }}
/* The settings column stays in view with its own scroll, and RUN / RESET stay pinned at its foot, so the progress
   and results that appear at the top of the right pane are on screen when RUN is pressed. */
[data-testid="stColumn"]:has(.st-key-nr_settings) {{ position: sticky; top: 3rem; align-self: flex-start; }}
.st-key-nr_settings {{ max-height: calc(100vh - 4rem); overflow-y: auto; overscroll-behavior: contain;
  scrollbar-width: thin; scrollbar-color: var(--fm-surface-3) transparent; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-nr_actions) {{ position: sticky; bottom: 0; z-index: 3;
  background: var(--fm-surface); padding: 0 0 1rem; }}
.st-key-nr_actions .fm-rule {{ margin-top: 0; }}
/* Below 1200px the settings and the preview stack (Streamlit itself stacks columns only below 640px). */
@media (max-width: 1200px) {{
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-nr_settings) {{ flex-direction: column; }}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-nr_settings) > [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 100% !important; }}
  [data-testid="stColumn"]:has(.st-key-nr_settings) {{ position: static; align-self: stretch; }}
  .st-key-nr_settings {{ max-height: none; overflow: visible; }}
}}
.st-key-nr_preview {{ padding: 0 .2rem; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
  background: var(--fm-surface-2) !important; border-color: var(--fm-border) !important; border-radius: 9px !important; }}
[data-testid="stDateInput"] input, [data-testid="stTextInput"] input {{ font-variant-numeric: tabular-nums; }}
[data-testid="stWidgetLabel"] p {{ color: var(--fm-muted); font-size: .8rem; font-weight: 500; letter-spacing: .01em; }}
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{ border-radius: 9px; font-weight: 550;
  border-color: var(--fm-border); transition: all .15s ease; }}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{ color: #1C2130; letter-spacing: .04em;
  background: var(--fm-accent) !important; border-color: var(--fm-accent) !important; box-shadow: 0 4px 14px rgba(124,156,245,.22); }}
.stButton button[kind="primary"]:hover:not(:disabled), .stDownloadButton button[kind="primary"]:hover {{
  background: #8DAAF7 !important; color: #1C2130 !important; }}
.stButton button[kind="primary"]:focus, .stButton button[kind="primary"]:active,
.stDownloadButton button[kind="primary"]:focus, .stDownloadButton button[kind="primary"]:active {{ color: #1C2130 !important; }}
.stButton button[kind="primary"]:disabled {{ background: var(--fm-surface-2) !important; border-color: var(--fm-border) !important;
  color: var(--fm-faint) !important; box-shadow: none; }}
.st-key-nr_chips button[kind="primary"] {{ background: var(--fm-accent-soft) !important; color: var(--fm-accent-text) !important;
  border-color: var(--fm-accent-line) !important; box-shadow: none; letter-spacing: 0; }}
.stButton button[kind="secondary"]:hover:not(:disabled) {{ border-color: var(--fm-accent-line); color: var(--fm-accent-text); }}
[data-testid="stButtonGroup"] button {{ border-color: var(--fm-border); }}
/* Segmented controls and pills (Streamlit 1.66 marks them data-variant / aria-checked): unselected segments sit on
   SURFACE_2 like the tiles and inputs beside them, the selected one is tinted like a ticked tile. */
.st-key-nr_settings [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{ background: var(--fm-surface-2) !important; }}
[data-testid="stButtonGroup"] button[data-variant][aria-checked="true"] {{ background: var(--fm-accent-soft) !important;
  color: var(--fm-accent-text) !important; border-color: var(--fm-accent-line) !important; }}
.st-key-nr_settings [data-testid="stButtonGroup"] > div {{ flex-wrap: nowrap; }}
.st-key-nr_settings [data-testid="stButtonGroup"] button p {{ white-space: nowrap; }}
.stDownloadButton button {{ height: auto; min-height: 2.5rem; }}
.stDownloadButton button p {{ white-space: normal; text-align: left; line-height: 1.3; }}
[data-testid="stExpander"] details {{ border-color: var(--fm-border); border-radius: var(--fm-radius); background: var(--fm-surface); }}
[data-testid="stExpander"] summary {{ color: var(--fm-muted); }}
[data-testid="stAlert"] {{ border-radius: var(--fm-radius); }}
[data-testid="stCode"] pre {{ background: #232938 !important; border-radius: 10px; }}

/* ---- selectable tiles (fund, check and operation checkboxes) ----------------------------------------- */
[class*="st-key-nrf_"] label, [class*="st-key-nrr_"] label, [class*="st-key-nr_op_"] label {{
  width: 100%; margin: 0; padding: .34rem .6rem; border: 1px solid var(--fm-border); border-radius: 8px;
  background: var(--fm-surface-2); transition: background .12s ease, border-color .12s ease; }}
[class*="st-key-nrf_"] label:hover, [class*="st-key-nrr_"] label:hover, [class*="st-key-nr_op_"] label:hover {{
  border-color: var(--fm-accent-line); }}
[class*="st-key-nrf_"]:has(input:checked) label, [class*="st-key-nrr_"]:has(input:checked) label,
[class*="st-key-nr_op_"]:has(input:checked) label {{ background: var(--fm-accent-soft); border-color: var(--fm-accent-line); }}
[class*="st-key-nrf_"]:has(input:checked) label p, [class*="st-key-nrr_"]:has(input:checked) label p,
[class*="st-key-nr_op_"]:has(input:checked) label p {{ color: var(--fm-accent-text); }}
[class*="st-key-nrr_"] label p, [class*="st-key-nr_op_"] label p {{ font-size: .86rem; }}
[class*="st-key-nrf_"] label p {{ font-size: .86rem; font-weight: 550; letter-spacing: .03em; font-variant-numeric: tabular-nums; }}
[class*="st-key-nrf_"], [class*="st-key-nrr_"], [class*="st-key-nr_op_"] {{ margin-bottom: -0.35rem; }}
.st-key-nr_chips button {{ padding: .1rem .7rem; min-height: 1.9rem; font-size: .8rem; border-radius: 999px; }}

/* ---- section headings inside the settings panel ------------------------------------------------------ */
.fm-sec {{ display: flex; flex-wrap: wrap; gap: .2rem .6rem; align-items: baseline; justify-content: space-between;
  margin: .35rem 0 .15rem; }}
.fm-sec h3 {{ font-size: 1.02rem; font-weight: 640; margin: 0; padding: 0; color: var(--fm-text); white-space: nowrap; }}
.fm-sec .fm-count {{ display: inline-block; margin-left: .45rem; padding: 0 .5rem; border-radius: 999px; font-size: .75rem;
  background: var(--fm-accent-soft); color: var(--fm-accent-text); font-weight: 600; vertical-align: 2px; }}
.fm-sec small {{ color: var(--fm-faint); font-size: .76rem; }}
.fm-rule {{ height: 1px; background: var(--fm-border); margin: .9rem -1.15rem .7rem; opacity: .8; }}
.fm-hint {{ color: var(--fm-faint); font-size: .78rem; margin: .1rem 0 .2rem; }}

/* ---- pills ------------------------------------------------------------------------------------------- */
.fm-pill {{ display: inline-block; min-width: 2.9rem; text-align: center; padding: .08rem .55rem; border-radius: 999px;
  font-size: .72rem; font-weight: 650; letter-spacing: .05em; line-height: 1.45rem; border: 1px solid transparent;
  white-space: nowrap; }}
.fm-pass {{ color: var(--fm-pass); background: var(--fm-pass-soft); border-color: rgba(127,203,155,.28); }}
.fm-fail {{ color: var(--fm-fail); background: var(--fm-fail-soft); border-color: rgba(238,142,142,.3); }}
.fm-warn {{ color: var(--fm-warn); background: var(--fm-warn-soft); border-color: rgba(230,192,123,.3); }}
.fm-err {{ color: #F2A7C0; background: rgba(242,167,192,.13); border-color: rgba(242,167,192,.3); }}
.fm-na, .fm-none {{ color: var(--fm-na); background: var(--fm-na-soft); border-color: rgba(163,172,189,.2); }}
.fm-run {{ color: var(--fm-accent-text); background: var(--fm-accent-soft); border-color: var(--fm-accent-line); }}
.fm-pill.sm {{ min-width: 2.5rem; padding: 0 .4rem; font-size: .7rem; line-height: 1.3rem; }}
.fm-dash {{ color: var(--fm-faint); }}

/* ---- metric cards and the preview sentence ----------------------------------------------------------- */
.fm-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .9rem; margin: .2rem 0 1rem; }}
.fm-metric {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; padding: .95rem 1rem .85rem;
  text-align: center; box-shadow: var(--fm-shadow); }}
.fm-metric .v {{ font-size: 1.85rem; font-weight: 650; color: var(--fm-text); line-height: 1.1; font-variant-numeric: tabular-nums; }}
.fm-metric .l {{ color: var(--fm-muted); font-size: .8rem; margin-top: .3rem; letter-spacing: .02em; }}
.fm-metric .s {{ color: var(--fm-faint); font-size: .74rem; margin-top: .15rem; line-height: 1.35; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
.fm-metric.accent .v {{ color: var(--fm-accent-text); }}
.fm-sentence {{ text-align: center; color: var(--fm-muted); font-size: .92rem; line-height: 1.6; margin: .2rem 0 1.2rem; }}
.fm-sentence b {{ color: var(--fm-text); font-weight: 650; }}
.fm-sentence .eq {{ color: var(--fm-accent-text); }}
.fm-sentence .est {{ color: var(--fm-faint); font-size: .82rem; }}

/* ---- callouts ---------------------------------------------------------------------------------------- */
.fm-callout {{ border-radius: 12px; padding: .75rem 1rem; margin: .2rem 0 1rem; border: 1px solid var(--fm-border);
  background: var(--fm-surface); font-size: .9rem; display: flex; gap: .7rem; align-items: flex-start; }}
.fm-callout .ic {{ font-weight: 800; width: 1.4rem; height: 1.4rem; border-radius: 50%; display: grid; place-items: center;
  font-size: .8rem; flex: none; margin-top: .05rem; }}
.fm-callout.error {{ border-color: rgba(238,142,142,.35); background: linear-gradient(0deg, var(--fm-fail-soft), var(--fm-fail-soft)), var(--fm-surface); }}
.fm-callout.error .ic {{ background: var(--fm-fail); color: #2A2025; }}
.fm-callout.info {{ border-color: var(--fm-accent-line); background: linear-gradient(0deg, var(--fm-accent-soft), var(--fm-accent-soft)), var(--fm-surface); }}
.fm-callout.info .ic {{ background: var(--fm-accent); color: #1C2130; }}
.fm-callout.warn {{ border-color: rgba(230,192,123,.35); background: linear-gradient(0deg, var(--fm-warn-soft), var(--fm-warn-soft)), var(--fm-surface); }}
.fm-callout.warn .ic {{ background: var(--fm-warn); color: #2A2517; }}
.fm-callout.ok {{ border-color: rgba(127,203,155,.35); background: linear-gradient(0deg, var(--fm-pass-soft), var(--fm-pass-soft)), var(--fm-surface); }}
.fm-callout.ok .ic {{ background: var(--fm-pass); color: #17261D; }}
.fm-callout ul {{ margin: .25rem 0 0 1rem; padding: 0; }}

/* ---- cards: rule guide, rule results, NAV / holdings / trading --------------------------------------- */
.fm-h2 {{ display: flex; align-items: baseline; gap: .6rem; margin: 1.4rem 0 .7rem; }}
.fm-h2 h2 {{ font-size: 1.12rem; font-weight: 640; margin: 0; padding: 0; color: var(--fm-text); }}
.fm-h2 span {{ color: var(--fm-faint); font-size: .82rem; }}
.fm-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem; align-items: start; }}
.fm-cell {{ min-width: 0; }}
.fm-cell.wide {{ grid-column: 1 / -1; }}
@media (max-width: 1100px) {{ .fm-grid {{ grid-template-columns: minmax(0, 1fr); }} }}
.fm-split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }}
.fm-split > div + div {{ border-left: 1px solid var(--fm-border); }}
@media (max-width: 900px) {{ .fm-split {{ grid-template-columns: minmax(0, 1fr); }} .fm-split > div + div {{ border-left: none; }} }}
.fm-card {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; overflow: hidden;
  box-shadow: var(--fm-shadow); }}
.fm-band {{ background: linear-gradient(180deg, #3E4F7D, var(--fm-band)); color: #EEF2FB; padding: .5rem 1rem;
  display: flex; align-items: center; justify-content: space-between; gap: .6rem; font-weight: 620; font-size: .93rem;
  min-height: 2.85rem; box-sizing: border-box; }}
.fm-band .fm-pill {{ background: var(--fm-surface); font-size: .7rem; line-height: 1.25rem; flex: none; }}
.fm-band .sub {{ color: #C3CEEA; font-weight: 450; font-size: .78rem; }}
.fm-band.plain {{ background: var(--fm-surface-2); color: var(--fm-text); border-bottom: 1px solid var(--fm-border); }}
.fm-body {{ padding: .3rem 0 .2rem; }}
.fm-card p.fm-desc {{ color: var(--fm-muted); font-size: .86rem; margin: .7rem 1rem .4rem; line-height: 1.5; }}
.fm-card ul.fm-conds {{ margin: .2rem 1rem .6rem 1.1rem; padding: 0 0 0 .8rem; color: var(--fm-text); font-size: .84rem; line-height: 1.55; }}
.fm-card ul.fm-conds li::marker {{ color: var(--fm-accent); }}
.fm-card .fm-applies {{ margin: 0 1rem .85rem; color: var(--fm-muted); font-size: .8rem; }}
table.fm-kv {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
table.fm-kv th {{ text-align: left; font-weight: 560; color: var(--fm-accent-text); padding: .45rem 1rem; width: auto;
  border-bottom: 1px solid rgba(54,63,82,.7); vertical-align: top; }}
table.fm-kv td {{ text-align: right; padding: .45rem 1rem; color: var(--fm-text); border-bottom: 1px solid rgba(54,63,82,.7);
  overflow-wrap: anywhere; }}
table.fm-kv td.nw {{ white-space: nowrap; overflow-wrap: normal; }}
table.fm-kv th {{ min-width: 7rem; }}
table.fm-kv td:has(> .long) {{ width: 58%; }}
table.fm-kv td.num, table.fm-table td.num {{ font-variant-numeric: tabular-nums; }}
table.fm-kv tr:last-child th, table.fm-kv tr:last-child td {{ border-bottom: none; }}
table.fm-kv tr:hover td, table.fm-kv tr:hover th {{ background: rgba(124,156,245,.04); }}
table.fm-kv td.v-pass {{ color: var(--fm-pass); font-weight: 600; }}
table.fm-kv td.v-fail {{ color: var(--fm-fail); font-weight: 600; }}
table.fm-kv td.v-warn {{ color: var(--fm-warn); font-weight: 600; }}
table.fm-kv td.v-na {{ color: var(--fm-na); }}
table.fm-kv td .why {{ display: block; color: var(--fm-muted); font-size: .76rem; font-weight: 400; }}
table.fm-kv td .long {{ display: block; max-height: 7.5em; overflow: auto; text-align: left; font-size: .78rem; color: var(--fm-muted); }}
.fm-sub {{ padding: .55rem 1rem .25rem; color: var(--fm-muted); font-size: .74rem; font-weight: 650; letter-spacing: .07em;
  text-transform: uppercase; border-top: 1px solid var(--fm-border); }}
.fm-sub.first {{ border-top: none; }}
.fm-card details {{ border-top: 1px solid var(--fm-border); }}
.fm-card details > summary {{ cursor: pointer; padding: .5rem 1rem; color: var(--fm-muted); font-size: .8rem; list-style: none; }}
.fm-card details > summary::-webkit-details-marker {{ display: none; }}
.fm-card details > summary::before {{ content: "\\25B8"; display: inline-block; margin-right: .45rem; transition: transform .15s; color: var(--fm-accent); }}
.fm-card details[open] > summary::before {{ transform: rotate(90deg); }}
.fm-notes {{ margin: .1rem 1rem .6rem; padding: .45rem .65rem; border-left: 2px solid var(--fm-accent-line); background: var(--fm-accent-soft);
  border-radius: 0 8px 8px 0; color: var(--fm-muted); font-size: .78rem; }}
.fm-foot {{ padding: .6rem 1rem .8rem; border-top: 1px solid var(--fm-border); background: rgba(31,36,48,.35); }}
.fm-foot .t {{ color: var(--fm-muted); font-size: .76rem; font-weight: 600; margin-bottom: .2rem; }}
.fm-card details.fm-foot {{ padding: 0 1rem .1rem; }}
.fm-card details.fm-foot > summary.t {{ padding: .55rem 0; margin: 0; font-size: .76rem; }}
.fm-card details.fm-foot[open] {{ padding-bottom: .8rem; }}
.fm-foot p {{ margin: .15rem 0; color: var(--fm-muted); font-size: .78rem; line-height: 1.5; }}
.fm-reason {{ margin: .7rem 1rem; color: var(--fm-muted); font-size: .84rem; }}
.fm-stats {{ display: flex; flex-wrap: wrap; gap: .45rem; padding: .7rem 1rem .4rem; }}
.fm-stat {{ background: var(--fm-surface-2); border: 1px solid var(--fm-border); border-radius: 9px; padding: .3rem .6rem; font-size: .78rem; color: var(--fm-muted); }}
.fm-stat b {{ color: var(--fm-text); margin-left: .3rem; font-variant-numeric: tabular-nums; }}
.fm-stat.bad b {{ color: var(--fm-fail); }} .fm-stat.warnish b {{ color: var(--fm-warn); }}

/* ---- data tables (matrix, history, GICS, discrepancies, trading) ------------------------------------- */
/* A wide table scrolls sideways; a soft shadow at the right edge says there is more (it is covered once the
   table is scrolled to its end: the cover scrolls with the content, the shadow stays with the box). */
.fm-tablewrap {{ overflow-x: auto;
  background: linear-gradient(to left, var(--fm-surface) 40%, rgba(38,45,59,0)) right center / 2.5rem 100% no-repeat local,
              radial-gradient(farthest-side at 100% 50%, rgba(8,10,16,.55), rgba(8,10,16,0)) right center / .9rem 100% no-repeat scroll; }}
table.fm-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .82rem; }}
table.fm-table th {{ text-align: left; color: var(--fm-muted); font-weight: 600; font-size: .74rem; letter-spacing: .04em;
  padding: .5rem .75rem; background: var(--fm-surface-2); border-bottom: 1px solid var(--fm-border); white-space: nowrap; }}
table.fm-table td {{ padding: .42rem .75rem; border-bottom: 1px solid rgba(54,63,82,.6); color: var(--fm-text);
  vertical-align: middle; }}
.fm-card table.fm-table th:first-child, .fm-card table.fm-table td:first-child {{ padding-left: 1rem; }}
.fm-card table.fm-table th:last-child, .fm-card table.fm-table td:last-child {{ padding-right: 1rem; }}
table.fm-table tr:last-child td {{ border-bottom: none; }}
table.fm-table tbody tr:hover td {{ background: rgba(124,156,245,.05); }}
table.fm-table td.num, table.fm-table th.num {{ text-align: right; }}
table.fm-table td.c, table.fm-table th.c {{ text-align: center; }}
table.fm-matrix th {{ white-space: normal; text-align: center; padding: .5rem .2rem; line-height: 1.2; font-size: .7rem; }}
table.fm-matrix th:first-child {{ text-align: left; padding-left: .9rem; }}
table.fm-matrix td {{ padding: .38rem .15rem; }}
table.fm-matrix th:first-child, table.fm-matrix td:first-child {{ position: sticky; left: 0; z-index: 1;
  background: var(--fm-surface); box-shadow: inset -1px 0 0 var(--fm-border); }}
table.fm-matrix th:first-child {{ background: var(--fm-surface-2); }}
table.fm-history td {{ font-size: .8rem; }}
table.fm-history td:nth-child(1), table.fm-history td:nth-child(3) {{ white-space: nowrap; }}
table.fm-history td:nth-child(4), table.fm-history th:nth-child(4) {{ text-align: left; }}
table.fm-history .nw {{ white-space: nowrap; }}
table.fm-history td:nth-child(8) {{ color: var(--fm-faint); font-size: .76rem; overflow-wrap: anywhere; }}
@media (max-width: 1100px) {{ table.fm-history th:nth-child(8), table.fm-history td:nth-child(8) {{ display: none; }}
  table.fm-history th, table.fm-history td {{ padding-left: .5rem; padding-right: .5rem; }}
  table.fm-history td:nth-child(5) {{ min-width: 11rem; }} }}
table.fm-matrix td:first-child {{ font-weight: 650; letter-spacing: .03em; padding-left: .9rem; }}
table.fm-table td.fund {{ font-weight: 650; letter-spacing: .03em; }}
table.fm-table tr.hl-bad td {{ background: var(--fm-fail-soft); }}
table.fm-table tr.hl-good td {{ background: var(--fm-pass-soft); }}
table.fm-table tr.hl-breach td {{ background: var(--fm-warn-soft); }}
table.fm-table tr.total td {{ font-weight: 650; background: var(--fm-surface-2); }}
table.fm-table td.muted {{ color: var(--fm-muted); }}
table.fm-table td .sub {{ display: block; color: var(--fm-faint); font-size: .72rem; }}
.fm-card .fm-table th {{ background: rgba(44,52,70,.6); }}
.fm-tablecap {{ padding: .45rem 1rem .6rem; color: var(--fm-faint); font-size: .74rem; }}
.fm-tabletitle {{ padding: .6rem 1rem .35rem; color: var(--fm-muted); font-size: .8rem; font-weight: 600; }}
.fm-panel {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; overflow: hidden; box-shadow: var(--fm-shadow); margin-bottom: 1rem; }}

/* ---- run banner, stepper, progress ------------------------------------------------------------------- */
.fm-banner {{ display: flex; align-items: center; gap: 1rem; padding: .9rem 1.1rem; border-radius: 14px; margin: .2rem 0 1rem;
  background: var(--fm-surface); border: 1px solid var(--fm-border); box-shadow: var(--fm-shadow); }}
.fm-banner > .fm-pill {{ font-size: .8rem; padding: .15rem .8rem; }}
.fm-banner .t {{ font-weight: 620; color: var(--fm-text); }}
.fm-banner .m {{ color: var(--fm-faint); font-size: .8rem; margin-top: .1rem; }}
.fm-steps {{ display: flex; align-items: center; gap: .35rem; flex-wrap: wrap; margin: .6rem 0 .4rem; }}
.fm-step {{ display: flex; align-items: center; gap: .4rem; color: var(--fm-faint); font-size: .8rem; }}
.fm-step .dot {{ width: 1.35rem; height: 1.35rem; border-radius: 50%; display: grid; place-items: center; font-size: .7rem;
  border: 1px solid var(--fm-border); background: var(--fm-surface-2); color: var(--fm-faint); }}
.fm-step.done {{ color: var(--fm-muted); }}
.fm-step.done .dot {{ background: var(--fm-pass-soft); border-color: rgba(127,203,155,.4); color: var(--fm-pass); }}
.fm-step.now {{ color: var(--fm-text); font-weight: 600; }}
.fm-step.now .dot {{ background: var(--fm-accent-soft); border-color: var(--fm-accent); color: var(--fm-accent-text);
  box-shadow: 0 0 0 4px rgba(124,156,245,.12); }}
.fm-step-line {{ width: 1.4rem; height: 1px; background: var(--fm-border); }}
.fm-live {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
.fm-live .t {{ font-size: 1.05rem; font-weight: 640; }}
.fm-live .m {{ color: var(--fm-faint); font-size: .8rem; }}
.fm-now {{ color: var(--fm-muted); font-size: .86rem; margin: .2rem 0 .1rem; }}
.fm-now b {{ color: var(--fm-accent-text); }}
.st-key-nr_progress {{ background: var(--fm-surface); border: 1px solid var(--fm-accent-line) !important; border-radius: 16px;
  padding: 1rem 1.15rem; box-shadow: 0 0 0 4px rgba(124,156,245,.06), var(--fm-shadow); }}
.fm-empty {{ text-align: center; color: var(--fm-faint); padding: 2.2rem 1rem; border: 1px dashed var(--fm-border); border-radius: 14px; }}
.fm-empty b {{ display: block; color: var(--fm-muted); font-size: 1rem; margin-bottom: .3rem; }}
.fm-files {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .5rem; }}
"""


def inject() -> None:
    """Add the app's CSS to the page (style-only ``st.html`` takes no layout space)."""
    st.html(f"<style>{CSS}</style>")
