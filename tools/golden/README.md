# Golden-master harness

Replays recorded DB data through the app and compares every output (Excel, PDF,
emails, run log, exit status) with an accepted baseline. Run from the repo root;
set `GOLDEN_WORK` to share one recording/baseline between worktrees.

| Command | What it does |
|---|---|
| `golden.py check [scenario ...]` | Replay into `_work/current`, print `OK`/`DIFF` per scenario; exit 1 on any diff. |
| `golden.py report OUT.md [--against SNAPSHOT] [--no-run] [scenario ...]` | Same comparison, written as a reviewer-friendly markdown file (cell before/after, PDF text diff, log and email diffs). `--no-run` reuses `_work/current`. |
| `golden.py snapshot NAME` | Copy the baseline and recorded data to `WORK/snapshots/NAME` before changing them. |
| `golden.py rebaseline [scenario ...]` | Accept `_work/current` as the new baseline. Equivalent-request scenarios are only checked against their base. |
| `golden.py capture [scenario ...]` | Re-record from the live DB (read-only) and rebuild the baseline. |

Output-changing fix workflow: `snapshot pre-fix`, make the change,
`report WORK/review/<fix>.md`, review it, `rebaseline`, then `check` (all OK) and commit.

Run roots record absolute paths in their logs and emails. `snapshot` and
`rebaseline` re-point those paths for you, so do not copy run roots by hand.
