# Promotion report

branch: fast/run-20260306-1910-linux-test-bootstrap
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Canonical checks fail to run on Linux automation hosts due to missing bootstrap path
- **Title:** Add Linux/macOS bootstrap flow so canonical test and replay evidence can be collected

## Why it outranked alternatives

Issue ranking from current run evidence:

1. Structural invariant violations: no current failing structural evidence was available before setup (assessment command could not run because runtime deps were missing).
2. Failing canonical tests: reproducible failure at command level on `agent-base`:
   - `python3 -m pytest -q` -> `/usr/bin/python3: No module named pytest`
   - `python3 tools/replay_verification_assess.py` -> `ModuleNotFoundError: No module named 'fastapi'`

Given this rank-2 failure, lower-ranked categories were not selected.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `3a871b09bf26f9afefec73fb2fdad16118138ea8`
- Baseline checks:
  - `python3 -m pytest -q` -> failed (`No module named pytest`)
  - `python3 tools/replay_verification_assess.py` -> failed (`No module named fastapi`)
  - Idempotence probe: only Windows bootstrap script existed (`scripts/bootstrap.ps1`), no Linux/macOS bootstrap script present

## Files changed

| Path | Change |
|------|--------|
| `scripts/bootstrap.sh` | Added Linux/macOS bootstrap script to create `.venv311`, install `requirements-dev.txt`, recover from incomplete venvs, and fall back to `virtualenv` when stdlib `venv` is unavailable. |
| `README.md` | Updated setup instructions to include Linux/macOS bootstrap command and retained Windows bootstrap path. |
| `automation/reports/fast-run-20260306-1910-linux-test-bootstrap.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-1910-linux-test-bootstrap.json` | Added machine-readable run report artifact. |

## Validation performed

- `./scripts/bootstrap.sh` (first run) -> exposed `venv` failure (`ensurepip` unavailable)
- `./scripts/bootstrap.sh` (after fallback/recovery fix) -> success; dependencies installed in `.venv311`
- `./scripts/bootstrap.sh` (repeat run) -> success using existing venv (idempotent behavior)
- `./.venv311/bin/python -m pytest -q` -> `379 passed in 7.70s`
- `./.venv311/bin/python tools/replay_verification_assess.py` -> success; `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`

## Before/after evidence

- **Before:** canonical commands on `agent-base` could not run in Linux automation env (`pytest` and runtime deps missing), preventing ranked evidence collection.
- **After:** repository now provides a Linux/macOS bootstrap entrypoint; after running it, canonical pytest and replay assessment execute successfully with clean diagnostics.

## Unresolved risks

- `python -m playwright install` can emit host dependency warnings when OS browser libs are missing (`playwright install-deps` not run). This does not block canonical pytest/replay checks validated in this run.
- This run scopes to setup/bootstrap for evidence collection and does not alter engine math or diagnostics logic.

## Stop reason

Stopped after the selected rank-2 issue was fixed with minimal bounded changes and repeated validation; additional setup hardening (system package installers per distro) would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
