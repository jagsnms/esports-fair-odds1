# Promotion report

branch: fast/run-20260306-2202-linux-test-bootstrap
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

1. Structural invariant violations: no current failing structural evidence was available before setup because replay assessment itself could not run due missing runtime dependencies.
2. Failing canonical tests: reproducible command-level failures on `agent-base`:
   - `python3 -m pytest -q` -> `/usr/bin/python3: No module named pytest`
   - `python3 tools/replay_verification_assess.py` -> `ModuleNotFoundError: No module named 'fastapi'`

Given this rank-2 failure, lower-ranked issue classes were not selected.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `789beead284a32665696c45286c143a70ed25705`
- Baseline checks:
  - `python3 -m pytest -q` -> failed (`No module named pytest`)
  - `python3 tools/replay_verification_assess.py` -> failed (`No module named fastapi`)
  - `scripts/bootstrap.sh` did not exist
  - `README.md` setup only documented Windows bootstrap (`scripts/bootstrap.ps1`)

## Files changed

| Path | Change |
|------|--------|
| `scripts/bootstrap.sh` | Added Linux/macOS bootstrap script to create `.venv311`, install `requirements-dev.txt`, recover from incomplete venvs, and fall back to `virtualenv` when stdlib `venv` is unavailable. |
| `README.md` | Updated setup instructions to include Linux/macOS bootstrap command while keeping Windows bootstrap instructions. |
| `automation/reports/fast-run-20260306-2202-linux-test-bootstrap.md` | Added human-readable report artifact for this run. |
| `automation/reports/fast-run-20260306-2202-linux-test-bootstrap.json` | Added machine-readable report artifact for this run. |

## Validation performed

- `chmod +x scripts/bootstrap.sh && ./scripts/bootstrap.sh` -> success; script recovered from missing `ensurepip` by falling back from `python -m venv` to `virtualenv` and completed install.
- `./scripts/bootstrap.sh` (repeat run) -> success using existing `.venv311` (idempotent).
- `./.venv311/bin/python -m pytest -q` -> `392 passed in 7.92s`
- `./.venv311/bin/python tools/replay_verification_assess.py` -> success with clean diagnostics (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`)

## Before/after evidence

- **Before:** canonical test/replay evidence collection failed immediately in clean Linux automation environments due missing runtime dependencies and no Linux bootstrap entrypoint.
- **After:** repository now provides `scripts/bootstrap.sh`; after running it, canonical pytest and replay assessment execute successfully with clean outputs.

## Unresolved risks

- Playwright reports host browser-library warnings (`playwright install-deps` not run), which may affect browser-driven workflows but did not block canonical pytest/replay validations in this run.
- This run addresses environment bootstrap for evidence collection and does not alter engine math, replay semantics, or invariant logic.

## Stop reason

Stopped after the selected rank-2 issue was fixed with minimal bounded changes and repeated issue-specific validation; additional distro-specific OS package automation would exceed single-issue maintenance scope.

## Recommendation

- `promote`
