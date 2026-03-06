# Promotion report

branch: fast/run-20260306-2305-linux-test-bootstrap
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** Canonical test/replay evidence could not run on fresh Linux environments
- **Title:** Missing Linux bootstrap path caused immediate dependency import failures (`pytest`, `fastapi`)

## Why it outranked alternatives

Issue ranking from current run evidence:

1. Structural invariant violations: no evidence of violations once canonical tooling is runnable (`structural_violations_total=0`).
2. Failing canonical tests: highest unresolved issue on fresh run startup (`python3 -m pytest -q` failed with `No module named pytest`).
3. Confirmed replay mismatches: not reached before dependency bootstrap; after bootstrap assessment is clean.
4. High-frequency diagnostic invariant failures: none after bootstrap (`behavioral_violations_total=0`, `invariant_violations_total=0`).

The selected issue is rank #2 and directly blocked baseline evidence collection.

## Baseline evidence

- Baseline branch: `fast/run-20260306-2305-linux-test-bootstrap` (from `agent-base`)
- Baseline commit: `95e93f599810d133c551e9286c76da40b64b2ee3`
- Baseline checks before code edits:
  - `python3 -m pytest -q` -> `/usr/bin/python3: No module named pytest`
  - `python3 tools/replay_verification_assess.py` -> `ModuleNotFoundError: No module named 'fastapi'`
  - Existing bootstrap script coverage probe: only `scripts/bootstrap.ps1` existed (no Linux shell bootstrap script).

## Files changed

| Path | Change |
|------|--------|
| `scripts/bootstrap.sh` | Added Linux/macOS bootstrap script to install required dev/test dependencies with Python 3.11/3 fallback. |
| `README.md` | Updated setup instructions to include Linux/macOS bootstrap path alongside Windows bootstrap. |
| `automation/reports/fast-run-20260306-2305-linux-test-bootstrap.md` | Added human-readable run report. |
| `automation/reports/fast-run-20260306-2305-linux-test-bootstrap.json` | Added machine-readable run report. |

## Validation performed

- `./scripts/bootstrap.sh` -> success (dependency install completed, exit code 0)
- `python3 -m pytest -q` -> `394 passed in 7.50s`
- `python3 tools/replay_verification_assess.py` -> clean summary (`structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`)

## Before/after evidence

- **Before:** Fresh Linux run could not execute canonical tests or replay assessment due missing dependencies in default environment (`pytest`, `fastapi` import failures), and there was no shell bootstrap path to standardize setup.
- **After:** `scripts/bootstrap.sh` provides a reproducible Linux/macOS dependency bootstrap path, and canonical evidence commands run successfully in this run (`394 passed`, replay assessment clean).

## Unresolved risks

- The bootstrap script installs into the active Python environment and does not enforce a virtualenv; teams may still prefer local venv isolation.
- Optional browser tooling installation (e.g., Playwright) remains outside this minimal Linux test-bootstrap fix.

## Stop reason

Stopped after the selected rank-2 issue was resolved with minimal bounded edits and repeated validation; further environment/tooling enhancements would exceed this single-issue maintenance scope.

## Recommendation

- `promote`
