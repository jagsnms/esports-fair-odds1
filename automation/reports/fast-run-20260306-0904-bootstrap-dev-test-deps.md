# Promotion report

branch: fast/run-20260306-0904-bootstrap-dev-test-deps
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** canonical test execution fails in clean bootstrap contexts because dev test dependencies are not installed
- **Title:** `scripts/bootstrap.ps1` installs `requirements.txt` only, so `pytest` is missing for canonical test runs

## Why it outranked alternatives

Issue ranking used the required ladder with current evidence:

1. Structural invariant violations: none observed in replay assessment (`structural_violations_total=0`).
2. Failing canonical tests: reproducible at baseline (`python3 -m pytest -q` failed with `No module named pytest`).
3. Confirmed replay mismatches: none observed.
4. High-frequency diagnostic invariant failures: none observed (`behavioral_violations_total=0`, `invariant_violations_total=0`).

Because a rank-2 issue was present, lower-ranked candidates were not selected.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `e2a39d4efc3759fdc39921cf2f7111e6874d518c`
- Baseline checks:
  - `python3 -m pytest -q` -> failed: `/usr/bin/python3: No module named pytest`
  - `python3 tools/replay_verification_assess.py` -> failed pre-bootstrap: `ModuleNotFoundError: No module named 'fastapi'`
  - `scripts/bootstrap.ps1` inspection -> installed `requirements.txt` via `pip install -r $ReqTxt` (no `requirements-dev.txt` install path)

## Files changed

| Path | Change |
|------|--------|
| `scripts/bootstrap.ps1` | Added `requirements-dev.txt` existence guard and switched install target from `requirements.txt` to `requirements-dev.txt` so bootstrap includes canonical test dependencies. |
| `automation/reports/fast-run-20260306-0904-bootstrap-dev-test-deps.md` | Added this human-readable promotion report. |
| `automation/reports/fast-run-20260306-0904-bootstrap-dev-test-deps.json` | Added machine-readable promotion report artifact. |

## Validation performed

- `rg "requirements-dev\\.txt|pip install -r \\$ReqDev" scripts/bootstrap.ps1` -> pass; bootstrap now references and installs `requirements-dev.txt`
- `python3 -m pytest -q` -> `365 passed in 7.20s`
- `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`

## Before/after evidence

- **Before:** clean-run canonical test command failed (`No module named pytest`), replay assessment also blocked by missing runtime deps (`No module named fastapi`), and bootstrap installed runtime requirements only.
- **After:** bootstrap script installs `requirements-dev.txt` (which includes test deps via `-r requirements.txt` + `pytest`), and canonical tests/replay assessment execute successfully in the sandbox.

## Unresolved risks

- PowerShell is not available in this Linux sandbox, so bootstrap behavior was validated by code-path inspection plus post-install test evidence rather than direct `bootstrap.ps1` execution.
- `requirements-dev.txt` leaves dev package versions unpinned (`pytest`, `pytest-asyncio`), so future upstream drift can affect clean bootstrap reproducibility.

## Stop reason

Stopped after the selected rank-2 issue was fixed with a minimal one-file code change and validated with issue-specific and regression checks; further work would exceed this single-issue bounded run.

## Recommendation

- `promote`
