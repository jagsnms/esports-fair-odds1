# Promotion report

branch: fast/run-20260306-0807-bootstrap-dev-test-deps
base_branch: agent-base
lane: fast
run_type: maintenance
status: implemented
recommendation: promote

## Selected issue

- **Issue:** canonical tests fail after documented bootstrap because test dependencies are not installed
- **Title:** `scripts/bootstrap.ps1` installs runtime deps only, causing `python3 -m pytest -q` failure in clean setup

## Why it outranked alternatives

Issue ranking evidence from this run:

1. Structural invariant violations: none observed in replay diagnostics (`structural_violations_total=0`).
2. Failing canonical tests: present at baseline; `python3 -m pytest -q` failed with `/usr/bin/python3: No module named pytest`.
3. Confirmed replay mismatches: none observed after running replay assessment.
4. High-frequency diagnostic invariant failures: none observed (`behavioral_violations_total=0`, `invariant_violations_total=0`).

Because a rank-2 issue was reproducible in this run, lower-ranked items were not selected.

## Baseline evidence

- Baseline branch: `agent-base`
- Baseline commit: `6d1374ca22aa63df08b20b916b9a5793bd5cbffc`
- Baseline checks:
  - `python3 -m pytest -q` -> failed: `/usr/bin/python3: No module named pytest`
  - Script inspection: `scripts/bootstrap.ps1` installed `requirements.txt` only (`pip install -r $ReqTxt`)
  - `python3 tools/replay_verification_assess.py` (after dependency bootstrap for diagnostics) -> structural/behavioral/invariant totals all `0`

## Files changed

| Path | Change |
|------|--------|
| `scripts/bootstrap.ps1` | Updated bootstrap to install `requirements-dev.txt` (and fail early if file missing), ensuring pytest dependencies are installed in bootstraped dev environments. |
| `automation/reports/fast-run-20260306-0807-bootstrap-dev-test-deps.md` | Added human-readable report for this run. |
| `automation/reports/fast-run-20260306-0807-bootstrap-dev-test-deps.json` | Added machine-readable report for this run. |

## Validation performed

- `rg "requirements-dev\\.txt|pip install -r" scripts/bootstrap.ps1` -> confirms bootstrap now installs `requirements-dev.txt`
- `python3 -m pytest -q` -> `361 passed in 7.31s`
- `python3 tools/replay_verification_assess.py` -> `structural_violations_total=0`, `behavioral_violations_total=0`, `invariant_violations_total=0`

## Before/after evidence

- **Before:** On `agent-base`, canonical pytest could not run in the current environment (`No module named pytest`), and bootstrap script installed runtime deps only.
- **After:** Bootstrap script now installs `requirements-dev.txt` (which includes `pytest`/`pytest-asyncio`), and canonical tests run/passed in this run (`361 passed`).

## Unresolved risks

- PowerShell runtime is not available in this Linux sandbox (`pwsh` missing), so bootstrap execution was validated by script diff/contract checks rather than direct script execution.
- `requirements-dev.txt` uses unpinned dev packages (`pytest`, `pytest-asyncio`), which may change over time.

## Stop reason

Stopped after the selected single rank-2 issue was addressed with a minimal one-file code change and validated; additional changes would be outside this issue's bounded scope.

## Recommendation

- `promote`
