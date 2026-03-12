# Current Status - `master`

Last updated: 2026-03-12

## Snapshot
- **Promoted `master` initiative:** Backend BO3 capture corpus contract correction is the current promoted `master` state.
- **Promoted `master` artifact truth:** backend BO3 capture accumulates first at `logs/bo3_backend_live_capture_contract.jsonl`; normal reset preserves that corpus; and `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl` remains a separate frozen snapshot cut instead of the primary accumulating artifact.
- **Current local-stage branch note:** `codex/backend-bo3-corpus-readiness-analyzer` adds one separate corpus-level readiness analyzer at `tools/run_backend_bo3_corpus_readiness_analyzer.py`, one shared helper at `tools/backend_bo3_capture_analysis_common.py`, focused unit coverage at `tests/unit/test_run_backend_bo3_corpus_readiness_analyzer.py`, and one machine-readable local-stage report artifact at `automation/reports/backend_bo3_corpus_readiness_report.json`. It does not replace the existing bounded single-match diagnostic.
- **Current local-stage report artifact note:** `automation/reports/backend_bo3_corpus_readiness_report.json` is a point-in-time run of the new analyzer against the mutable local corpus state that was present when the tool was executed on this branch. It is useful local evidence, but its row counts are not promoted `master` truth and may change as the corpus keeps growing.
- **Current local-stage analyzer result captured in the carried report artifact:** the carried report records `blockage_assessment = easing_multi_match`, `distinct_match_count = 2`, `eligible_compared_row_count = 298`, `distinct_raw_event_count = 93`, and `distinct_signal_active_raw_event_count = 90`. Those counts belong to that point-in-time local-stage report artifact, not to a frozen promoted corpus state.
- **Bounded diagnostic boundary remains intact:** `tools/run_backend_bo3_live_parity_diagnostic.py` is still the separate one-match bounded truth surface reading the frozen snapshot path under `automation/reports/`. It is not the corpus-level readiness analyzer.

## Main red flags
1. **This new analyzer is not calibration implementation.** It only says whether the corpus looks broad and clean enough to justify later work.
2. **This new analyzer is not live parity implementation.** It does not compare live against replay or open replay/live decision logic.
3. **The bounded single-match diagnostic still matters.** It remains the narrow one-match sanity tool and should not be treated as the corpus analyzer.
4. **This is still BO3-only on purpose.** It does not redesign GRID, broader logging, or a generic analytics platform.

## Most recent completed checks
- `tests/unit/test_backend_bo3_capture_contract.py` remains the focused corpus-contract test surface for the persistent BO3 capture path.
- `tests/unit/test_run_backend_bo3_live_parity_diagnostic.py` passed and confirms the existing bounded diagnostic still excludes `non_primary_match` rows explicitly and remains a separate one-match tool.
- `tests/unit/test_run_backend_bo3_corpus_readiness_analyzer.py` passed and confirms the new analyzer reads the persistent corpus path by default, rolls up multi-match readiness truthfully, and surfaces exclusion reasons across matches instead of hiding them behind `non_primary_match`.

## Current initiative status
- **Actual current runtime BO3 ingestion path:** `backend/services/runner.py`.
- **Promoted `master` artifact contract:** persistent accumulating corpus at `logs/bo3_backend_live_capture_contract.jsonl`; separate optional frozen snapshot at `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl`.
- **Current local-stage analyzer contract:** the new readiness analyzer reads the persistent corpus, reports corpus-wide and match-by-match readiness contributions, and calls out dominant blockers without changing the bounded diagnostic's role.
- **Truth boundary:** this branch adds one narrow corpus-readiness analyzer only. It does not implement calibration, parity redesign, replay/live redesign, or a dashboard/reporting platform.

## Next likely step
- Re-review whether the new corpus-level analyzer is narrow, truthful, and useful enough to support promotion-readiness as a separate readiness surface while preserving the old bounded one-match diagnostic.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit when a branch has newer work than promoted `master`.
