# Current Status - `master`

Last updated: 2026-03-12

## Snapshot
- **Promoted `master` initiative:** Backend-native BO3 live-capture/source contract for replay-anchored parity work is now the current promoted `master` state.
- **Branch-state assessment:** promoted `master` currently carries a lifecycle split where the live backend writer appends to the disposable runtime path `logs/runtime/bo3_backend_live_capture_contract.jsonl` and the committed frozen evidence artifact is `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl`. That split is internally clean, but it over-weights snapshot neatness relative to the actual corpus-growth mission.
- **Current corpus-correction branch note:** `codex/backend-bo3-corpus-contract-correction` restores the mission-first contract on top of promoted `master`. The canonical BO3 capture artifact on this branch is again the persistent accumulating corpus at `logs/bo3_backend_live_capture_contract.jsonl`; normal reset preserves that corpus; and `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl` remains only a separate frozen cut that supports the corpus instead of replacing it.
- **Current diagnostic-stage branch note:** this branch still carries `automation/reports/backend_bo3_live_parity_diagnostic_report.json`, which consumes the frozen evidence snapshot `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl` cut from the corpus, selects dominant match `113437` (`459/463` rows), excludes `409` rows explicitly, keeps the per-tick compared count visible at `54`, and derives its final decision from `17` distinct truthfully comparable raw events rather than treating duplicate ticks as independent evidence. The current bounded result is `decision = inconclusive`. This is still one bounded diagnostic only. It is not live parity implementation, replay/live linkage, or broad representativeness proof.

## Main red flags
1. **This work is still not live parity implementation.** It does not compare live against replay, does not claim parity, and does not open broad live/replay decision logic.
2. **This correction is BO3-only on purpose.** It does not redesign GRID, broader logging, or general artifact policy.
3. **Frozen snapshots are secondary only.** They can support bounded audit or diagnostics, but they must not replace the accumulating corpus.
4. **The repo is still otherwise narrow.** The replay/simulation, simulation-evidence, and replay-anchored decision lanes remain truthful but intentionally bounded.

## Most recent completed checks
- `tests/unit/test_backend_bo3_capture_contract.py` is the focused corpus-contract test surface for this branch and should confirm that accepted BO3 live frames append a backend-native JSONL capture artifact while the default backend capture path is the persistent corpus `logs/bo3_backend_live_capture_contract.jsonl`.
- The bounded diagnostic remains a separate consumer of the frozen snapshot path under `automation/reports/`; it is not the primary collection artifact.

## Current initiative status
- **Actual current runtime BO3 ingestion path:** `backend/services/runner.py`.
- **Promoted `master` artifact split before this correction stage:** disposable runtime writer path `logs/runtime/bo3_backend_live_capture_contract.jsonl`, frozen evidence snapshot `automation/reports/backend_bo3_live_capture_contract_snapshot_v1.jsonl`.
- **Local-stage corrected corpus/snapshot split on this branch:** backend capture now appends to the persistent corpus `logs/bo3_backend_live_capture_contract.jsonl`; frozen evidence snapshots remain separate under `automation/reports/`; normal reset preserves the corpus and treats only the runtime logs as disposable.
- **Corpus-first truth boundary:** this branch corrects artifact roles only. It does not redesign parity, replay/live linkage, or broader artifact management.

## Next likely step
- Validate the restored corpus contract on this branch and review whether the correction truly protects accumulation first without reintroducing corpus/snapshot ambiguity.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit when a branch has newer work than promoted `master`.
