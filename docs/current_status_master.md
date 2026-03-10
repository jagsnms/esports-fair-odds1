# Current Status - `master`

Last updated: 2026-03-10

## Snapshot
- **Promoted `master` initiative:** Backend-native BO3 live-capture/source contract for replay-anchored parity work is now the current promoted `master` state.
- **Branch-state assessment:** promoted `master` now carries one real-runtime BO3 capture contract on the FastAPI/backend path centered on `backend/services/runner.py`, with one append-only JSONL artifact at `logs/bo3_backend_live_capture_contract.jsonl`. The earlier legacy Streamlit parquet step remains in history, but it is not the current runtime contract.

## Main red flags
1. **This work is still not live parity implementation.** It does not compare live against replay, does not claim parity, and does not open broad live/replay decision logic.
2. **This work is BO3-only on purpose.** It does not unify BO3 and GRID, and it does not claim BO3 is the final best source for full live parity.
3. **The earlier legacy-path parquet step remains in repo history.** It should not be confused with the now-promoted real-runtime backend JSONL contract.
4. **The repo is still otherwise narrow.** The replay/simulation, simulation-evidence, and replay-anchored decision lanes remain truthful but intentionally bounded.

## Most recent completed checks
- `tests/unit/test_backend_bo3_capture_contract.py` passed (`2 passed`) on the real backend runner path and confirmed that accepted BO3 live frames append a backend-native JSONL capture artifact.
- The focused backend artifact-generation check `tests/unit/test_backend_bo3_capture_contract.py -k 'appends_jsonl_rows'` passed and confirmed that the canonical backend artifact is produced, append-only, preserves raw-event linkage, includes normalized frame fields, and includes derived diagnostics.
- A bounded real FastAPI/backend runtime verification run against live BO3 match `111953` (`K27` vs `Metizport`) succeeded, confirming that the selected match was actually polled on the real runtime path, raw BO3 JSONL appended, backend capture JSONL appended, `bo3_health_reason = null` is the healthy-state value, `clamp_reason = "ok"` is the truthful unclamped value, and the raw-row vs capture-row count difference is expected on this path.
- The local `.venv311` launcher still required the known outside-sandbox workaround for pytest invocation; this remained an environment quirk, not a product failure.

## Current initiative status
- **Actual current runtime BO3 ingestion path:** `backend/services/runner.py`.
- **Promoted backend artifact path:** `logs/bo3_backend_live_capture_contract.jsonl`.
- **Actual backend-runtime persisted live outputs on `master`:** raw BO3 snapshot JSONL (`logs/bo3_raw_match_<match_id>.jsonl` or `logs/bo3_raw.jsonl`), backend capture-contract JSONL (`logs/bo3_backend_live_capture_contract.jsonl`), plus backend history JSONL (`logs/history_points.jsonl`, `logs/history_score_points.jsonl`).
- **Backend-native BO3 capture contract now persisted on `master`:**
  - source identity: `schema_version = "backend_bo3_live_capture_contract.v1"`, `live_source = "BO3"`, `capture_ts_iso`, `match_id`, `team_a_is_team_one`
  - raw linkage: `raw_provider_event_id`, `raw_seq_index`, `raw_sent_time`, `raw_updated_at`, `raw_snapshot_ts`, `raw_record_path`
  - replay-anchorable identity: `game_number`, `map_index`, `round_number`, `round_phase`, team ids/provider ids, and side mapping actually used by the engine
  - normalized frame fields: map/round scores, `a_side`, `bomb_planted`, `round_time_remaining_s`, alive counts, HP totals, cash/loadout/armor totals, loadout source, and round-time normalization flags
  - derived diagnostics: `p_hat`, `rail_low`, `rail_high`, `series_low`, `series_high`, `bo3_snapshot_status`, `bo3_health`, optional/null `bo3_health_reason` and `bo3_feed_error`, `q_intra_total`, `midround_weight`, `clamp_reason` (`"ok"` when the live row is not clamped), and `dominance_score`
- **Capture-count behavior on the real backend path:** raw BO3 JSONL dedupes identical source snapshots by signature, while the backend capture contract and history logs append one row per accepted backend compute tick. That means capture/history row counts can legitimately exceed raw BO3 row counts during steady live polling.
- **Truth boundary:** this promoted step adds a real-runtime capture contract only. It is not live parity and not replay/live comparison completion.

## Next likely step
- Re-rank the next justified project from current `master` reality rather than assuming automatic live-parity follow-on work.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit when a branch has newer work than promoted `master`.


