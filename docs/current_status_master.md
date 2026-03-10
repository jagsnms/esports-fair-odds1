# Current Status - `master`

Last updated: 2026-03-10

## Snapshot
- **Promoted `master` initiative:** Bounded replay-anchored two-source decision contract on the canonical Phase 2 lane remains the current promoted `master` state.
- **Local-stage assessment:** A bounded BO3-authoritative live-capture/source contract now sits on top of `master` locally. Real BO3 auto-pull activation now auto-enables the snapshot-lock gate for this bounded contract, so BO3 live runs produce one append-only canonical live artifact row in `data/processed/cs2_replay_snapshots.parquet` without a separate manual lock step. The row preserves explicit raw-event linkage, normalized engine-consumed frame fields, and derived intraround/parity diagnostics suitable for later replay-anchored parity work.

## Main red flags
1. **This local stage is not live parity implementation.** It does not compare live against replay, does not claim parity, and does not open broad live/replay decision logic.
2. **This local stage is BO3-only on purpose.** It does not unify BO3 and GRID, and it does not claim BO3 is the final best source for full live parity.
3. **This local stage is still bounded capture-contract work only.** It creates one reusable evidence artifact; it does not establish broad representativeness or broader replay/live resolution.
4. **Promoted `master` is still otherwise narrow.** The previously landed replay/simulation and simulation-evidence lanes remain truthful but intentionally bounded.

## Most recent completed checks
- `tests/unit/test_bo3_live_capture_contract.py` passed (`5 passed`), covering raw-linkage preservation, append-only artifact generation, live-only persistence gating, the BO3 auto-activation lock/default-capture regression case, and a parse smoke check for `legacy/app/app35_ml.py`.
- The focused bounded artifact-generation check `tests/unit/test_bo3_live_capture_contract.py::test_bo3_live_capture_contract_persists_append_only_artifact` passed and confirmed that the canonical BO3 live artifact is produced, append-only, preserves raw-event linkage, includes normalized frame fields, includes derived diagnostics, and does not rely on the old broad `cs2_inplay_persist` toggle.
- The local `.venv311` launcher still required the known outside-sandbox workaround for pytest invocation; this remained an environment quirk, not a product failure.

## Current initiative status
- **Canonical live artifact path:** `data/processed/cs2_replay_snapshots.parquet`.
- **Authoritative live source for this stage:** BO3 only.
- **Minimum contract now persisted for BO3 live rows:**
  - source identity: `schema_version = "bo3_live_capture_contract.v1"`, `live_source = "BO3"`, `capture_ts_iso`, `match_id`, `team_a_is_team_one`
  - raw linkage: `raw_ts_utc`, `raw_provider_event_id`, `raw_seq_index`, `raw_sent_time`, `raw_updated_at`, `raw_record_path`
  - replay-anchorable identity: `game_number`, `round_number`, `round_phase`, `round_key_current`, team ids/provider ids, side mapping used by the engine
  - normalized frame fields: map/round scores, `a_side`, `bomb_planted`, `round_time_remaining_s`, alive counts, HP totals, cash/loadout/armor totals, and `intraround_state_source`
  - derived diagnostics: `p_hat`, `p_hat_map`, `rail_low`, `rail_high`, `q_intra_round_win_a`, `q_intra_round_win_a_source`, plus existing intraround score terms already emitted by the snapshot row
- **Default capture behavior change:** BO3 live auto activation now always records raw BO3 pulls to `logs/bo3_pulls.jsonl`, automatically enables the snapshot-lock gate for this bounded contract, and canonical BO3 live snapshot persistence still stays decoupled from the old broad `Persist snapshots + results` toggle.
- **Truth boundary:** this stage makes real-match BO3 collection reusable later; it does not claim that the current repo can already answer replay/live parity questions.

## Next likely step
- Review whether this bounded BO3 live-capture/source-contract step is clean enough for promotion before opening any replay/live parity project.

## Process note for future pushes
- Append one new entry to `docs/branch_history_master.md` per final push.
- Update this status note to the new `master` branch state each time.
- Keep local-stage notes explicit when a branch has newer work than promoted `master`.

