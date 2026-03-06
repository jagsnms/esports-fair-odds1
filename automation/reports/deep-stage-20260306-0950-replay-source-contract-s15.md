# Deep Stage 1.5 — Replay Source Contract Signaling (approved scope)

## Scope (locked)

Implement replay source contract signaling so non-canonical point sources remain visible but are **never silently presented as canonical-selectable** under default `reject_point_like`, while keeping **runner gate enforcement unchanged**.

This stage adds signaling only:
- `/replay/sources`: per-source eligibility metadata + deterministic reason codes.
- `/replay/load`: lightweight preflight signaling in the response (no enforcement).

No runner or compute changes.

## Files changed

- `backend/api/routes_replay.py`
- `tests/unit/test_replay_status_contract_gate.py`
- `tests/unit/test_replay_sources_contract_signaling.py` (new)

## Behavior changed (concrete)

### `/replay/sources`

Each source row now includes:
- `selectable`
- `contract_class`
- `reason_code`
- `requires_transition_mode`
- `transition_window_valid`

Rules:
- **raw sources**: `selectable=True`, `contract_class="canonical_raw"`, `reason_code=None`.
- **point sources**:
  - default config: `selectable=False`, `contract_class="non_canonical_point"`, `reason_code="POINT_REPLAY_REJECTED_DEFAULT_POLICY"`, `requires_transition_mode=True`
  - transition enabled + sunset valid: `selectable=True`, `reason_code=None`, `transition_window_valid=True`

Reason codes reuse the existing runner constants to avoid API-vs-runner drift.

### `/replay/load`

Response now includes:
- `replay_load_preflight: { selectable, contract_class, reason_code, requires_transition_mode, transition_window_valid }`

This is signaling only; runner remains the final enforcement authority.

## Acceptance criteria status

- `/replay/sources` includes the new contract fields for every source: **met**
- default config: point rows non-selectable with deterministic reason: **met**
- transition-enabled + valid-sunset: point rows may become selectable with deterministic signaling: **met**
- raw sources remain selectable canonical rows: **met**
- `/replay/load` always includes deterministic `replay_load_preflight` metadata: **met**
- existing runner gate tests still pass unchanged: **met** (runner not modified; test suite for runner gate remains green)
- no runner or compute semantics changed: **met** (runner and engine/compute untouched)

## Validation

Commands and results:
- `python -m pytest -q tests/unit/test_replay_sources_contract_signaling.py` → **4 passed**
- `python -m pytest -q tests/unit/test_replay_status_contract_gate.py` → **2 passed**
- `python -m pytest -q tests/unit/test_runner_replay_contract_mode.py` → **6 passed**
- `python -m pytest -q tests/unit/test_replay_verification_assess_stage1.py` → **1 passed**
- `python -m pytest -q` → **5 failed, 364 passed**
  - Failures are in `tests/unit/test_runner_map_identity.py` due to Windows asyncio event loop expectations; pre-existing and not caused by this stage’s API signaling work.

## Risks / scope pressure encountered

- Minimal risk of policy drift was addressed by reusing the runner’s reason-code constants in the API layer (bounded import only; no new framework).
- `/replay/load` preflight uses a lightweight kind inference (discovered sources preferred; filename fallback) to avoid broad discovery redesign.

## Explicit out-of-scope preserved

- **Runner behavior unchanged** (no edits to `backend/services/runner.py`)
- **No engine/compute changes**
- No replay conversion/migration mechanics
- No replay source discovery redesign beyond adding metadata on existing discovery rows

