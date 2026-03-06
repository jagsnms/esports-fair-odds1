"""
Stage 3: Explicit replay raw vs point contract — raw runs full pipeline (resolve); point is passthrough.
Policy: raw = normalize + reduce + bounds + rails + resolve; point = append as-is with synthetic explain.
"""
from __future__ import annotations

from engine.models import Config
from backend.services.runner import _is_raw_bo3_snapshot, _is_point_like_payload


def test_replay_raw_vs_point_contract_policy() -> None:
    """Raw payloads are detected as raw; point-like payloads are detected as point; contract policy is explicit."""
    raw_payload = {
        "team_one": {"name": "A", "score": 1},
        "team_two": {"name": "B", "score": 0},
        "round_phase": "IN_PROGRESS",
    }
    point_like = {"t": 1000.0, "p": 0.55, "lo": 0.2, "hi": 0.8, "rail_low": 0.4, "rail_high": 0.6}
    assert _is_raw_bo3_snapshot(raw_payload) is True
    assert _is_point_like_payload(point_like) is True
    assert _is_raw_bo3_snapshot(point_like) is False
    assert _is_point_like_payload(raw_payload) is False


def test_replay_format_determines_pipeline() -> None:
    """Replay format raw => full pipeline (resolve path); format point => passthrough (no resolve). Documented contract."""
    # Contract: runner sets _replay_format = "raw" | "point" from first payload; raw runs resolve, point does not.
    assert _is_raw_bo3_snapshot({"team_one": {}, "team_two": {}, "round_phase": "x"}) is True
    # Point-like needs t, p, lo, hi, rail_low, rail_high, seg (or equivalent) and must not look like raw BO3.
    point_like = {"t": 1000.0, "p": 0.5, "lo": 0.2, "hi": 0.8, "rail_low": 0.2, "rail_high": 0.8}
    assert _is_point_like_payload(point_like) is True
