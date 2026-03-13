"""
Tests for corridor naming/alias semantics in state/derived serialization.
"""
from __future__ import annotations

from backend.store.memory_store import _history_point_to_wire, _state_derived_to_dict
from engine.models import Config, Derived, Frame, HistoryPoint, State


def test_derived_state_includes_series_and_map_aliases() -> None:
  """Serialized current state.derived includes series_low/high and map_low/high."""
  cfg = Config()
  frame = Frame()
  state = State(config=cfg, last_frame=frame, map_index=0, last_total_rounds=0, segment_id=0)
  derived = Derived(
    p_hat=0.5,
    bound_low=0.25,
    bound_high=0.75,
    rail_low=0.3,
    rail_high=0.7,
    kappa=0.0,
  )
  payload = _state_derived_to_dict(state, derived)
  d = payload["derived"]
  assert d["bound_low"] == 0.25
  assert d["bound_high"] == 0.75
  assert d["rail_low"] == 0.3
  assert d["rail_high"] == 0.7
  assert d["series_low"] == 0.25
  assert d["series_high"] == 0.75
  assert d["map_low"] == 0.3
  assert d["map_high"] == 0.7


def test_history_point_wire_includes_series_and_map_aliases() -> None:
  """Wire point payload includes both legacy and new corridor keys."""
  p = HistoryPoint(
    time=123.0,
    p_hat=0.6,
    bound_low=0.2,
    bound_high=0.8,
    rail_low=0.25,
    rail_high=0.75,
    market_mid=None,
    segment_id=1,
    match_id=321,
  )
  out = _history_point_to_wire(p)
  # Legacy keys
  assert out["lo"] == 0.2
  assert out["hi"] == 0.8
  assert out["rail_low"] == 0.25
  assert out["rail_high"] == 0.75
  # New aliases
  assert out["series_low"] == 0.2
  assert out["series_high"] == 0.8
  assert out["map_low"] == 0.25
  assert out["map_high"] == 0.75
  assert out["match_id"] == 321


