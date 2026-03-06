from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.api.routes_replay import replay_load, replay_sources
from backend.services.runner import (
    REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY,
)
from backend.store.memory_store import MemoryStore
from engine.models import Config, Derived, State


def _mock_sources() -> dict:
    return {
        "default_path": "logs/bo3_pulls.jsonl",
        "sources": [
            {"label": "BO3 raw", "path": "logs/bo3_raw.jsonl", "kind": "raw", "mtime": 0.0, "size": 123},
            {"label": "History points", "path": "logs/history_points.jsonl", "kind": "point", "mtime": 0.0, "size": 456},
        ],
    }


@pytest.mark.asyncio
async def test_replay_sources_contract_metadata_default_reject_is_deterministic() -> None:
    store = MemoryStore(max_history=10)
    await store.set_current(
        State(config=Config(source="REPLAY", replay_path="logs/bo3_raw.jsonl", replay_loop=False), segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )

    with patch("backend.api.routes_replay._discover_replay_sources", return_value=_mock_sources()):
        with patch("backend.api.routes_replay.time.time", return_value=1000.0):
            first = await replay_sources(store=store)
            second = await replay_sources(store=store)

    assert first == second, "replay/sources contract signaling must be deterministic for same policy + sources"
    rows = first.get("sources") or []
    assert len(rows) == 2
    raw = next(r for r in rows if r.get("kind") == "raw")
    pt = next(r for r in rows if r.get("kind") == "point")

    assert raw["selectable"] is True
    assert raw["contract_class"] == "canonical_raw"
    assert raw["reason_code"] is None
    assert raw["requires_transition_mode"] is False

    assert pt["selectable"] is False
    assert pt["contract_class"] == "non_canonical_point"
    assert pt["reason_code"] == REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY
    assert pt["requires_transition_mode"] is True
    assert pt["transition_window_valid"] is False


@pytest.mark.asyncio
async def test_replay_sources_transition_enabled_valid_sunset_makes_point_selectable() -> None:
    store = MemoryStore(max_history=10)
    await store.set_current(
        State(config=Config(source="REPLAY", replay_path="logs/bo3_raw.jsonl", replay_loop=False), segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    await store.update_config(
        {
            "replay_point_transition_enabled": True,
            "replay_point_transition_sunset_epoch": 2000.0,
        }
    )

    with patch("backend.api.routes_replay._discover_replay_sources", return_value=_mock_sources()):
        with patch("backend.api.routes_replay.time.time", return_value=1000.0):
            out = await replay_sources(store=store)

    rows = out.get("sources") or []
    pt = next(r for r in rows if r.get("kind") == "point")
    assert pt["selectable"] is True
    assert pt["requires_transition_mode"] is True
    assert pt["transition_window_valid"] is True
    assert pt["reason_code"] is None


@pytest.mark.asyncio
async def test_replay_load_includes_preflight_metadata_for_point_source_default_reject() -> None:
    store = MemoryStore(max_history=10)
    await store.set_current(
        State(config=Config(source="REPLAY", replay_path="logs/bo3_raw.jsonl", replay_loop=False), segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    body = {"path": "logs/history_points.jsonl"}
    with patch("backend.api.routes_replay._discover_replay_sources", return_value=_mock_sources()):
        with patch("backend.api.routes_replay.time.time", return_value=1000.0):
            resp = await replay_load(body=body, store=store)
    pf = resp.get("replay_load_preflight")
    assert isinstance(pf, dict)
    assert pf["contract_class"] == "non_canonical_point"
    assert pf["selectable"] is False
    assert pf["reason_code"] == REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY
    assert pf["requires_transition_mode"] is True


@pytest.mark.asyncio
async def test_replay_load_preflight_reflects_transition_enabled_valid_sunset() -> None:
    store = MemoryStore(max_history=10)
    await store.set_current(
        State(config=Config(source="REPLAY", replay_path="logs/bo3_raw.jsonl", replay_loop=False), segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    body = {
        "path": "logs/history_points.jsonl",
        "replay_point_transition_enabled": True,
        "replay_point_transition_sunset_epoch": 2000.0,
    }
    with patch("backend.api.routes_replay._discover_replay_sources", return_value=_mock_sources()):
        with patch("backend.api.routes_replay.time.time", return_value=1000.0):
            resp = await replay_load(body=body, store=store)
    pf = resp.get("replay_load_preflight")
    assert isinstance(pf, dict)
    assert pf["contract_class"] == "non_canonical_point"
    assert pf["selectable"] is True
    assert pf["reason_code"] is None
    assert pf["requires_transition_mode"] is True
    assert pf["transition_window_valid"] is True

