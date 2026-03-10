"""
Unit tests for BO3 readiness cache: TTL/budget in filter_needs_probe, select_telemetry_ready_match_ids.
No FastAPI.
"""
from __future__ import annotations

from engine.ingest.bo3_readiness_cache import (
    DEFAULT_TTL_S,
    filter_needs_probe,
    select_telemetry_ready_match_ids,
    tier_rank,
    update_cache_from_results,
)


def test_filter_needs_probe_empty_cache() -> None:
    """All match_ids need probe when cache is empty."""
    out = filter_needs_probe([1, 2, 3], {}, now_ts=1000.0, ttl_s=60.0, budget=10)
    assert out == [1, 2, 3]


def test_filter_needs_probe_respects_budget() -> None:
    """Result is capped at budget."""
    out = filter_needs_probe([1, 2, 3, 4, 5], {}, now_ts=1000.0, ttl_s=60.0, budget=2)
    assert out == [1, 2]


def test_filter_needs_probe_ttl_fresh_entries_skip() -> None:
    """Entries probed within TTL are not in need_probe."""
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 950.0},
        2: {"telemetry_ready": False, "last_probe_ts_unix": 960.0},
    }
    out = filter_needs_probe([1, 2, 3], cache, now_ts=1000.0, ttl_s=60.0, budget=10)
    assert out == [3]


def test_filter_needs_probe_ttl_stale_entries_included() -> None:
    """Entries older than TTL need probe again."""
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 900.0},
        2: {"telemetry_ready": False, "last_probe_ts_unix": 939.0},
    }
    out = filter_needs_probe([1, 2], cache, now_ts=1000.0, ttl_s=60.0, budget=10)
    assert 1 in out and 2 in out
    assert len(out) == 2


def test_filter_needs_probe_missing_last_probe_ts_unix_treated_stale() -> None:
    """Entry without last_probe_ts_unix is treated as stale."""
    cache = {1: {"telemetry_ready": True}}
    out = filter_needs_probe([1], cache, now_ts=1000.0, ttl_s=60.0, budget=10)
    assert out == [1]


def test_update_cache_from_results() -> None:
    """update_cache_from_results merges results and sets last_probe_ts_unix."""
    cache: dict[int, dict] = {}
    results = [
        {"match_id": 10, "telemetry_ready": True, "status_code": 200, "reason": "ok", "last_probe_ts": "2026-01-01T12:00:00Z"},
        {"match_id": 20, "telemetry_ready": False, "status_code": 404, "reason": "not_ready_404", "last_probe_ts": "2026-01-01T12:00:00Z"},
    ]
    update_cache_from_results(cache, results, now_ts=1000.0)
    assert cache[10]["telemetry_ready"] is True
    assert cache[10]["last_probe_ts_unix"] == 1000.0
    assert cache[20]["telemetry_ready"] is False
    assert cache[20]["last_probe_ts_unix"] == 1000.0


def test_select_telemetry_ready_match_ids_respects_limit() -> None:
    """Selection is capped at limit."""
    candidates = [
        {"id": 1, "live_coverage": True, "tier": 1},
        {"id": 2, "live_coverage": True, "tier": 1},
        {"id": 3, "live_coverage": False, "tier": 2},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 0},
        2: {"telemetry_ready": True, "last_probe_ts_unix": 0},
        3: {"telemetry_ready": True, "last_probe_ts_unix": 0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=2)
    assert len(out) == 2


def test_select_telemetry_ready_match_ids_only_ready() -> None:
    """Only candidates with telemetry_ready in cache are returned."""
    candidates = [
        {"id": 1, "live_coverage": True, "tier": 1},
        {"id": 2, "live_coverage": True, "tier": 1},
        {"id": 3, "live_coverage": True, "tier": 1},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 0},
        2: {"telemetry_ready": False, "last_probe_ts_unix": 0},
        3: {"telemetry_ready": True, "last_probe_ts_unix": 0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=10)
    assert set(out) == {1, 3}


def test_select_telemetry_ready_match_ids_prefer_live_coverage() -> None:
    """Among ready, live_coverage True is preferred (comes first); then lower tier."""
    candidates = [
        {"id": 1, "live_coverage": False, "tier": 1},
        {"id": 2, "live_coverage": True, "tier": 1},
        {"id": 3, "live_coverage": True, "tier": 2},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 0},
        2: {"telemetry_ready": True, "last_probe_ts_unix": 0},
        3: {"telemetry_ready": True, "last_probe_ts_unix": 0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=10)
    assert len(out) == 3
    assert 2 in out and 3 in out and 1 in out
    idx_live = min(out.index(2), out.index(3))
    idx_not_live = out.index(1)
    assert idx_live < idx_not_live


def test_default_ttl_constant() -> None:
    assert DEFAULT_TTL_S == 60.0


def test_tier_rank_mapping() -> None:
    """tier_rank maps s/a/b/c/d (and uppercase) to 5/4/3/2/1; None/unknown -> 0."""
    assert tier_rank("s") == 5
    assert tier_rank("S") == 5
    assert tier_rank("a") == 4
    assert tier_rank("A") == 4
    assert tier_rank("b") == 3
    assert tier_rank("c") == 2
    assert tier_rank("d") == 1
    assert tier_rank(None) == 0
    assert tier_rank("") == 0
    assert tier_rank("x") == 0
    assert tier_rank("unknown") == 0
    assert tier_rank(1) == 1
    assert tier_rank(5) == 5


def test_tier_rank_unknown_does_not_crash() -> None:
    """Unknown tier values return 0 and do not raise."""
    assert tier_rank("xyz") == 0
    assert tier_rank(99) == 0
    assert tier_rank([]) == 0


def test_selection_prefers_recent_ready() -> None:
    """Among telemetry_ready, more recent last_probe_ts_unix is preferred (comes first)."""
    candidates = [
        {"id": 1, "live_coverage": True, "tier": "a"},
        {"id": 2, "live_coverage": True, "tier": "a"},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 100.0},
        2: {"telemetry_ready": True, "last_probe_ts_unix": 200.0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=10)
    assert out == [2, 1]


def test_selection_prefers_live_coverage_when_same_freshness() -> None:
    """When last_probe_ts_unix is equal, live_coverage True is preferred."""
    candidates = [
        {"id": 1, "live_coverage": False, "tier": "a"},
        {"id": 2, "live_coverage": True, "tier": "a"},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 100.0},
        2: {"telemetry_ready": True, "last_probe_ts_unix": 100.0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=10)
    assert out == [2, 1]


def test_selection_tier_rank_ordering() -> None:
    """When ready and freshness equal, higher tier_rank (e.g. S over A) comes first."""
    candidates = [
        {"id": 1, "live_coverage": True, "tier": "a"},
        {"id": 2, "live_coverage": True, "tier": "s"},
    ]
    cache = {
        1: {"telemetry_ready": True, "last_probe_ts_unix": 100.0},
        2: {"telemetry_ready": True, "last_probe_ts_unix": 100.0},
    }
    out = select_telemetry_ready_match_ids(candidates, cache, limit=10)
    assert out == [2, 1]
