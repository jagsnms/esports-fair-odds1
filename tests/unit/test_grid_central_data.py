"""
Unit tests for GRID Central Data normalizer, service levels, ordering. No FastAPI.
"""
from __future__ import annotations

from engine.ingest.grid_central_data import (
    DEFAULT_ORDER_DIRECTION,
    GRID_CS2_TITLE_ID,
    _normalize_node,
    _sort_candidates_soonest_relevant,
    clear_cs2_candidates_cache,
    live_data_feed_level,
    normalize_service_levels,
    select_best_series_ids,
    service_level_rank,
)


def test_title_id_constant() -> None:
    assert GRID_CS2_TITLE_ID == "28"


def test_ordering_default_asc() -> None:
    """Default order direction is ASC (soonest first)."""
    assert DEFAULT_ORDER_DIRECTION == "ASC"


def test_sort_candidates_soonest_relevant() -> None:
    """Soonest start_time first, then higher live_data_feed_rank first."""
    later = {"start_time": "2026-03-02T18:00:00Z", "live_data_feed_rank": 3}
    sooner = {"start_time": "2026-03-01T18:00:00Z", "live_data_feed_rank": 1}
    same_time_full = {"start_time": "2026-03-01T18:00:00Z", "live_data_feed_rank": 3}
    sorted_list = _sort_candidates_soonest_relevant([later, sooner, same_time_full])
    assert sorted_list[0]["start_time"] == "2026-03-01T18:00:00Z"
    assert sorted_list[0]["live_data_feed_rank"] == 3
    assert sorted_list[1]["live_data_feed_rank"] == 1
    assert sorted_list[2]["start_time"] == "2026-03-02T18:00:00Z"


def test_service_level_normalization_handles_missing() -> None:
    """normalize_service_levels returns {} for missing/odd structures."""
    assert normalize_service_levels(None) == {}
    assert normalize_service_levels([]) == {}
    assert normalize_service_levels("FULL") == {}
    assert normalize_service_levels([["FULL"]]) == {}
    assert normalize_service_levels([{}]) == {}


def test_service_level_normalization_parses_objects() -> None:
    """normalize_service_levels parses list of product/level objects."""
    assert normalize_service_levels([
        {"product": "liveDataFeed", "level": "FULL"},
        {"name": "centralData", "level": "PARTIAL"},
    ]) == {"liveDataFeed": "FULL", "centralData": "PARTIAL"}


def test_service_level_rank_full_beats_none() -> None:
    """FULL has highest rank, NONE lowest."""
    assert service_level_rank("FULL") == 3
    assert service_level_rank("PARTIAL") == 2
    assert service_level_rank("UNKNOWN") == 1
    assert service_level_rank("NONE") == 0
    assert service_level_rank("other") == 1


def test_live_data_feed_level_from_normalized() -> None:
    """live_data_feed_level returns key from normalized dict or UNKNOWN."""
    assert live_data_feed_level({"liveDataFeed": "FULL"}) == "FULL"
    assert live_data_feed_level({"live_data_feed": "PARTIAL"}) == "PARTIAL"
    assert live_data_feed_level({}) == "UNKNOWN"
    assert live_data_feed_level({"centralData": "FULL"}) == "UNKNOWN"


def test_normalize_node_sets_live_data_feed_level() -> None:
    """_normalize_node sets live_data_feed_level and live_data_feed_rank."""
    node = {
        "id": "1",
        "productServiceLevels": [
            {"product": "liveDataFeed", "level": "FULL"},
        ],
    }
    out = _normalize_node(node)
    assert out["live_data_feed_level"] == "FULL"
    assert out["live_data_feed_rank"] == 3
    assert out.get("product_service_levels") is not None


def test_normalize_node_handles_missing_fields() -> None:
    """Normalizer returns safe defaults when fields are missing."""
    out = _normalize_node({})
    assert out["series_id"] is None
    assert out["name"] is None
    assert out["tournament_name"] is None
    assert out["start_time"] is None
    assert out["updated_at"] is None
    assert out.get("product_service_levels") is None
    assert out["live_data_feed_level"] == "UNKNOWN"
    assert out["live_data_feed_rank"] == 1


def test_normalize_node_partial_node() -> None:
    """Normalizer fills what is present; ignores missing."""
    node = {"id": "123", "title": {"nameShortened": "cs2"}}
    out = _normalize_node(node)
    assert out["series_id"] == "123"
    assert out["name"] == "cs2"
    assert out["tournament_name"] is None
    assert out["start_time"] is None


def test_normalize_node_tournament_and_times() -> None:
    node = {
        "id": "456",
        "title": {"name": "Counter Strike 2", "nameShortened": "cs2"},
        "tournament": {"id": "t1", "name": "Blast Premier"},
        "startTimeScheduled": "2026-03-01T18:00:00Z",
        "updatedAt": "2026-03-01T17:00:00Z",
        "type": "ESPORT",
        "productServiceLevels": ["FULL"],
    }
    out = _normalize_node(node)
    assert out["series_id"] == "456"
    assert out["name"] == "Counter Strike 2"
    assert out["tournament_name"] == "Blast Premier"
    assert out["start_time"] == "2026-03-01T18:00:00Z"
    assert out["updated_at"] == "2026-03-01T17:00:00Z"
    assert out["type"] == "ESPORT"
    assert out["product_service_levels"] == ["FULL"]
    # List of plain strings doesn't give product key -> UNKNOWN
    assert out["live_data_feed_level"] == "UNKNOWN"
    assert out["live_data_feed_rank"] == 1


def test_normalize_node_title_not_dict() -> None:
    """When title is a string, name is that string."""
    out = _normalize_node({"id": "1", "title": "CS2"})
    assert out["series_id"] == "1"
    assert out["name"] == "CS2"


def test_clear_cache() -> None:
    """clear_cs2_candidates_cache resets cache."""
    clear_cs2_candidates_cache()
    # After clear, next get_cs2_series_candidates would refetch (no assertion on API)
    assert True


# --- select_best_series_ids (auto-track) ---


def test_select_best_full_chosen_over_partial_unknown() -> None:
    """FULL is chosen before PARTIAL and UNKNOWN when within limit."""
    candidates = [
        {"series_id": "id-partial", "live_data_feed_rank": 2, "start_time": "2026-03-01T12:00:00Z", "updated_at": None},
        {"series_id": "id-full", "live_data_feed_rank": 3, "start_time": "2026-03-01T10:00:00Z", "updated_at": None},
        {"series_id": "id-unknown", "live_data_feed_rank": 1, "start_time": "2026-03-01T09:00:00Z", "updated_at": None},
    ]
    out = select_best_series_ids(candidates, limit=3, min_rank=2)
    assert out == ["id-full", "id-partial"]
    assert "id-unknown" not in out


def test_select_best_respects_limit() -> None:
    """Selection is capped at limit."""
    candidates = [
        {"series_id": f"id-{i}", "live_data_feed_rank": 3, "start_time": "2026-03-01T10:00:00Z", "updated_at": None}
        for i in range(10)
    ]
    out = select_best_series_ids(candidates, limit=3, min_rank=2)
    assert len(out) == 3
    assert out == ["id-0", "id-1", "id-2"]


def test_select_best_start_time_tie_breaker_soonest_first() -> None:
    """When same rank, sooner start_time comes first."""
    candidates = [
        {"series_id": "later", "live_data_feed_rank": 3, "start_time": "2026-03-02T18:00:00Z", "updated_at": None},
        {"series_id": "sooner", "live_data_feed_rank": 3, "start_time": "2026-03-01T18:00:00Z", "updated_at": None},
    ]
    out = select_best_series_ids(candidates, limit=2, min_rank=2)
    assert out == ["sooner", "later"]


def test_select_best_ignores_missing_series_id() -> None:
    """Candidates with missing or empty series_id are skipped."""
    candidates = [
        {"series_id": "ok", "live_data_feed_rank": 3, "start_time": "2026-03-01T10:00:00Z", "updated_at": None},
        {"series_id": None, "live_data_feed_rank": 3, "start_time": "2026-03-01T11:00:00Z", "updated_at": None},
        {"series_id": "", "live_data_feed_rank": 3, "start_time": "2026-03-01T12:00:00Z", "updated_at": None},
    ]
    out = select_best_series_ids(candidates, limit=5, min_rank=2)
    assert out == ["ok"]


def test_select_best_unknown_fallback_when_no_full_or_partial() -> None:
    """When no candidate has rank >= 2, fallback to UNKNOWN (rank 1) if allow_unknown_fallback."""
    candidates = [
        {"series_id": "id-unknown", "live_data_feed_rank": 1, "start_time": "2026-03-01T10:00:00Z", "updated_at": None},
    ]
    out = select_best_series_ids(candidates, limit=5, min_rank=2, allow_unknown_fallback=True)
    assert out == ["id-unknown"]
    out_no_fallback = select_best_series_ids(candidates, limit=5, min_rank=2, allow_unknown_fallback=False)
    assert out_no_fallback == []
