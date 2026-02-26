"""
Config defaults and safe merge of partial updates.

Used by the backend when applying UI/control updates without overwriting
fields that were not sent.
"""
from __future__ import annotations

from typing import Any, Optional

from engine.models import Config


DEFAULTS: dict[str, Any] = {
    "source": "BO3",
    "match_id": None,
    "poll_interval_s": 5.0,
    "contract_scope": "",
    "series_fmt": "",
    "prematch_map": None,
    "prematch_series": None,
    "lock_team_mapping": False,
    "market_delay_s": 0.0,
    "team_a_is_team_one": True,
}


def _coerce_match_id(value: Any) -> Optional[int]:
    """Coerce match_id: None/'' -> None; int -> int; string of digits -> int. Else raise ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"invalid match_id: {value!r}")


def merge_config(current: Config, partial: dict[str, Any]) -> Config:
    """
    Return a new Config with only the keys present in `partial` updated.
    Unknown keys in `partial` are ignored. Does not mutate `current`.
    Normalizes source to uppercase and match_id to int or None.
    """
    allowed = set(DEFAULTS)
    updates = {k: v for k, v in partial.items() if k in allowed}
    if not updates:
        return current
    if "source" in updates and isinstance(updates["source"], str):
        updates["source"] = (updates["source"] or "BO3").strip().upper()
        if updates["source"] not in ("BO3", "GRID"):
            updates["source"] = "BO3"
    if "match_id" in updates:
        updates["match_id"] = _coerce_match_id(updates["match_id"])
    d = {
        "source": getattr(current, "source"),
        "match_id": getattr(current, "match_id"),
        "poll_interval_s": getattr(current, "poll_interval_s"),
        "contract_scope": getattr(current, "contract_scope"),
        "series_fmt": getattr(current, "series_fmt"),
        "prematch_map": getattr(current, "prematch_map"),
        "prematch_series": getattr(current, "prematch_series"),
        "lock_team_mapping": getattr(current, "lock_team_mapping"),
        "market_delay_s": getattr(current, "market_delay_s"),
        "team_a_is_team_one": getattr(current, "team_a_is_team_one", True),
    }
    d.update(updates)
    return Config(**d)
