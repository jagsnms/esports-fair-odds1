"""
Config defaults and safe merge of partial updates.

Used by the backend when applying UI/control updates without overwriting
fields that were not sent.
"""
from __future__ import annotations

from typing import Any

from engine.models import Config


DEFAULTS: dict[str, Any] = {
    "source": "BO3",
    "match_id": "",
    "poll_interval_s": 5.0,
    "contract_scope": "",
    "series_fmt": "",
    "prematch_map": None,
    "prematch_series": None,
    "lock_team_mapping": False,
    "market_delay_s": 0.0,
}


def merge_config(current: Config, partial: dict[str, Any]) -> Config:
    """
    Return a new Config with only the keys present in `partial` updated.
    Unknown keys in `partial` are ignored. Does not mutate `current`.
    """
    allowed = set(DEFAULTS)
    updates = {k: v for k, v in partial.items() if k in allowed}
    if not updates:
        return current
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
    }
    d.update(updates)
    return Config(**d)
