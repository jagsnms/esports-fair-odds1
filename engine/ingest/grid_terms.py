"""
GRID field/event name mapping for canonical contract. MVP: dictionary-based translator.
No FastAPI deps.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# GraphQL (camelCase) -> normalized (snake_case) for known fields
GRID_FIELD_ALIASES: dict[str, str] = {
    "currentHealth": "current_health",
    "currentArmor": "current_armor",
    "loadoutValue": "loadout_value",
    "sequenceNumber": "sequence_number",
    "ticksBackwards": "ticks_backwards",
    "nameShortened": "name_shortened",
}


def normalize_field_name(key: str) -> str:
    """Return canonical name for a GRID field; else return key as-is."""
    return GRID_FIELD_ALIASES.get(key, key)


def log_unknown_event_type(event_type: str, payload: Any) -> None:
    """Log unknown event type for debugging (MVP: console/log)."""
    logger.debug("GRID unknown event type: %s (payload keys: %s)", event_type, list(payload.keys()) if isinstance(payload, dict) else "n/a")
