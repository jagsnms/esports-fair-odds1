"""
Initial/neutral state for reduce_state when no previous state exists (e.g. session-local state).
No backend or FastAPI dependencies.
"""
from __future__ import annotations

from engine.models import Config, State


def initial_state(config: Config) -> State:
    """
    Return a neutral State suitable as old_state for reduce_state when the session
    has no prior state (same structure as store default when empty).
    """
    return State(config=config)
