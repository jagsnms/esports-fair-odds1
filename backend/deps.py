"""
Dependencies: singleton store, broadcaster, runner. Set at app startup.
"""
from __future__ import annotations

from backend.services.broadcaster import Broadcaster
from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore

_store: MemoryStore | None = None
_broadcaster: Broadcaster | None = None
_runner: Runner | None = None


def set_store(store: MemoryStore) -> None:
    global _store
    _store = store


def get_store() -> MemoryStore:
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


def set_broadcaster(broadcaster: Broadcaster) -> None:
    global _broadcaster
    _broadcaster = broadcaster


def get_broadcaster() -> Broadcaster:
    if _broadcaster is None:
        raise RuntimeError("Broadcaster not initialized")
    return _broadcaster


def set_runner(runner: Runner) -> None:
    global _runner
    _runner = runner


def get_runner() -> Runner:
    if _runner is None:
        raise RuntimeError("Runner not initialized")
    return _runner
