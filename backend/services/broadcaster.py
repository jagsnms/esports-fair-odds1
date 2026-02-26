"""
Simple broadcaster: holds set of websockets, broadcasts messages, prunes dead sockets.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class Broadcaster:
    """Maintains active WebSocket connections and broadcasts JSON messages."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send message to all connections; remove those that fail."""
        async with self._lock:
            dead: set[WebSocket] = set()
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self._connections.discard(ws)
