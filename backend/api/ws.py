"""
WebSocket /api/v1/stream: snapshot on connect, then incremental point updates via broadcaster.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.deps import get_broadcaster, get_store

router = APIRouter()


@router.websocket("/stream")
async def stream_ws(websocket: WebSocket) -> None:
    """Accept, send snapshot (current + history 500), register for broadcast; unregister on disconnect."""
    await websocket.accept()
    store = get_store()
    broadcaster = get_broadcaster()
    await broadcaster.register(websocket)
    try:
        current = await store.get_current()
        history = await store.get_history(limit=500)
        await websocket.send_json({"type": "snapshot", "current": current, "history": history})
        # Keep connection open; broadcaster will send point updates
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.unregister(websocket)
