"""
WebSocket /api/v1/stream: incremental updates and heartbeat (stub).
"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/stream")
async def stream_ws(websocket: WebSocket) -> None:
    """
    Accept connections and send a simple heartbeat every 2s.
    Stub; will later push State/Derived/HistoryPoint updates.
    """
    await websocket.accept()
    try:
        while True:
            # Send heartbeat
            await websocket.send_json({"type": "heartbeat", "ok": True})
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
