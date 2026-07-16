from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os
import asyncio

app = FastAPI(title="AGRA Mission Control")

# Track all connected WebSocket clients
connections: list[WebSocket] = []

# ── Routes ──

@app.get("/")
async def dashboard():
    """Serve the main mission control dashboard (laptop viewer)."""
    filepath = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(filepath, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/broadcaster")
async def broadcaster():
    """Serve the iPhone broadcaster page."""
    filepath = os.path.join(os.path.dirname(__file__), "broadcaster.html")
    with open(filepath, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "connections": len(connections)}


# ── WebSocket Signaling Server ──

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebRTC signaling relay.
    
    Relays SDP offers/answers and ICE candidates between
    the iPhone broadcaster and the laptop dashboard viewer.
    Supports keepalive pings to prevent connection drops.
    """
    await websocket.accept()
    connections.append(websocket)
    print(f"[WS] Client connected. Total: {len(connections)}")

    try:
        while True:
            data = await websocket.receive_text()

            # Relay signaling messages to all OTHER connected peers
            disconnected = []
            for conn in connections:
                if conn != websocket:
                    try:
                        await conn.send_text(data)
                    except Exception:
                        disconnected.append(conn)

            # Clean up any dead connections
            for conn in disconnected:
                if conn in connections:
                    connections.remove(conn)

    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(connections)}")
    except Exception as e:
        if websocket in connections:
            connections.remove(websocket)
        print(f"[WS] Error: {e}. Total: {len(connections)}")
