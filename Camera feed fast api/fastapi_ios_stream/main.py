from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
connections = []

@app.get("/")
async def get_index():
    return HTMLResponse("<h1>Dashboards</h1><a href='/broadcaster'>1. iPhone Broadcaster</a><br><br><a href='/viewer'>2. Laptop Viewer</a>")

@app.get("/broadcaster")
async def get_broadcaster():
    with open("broadcaster.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/viewer")
async def get_viewer():
    with open("viewer.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Relay signaling messages to the other peer
            for conn in connections:
                if conn != websocket:
                    await conn.send_text(data)
    except WebSocketDisconnect:
        connections.remove(websocket)
