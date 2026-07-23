from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os
import asyncio
import json

try:
    from cv_pipeline import ObjectDetector
    ai_detector = ObjectDetector()
except ImportError:
    print("[WARN] Could not import cv_pipeline. AI features disabled.")
    ai_detector = None

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

# ── AI Vision Endpoint ──

@app.websocket("/ws/cv")
async def cv_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Computer Vision processing.
    Handles message types:
      - "set_general_detection" - toggle general object detection on/off
      - regular frames - run detection and return results
    """
    await websocket.accept()
    print("[CV] Dashboard connected to AI vision pipeline.")
    
    try:
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)

                # TOGGLE GENERAL DETECTION
                if data.get("type") == "set_general_detection":
                    enabled = data.get("enabled", False)
                    if ai_detector:
                        ai_detector.set_general_detection(enabled)
                        await websocket.send_json({
                            "type": "general_detection_status",
                            "enabled": enabled
                        })
                    continue

                # REGULAR DETECTION FRAME
                image_b64 = data.get("image", "")
                threshold = data.get("threshold", 0.4)
                
                if ai_detector and image_b64:
                    results = ai_detector.detect_objects(image_b64, confidence_threshold=threshold)
                    
                    grid = data.get("grid")
                    if grid and len(results.get("detections", [])) > 0:
                        instruction = ai_detector.check_geofence(results["detections"], grid)
                        results["instruction"] = instruction

                    if data.get("auto_grid") == True:
                        line_results = ai_detector.detect_floor_grid(image_b64)
                        results["physical_lines"] = line_results.get("lines", [])

                    await websocket.send_json(results)
                else:
                    await websocket.send_json({"error": "AI disabled or no image provided", "detections": []})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON", "detections": []})
                
    except WebSocketDisconnect:
        print("[CV] Dashboard disconnected from AI vision pipeline.")
    except Exception as e:
        print(f"[CV] Error in CV pipeline: {e}")
