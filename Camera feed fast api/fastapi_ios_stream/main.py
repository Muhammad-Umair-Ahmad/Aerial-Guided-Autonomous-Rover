from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import json
import base64

try:
    from cv_pipeline import ObjectDetector
    ai_detector = ObjectDetector()
except ImportError:
    print("[WARN] Could not import cv_pipeline. AI features disabled.")
    ai_detector = None

try:
    from autopilot import AutopilotEngine
    autopilot = AutopilotEngine()
    print("[AUTOPILOT] Autopilot engine initialized.")
except ImportError as e:
    print(f"[WARN] Could not import autopilot: {e}")
    autopilot = None

app = FastAPI(title="AGRA Mission Control")

# ── CORS (allow dashboard to talk to ESP32 through proxy if needed) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track all connected WebSocket clients
connections: list[WebSocket] = []

# Store the latest camera frame (base64) for snapshot endpoint
latest_frame_b64: str = ""

# Track autopilot telemetry WebSocket clients
autopilot_ws_clients: list[WebSocket] = []

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
    return {
        "status": "ok",
        "frame_available": bool(latest_frame_b64),
        "connections": len(connections),
        "autopilot_available": autopilot is not None,
    }


@app.get("/snapshot")
async def snapshot():
    """Return the latest camera frame as JPEG (for debugging)."""
    if not latest_frame_b64:
        return JSONResponse({"error": "No frame available yet"}, status_code=404)

    b64_data = latest_frame_b64
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(b64_data)
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════
#  WebSocket Signaling Server (unchanged)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  AI Vision Endpoint (updated — feeds autopilot)
# ══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/cv")
async def cv_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Computer Vision processing.
    Handles message types:
      - "set_general_detection" - toggle general object detection on/off
      - regular frames - run detection and return results
    
    Now also feeds detection results to the autopilot tracker.
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

                # Store latest frame globally for snapshot endpoint
                if image_b64:
                    global latest_frame_b64
                    latest_frame_b64 = image_b64
                
                if ai_detector and image_b64:
                    results = ai_detector.detect_objects(image_b64, confidence_threshold=threshold)
                    
                    # ── Feed detection to autopilot tracker ──
                    if autopilot and results.get("detections"):
                        autopilot.feed_detection(results)

                    # ── Add rover center coordinates for convenience ──
                    if results.get("detections") and len(results["detections"]) > 0:
                        box = results["detections"][0]["box"]
                        results["rover_center"] = {
                            "x": box["x"] + box["width"] / 2,
                            "y": box["y"] + box["height"] / 2,
                        }

                    grid = data.get("grid")
                    if grid and len(results.get("detections", [])) > 0:
                        instruction = ai_detector.check_geofence(results["detections"], grid)
                        results["instruction"] = instruction

                    if data.get("auto_grid") == True:
                        line_results = ai_detector.detect_floor_grid(image_b64)
                        results["physical_lines"] = line_results.get("lines", [])

                    # ── Include autopilot state in CV response ──
                    if autopilot:
                        results["autopilot_state"] = autopilot.state.value

                    await websocket.send_json(results)
                else:
                    await websocket.send_json({"error": "AI disabled or no image provided", "detections": []})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON", "detections": []})
                
    except WebSocketDisconnect:
        print("[CV] Dashboard disconnected from AI vision pipeline.")
    except Exception as e:
        print(f"[CV] Error in CV pipeline: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  AUTOPILOT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/autopilot")
async def autopilot_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for real-time autopilot telemetry.
    Streams state updates to the dashboard every ~300ms.
    """
    await websocket.accept()
    autopilot_ws_clients.append(websocket)
    print("[AUTOPILOT] Dashboard connected to autopilot telemetry.")

    try:
        # Register a callback to push telemetry to this client
        async def send_telemetry(telemetry: dict):
            try:
                await websocket.send_json(telemetry)
            except Exception:
                pass

        if autopilot:
            autopilot.add_telemetry_callback(send_telemetry)

        # Also handle incoming messages (config updates, commands)
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "start_mission" and autopilot:
                    grid_config = data.get("grid_config", {})
                    esp32_ip = data.get("esp32_ip", "")
                    waypoints = data.get("waypoints")
                    asyncio.create_task(autopilot.start_mission(grid_config, esp32_ip, waypoints))
                    await websocket.send_json({"type": "ack", "msg": "Mission starting..."})

                elif msg_type == "stop_mission" and autopilot:
                    asyncio.create_task(autopilot.stop_mission())
                    await websocket.send_json({"type": "ack", "msg": "Mission stopped."})

                elif msg_type == "pause_mission" and autopilot:
                    asyncio.create_task(autopilot.pause_mission())
                    await websocket.send_json({"type": "ack", "msg": "Pause toggled."})

                elif msg_type == "update_config" and autopilot:
                    config = data.get("config", {})
                    autopilot.set_config(config)
                    # If speed changed, also send to ESP32
                    if "speed" in config:
                        asyncio.create_task(autopilot.motor.set_speed(config["speed"]))
                    await websocket.send_json({"type": "ack", "msg": "Config updated."})

                elif msg_type == "get_telemetry" and autopilot:
                    await websocket.send_json(autopilot.get_telemetry())

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        if autopilot:
            # Remove callback (find by reference won't work here, but cleanup)
            pass
        if websocket in autopilot_ws_clients:
            autopilot_ws_clients.remove(websocket)
        print("[AUTOPILOT] Dashboard disconnected from autopilot telemetry.")
    except Exception as e:
        if websocket in autopilot_ws_clients:
            autopilot_ws_clients.remove(websocket)
        print(f"[AUTOPILOT] Error: {e}")


# ── REST endpoints for autopilot (alternative to WebSocket) ──

@app.post("/autopilot/start")
async def start_autopilot(data: dict):
    """Start an autonomous mission via REST."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    
    grid_config = data.get("grid_config", {})
    esp32_ip = data.get("esp32_ip", "")
    
    if not grid_config or not esp32_ip:
        return JSONResponse({"error": "Missing grid_config or esp32_ip"}, status_code=400)
    
    asyncio.create_task(autopilot.start_mission(grid_config, esp32_ip))
    return {"status": "starting", "message": "Mission starting..."}


@app.post("/autopilot/stop")
async def stop_autopilot():
    """Stop the autonomous mission."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    await autopilot.stop_mission()
    return {"status": "stopped"}


@app.post("/autopilot/pause")
async def pause_autopilot():
    """Pause/resume the autonomous mission."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    await autopilot.pause_mission()
    return {"status": autopilot.state.value}


@app.get("/autopilot/status")
async def autopilot_status():
    """Get current autopilot telemetry."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    return autopilot.get_telemetry()


# ── Cleanup on shutdown ──

@app.on_event("shutdown")
async def shutdown_event():
    if autopilot:
        await autopilot.cleanup()
