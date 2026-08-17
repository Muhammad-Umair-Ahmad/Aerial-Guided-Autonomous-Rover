from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import json
import base64
import cv2
import numpy as np
from terrain_analyzer import UnsupervisedTerrainAnalyzer

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

@app.on_event("startup")
async def startup_event():
    if autopilot:
        autopilot.start()

# ── CORS (allow dashboard to talk to ESP32 through proxy if needed) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static files ──
os.makedirs("static/heatmaps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Track all connected WebSocket clients
connections: list[WebSocket] = []

# Store the latest camera frame (base64) for snapshot endpoint
latest_frame_b64: str = ""
latest_image_b64: str = None

# Track autopilot telemetry WebSocket clients
autopilot_ws_clients: list[WebSocket] = []

# Initialize terrain analyzer
terrain_analyzer = UnsupervisedTerrainAnalyzer(patch_size=32, field_width_m=10.0, field_height_m=8.0)

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

    except (WebSocketDisconnect, ConnectionResetError):
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
                    global latest_frame_b64, latest_image_b64
                    latest_frame_b64 = image_b64
                    latest_image_b64 = image_b64
                
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
                        results["autopilot_state"] = autopilot.state_value

                    await websocket.send_json(results)
                else:
                    await websocket.send_json({"error": "AI disabled or no image provided", "detections": []})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON", "detections": []})
                
    except (WebSocketDisconnect, ConnectionResetError):
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
                
                # Update global latest image for analysis
                if data.get("image"):
                    global latest_image_b64
                    latest_image_b64 = data.get("image")

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

                elif msg_type == "upload_sequence" and autopilot:
                    sequence = data.get("sequence", [])
                    version = data.get("version", 0)
                    result = await autopilot.upload_sequence(sequence, version)
                    await websocket.send_json({"type": "upload_result", **result})

                elif msg_type == "run_analysis":
                    if not latest_image_b64:
                        await websocket.send_json({"type": "analysis_error", "msg": "No camera feed available."})
                        continue
                    
                    try:
                        # Decode base64 to numpy array
                        img_data = base64.b64decode(latest_image_b64.split(",")[1] if "," in latest_image_b64 else latest_image_b64)
                        nparr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None or frame.size == 0:
                            await websocket.send_json({"type": "analysis_error", "msg": "Failed to decode camera frame."})
                            continue
                            
                        # Crop to boundary if provided
                        boundary = data.get("boundary")
                        if boundary:
                            x1 = min(boundary.get("x1", 0), boundary.get("x2", frame.shape[1]))
                            x2 = max(boundary.get("x1", 0), boundary.get("x2", frame.shape[1]))
                            y1 = min(boundary.get("y1", 0), boundary.get("y2", frame.shape[0]))
                            y2 = max(boundary.get("y1", 0), boundary.get("y2", frame.shape[0]))
                            
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                frame = frame[y1:y2, x1:x2].copy()
                                
                        # Run the analysis (synchronously, but it's fast enough or we could use run_in_executor)
                        enhance = data.get("enhance", True)
                        scan, images = terrain_analyzer.analyze(frame, enhance_input=enhance)
                        
                        # Save the images
                        urls = {}
                        for key, img in images.items():
                            filename = f"heatmaps/{key}.jpg"
                            cv2.imwrite(f"static/{filename}", img)
                            urls[key] = f"/static/{filename}"
                        
                        # Count anomaly patches
                        anomaly_count = sum(1 for p in scan.patches if p.anomaly_score > 0.5)
                            
                        await websocket.send_json({
                            "type": "analysis_complete",
                            "heatmaps": urls,
                            "stats": {
                                "life_percent": scan.field_life_percent,
                                "liquid_percent": scan.field_liquid_percent,
                                "processing_time": scan.processing_time_s,
                                "grid_shape": list(scan.grid_shape),
                                "patch_size": scan.patch_size,
                                "anomaly_count": anomaly_count,
                                "total_patches": len(scan.patches),
                                "surface_classes": scan.surface_class_names
                            }
                        })
                    except Exception as e:
                        print(f"Analysis error: {e}")
                        await websocket.send_json({"type": "analysis_error", "msg": str(e)})

                elif msg_type == "update_config" and autopilot:
                    config = data.get("config", {})
                    autopilot.set_config(config)
                    await websocket.send_json({"type": "ack", "msg": "Config updated."})

                elif msg_type == "update_ip" and autopilot:
                    ip = data.get("ip", "")
                    if ip:
                        autopilot.esp32_ip = ip
                    await websocket.send_json({"type": "ack", "msg": "IP updated."})

                elif msg_type == "resume_mission" and autopilot:
                    asyncio.create_task(autopilot.resume_mission())
                    await websocket.send_json({"type": "ack", "msg": "Mission resumed."})

                elif msg_type == "get_sequence_status" and autopilot:
                    status = await autopilot.query_esp32_status()
                    await websocket.send_json({"type": "sequence_status", **status})

                elif msg_type == "save_sequence" and autopilot:
                    sequence = data.get("sequence", [])
                    version = data.get("version", 0)
                    autopilot._current_sequence = sequence
                    if version > 0:
                        autopilot._sequence_version = version
                    else:
                        autopilot._sequence_version += 1
                    autopilot._save_sequence_to_file()
                    await websocket.send_json({"type": "ack", "msg": f"Sequence saved (v{autopilot._sequence_version})"})

                elif msg_type == "set_recording":
                    enabled = data.get("enabled", False)
                    data_logger.set_recording(enabled)
                    await websocket.send_json({"type": "ack", "msg": f"Recording {'started' if enabled else 'stopped'}"})

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except (WebSocketDisconnect, ConnectionResetError):
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
    
    if not esp32_ip:
        return JSONResponse({"error": "Missing esp32_ip"}, status_code=400)
    
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


@app.get("/sequence")
async def get_sequence():
    """Get the currently saved sequence."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    return autopilot.get_sequence_status()


@app.get("/sequence/status")
async def sequence_status():
    """Get sequence sync status (local vs ESP32)."""
    if not autopilot:
        return JSONResponse({"error": "Autopilot not available"}, status_code=503)
    return await autopilot.query_esp32_status()


# ── Cleanup on shutdown ──

@app.on_event("shutdown")
async def shutdown_event():
    if autopilot:
        await autopilot.cleanup()


