import asyncio
import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mars_rover.config import Config
from mars_rover.navigation.world_model import WorldModel
from mars_rover.perception.perception_manager import PerceptionManager
from mars_rover.navigation.state_machine import NavigationStateMachine
from mars_rover.communication.esp32 import ESP32Controller
from mars_rover.safety.failsafe import SafetySystem

app = FastAPI(title="NASA Mars Rover Autonomy - Mission Control")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Static Files for Modular Dashboard ──
dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.exists(dashboard_dir):
    app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")

# ── Global State & Modules ──
world_model = WorldModel()
perception = PerceptionManager()
motor_controller = ESP32Controller(esp32_ip=Config.ESP32_IP)
state_machine = NavigationStateMachine(world_model, motor_controller)
safety = SafetySystem(world_model, motor_controller)

connections: list[WebSocket] = []
autopilot_ws_clients: list[WebSocket] = []
latest_frame_b64: str = ""

# ── Async Orchestrator Loop ──
async def autonomy_loop():
    """Main execution loop decoupling perception from navigation."""
    while True:
        try:
            # Run safety checks first
            await safety.check_all()
            
            # Step the state machine
            if world_model.mission_active:
                await state_machine.step()
                
        except Exception as e:
            print(f"[ERROR] Autonomy Loop: {e}")
        finally:
            await asyncio.sleep(0.05)  # 20 Hz loop

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(autonomy_loop())
    print("[SYSTEM] Mars Rover Autonomy Stack Initialized")

# ── Routes ──
@app.get("/")
async def dashboard():
    filepath = os.path.join(os.path.dirname(__file__), "dashboard", "dashboard.html")
    if not os.path.exists(filepath):
        return HTMLResponse("<h1>Dashboard not found</h1><p>Building...</p>")
    with open(filepath, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/broadcaster")
async def broadcaster():
    # Use the original broadcaster for now
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "broadcaster.html")
    with open(filepath, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "frame_available": bool(latest_frame_b64),
        "connections": len(connections),
        "state": world_model.state.value
    }

# (WebSocket logic to follow...)
# --------------------------------------------------------------------------
#  WebSocket Endpoints
# --------------------------------------------------------------------------
import json

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            for conn in connections:
                if conn != websocket:
                    try:
                        await conn.send_text(data)
                    except:
                        pass
    except:
        if websocket in connections:
            connections.remove(websocket)

@app.websocket("/ws/cv")
async def cv_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
                if "image" in data:
                    global latest_frame_b64
                    latest_frame_b64 = data["image"]
                    
                    # 1. Perception Layer (Process Frame -> Pose)
                    pose_data = perception.process_frame(latest_frame_b64)
                    
                    # 2. Update World Model
                    if pose_data.get("rover_pose"):
                        world_model.rover_pose = pose_data["rover_pose"]
                    world_model.fps = pose_data.get("fps", 0)
                    
                    # 3. Send response back to dashboard
                    pose_data["autopilot_state"] = world_model.state.value
                    await websocket.send_json(pose_data)
            except Exception as e:
                print(f"[CV WS Error] {e}")
    except:
        pass

@app.websocket("/ws/autopilot")
async def autopilot_telemetry(websocket: WebSocket):
    await websocket.accept()
    autopilot_ws_clients.append(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "start":
                world_model.mission_active = True
                print("[MISSION] Started")
            elif msg_type == "stop":
                world_model.mission_active = False
                await motor_controller.stop()
                print("[MISSION] Stopped")
            elif msg_type == "manual_override":
                world_model.mission_active = False
                cmd = data.get("command")
                if cmd == "forward": await motor_controller.forward()
                elif cmd == "reverse": await motor_controller.reverse()
                elif cmd == "left": await motor_controller.left()
                elif cmd == "right": await motor_controller.right()
                elif cmd == "stop": await motor_controller.stop()
    except:
        if websocket in autopilot_ws_clients:
            autopilot_ws_clients.remove(websocket)

async def broadcast_telemetry():
    while True:
        if autopilot_ws_clients:
            tel = {
                "type": "telemetry",
                "state": world_model.state.value,
                "battery": world_model.battery_voltage,
                "target_heading": world_model.target_heading,
                "heading_error": world_model.heading_error,
                "confidence": world_model.rover_pose.confidence if world_model.rover_pose else 0,
            }
            dead = []
            for ws in autopilot_ws_clients:
                try:
                    await ws.send_json(tel)
                except:
                    dead.append(ws)
            for ws in dead:
                autopilot_ws_clients.remove(ws)
        await asyncio.sleep(0.3)

@app.on_event("startup")
async def startup_telemetry():
    asyncio.create_task(broadcast_telemetry())
