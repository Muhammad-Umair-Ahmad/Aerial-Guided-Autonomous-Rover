"""
AGRA — Geofence & Sequence Monitor (Phase 2 & 3)
==================================================================
Replaces the old Autopilot engine.
Now simply tracks rover position via YOLO and enforces the Geofence.
Sends Sequence start/pause/stop commands to the ESP32.
"""

import time
import httpx
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable
import json as json_lib
from pathlib import Path

logger = logging.getLogger("AGRA.Geofence")
logger.setLevel(logging.INFO)

@dataclass
class Position:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)

class RoverTracker:
    def __init__(self):
        self._last_position: Optional[Position] = None
        self._detected = False
        self.miss_count = 0

    def update(self, detection_result: dict):
        detections = detection_result.get("detections", [])
        if detections:
            box = detections[0]["box"]
            self._last_position = Position(x=box["x"] + box["width"] / 2.0, y=box["y"] + box["height"] / 2.0)
            self._detected = True
            self.miss_count = 0
        else:
            self.miss_count += 1
            self._detected = False

    @property
    def position(self) -> Optional[Position]:
        return self._last_position

class Geofence:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    def is_inside(self, pos: Position) -> bool:
        return (self.x1 <= pos.x <= self.x2 and
                self.y1 <= pos.y <= self.y2)

class AutopilotEngine:
    def __init__(self):
        self.tracker = RoverTracker()
        self.geofence: Optional[Geofence] = None
        self.esp32_ip = "192.168.137.10"
        self.esp32_connected = False
        self.out_of_bounds = False
        self.client = httpx.AsyncClient(timeout=2.0)
        self._telemetry_callbacks: list[Callable] = []
        self._sequence_file = Path(__file__).parent / "sequence_data.json"
        self._current_sequence: list = []
        self._sequence_version: int = 0
        self._esp32_version: int = 0
        self._load_sequence_from_file()

    def start(self):
        asyncio.create_task(self._health_check_loop())

    @property
    def state_value(self):
        return "GEOFENCE_ALERT" if self.out_of_bounds else "ACTIVE"

    def set_config(self, config: dict):
        pass # Ignored for Phase 2

    def add_telemetry_callback(self, cb: Callable):
        self._telemetry_callbacks.append(cb)

    def _load_sequence_from_file(self):
        """Load saved sequence from disk on startup."""
        try:
            if self._sequence_file.exists():
                data = json_lib.loads(self._sequence_file.read_text())
                self._current_sequence = data.get("sequence", [])
                self._sequence_version = data.get("version", 0)
                logger.info(f"Loaded saved sequence: {len(self._current_sequence)} steps, v{self._sequence_version}")
        except Exception as e:
            logger.error(f"Failed to load sequence file: {e}")

    def _save_sequence_to_file(self):
        """Persist sequence to disk."""
        try:
            data = {
                "sequence": self._current_sequence,
                "version": self._sequence_version,
                "last_saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._sequence_file.write_text(json_lib.dumps(data, indent=2))
            logger.info(f"Saved sequence to file: {len(self._current_sequence)} steps, v{self._sequence_version}")
        except Exception as e:
            logger.error(f"Failed to save sequence file: {e}")

    def get_sequence_status(self) -> dict:
        """Return current sequence status for API consumers."""
        return {
            "sequence": self._current_sequence,
            "version": self._sequence_version,
            "esp32_version": self._esp32_version,
            "step_count": len(self._current_sequence),
            "synced": self._sequence_version == self._esp32_version and self._sequence_version > 0,
        }

    async def start_mission(self, grid_config: dict, esp32_ip: str, waypoints=None):
        if esp32_ip:
            self.esp32_ip = esp32_ip
        if grid_config:
            self.set_grid(grid_config)
        else:
            self.geofence = None
        self.out_of_bounds = False
        await self.send_command("/sequence/start")

    async def stop_mission(self):
        await self.send_command("/sequence/stop")

    async def pause_mission(self):
        await self.send_command("/sequence/pause")
        
    async def resume_mission(self):
        await self.send_command("/sequence/resume")

    async def upload_sequence(self, sequence_data: list, version: int = 0) -> dict:
        """Upload sequence to ESP32 with verification. Returns result dict."""
        if not self.esp32_ip:
            return {"ok": False, "error": "No ESP32 IP configured"}

        # Save locally first
        self._current_sequence = sequence_data
        if version > 0:
            self._sequence_version = version
        else:
            self._sequence_version += 1
        self._save_sequence_to_file()

        # Format for ESP32
        str_parts = []
        for step in sequence_data:
            pwm = step.get('pwm', 150)
            direction = str(step.get('direction', 'FORWARD')).upper()
            duration = step.get('duration', 1000)
            str_parts.append(f"{pwm},{direction},{duration}")
        payload = ";".join(str_parts)
        url = f"http://{self.esp32_ip}/sequence/upload"

        try:
            response = await self.client.post(url, content=payload, headers={"Content-Type": "text/plain"})
            logger.info(f"ESP32 upload response: {response.text}")

            # Parse ESP32 JSON response
            try:
                esp_result = response.json()
            except Exception:
                esp_result = {"ok": response.status_code == 200, "raw": response.text}

            if esp_result.get("ok"):
                self._esp32_version = esp_result.get("version", self._sequence_version)
                # Verify by querying /sequence/status
                verify = await self._verify_esp32_sequence()
                return {
                    "ok": True,
                    "steps_sent": len(sequence_data),
                    "esp32_steps": esp_result.get("steps", 0),
                    "version": self._sequence_version,
                    "esp32_version": esp_result.get("version", 0),
                    "esp32_checksum": esp_result.get("checksum", ""),
                    "verified": verify.get("verified", False),
                }
            else:
                return {"ok": False, "error": esp_result.get("error", "ESP32 rejected sequence")}

        except Exception as e:
            logger.error(f"Failed to upload sequence: {e}")
            return {"ok": False, "error": str(e)}

    async def _verify_esp32_sequence(self) -> dict:
        """Query ESP32 /sequence/status to verify what's actually stored."""
        if not self.esp32_ip:
            return {"verified": False}
        try:
            url = f"http://{self.esp32_ip}/sequence/status"
            response = await self.client.get(url)
            data = response.json()
            stored_count = data.get("step_count", 0)
            stored_version = data.get("version", 0)
            verified = stored_count == len(self._current_sequence)
            return {
                "verified": verified,
                "esp32_step_count": stored_count,
                "esp32_version": stored_version,
                "esp32_checksum": data.get("checksum", ""),
            }
        except Exception as e:
            logger.error(f"Failed to verify ESP32 sequence: {e}")
            return {"verified": False, "error": str(e)}

    async def query_esp32_status(self) -> dict:
        """Public method to query ESP32 sequence status."""
        verify = await self._verify_esp32_sequence()
        return {
            **self.get_sequence_status(),
            **verify,
        }

    def set_grid(self, grid_config: dict):
        bbox = grid_config.get("bbox", [])
        if len(bbox) == 4:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            self.geofence = Geofence(x1, y1, x2, y2)

    def feed_detection(self, results):
        self.tracker.update(results)
        
        if self.tracker.position and self.geofence:
            was_oob = self.out_of_bounds
            self.out_of_bounds = not self.geofence.is_inside(self.tracker.position)
            
            # If just went out of bounds, STOP sequence immediately
            if self.out_of_bounds and not was_oob:
                logger.warning("GEOFENCE VIOLATION! Sending STOP to ESP32.")
                asyncio.create_task(self.send_command("/sequence/stop"))
                asyncio.create_task(self.send_command("/stop"))

        # Send telemetry to UI
        asyncio.create_task(self._push_telemetry())
                
    async def send_command(self, endpoint: str):
        if not self.esp32_ip: return
        url = f"http://{self.esp32_ip}{endpoint}"
        try:
            await self.client.get(url)
            logger.info(f"Sent {endpoint} to {self.esp32_ip}")
        except Exception as e:
            logger.error(f"Failed to send {endpoint}: {e}")

    async def _push_telemetry(self):
        pos = self.tracker.position
        data = {
            "type": "telemetry",
            "state": self.state_value,
            "position": {"x": pos.x, "y": pos.y} if pos else None,
            "out_of_bounds": self.out_of_bounds,
            "geofence_active": self.geofence is not None,
            "esp32_connected": self.esp32_connected
        }
        for cb in self._telemetry_callbacks:
            try:
                await cb(data)
            except Exception:
                pass

    async def _health_check_loop(self):
        while True:
            if self.esp32_ip:
                url = f"http://{self.esp32_ip}/status"
                try:
                    await self.client.get(url, timeout=1.0)
                    self.esp32_connected = True
                except Exception:
                    self.esp32_connected = False
            await asyncio.sleep(2.0)
