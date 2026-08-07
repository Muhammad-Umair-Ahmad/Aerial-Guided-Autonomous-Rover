"""
AGRA — Autopilot Controller v2 (Turn-Then-Drive Navigation)
==================================================================
Closes the loop between YOLO rover detection and ESP32 motor commands.
Ensures the rover NEVER leaves the grid boundary.

Architecture:
  iPhone camera → Dashboard (WebRTC) → base64 frames → YOLO detection →
  RoverTracker (position) → AutopilotEngine (decisions) →
  MotorController → ESP32 HTTP → Motors

Control Philosophy — "Observe-Then-Act" with Geofencing and Turn-Then-Drive:
  1. Observe stable rover position via YOLO
  2. CHECK: Is rover inside grid? If not, correct first.
  3. CHECK: Is rover near grid edge? If so, bias steering inward.
  4. Compare current heading to desired angle
  5. If heading error > 30°: Turn Left/Right
  6. If heading error <= 30°: Drive Forward
  7. Stop and re-observe
  8. Verify the rover actually moved (Stuck detection)

Hardware Notes:
  - This is a differential-drive RC car (NOT holonomic).
  - Left/Right commands rotate the car in place.
  - Forward/Reverse commands are physically swapped in the car's wiring.
"""

import asyncio
import time
import math
import logging
from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────
logger = logging.getLogger("AGRA.Autopilot")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)


# ══════════════════════════════════════════════════════════════════════════
#  DATA TYPES
# ══════════════════════════════════════════════════════════════════════════

class AutopilotState(str, Enum):
    IDLE        = "IDLE"
    CALIBRATING = "CALIBRATING"
    NAVIGATING  = "NAVIGATING"
    CORRECTING  = "CORRECTING"
    GEOFENCE    = "GEOFENCE"
    PAUSED      = "PAUSED"
    COMPLETE    = "COMPLETE"
    ERROR       = "ERROR"


class Direction(str, Enum):
    FORWARD  = "forward"
    REVERSE  = "reverse"
    LEFT     = "left"
    RIGHT    = "right"
    STOP     = "stop"


@dataclass
class Position:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)

    def distance_to(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "Position") -> float:
        """Angle from self to other in degrees. 0°=right, 90°=down (image coords)."""
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))


@dataclass
class Waypoint:
    index: int
    grid_col: int
    grid_row: int
    pixel_x: float
    pixel_y: float
    visited: bool = False


# ══════════════════════════════════════════════════════════════════════════
#  ROVER TRACKER — Maintains rover state from YOLO detections
# ══════════════════════════════════════════════════════════════════════════

class RoverTracker:
    """
    Tracks rover position from YOLO detection results.
    Maintains a history for velocity estimation and stuck detection.
    """

    def __init__(self, history_size: int = 30, lost_threshold: int = 20):
        self.history: deque[Position] = deque(maxlen=history_size)
        self.lost_threshold = lost_threshold
        self.miss_count = 0
        self._last_position: Optional[Position] = None
        self._detected = False

    def update(self, detection_result: dict):
        detections = detection_result.get("detections", [])

        if detections and len(detections) > 0:
            rover = detections[0]
            box = rover["box"]
            cx = box["x"] + box["width"] / 2.0
            cy = box["y"] + box["height"] / 2.0

            pos = Position(x=cx, y=cy)
            self.history.append(pos)
            self._last_position = pos
            self._detected = True
            self.miss_count = 0
        else:
            self.miss_count += 1
            self._detected = False

    @property
    def position(self) -> Optional[Position]:
        return self._last_position

    @property
    def is_detected(self) -> bool:
        return self._detected and self.miss_count < 3

    @property
    def is_lost(self) -> bool:
        return self.miss_count >= self.lost_threshold

    def get_stable_position(self, n: int = 3) -> Optional[Position]:
        """Average the last N positions for a more stable reading."""
        if len(self.history) < n:
            return self._last_position

        recent = list(self.history)[-n:]
        avg_x = sum(p.x for p in recent) / len(recent)
        avg_y = sum(p.y for p in recent) / len(recent)
        return Position(x=avg_x, y=avg_y)

    def get_heading(self) -> Optional[float]:
        """
        Estimate heading (angle in degrees) from recent position history.
        0° = right, 90° = down, 180° = left, 270° = up (image coords).
        Returns None if insufficient history or movement is too small.
        """
        if len(self.history) < 3:
            return None

        recent = list(self.history)[-5:]
        if len(recent) < 2:
            return None

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y

        dist = math.sqrt(dx**2 + dy**2)
        if dist < 10.0:  # Ignore tiny movements (YOLO jitter)
            return None

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return angle

    def reset(self):
        self.history.clear()
        self._last_position = None
        self._detected = False
        self.miss_count = 0


# ══════════════════════════════════════════════════════════════════════════
#  GEOFENCE — Grid boundary enforcement
# ══════════════════════════════════════════════════════════════════════════

class Geofence:
    """
    Enforces that the rover stays within the grid boundary.
    Warning zone margin: 8% (steers inward)
    Waypoint clamping margin: 15% (keeps waypoints safely inside)
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int, margin_pct: float = 0.08):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.margin_pct = margin_pct

        w = self.x2 - self.x1
        h = self.y2 - self.y1
        self.margin_x = w * margin_pct
        self.margin_y = h * margin_pct

    @property
    def center(self) -> Position:
        return Position(
            x=(self.x1 + self.x2) / 2,
            y=(self.y1 + self.y2) / 2,
        )

    def is_inside(self, pos: Position) -> bool:
        return (self.x1 <= pos.x <= self.x2 and
                self.y1 <= pos.y <= self.y2)

    def is_in_warning_zone(self, pos: Position) -> bool:
        if not self.is_inside(pos):
            return True

        near_left   = (pos.x - self.x1) < self.margin_x
        near_right  = (self.x2 - pos.x) < self.margin_x
        near_top    = (pos.y - self.y1) < self.margin_y
        near_bottom = (self.y2 - pos.y) < self.margin_y

        return near_left or near_right or near_top or near_bottom

    def get_correction_vector(self, pos: Position) -> Tuple[float, float]:
        if self.is_inside(pos) and not self.is_in_warning_zone(pos):
            return (0.0, 0.0)

        safe_x = max(self.x1 + self.margin_x, min(pos.x, self.x2 - self.margin_x))
        safe_y = max(self.y1 + self.margin_y, min(pos.y, self.y2 - self.margin_y))

        dx = safe_x - pos.x
        dy = safe_y - pos.y

        if not self.is_inside(pos):
            dx *= 2.0
            dy *= 2.0

        return (dx, dy)

    def clamp_waypoint(self, px: float, py: float) -> Tuple[float, float]:
        """Clamp a waypoint to be within the safe zone using 15% margin."""
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        wp_margin_x = w * 0.15
        wp_margin_y = h * 0.15
        
        clamped_x = max(self.x1 + wp_margin_x, min(px, self.x2 - wp_margin_x))
        clamped_y = max(self.y1 + wp_margin_y, min(py, self.y2 - wp_margin_y))
        return (clamped_x, clamped_y)


# ══════════════════════════════════════════════════════════════════════════
#  WAYPOINT NAVIGATOR — Path following logic
# ══════════════════════════════════════════════════════════════════════════

class WaypointNavigator:
    def __init__(self):
        self.waypoints: list[Waypoint] = []
        self.current_index: int = 0

    def generate_path(self, grid_x1: int, grid_y1: int, grid_x2: int, grid_y2: int,
                      rows: int, cols: int, geofence: Optional[Geofence] = None,
                      precomputed_waypoints: Optional[list] = None) -> list[Waypoint]:
        self.waypoints = []
        self.current_index = 0

        if precomputed_waypoints and len(precomputed_waypoints) > 0:
            for i, wp_data in enumerate(precomputed_waypoints):
                px = wp_data.get("x", 0)
                py = wp_data.get("y", 0)
                if geofence:
                    px, py = geofence.clamp_waypoint(px, py)
                idx = wp_data.get("cellIdx", 0)
                r = idx // cols if cols else 0
                c = idx % cols if cols else 0
                
                wp = Waypoint(
                    index=i,
                    grid_col=c,
                    grid_row=r,
                    pixel_x=px,
                    pixel_y=py,
                )
                self.waypoints.append(wp)
            logger.info(f"Loaded {len(self.waypoints)} precomputed waypoints from JS")
            return self.waypoints

        cell_w = (grid_x2 - grid_x1) / cols
        cell_h = (grid_y2 - grid_y1) / rows

        idx = 0
        for r in range(rows):
            col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_range:
                px = grid_x1 + (c + 0.5) * cell_w
                py = grid_y1 + (r + 0.5) * cell_h

                if geofence:
                    px, py = geofence.clamp_waypoint(px, py)

                wp = Waypoint(
                    index=idx,
                    grid_col=c,
                    grid_row=r,
                    pixel_x=px,
                    pixel_y=py,
                )
                self.waypoints.append(wp)
                idx += 1

        logger.info(f"Generated {len(self.waypoints)} waypoints")
        return self.waypoints

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.waypoints)

    @property
    def progress_percent(self) -> float:
        if not self.waypoints:
            return 0.0
        return (self.current_index / len(self.waypoints)) * 100.0

    def advance(self) -> Optional[Waypoint]:
        if self.current_waypoint:
            self.current_waypoint.visited = True
            logger.info(f"Arrived at waypoint {self.current_index}")
        self.current_index += 1
        return self.current_waypoint

    def has_arrived(self, position: Position, threshold: float = 30.0) -> bool:
        wp = self.current_waypoint
        if wp is None:
            return False
        target = Position(x=wp.pixel_x, y=wp.pixel_y)
        dist = position.distance_to(target)
        return dist <= threshold

    def get_visited_cells(self) -> list[dict]:
        return [{"col": wp.grid_col, "row": wp.grid_row} for wp in self.waypoints if wp.visited]

    def reset(self):
        for wp in self.waypoints:
            wp.visited = False
        self.current_index = 0


# ══════════════════════════════════════════════════════════════════════════
#  DIFFERENTIAL STEERING CONTROLLER — Turn-Then-Drive
# ══════════════════════════════════════════════════════════════════════════

class DifferentialSteeringController:
    """
    Decides between turning in place or driving forward based on heading error.
    Used for differential-drive rovers.
    """

    def __init__(self, heading_tolerance: float = 30.0):
        self.heading_tolerance = heading_tolerance
        self.forward_pixel_angle = 0.0
        self.calibrated = False

    def decide_action(self, current_pos: Position, current_heading: float, target_pos: Position, 
                      drive_duration: int = 350, turn_duration: int = 250) -> Tuple[Direction, int]:
        desired_angle = current_pos.angle_to(target_pos) % 360
        heading_error = self.angle_diff(desired_angle, current_heading)

        if abs(heading_error) > self.heading_tolerance:
            if heading_error > 0:
                return Direction.RIGHT, turn_duration
            else:
                return Direction.LEFT, turn_duration
        else:
            return Direction.FORWARD, drive_duration

    @staticmethod
    def angle_diff(target: float, current: float) -> float:
        """Compute shortest angular difference between two angles [-180, 180]."""
        return (target - current + 180) % 360 - 180


# ══════════════════════════════════════════════════════════════════════════
#  MOTOR CONTROLLER — HTTP interface to ESP32
# ══════════════════════════════════════════════════════════════════════════

class MotorController:
    def __init__(self, esp32_ip: str = "", default_speed: int = 140):
        self.esp32_ip = esp32_ip
        self.speed = default_speed
        self.last_command: str = "stop"
        self.last_command_time: float = 0
        self.min_command_interval: float = 0.10
        self.invert_forward_reverse: bool = False
        self._connected = False
        self._client = None

    @property
    def base_url(self) -> str:
        return f"http://{self.esp32_ip}"

    @property
    def is_configured(self) -> bool:
        return bool(self.esp32_ip)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _get_client(self):
        if self._client is None:
            if not HTTPX_AVAILABLE:
                return None
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def send_command(self, direction: Direction, duration_ms: int = 0) -> bool:
        if not self.is_configured:
            logger.warning("ESP32 IP not configured — command not sent")
            return False

        now = time.time()
        if now - self.last_command_time < self.min_command_interval:
            return True

        physical_direction = direction.value
        if self.invert_forward_reverse:
            if direction == Direction.FORWARD:
                physical_direction = "reverse"
            elif direction == Direction.REVERSE:
                physical_direction = "forward"

        url = f"{self.base_url}/{physical_direction}"
        if duration_ms > 0:
            url += f"?ms={duration_ms}"

        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                success = resp.status_code == 200
            else:
                success = await self._fallback_request(url)

            if success:
                self.last_command = physical_direction
                self.last_command_time = now
                self._connected = True
                return True
            else:
                logger.error(f"ESP32 returned non-200 for {physical_direction}")
                return False

        except Exception as e:
            self._connected = False
            logger.error(f"ESP32 communication failed: {e}")
            return False

    async def _fallback_request(self, url: str) -> bool:
        import urllib.request
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=2))
            return True
        except Exception:
            return False

    async def stop(self) -> bool:
        return await self.send_command(Direction.STOP)

    async def set_speed(self, speed: int) -> bool:
        speed = max(0, min(255, speed))
        self.speed = speed

        if not self.is_configured:
            return False

        url = f"{self.base_url}/speed?v={speed}"
        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                return resp.status_code == 200
            else:
                return await self._fallback_request(url)
        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False

    async def check_heartbeat(self) -> bool:
        if not self.is_configured:
            return False

        url = f"{self.base_url}/status"
        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                self._connected = resp.status_code == 200
            else:
                self._connected = await self._fallback_request(url)
            return self._connected
        except Exception:
            self._connected = False
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ══════════════════════════════════════════════════════════════════════════
#  AUTOPILOT ENGINE — Main state machine
# ══════════════════════════════════════════════════════════════════════════

class AutopilotEngine:
    DEFAULT_ESP32_IP = "192.168.137.249"

    def __init__(self):
        self.tracker = RoverTracker()
        self.navigator = WaypointNavigator()
        self.motor = MotorController(default_speed=140)
        self.steering = DifferentialSteeringController(heading_tolerance=30.0)
        self.geofence: Optional[Geofence] = None

        self.state = AutopilotState.IDLE
        self.current_direction = Direction.STOP
        self.message = "Waiting for mission start"

        # ── Tuning parameters ──
        self.command_duration_ms: int = 350
        self.turn_duration_ms: int = 250
        self.observe_delay_s: float = 0.8
        self.arrival_threshold: float = 30.0
        self.stuck_threshold: float = 8.0
        self.max_stuck_retries: int = 8
        self.calibration_nudge_ms: int = 800

        # ── Internal state ──
        self._stuck_count: int = 0
        self._consecutive_stuck: int = 0
        self._consecutive_geofence: int = 0
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._telemetry_callbacks: list[Callable] = []
        self._log_entries: deque = deque(maxlen=100)

        # ── Grid config ──
        self._grid_rows: int = 0
        self._grid_cols: int = 0

    def add_telemetry_callback(self, callback: Callable):
        self._telemetry_callbacks.append(callback)

    def remove_telemetry_callback(self, callback: Callable):
        if callback in self._telemetry_callbacks:
            self._telemetry_callbacks.remove(callback)

    def _log(self, msg: str):
        entry = {"time": time.strftime("%H:%M:%S"), "msg": msg}
        self._log_entries.append(entry)
        logger.info(msg)

    async def _emit_telemetry(self):
        telemetry = self.get_telemetry()
        for cb in self._telemetry_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(telemetry)
                else:
                    cb(telemetry)
            except Exception as e:
                logger.error(f"Telemetry callback error: {e}")

    def get_telemetry(self) -> dict:
        pos = self.tracker.position
        wp = self.navigator.current_waypoint

        geofence_status = "N/A"
        if self.geofence and pos:
            if not self.geofence.is_inside(pos):
                geofence_status = "VIOLATION"
            elif self.geofence.is_in_warning_zone(pos):
                geofence_status = "WARNING"
            else:
                geofence_status = "SAFE"

        return {
            "state": self.state.value,
            "current_waypoint_index": self.navigator.current_index,
            "total_waypoints": len(self.navigator.waypoints),
            "rover_position": {"x": pos.x, "y": pos.y} if pos else None,
            "target_position": {"x": wp.pixel_x, "y": wp.pixel_y} if wp else None,
            "direction": self.current_direction.value,
            "speed": self.motor.speed,
            "visited_cells": self.navigator.get_visited_cells(),
            "progress_percent": round(self.navigator.progress_percent, 1),
            "esp32_connected": self.motor.is_connected,
            "rover_detected": self.tracker.is_detected,
            "stuck_count": self._stuck_count,
            "geofence_status": geofence_status,
            "calibrated": self.steering.calibrated,
            "heading": self.tracker.get_heading(),
            "message": self.message,
            "log": list(self._log_entries)[-15:],
            "grid_config": {
                "rows": self._grid_rows,
                "cols": self._grid_cols,
            }
        }

    def feed_detection(self, detection_result: dict):
        self.tracker.update(detection_result)

    async def start_mission(self, grid_config: dict, esp32_ip: str, waypoints: Optional[list] = None):
        if self.state not in (AutopilotState.IDLE, AutopilotState.COMPLETE, AutopilotState.ERROR):
            self._log("Cannot start: mission already in progress")
            return

        self.motor.esp32_ip = esp32_ip or self.DEFAULT_ESP32_IP
        self._log(f"ESP32 target: {self.motor.esp32_ip}")

        self._grid_rows = grid_config.get("rows", 2)
        self._grid_cols = grid_config.get("cols", 3)

        self.geofence = Geofence(
            x1=grid_config["x1"],
            y1=grid_config["y1"],
            x2=grid_config["x2"],
            y2=grid_config["y2"],
            margin_pct=0.08,
        )
        self._log(f"Geofence set: [{self.geofence.x1},{self.geofence.y1}]-[{self.geofence.x2},{self.geofence.y2}]")

        self.navigator.generate_path(
            grid_x1=grid_config["x1"],
            grid_y1=grid_config["y1"],
            grid_x2=grid_config["x2"],
            grid_y2=grid_config["y2"],
            rows=self._grid_rows,
            cols=self._grid_cols,
            geofence=self.geofence,
            precomputed_waypoints=waypoints
        )

        self._log(f"Mission configured: {self._grid_rows}x{self._grid_cols} grid, {len(self.navigator.waypoints)} waypoints")

        await self.motor.set_speed(140)
        self._log("PWM speed set to 140")

        self._stuck_count = 0
        self._consecutive_stuck = 0
        self._consecutive_geofence = 0
        self.tracker.reset()
        self.steering = DifferentialSteeringController(heading_tolerance=30.0)

        self._running = True
        self.state = AutopilotState.CALIBRATING
        self.message = "Calibrating — looking for rover..."
        self._log("Mission started — calibrating...")

        self._task = asyncio.create_task(self._control_loop())

    async def stop_mission(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self.motor.stop()
        self.state = AutopilotState.IDLE
        self.message = "Mission aborted"
        self._log("Mission aborted by user")
        await self._emit_telemetry()

    async def pause_mission(self):
        if self.state == AutopilotState.PAUSED:
            self.state = AutopilotState.NAVIGATING
            self.message = "Resumed"
            self._log("Mission resumed")
        elif self.state in (AutopilotState.NAVIGATING, AutopilotState.CORRECTING, AutopilotState.GEOFENCE):
            await self.motor.stop()
            self.state = AutopilotState.PAUSED
            self.message = "Paused"
            self._log("Mission paused")
        await self._emit_telemetry()

    async def _control_loop(self):
        try:
            await self._wait_for_rover()
            if self.state == AutopilotState.ERROR or not self._running:
                return

            await self._calibrate()
            if self.state == AutopilotState.ERROR or not self._running:
                return

            self.state = AutopilotState.NAVIGATING
            self.message = "Navigating — sweeping grid"

            while self._running and not self.navigator.is_complete:
                while self.state == AutopilotState.PAUSED:
                    await asyncio.sleep(0.3)
                    if not self._running:
                        return

                if not self._running:
                    return

                await self._navigate_step()
                await self._emit_telemetry()

            if self.navigator.is_complete:
                await self.motor.stop()
                self.state = AutopilotState.COMPLETE
                self.message = "Mission complete! All waypoints visited."
                self._log("MISSION COMPLETE — all waypoints visited!")
                await self._emit_telemetry()

        except asyncio.CancelledError:
            logger.info("Control loop cancelled")
        except Exception as e:
            self.state = AutopilotState.ERROR
            self.message = f"Error: {str(e)}"
            self._log(f"Error: {e}")
            await self.motor.stop()
            await self._emit_telemetry()
            logger.exception("Control loop error")

    async def _wait_for_rover(self):
        self._log("Looking for rover in camera feed...")
        detection_count = 0
        max_wait = 60

        start = time.time()
        while self._running and detection_count < 5:
            if time.time() - start > max_wait:
                self.state = AutopilotState.ERROR
                self.message = "Calibration timeout — rover not found"
                self._log("Calibration failed: rover not detected within 60s")
                return

            if self.tracker.is_detected:
                detection_count += 1
                self._log(f"Rover detected ({detection_count}/5 confirmations)")
            else:
                detection_count = max(0, detection_count - 1)

            await self._emit_telemetry()
            await asyncio.sleep(0.4)

        if not self._running:
            return

        pos = self.tracker.position
        if pos:
            self._log(f"Rover locked at ({pos.x:.0f}, {pos.y:.0f})")

        connected = await self.motor.check_heartbeat()
        if connected:
            self._log("ESP32 rover connected")
        else:
            self._log("ESP32 not responding — will retry on first command")

    async def _calibrate(self):
        """Auto-calibration: nudge the rover FORWARD to learn orientation."""
        self.state = AutopilotState.CALIBRATING
        self._log("Starting auto-calibration (FORWARD only)...")
        await self._emit_telemetry()

        fwd_angle = await self._nudge_and_measure(Direction.FORWARD)
        if fwd_angle is None:
            self._log("Forward calibration failed — will rely on dynamic heading")
            self.steering.calibrated = False
            self.steering.forward_pixel_angle = 0.0
        else:
            self.steering.forward_pixel_angle = fwd_angle
            self.steering.calibrated = True
            self._log(f"Calibration complete! Forward → {fwd_angle:.1f}°")

    async def _nudge_and_measure(self, direction: Direction) -> Optional[float]:
        await asyncio.sleep(0.3)
        pre_pos = self.tracker.get_stable_position(n=5)
        if pre_pos is None:
            self._log(f"Cannot calibrate {direction.value}: rover not detected")
            return None

        self._log(f"Nudging {direction.value}... (pre: {pre_pos.x:.0f},{pre_pos.y:.0f})")

        sent = await self.motor.send_command(direction, duration_ms=self.calibration_nudge_ms)
        if not sent:
            self._log(f"Failed to send {direction.value} nudge")
            return None

        await asyncio.sleep(self.calibration_nudge_ms / 1000.0 + 0.3)
        await self.motor.stop()
        await asyncio.sleep(self.observe_delay_s)

        post_pos = self.tracker.get_stable_position(n=5)
        if post_pos is None:
            self._log(f"Lost rover after {direction.value} nudge")
            return None

        dx = post_pos.x - pre_pos.x
        dy = post_pos.y - pre_pos.y
        dist = math.sqrt(dx**2 + dy**2)

        self._log(f"  {direction.value} nudge result: dx={dx:.1f}, dy={dy:.1f}, dist={dist:.1f}px")

        if dist < 5.0:
            self._log(f"  {direction.value} nudge too small — rover may not have moved")
            return None

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return angle

    async def _navigate_step(self):
        # 1. Get stable rover position (average of last 5)
        pos = self.tracker.get_stable_position(n=5)
        if pos is None:
            if self.tracker.is_lost:
                self.state = AutopilotState.ERROR
                self.message = "Rover lost — cannot detect"
                self._log("Rover lost for too long!")
                await self.motor.stop()
                return
            self.message = "Waiting for rover detection..."
            await asyncio.sleep(0.3)
            return

        wp = self.navigator.current_waypoint
        if wp is None:
            return

        # 2. GEOFENCE CHECK
        if self.geofence and not self.geofence.is_inside(pos):
            self._consecutive_geofence += 1
            self.state = AutopilotState.GEOFENCE
            self._log(f"⚠ GEOFENCE VIOLATION! Rover at ({pos.x:.0f},{pos.y:.0f}) — correcting...")

            if self._consecutive_geofence > 30:
                self.state = AutopilotState.ERROR
                self.message = "Cannot get rover back in bounds"
                self._log("Geofence correction failed after 30 attempts")
                await self.motor.stop()
                return

            await self._steer_toward(pos, self.geofence.center)
            return

        if self.geofence and self.geofence.is_inside(pos):
            self._consecutive_geofence = 0

        # 3. Check Arrival
        if self.navigator.has_arrived(pos, threshold=self.arrival_threshold):
            self.navigator.advance()
            self._stuck_count = 0
            self._consecutive_stuck = 0
            self.state = AutopilotState.NAVIGATING
            self.message = f"Arrived at waypoint {wp.index} — advancing"

            if self.navigator.is_complete:
                return

            next_wp = self.navigator.current_waypoint
            if next_wp:
                self._log(f"→ Moving to waypoint {next_wp.index} [col={next_wp.grid_col}, row={next_wp.grid_row}]")
            return

        # 4. Target Generation & Edge Bias
        target = Position(x=wp.pixel_x, y=wp.pixel_y)
        dist = pos.distance_to(target)

        if self.geofence and self.geofence.is_in_warning_zone(pos):
            corr_dx, corr_dy = self.geofence.get_correction_vector(pos)
            if abs(corr_dx) > 1 or abs(corr_dy) > 1:
                wp_dx = target.x - pos.x
                wp_dy = target.y - pos.y
                blended_dx = 0.6 * wp_dx + 0.4 * corr_dx
                blended_dy = 0.6 * wp_dy + 0.4 * corr_dy
                target = Position(x=pos.x + blended_dx, y=pos.y + blended_dy)
                self.message = f"⚠ Near edge — biasing toward center | WP{wp.index} (dist: {dist:.0f}px)"
            else:
                self.message = f"Moving → WP{wp.index} (dist: {dist:.0f}px)"
        else:
            self.state = AutopilotState.NAVIGATING
            self.message = f"Moving → WP{wp.index} (dist: {dist:.0f}px)"

        # 5. Determine current heading
        current_heading = self.tracker.get_heading()
        if current_heading is None:
            current_heading = self.steering.forward_pixel_angle if self.steering.calibrated else 0.0

        # 6. Decide Action
        direction, dur = self.steering.decide_action(
            current_pos=pos,
            current_heading=current_heading,
            target_pos=target,
            drive_duration=self.command_duration_ms,
            turn_duration=self.turn_duration_ms
        )
        self.current_direction = direction

        if direction == Direction.STOP:
            await asyncio.sleep(0.2)
            return

        # 7. Log intent and send command
        desired_angle = pos.angle_to(target) % 360
        heading_error = self.steering.angle_diff(desired_angle, current_heading)
        if direction in (Direction.LEFT, Direction.RIGHT):
            self._log(f"TURN {direction.value.upper()} (heading error: {heading_error:.1f}°)")
        else:
            self._log(f"DRIVE FORWARD (heading aligned, dist: {dist:.0f}px)")

        pre_pos = Position(x=pos.x, y=pos.y, timestamp=pos.timestamp)
        sent = await self.motor.send_command(direction, duration_ms=dur)
        if not sent:
            self._log("Failed to send command to ESP32")
            await asyncio.sleep(1.0)
            return

        await asyncio.sleep(dur / 1000.0 + 0.05)
        await self.motor.stop()
        await asyncio.sleep(self.observe_delay_s)

        # 8. Stuck Detection
        new_pos = self.tracker.get_stable_position(n=3)
        if new_pos and pre_pos:
            moved = pre_pos.distance_to(new_pos)
            if moved < self.stuck_threshold:
                self._consecutive_stuck += 1
                if self._consecutive_stuck >= 3:
                    self._stuck_count += 1
                    self._consecutive_stuck = 0
                    self._log(f"Rover may be stuck ({self._stuck_count}/{self.max_stuck_retries}) — moved only {moved:.1f}px for 3 steps")
                    self.state = AutopilotState.CORRECTING

                    if self._stuck_count >= self.max_stuck_retries:
                        self.state = AutopilotState.ERROR
                        self.message = "Rover stuck — manual intervention needed"
                        self._log("Rover stuck! Max retries exceeded.")
                        await self.motor.stop()
                        return

                    await self._unstick()
            else:
                self._consecutive_stuck = 0
                self._stuck_count = max(0, self._stuck_count - 1)

    async def _steer_toward(self, current_pos: Position, target: Position):
        current_heading = self.tracker.get_heading()
        if current_heading is None:
            current_heading = self.steering.forward_pixel_angle if self.steering.calibrated else 0.0

        direction, dur = self.steering.decide_action(
            current_pos=current_pos,
            current_heading=current_heading,
            target_pos=target,
            drive_duration=250,
            turn_duration=250
        )
        self.current_direction = direction

        desired_angle = current_pos.angle_to(target) % 360
        error = self.steering.angle_diff(desired_angle, current_heading)
        self.message = f"🚨 GEOFENCE — steering {direction.value} back to grid"
        self._log(f"Geofence correction: {direction.value} (error: {error:.1f}°)")

        sent = await self.motor.send_command(direction, duration_ms=dur)
        if sent:
            await asyncio.sleep(dur / 1000.0 + 0.05)
            await self.motor.stop()
            await asyncio.sleep(self.observe_delay_s)

    async def _unstick(self):
        """Stuck recovery: reverse for 500ms."""
        self.message = f"Correcting — unstick attempt {self._stuck_count}"
        self._log("Unstick: reversing for 500ms")
        
        await self.motor.send_command(Direction.REVERSE, duration_ms=500)
        await asyncio.sleep(0.55)
        await self.motor.stop()
        await asyncio.sleep(self.observe_delay_s)

    def set_config(self, config: dict):
        if "speed" in config:
            self.motor.speed = max(0, min(255, config["speed"]))
        if "arrival_threshold" in config:
            self.arrival_threshold = max(10, min(100, config["arrival_threshold"]))
        if "command_duration_ms" in config:
            self.command_duration_ms = max(100, min(2000, config["command_duration_ms"]))
        if "observe_delay_s" in config:
            self.observe_delay_s = max(0.2, min(3.0, config["observe_delay_s"]))
        if "esp32_ip" in config and config["esp32_ip"]:
            self.motor.esp32_ip = config["esp32_ip"]

        logger.info(f"Config updated: speed={self.motor.speed}, "
                    f"arrival={self.arrival_threshold}px, "
                    f"duration={self.command_duration_ms}ms")

    async def cleanup(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self.motor.stop()
        await self.motor.close()
