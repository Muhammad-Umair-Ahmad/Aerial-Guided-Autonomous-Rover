"""
AGRA — Autopilot Controller v2 (Grid-Constrained Lawnmower Sweep)
==================================================================
Closes the loop between YOLO rover detection and ESP32 motor commands.
Ensures the rover NEVER leaves the grid boundary.

Architecture:
  iPhone camera → Dashboard (WebRTC) → base64 frames → YOLO detection →
  RoverTracker (position) → AutopilotEngine (decisions) →
  MotorController → ESP32 HTTP → Motors

Control Philosophy — "Observe-Then-Act" with Geofencing:
  1. Observe rover position via YOLO
  2. CHECK: Is rover inside grid? If not, correct first.
  3. CHECK: Is rover near grid edge? If so, bias steering inward.
  4. Compute direction to next waypoint
  5. Send motor command for a short burst
  6. Stop and re-observe
  7. Verify the rover actually moved
  8. Repeat

Auto-Calibration:
  On mission start, the rover does a "nudge test" to learn which
  motor command maps to which pixel-space direction. This means the
  system works regardless of how the iPhone camera is oriented.
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
    GEOFENCE    = "GEOFENCE"       # actively correcting out-of-bounds
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


@dataclass
class CalibrationResult:
    """Maps each ESP32 motor direction to a pixel-space angle."""
    forward_angle: float = 0.0    # angle in degrees that "forward" moves in pixel space
    left_angle: float = 0.0
    right_angle: float = 0.0
    reverse_angle: float = 0.0
    calibrated: bool = False


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
        self.lost_threshold = lost_threshold   # frames with no detection → lost
        self.miss_count = 0
        self._last_position: Optional[Position] = None
        self._detected = False

    def update(self, detection_result: dict):
        """
        Feed a CV detection result into the tracker.
        Expected format: {"detections": [{"box": {"x":, "y":, "width":, "height:"}, ...}]}
        """
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
        """Current (latest) rover position, or None if never detected."""
        return self._last_position

    @property
    def is_detected(self) -> bool:
        """Is the rover currently being detected?"""
        return self._detected and self.miss_count < 3

    @property
    def is_lost(self) -> bool:
        """Has the rover been undetected for too long?"""
        return self.miss_count >= self.lost_threshold

    def has_moved(self, min_distance: float = 8.0) -> bool:
        """
        Check if the rover has moved significantly between the last two positions.
        Used to detect if the rover is stuck.
        """
        if len(self.history) < 2:
            return True  # can't tell yet, assume it moved
        recent = self.history[-1]
        previous = self.history[-2]
        return recent.distance_to(previous) >= min_distance

    def get_stable_position(self, n: int = 3) -> Optional[Position]:
        """
        Average the last N positions for a more stable reading.
        Helps reduce jitter from YOLO detection noise.
        """
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

        # Average over last few positions for stability
        recent = list(self.history)[-5:]
        if len(recent) < 2:
            return None

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y

        if abs(dx) < 2 and abs(dy) < 2:
            return None  # too small to determine heading

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return angle

    def get_velocity(self) -> float:
        """Pixels per second estimate."""
        if len(self.history) < 2:
            return 0.0
        p1 = self.history[-2]
        p2 = self.history[-1]
        dt = p2.timestamp - p1.timestamp
        if dt <= 0:
            return 0.0
        return p1.distance_to(p2) / dt

    def reset(self):
        """Clear all history."""
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
    
    The grid is defined by 4 corners in pixel space (x1,y1)-(x2,y2).
    The geofence has two zones:
      - WARNING zone: within `margin_pct` of the edge → bias steering inward
      - VIOLATION zone: outside the grid → immediately correct
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int, margin_pct: float = 0.12):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.margin_pct = margin_pct

        # Compute margin in pixels
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
        """Check if position is within grid bounds."""
        return (self.x1 <= pos.x <= self.x2 and
                self.y1 <= pos.y <= self.y2)

    def is_in_warning_zone(self, pos: Position) -> bool:
        """Check if position is near a grid edge (within margin)."""
        if not self.is_inside(pos):
            return True  # outside is definitely a warning

        near_left   = (pos.x - self.x1) < self.margin_x
        near_right  = (self.x2 - pos.x) < self.margin_x
        near_top    = (pos.y - self.y1) < self.margin_y
        near_bottom = (self.y2 - pos.y) < self.margin_y

        return near_left or near_right or near_top or near_bottom

    def get_correction_vector(self, pos: Position) -> Tuple[float, float]:
        """
        Returns a (dx, dy) vector pointing from the rover's position TOWARD
        the grid center. Magnitude increases the further outside the rover is.
        
        If inside grid, returns (0, 0).
        If in warning zone, returns a gentle bias vector.
        If outside grid, returns a strong correction vector.
        """
        if self.is_inside(pos) and not self.is_in_warning_zone(pos):
            return (0.0, 0.0)

        # Compute target: the nearest safe interior point (or grid center if far out)
        safe_x = max(self.x1 + self.margin_x, min(pos.x, self.x2 - self.margin_x))
        safe_y = max(self.y1 + self.margin_y, min(pos.y, self.y2 - self.margin_y))

        dx = safe_x - pos.x
        dy = safe_y - pos.y

        # If outside bounds, amplify the correction
        if not self.is_inside(pos):
            dx *= 2.0
            dy *= 2.0

        return (dx, dy)

    def clamp_waypoint(self, px: float, py: float) -> Tuple[float, float]:
        """Clamp a waypoint to be within the safe zone (inside margins)."""
        clamped_x = max(self.x1 + self.margin_x, min(px, self.x2 - self.margin_x))
        clamped_y = max(self.y1 + self.margin_y, min(py, self.y2 - self.margin_y))
        return (clamped_x, clamped_y)


# ══════════════════════════════════════════════════════════════════════════
#  WAYPOINT NAVIGATOR — Path following logic
# ══════════════════════════════════════════════════════════════════════════

class WaypointNavigator:
    """
    Generates a boustrophedon (serpentine/lawnmower) path over a grid
    region and computes movement directions to follow it.
    
    Grid is configurable — works for any rows × cols.
    """

    def __init__(self):
        self.waypoints: list[Waypoint] = []
        self.current_index: int = 0

    def generate_path(self, grid_x1: int, grid_y1: int, grid_x2: int, grid_y2: int,
                      rows: int, cols: int, geofence: Optional[Geofence] = None,
                      precomputed_waypoints: Optional[list] = None) -> list[Waypoint]:
        """
        Generate boustrophedon waypoints in pixel space, OR use precomputed ones from JS.
        """
        self.waypoints = []
        self.current_index = 0

        # If JS gave us the exact path it drew, use it directly!
        if precomputed_waypoints and len(precomputed_waypoints) > 0:
            for i, wp_data in enumerate(precomputed_waypoints):
                px = wp_data.get("x", 0)
                py = wp_data.get("y", 0)
                if geofence:
                    px, py = geofence.clamp_waypoint(px, py)
                # Calculate grid col/row backwards from cellIdx if needed, or just dummy values
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

                # Clamp waypoints to be within geofence safe zone
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

        logger.info(f"Generated {len(self.waypoints)} waypoints "
                    f"({rows}x{cols} grid in region [{grid_x1},{grid_y1}]-[{grid_x2},{grid_y2}])")
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
        """Mark current waypoint as visited and move to the next."""
        if self.current_waypoint:
            self.current_waypoint.visited = True
            logger.info(f"Arrived at waypoint {self.current_index} "
                        f"(grid [{self.current_waypoint.grid_col},{self.current_waypoint.grid_row}])")
        self.current_index += 1
        return self.current_waypoint

    def has_arrived(self, position: Position, threshold: float = 30.0) -> bool:
        """Check if the rover position is close enough to the current waypoint."""
        wp = self.current_waypoint
        if wp is None:
            return False
        target = Position(x=wp.pixel_x, y=wp.pixel_y)
        dist = position.distance_to(target)
        return dist <= threshold

    def get_visited_cells(self) -> list[dict]:
        """Return list of visited grid cells for the minimap."""
        return [
            {"col": wp.grid_col, "row": wp.grid_row}
            for wp in self.waypoints if wp.visited
        ]

    def reset(self):
        """Reset all waypoints to unvisited."""
        for wp in self.waypoints:
            wp.visited = False
        self.current_index = 0


# ══════════════════════════════════════════════════════════════════════════
#  DIRECTION MAPPER — Translates pixel-space angles to motor commands
# ══════════════════════════════════════════════════════════════════════════

class DirectionMapper:
    """
    Maps pixel-space movement vectors to ESP32 motor commands using
    calibration data.
    
    During calibration, we nudge the rover with each command and record
    the resulting pixel displacement. This builds a mapping of:
      "forward" → moves at angle X° in pixel space
      "left"    → moves at angle Y° in pixel space
      etc.
    
    During navigation, given a desired pixel-space angle, we find the
    motor command whose calibrated angle is closest.
    """

    def __init__(self):
        self.calibration = CalibrationResult()
        # Mapping: Direction → pixel-space angle (degrees)
        self._direction_angles: dict[Direction, float] = {}

    def set_calibration(self, direction: Direction, pixel_angle: float):
        """Record the pixel-space angle for a motor command."""
        self._direction_angles[direction] = pixel_angle
        logger.info(f"Calibrated {direction.value} → {pixel_angle:.1f}° in pixel space")

        if direction == Direction.FORWARD:
            self.calibration.forward_angle = pixel_angle
        elif direction == Direction.LEFT:
            self.calibration.left_angle = pixel_angle
        elif direction == Direction.RIGHT:
            self.calibration.right_angle = pixel_angle
        elif direction == Direction.REVERSE:
            self.calibration.reverse_angle = pixel_angle

    def infer_remaining(self):
        """
        If we only calibrated forward and left, infer right and reverse.
        Right is opposite of left, reverse is opposite of forward.
        """
        if Direction.FORWARD in self._direction_angles:
            fwd = self._direction_angles[Direction.FORWARD]

            if Direction.REVERSE not in self._direction_angles:
                rev = (fwd + 180) % 360
                self._direction_angles[Direction.REVERSE] = rev
                self.calibration.reverse_angle = rev
                logger.info(f"Inferred reverse → {rev:.1f}° (opposite of forward)")

        if Direction.LEFT in self._direction_angles:
            left = self._direction_angles[Direction.LEFT]

            if Direction.RIGHT not in self._direction_angles:
                right = (left + 180) % 360
                self._direction_angles[Direction.RIGHT] = right
                self.calibration.right_angle = right
                logger.info(f"Inferred right → {right:.1f}° (opposite of left)")

        self.calibration.calibrated = True

    def get_best_direction(self, desired_angle: float) -> Direction:
        """
        Given a desired movement angle in pixel space, find the motor
        command that moves closest to that angle.
        """
        if not self._direction_angles:
            # Fallback: assume standard orientation (up = forward)
            return self._fallback_direction(desired_angle)

        best_dir = Direction.STOP
        best_diff = 999.0

        for direction, cal_angle in self._direction_angles.items():
            if direction == Direction.STOP:
                continue
            # Angular difference (handle wraparound)
            diff = abs(self._angle_diff(desired_angle, cal_angle))
            if diff < best_diff:
                best_diff = diff
                best_dir = direction

        # If the best match is more than 60° off, we might need to combine
        # For now, just use the closest single direction
        return best_dir

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Compute shortest angular difference between two angles."""
        diff = (a - b + 180) % 360 - 180
        return diff

    @staticmethod
    def _fallback_direction(angle: float) -> Direction:
        """
        Fallback if no calibration: assume standard image coords where
        up (negative y) = forward.
        0° = right, 90° = down, 180° = left, 270° = up
        """
        # Normalize to 0-360
        angle = angle % 360

        if 225 <= angle <= 315:
            return Direction.FORWARD   # up in image = forward
        elif 45 <= angle <= 135:
            return Direction.REVERSE   # down in image = reverse
        elif 135 < angle < 225:
            return Direction.LEFT      # left in image
        else:
            return Direction.RIGHT     # right in image


# ══════════════════════════════════════════════════════════════════════════
#  MOTOR CONTROLLER — HTTP interface to ESP32
# ══════════════════════════════════════════════════════════════════════════

class MotorController:
    """
    Sends HTTP commands to the ESP32 rover.
    Handles connection monitoring, rate limiting, and error recovery.
    """

    def __init__(self, esp32_ip: str = "", default_speed: int = 140):
        self.esp32_ip = esp32_ip
        self.speed = default_speed
        self.last_command: str = "stop"
        self.last_command_time: float = 0
        self.min_command_interval: float = 0.10  # seconds between commands
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
        """Lazy-init the HTTP client."""
        if self._client is None:
            if not HTTPX_AVAILABLE:
                return None
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def send_command(self, direction: Direction, duration_ms: int = 0) -> bool:
        """
        Send a movement command to the ESP32.

        Args:
            direction: Which direction to move
            duration_ms: If > 0, the ESP32 will auto-stop after this many ms.
                        If 0, motors stay on until /stop is called.
        Returns:
            True if command was sent successfully
        """
        if not self.is_configured:
            logger.warning("ESP32 IP not configured — command not sent")
            return False

        # Rate limiting
        now = time.time()
        if now - self.last_command_time < self.min_command_interval:
            return True  # skip, too soon

        # FIX FOR PHYSICAL HARDWARE: The user noted that the front of the car is actually the back.
        # We swap FORWARD and REVERSE here so the AI's concept of "forward" makes the car physically move "forward"
        # relative to its chassis (which is wired backwards).
        physical_direction = direction.value
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
        """Fallback HTTP request using asyncio + urllib."""
        import urllib.request
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(url, timeout=2)
            )
            return True
        except Exception:
            return False

    async def stop(self) -> bool:
        """Send stop command."""
        return await self.send_command(Direction.STOP)

    async def set_speed(self, speed: int) -> bool:
        """Set motor PWM speed (0-255)."""
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
        """Ping the ESP32 /status endpoint."""
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
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ══════════════════════════════════════════════════════════════════════════
#  AUTOPILOT ENGINE — Main state machine
# ══════════════════════════════════════════════════════════════════════════

class AutopilotEngine:
    """
    The brain. Runs an async control loop that:
    1. Observes rover position (from YOLO via RoverTracker)
    2. Checks geofence (is rover inside grid?)
    3. Decides next action (via WaypointNavigator + DirectionMapper)
    4. Sends motor commands (via MotorController)
    5. Verifies movement
    
    Key features:
    - Auto-calibration: nudges rover to learn camera-motor mapping
    - Geofence enforcement: every step checks boundaries
    - Configurable grid: works for any rows × cols
    - PWM speed default: 140
    """

    # Default ESP32 IP for your RC car
    DEFAULT_ESP32_IP = "192.168.137.37"

    def __init__(self):
        self.tracker = RoverTracker()
        self.navigator = WaypointNavigator()
        self.motor = MotorController(default_speed=140)
        self.direction_mapper = DirectionMapper()
        self.geofence: Optional[Geofence] = None

        self.state = AutopilotState.IDLE
        self.current_direction = Direction.STOP
        self.message = "Waiting for mission start"

        # ── Tuning parameters (adjustable from dashboard) ──
        self.command_duration_ms: int = 350     # how long to pulse motors per step
        self.observe_delay_s: float = 0.6       # wait after stopping to let camera catch up
        self.arrival_threshold: float = 25.0    # pixels — how close = "arrived" at waypoint
        self.stuck_threshold: float = 5.0       # pixels — minimum movement to not be "stuck"
        self.max_stuck_retries: int = 5         # retries before ERROR
        self.calibration_nudge_ms: int = 400    # how long to nudge during calibration

        # ── Internal state ──
        self._stuck_count: int = 0
        self._consecutive_geofence: int = 0     # how many steps in geofence correction
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._telemetry_callbacks: list[Callable] = []
        self._log_entries: deque = deque(maxlen=100)

        # ── Grid config stored for telemetry ──
        self._grid_rows: int = 0
        self._grid_cols: int = 0

    # ── Telemetry ─────────────────────────────────────────────────────

    def add_telemetry_callback(self, callback: Callable):
        """Register a callback that receives telemetry updates."""
        self._telemetry_callbacks.append(callback)

    def remove_telemetry_callback(self, callback: Callable):
        if callback in self._telemetry_callbacks:
            self._telemetry_callbacks.remove(callback)

    def _log(self, msg: str):
        """Add a log entry."""
        entry = {"time": time.strftime("%H:%M:%S"), "msg": msg}
        self._log_entries.append(entry)
        logger.info(msg)

    async def _emit_telemetry(self):
        """Send current state to all registered callbacks."""
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
        """Get current autopilot state as a dict for JSON serialization."""
        pos = self.tracker.position
        wp = self.navigator.current_waypoint

        # Geofence status
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
            "calibrated": self.direction_mapper.calibration.calibrated,
            "heading": self.tracker.get_heading(),
            "message": self.message,
            "log": list(self._log_entries)[-15:],  # last 15 log entries
            "grid_config": {
                "rows": self._grid_rows,
                "cols": self._grid_cols,
            }
        }

    # ── Feed CV detections ────────────────────────────────────────────

    def feed_detection(self, detection_result: dict):
        """
        Called by the CV pipeline whenever a new detection comes in (~5 FPS).
        This feeds the tracker with fresh position data.
        """
        self.tracker.update(detection_result)

    # ── Mission Control ───────────────────────────────────────────────

    async def start_mission(self, grid_config: dict, esp32_ip: str, waypoints: Optional[list] = None):
        """
        Start an autonomous mission.

        grid_config: {
            "x1": int, "y1": int, "x2": int, "y2": int,
            "rows": int, "cols": int
        }
        esp32_ip: IP address of the ESP32 rover (e.g., "192.168.137.37")
        waypoints: List of precomputed JS waypoints (optional)
        """
        if self.state not in (AutopilotState.IDLE, AutopilotState.COMPLETE, AutopilotState.ERROR):
            self._log("Cannot start: mission already in progress")
            return

        # Configure motor controller
        self.motor.esp32_ip = esp32_ip or self.DEFAULT_ESP32_IP
        self._log(f"ESP32 target: {self.motor.esp32_ip}")

        # Store grid config
        self._grid_rows = grid_config.get("rows", 2)
        self._grid_cols = grid_config.get("cols", 3)

        # Create geofence
        self.geofence = Geofence(
            x1=grid_config["x1"],
            y1=grid_config["y1"],
            x2=grid_config["x2"],
            y2=grid_config["y2"],
            margin_pct=0.08,  # 8% margin from edges
        )
        self._log(f"Geofence set: [{self.geofence.x1},{self.geofence.y1}]-"
                  f"[{self.geofence.x2},{self.geofence.y2}]")

        # Generate waypoints (clamped to geofence safe zone)
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

        self._log(f"Mission configured: {self._grid_rows}x{self._grid_cols} grid, "
                  f"{len(self.navigator.waypoints)} waypoints")

        # Set speed to 140
        await self.motor.set_speed(140)
        self._log("PWM speed set to 140")

        # Reset state
        self._stuck_count = 0
        self._consecutive_geofence = 0
        self.tracker.reset()
        self.direction_mapper = DirectionMapper()  # fresh calibration

        # Start the control loop
        self._running = True
        self.state = AutopilotState.CALIBRATING
        self.message = "Calibrating — looking for rover..."
        self._log("Mission started — calibrating...")

        self._task = asyncio.create_task(self._control_loop())

    async def stop_mission(self):
        """Abort the mission."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self.motor.stop()
        self.state = AutopilotState.IDLE
        self.message = "Mission aborted"
        self._log("Mission aborted by user")
        await self._emit_telemetry()

    async def pause_mission(self):
        """Pause/resume the mission."""
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

    # ── Main Control Loop ─────────────────────────────────────────────

    async def _control_loop(self):
        """
        The main autonomous control loop.
        Runs as an async task until the mission completes or is aborted.
        """
        try:
            # ── Phase 1: Wait for rover detection ──
            await self._wait_for_rover()
            if self.state == AutopilotState.ERROR or not self._running:
                return

            # ── Phase 2: Auto-calibration (nudge test) ──
            await self._calibrate()
            if self.state == AutopilotState.ERROR or not self._running:
                return

            # ── Phase 3: Navigation with geofencing ──
            self.state = AutopilotState.NAVIGATING
            self.message = "Navigating — sweeping grid"

            while self._running and not self.navigator.is_complete:
                # Handle pause
                while self.state == AutopilotState.PAUSED:
                    await asyncio.sleep(0.3)
                    if not self._running:
                        return

                # Abort check
                if not self._running:
                    return

                await self._navigate_step()
                await self._emit_telemetry()

            # ── Phase 4: Complete ──
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

    # ── Phase 1: Wait for rover detection ─────────────────────────────

    async def _wait_for_rover(self):
        """Wait until the rover is detected reliably (5 consecutive confirmations)."""
        self._log("Looking for rover in camera feed...")
        detection_count = 0
        max_wait = 60  # seconds

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

        # Check ESP32 connectivity
        connected = await self.motor.check_heartbeat()
        if connected:
            self._log("ESP32 rover connected")
        else:
            self._log("ESP32 not responding — will retry on first command")

    # ── Phase 2: Auto-calibration ─────────────────────────────────────

    async def _calibrate(self):
        """
        Auto-calibration: nudge the rover to learn camera-motor mapping.
        
        Steps:
        1. Record current position
        2. Send "forward" for calibration_nudge_ms
        3. Stop, wait for camera, record new position
        4. Compute pixel-space angle of displacement → that's what "forward" does
        5. Repeat for "left"
        6. Infer "reverse" and "right" as opposites
        """
        self.state = AutopilotState.CALIBRATING
        self._log("Starting auto-calibration (nudge test)...")
        await self._emit_telemetry()

        # ── Calibrate FORWARD ──
        fwd_angle = await self._nudge_and_measure(Direction.FORWARD)
        if fwd_angle is None:
            self._log("Forward calibration failed — using fallback orientation")
            self.direction_mapper.calibration.calibrated = False
            # Continue with fallback (standard image coords)
            return

        self.direction_mapper.set_calibration(Direction.FORWARD, fwd_angle)

        # Wait a bit before next nudge
        await asyncio.sleep(0.5)

        # ── Calibrate LEFT ──
        left_angle = await self._nudge_and_measure(Direction.LEFT)
        if left_angle is None:
            # Infer left as forward - 90°
            inferred_left = (fwd_angle - 90) % 360
            self.direction_mapper.set_calibration(Direction.LEFT, inferred_left)
            self._log(f"Left calibration failed — inferred as {inferred_left:.1f}°")
        else:
            self.direction_mapper.set_calibration(Direction.LEFT, left_angle)

        # ── Infer REVERSE and RIGHT ──
        self.direction_mapper.infer_remaining()

        self._log("Calibration complete!")
        self._log(f"  Forward → {self.direction_mapper.calibration.forward_angle:.1f}°")
        self._log(f"  Left    → {self.direction_mapper.calibration.left_angle:.1f}°")
        self._log(f"  Right   → {self.direction_mapper.calibration.right_angle:.1f}°")
        self._log(f"  Reverse → {self.direction_mapper.calibration.reverse_angle:.1f}°")

    async def _nudge_and_measure(self, direction: Direction) -> Optional[float]:
        """
        Nudge the rover in a direction and measure the pixel displacement.
        Returns the angle of displacement in pixel space, or None if failed.
        """
        # Get stable position before nudge
        await asyncio.sleep(0.3)
        pre_pos = self.tracker.get_stable_position(n=3)
        if pre_pos is None:
            self._log(f"Cannot calibrate {direction.value}: rover not detected")
            return None

        self._log(f"Nudging {direction.value}... (pre: {pre_pos.x:.0f},{pre_pos.y:.0f})")

        # Nudge
        sent = await self.motor.send_command(direction, duration_ms=self.calibration_nudge_ms)
        if not sent:
            self._log(f"Failed to send {direction.value} nudge")
            return None

        # Wait for movement + camera to catch up
        await asyncio.sleep(self.calibration_nudge_ms / 1000.0 + 0.3)
        await self.motor.stop()
        await asyncio.sleep(self.observe_delay_s)

        # Get stable position after nudge
        post_pos = self.tracker.get_stable_position(n=3)
        if post_pos is None:
            self._log(f"Lost rover after {direction.value} nudge")
            return None

        # Compute displacement
        dx = post_pos.x - pre_pos.x
        dy = post_pos.y - pre_pos.y
        dist = math.sqrt(dx**2 + dy**2)

        self._log(f"  {direction.value} nudge result: dx={dx:.1f}, dy={dy:.1f}, dist={dist:.1f}px")

        if dist < 5.0:
            self._log(f"  {direction.value} nudge too small — rover may not have moved")
            return None

        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to 0-360
        angle = angle % 360
        return angle

    # ── Phase 3: Navigation with geofencing ───────────────────────────

    async def _navigate_step(self):
        """
        One step of the navigation loop:
        1. Get current position
        2. CHECK GEOFENCE — if outside, correct FIRST (priority #1)
        3. Check if arrived at waypoint
        4. Compute direction to waypoint
        5. Apply geofence bias if near edge
        6. Send command for short burst
        7. Stop and observe
        8. Verify movement
        """
        pos = self.tracker.get_stable_position(n=3)
        if pos is None:
            # Rover lost — wait for re-detection
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

        # ════════════════════════════════════════════════════════════
        #  GEOFENCE CHECK — HIGHEST PRIORITY
        # ════════════════════════════════════════════════════════════

        if self.geofence and not self.geofence.is_inside(pos):
            # ── VIOLATION: Rover is OUTSIDE the grid! ──
            self._consecutive_geofence += 1
            self.state = AutopilotState.GEOFENCE
            self._log(f"⚠ GEOFENCE VIOLATION! Rover at ({pos.x:.0f},{pos.y:.0f}) — correcting...")

            if self._consecutive_geofence > 30:
                self.state = AutopilotState.ERROR
                self.message = "Cannot get rover back in bounds"
                self._log("Geofence correction failed after 30 attempts")
                await self.motor.stop()
                return

            # Steer toward grid center
            await self._steer_toward(pos, self.geofence.center)
            return

        # Reset geofence counter if we're safely inside
        if self.geofence and self.geofence.is_inside(pos):
            self._consecutive_geofence = 0

        # ════════════════════════════════════════════════════════════
        #  ARRIVAL CHECK
        # ════════════════════════════════════════════════════════════

        if self.navigator.has_arrived(pos, threshold=self.arrival_threshold):
            self.navigator.advance()
            self._stuck_count = 0
            self.state = AutopilotState.NAVIGATING
            self.message = f"Arrived at waypoint {wp.index} — advancing"

            if self.navigator.is_complete:
                return

            next_wp = self.navigator.current_waypoint
            if next_wp:
                self._log(f"→ Moving to waypoint {next_wp.index} "
                          f"[col={next_wp.grid_col}, row={next_wp.grid_row}]")
            return

        # ════════════════════════════════════════════════════════════
        #  COMPUTE DIRECTION TO WAYPOINT
        # ════════════════════════════════════════════════════════════

        target = Position(x=wp.pixel_x, y=wp.pixel_y)
        dist = pos.distance_to(target)

        # Compute desired pixel-space angle to waypoint
        desired_angle = pos.angle_to(target) % 360

        # ── GEOFENCE EDGE BIAS ──
        # If rover is in the warning zone, blend the direction toward center
        if self.geofence and self.geofence.is_in_warning_zone(pos):
            corr_dx, corr_dy = self.geofence.get_correction_vector(pos)
            if abs(corr_dx) > 1 or abs(corr_dy) > 1:
                # Blend: 60% waypoint direction + 40% geofence correction
                wp_dx = target.x - pos.x
                wp_dy = target.y - pos.y

                blended_dx = 0.6 * wp_dx + 0.4 * corr_dx
                blended_dy = 0.6 * wp_dy + 0.4 * corr_dy

                desired_angle = math.degrees(math.atan2(blended_dy, blended_dx)) % 360
                self.message = f"⚠ Near edge — biasing toward center | WP{wp.index} (dist: {dist:.0f}px)"
            else:
                self.message = f"Moving → WP{wp.index} (dist: {dist:.0f}px)"
        else:
            self.state = AutopilotState.NAVIGATING
            self.message = f"Moving → WP{wp.index} (dist: {dist:.0f}px)"

        # Map desired pixel-angle to the best motor command
        direction = self.direction_mapper.get_best_direction(desired_angle)
        self.current_direction = direction

        if direction == Direction.STOP:
            await asyncio.sleep(0.2)
            return

        # ════════════════════════════════════════════════════════════
        #  SEND MOTOR COMMAND (short burst)
        # ════════════════════════════════════════════════════════════

        pre_pos = Position(x=pos.x, y=pos.y, timestamp=pos.timestamp)

        sent = await self.motor.send_command(direction, duration_ms=self.command_duration_ms)
        if not sent:
            self._log("Failed to send command to ESP32")
            await asyncio.sleep(1.0)
            return

        # Wait for the rover to move
        await asyncio.sleep(self.command_duration_ms / 1000.0 + 0.05)

        # Ensure stopped (safety)
        await self.motor.stop()

        # Observe delay — let camera catch up
        await asyncio.sleep(self.observe_delay_s)

        # ════════════════════════════════════════════════════════════
        #  VERIFY MOVEMENT (stuck detection)
        # ════════════════════════════════════════════════════════════

        new_pos = self.tracker.get_stable_position(n=2)
        if new_pos and pre_pos:
            moved = pre_pos.distance_to(new_pos)
            if moved < self.stuck_threshold:
                self._stuck_count += 1
                self._log(f"Rover may be stuck ({self._stuck_count}/{self.max_stuck_retries}) "
                          f"— moved only {moved:.1f}px")
                self.state = AutopilotState.CORRECTING

                if self._stuck_count >= self.max_stuck_retries:
                    self.state = AutopilotState.ERROR
                    self.message = "Rover stuck — manual intervention needed"
                    self._log("Rover stuck! Max retries exceeded.")
                    await self.motor.stop()
                    return

                # ── Stuck recovery: try different approaches ──
                await self._unstick(self._stuck_count, direction)
            else:
                # Good movement — decay stuck counter
                self._stuck_count = max(0, self._stuck_count - 1)

    async def _steer_toward(self, current_pos: Position, target: Position):
        """
        Emergency steering: move the rover from current_pos toward target.
        Used for geofence corrections.
        """
        desired_angle = current_pos.angle_to(target) % 360
        direction = self.direction_mapper.get_best_direction(desired_angle)
        self.current_direction = direction

        self.message = f"🚨 GEOFENCE — steering {direction.value} back to grid"
        self._log(f"Geofence correction: {direction.value} "
                  f"(rover at {current_pos.x:.0f},{current_pos.y:.0f})")

        # Shorter burst for corrections (more careful)
        sent = await self.motor.send_command(direction, duration_ms=250)
        if sent:
            await asyncio.sleep(0.30)
            await self.motor.stop()
            await asyncio.sleep(self.observe_delay_s)

    async def _unstick(self, attempt: int, last_direction: Direction):
        """
        Try to get the rover unstuck with progressively more aggressive tactics.
        """
        self.message = f"Correcting — unstick attempt {attempt}"

        if attempt <= 2:
            # Try reversing briefly
            self._log("Unstick: trying brief reverse")
            opposite = self._opposite_direction(last_direction)
            await self.motor.send_command(opposite, duration_ms=300)
            await asyncio.sleep(0.35)
            await self.motor.stop()
            await asyncio.sleep(self.observe_delay_s)

        elif attempt <= 4:
            # Try a perpendicular direction
            self._log("Unstick: trying perpendicular direction")
            perp = self._perpendicular_direction(last_direction)
            await self.motor.send_command(perp, duration_ms=400)
            await asyncio.sleep(0.45)
            await self.motor.stop()
            await asyncio.sleep(self.observe_delay_s)

        else:
            # Last resort: reverse longer, then try again
            self._log("Unstick: aggressive reverse")
            opposite = self._opposite_direction(last_direction)
            await self.motor.send_command(opposite, duration_ms=600)
            await asyncio.sleep(0.65)
            await self.motor.stop()
            await asyncio.sleep(self.observe_delay_s)

    @staticmethod
    def _opposite_direction(d: Direction) -> Direction:
        opposites = {
            Direction.FORWARD: Direction.REVERSE,
            Direction.REVERSE: Direction.FORWARD,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites.get(d, Direction.REVERSE)

    @staticmethod
    def _perpendicular_direction(d: Direction) -> Direction:
        perps = {
            Direction.FORWARD: Direction.LEFT,
            Direction.REVERSE: Direction.RIGHT,
            Direction.LEFT: Direction.REVERSE,
            Direction.RIGHT: Direction.FORWARD,
        }
        return perps.get(d, Direction.LEFT)

    # ── Configuration ─────────────────────────────────────────────────

    def set_config(self, config: dict):
        """Update tuning parameters from the dashboard."""
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
        """Clean shutdown."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self.motor.stop()
        await self.motor.close()
