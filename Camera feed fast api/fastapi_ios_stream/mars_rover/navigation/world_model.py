import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

class AutopilotState(Enum):
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    PLAN = "PLAN"
    ALIGN = "ALIGN"
    DRIVE = "DRIVE"
    VERIFY = "VERIFY"
    NEXT_WAYPOINT = "NEXT_WAYPOINT"
    CORRECTING = "CORRECTING"
    GEOFENCE = "GEOFENCE"
    ERROR = "ERROR"
    COMPLETE = "COMPLETE"

@dataclass
class Position:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class Pose:
    position: Position
    heading: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class Waypoint:
    x: float
    y: float
    grid_col: int
    grid_row: int
    index: int
    visited: bool = False

class WorldModel:
    """
    Central state store for the Mars Rover.
    All modules read and write to this model.
    """
    def __init__(self):
        self.rover_pose: Optional[Pose] = None
        self.obstacles: List[Position] = []
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx: int = 0
        
        self.state: AutopilotState = AutopilotState.IDLE
        self.battery_voltage: float = 12.0
        
        self.fps: int = 0
        self.latency_ms: float = 0.0
        
        # Telemetry info
        self.target_heading: float = 0.0
        self.heading_error: float = 0.0
        self.current_velocity: float = 0.0
        
        self.mission_active: bool = False
        
    def get_current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
        
    def advance_waypoint(self):
        wp = self.get_current_waypoint()
        if wp:
            wp.visited = True
        self.current_waypoint_idx += 1
