from dataclasses import dataclass, field
import time
from typing import Optional, List, Dict, Any

@dataclass
class Pose:
    x: float
    y: float
    heading: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class Waypoint:
    x: float
    y: float
    visited: bool = False

@dataclass
class WorldModel:
    rover_pose: Optional[Pose] = None
    target_waypoint: Optional[Waypoint] = None
    waypoints: List[Waypoint] = field(default_factory=list)
    confidence: float = 0.0
    obstacles: List[Any] = field(default_factory=list)
    is_lost: bool = True
    out_of_bounds: bool = False
    battery_low: bool = False
