"""
Global Configuration for Mars Rover Architecture
"""

class Config:
    # ── ESP32 Hardware ──
    ESP32_IP = "192.168.137.249"
    DEFAULT_SPEED = 140
    MIN_COMMAND_INTERVAL = 0.10
    INVERT_FORWARD_REVERSE = False
    
    # ── Navigation & Control ──
    HEADING_TOLERANCE = 30.0
    COMMAND_DURATION_MS = 350
    TURN_DURATION_MS = 250
    OBSERVE_DELAY_S = 0.8
    ARRIVAL_THRESHOLD = 30.0
    
    # ── Safety & Failsafe ──
    STUCK_THRESHOLD = 8.0
    MAX_STUCK_RETRIES = 8
    
    # ── Geofence ──
    WARNING_MARGIN_PCT = 0.08
    WAYPOINT_MARGIN_PCT = 0.15
    
    # ── Perception ──
    LOST_THRESHOLD = 20
    CONFIDENCE_THRESHOLD = 0.50
