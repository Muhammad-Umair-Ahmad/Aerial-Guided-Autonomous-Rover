import time
import math
from collections import deque
from typing import Optional, Dict, Any
from mars_rover.navigation.world_model import Pose

try:
    # Adjust this import based on the actual path to cv_pipeline if needed
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Camera feed fast api', 'fastapi_ios_stream'))
    from cv_pipeline import ObjectDetector
    cv_available = True
except ImportError:
    cv_available = False

class PerceptionManager:
    def __init__(self, history_size: int = 30, lost_threshold: int = 20):
        if cv_available:
            self.detector = ObjectDetector()
        else:
            self.detector = None
        
        self.history: deque[Pose] = deque(maxlen=history_size)
        self.lost_threshold = lost_threshold
        self.miss_count = 0
        self._last_pose: Optional[Pose] = None
        self._detected = False

    def get_heading(self) -> Optional[float]:
        """
        Estimate heading (angle in degrees) from recent position history.
        """
        if len(self.history) < 3:
            return None

        recent = list(self.history)[-5:]
        if len(recent) < 2:
            return None

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y

        if abs(dx) < 2 and abs(dy) < 2:
            return None

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return angle

    def process_frame(self, frame_b64: str, threshold: float = 0.4) -> Dict[str, Any]:
        """
        Process a base64 frame, detect objects, and update the tracker.
        Returns a dictionary containing the rover pose, heading, confidence, obstacles, and timestamp.
        """
        timestamp = time.time()
        result = {
            "rover_pose": None,
            "heading": None,
            "confidence": 0.0,
            "obstacles": [],
            "timestamp": timestamp
        }

        if not self.detector or not frame_b64:
            self.miss_count += 1
            return result

        cv_results = self.detector.detect_objects(frame_b64, confidence_threshold=threshold)
        detections = cv_results.get("detections", [])
        
        # In this simplistic logic, assume rover is the first detection or a specific class
        # Add obstacle parsing as needed based on your YOLO classes
        rover_det = None
        obstacles = []
        for det in detections:
            # You might filter by det["class_name"] == "rover" etc.
            # Assuming first detection is rover for now to match old code
            if rover_det is None:
                rover_det = det
            else:
                obstacles.append(det)

        if rover_det:
            box = rover_det["box"]
            cx = box["x"] + box["width"] / 2.0
            cy = box["y"] + box["height"] / 2.0
            conf = rover_det.get("confidence", 0.0)
            
            pose = Pose(x=cx, y=cy, timestamp=timestamp)
            self.history.append(pose)
            
            # Recompute heading
            pose.heading = self.get_heading()
            
            self._last_pose = pose
            self._detected = True
            self.miss_count = 0
            
            result["rover_pose"] = pose
            result["heading"] = pose.heading
            result["confidence"] = conf
        else:
            self.miss_count += 1
            self._detected = False
            
        result["obstacles"] = obstacles
        return result
