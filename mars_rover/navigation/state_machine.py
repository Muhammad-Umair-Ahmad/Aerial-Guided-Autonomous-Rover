import math
import logging
from enum import Enum
from mars_rover.navigation.world_model import WorldModel
from mars_rover.config import Config

logger = logging.getLogger("MarsRover.StateMachine")

class NavState(str, Enum):
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    PLAN = "PLAN"
    ALIGN = "ALIGN"
    DRIVE = "DRIVE"
    VERIFY = "VERIFY"
    NEXT_WAYPOINT = "NEXT_WAYPOINT"
    ERROR = "ERROR"

class NavigationStateMachine:
    def __init__(self):
        self.state = NavState.IDLE
        self.world_model = WorldModel()

    def update(self, world_model: WorldModel) -> NavState:
        self.world_model = world_model
        
        # Failsafe checks transition to ERROR or IDLE if needed, but handled outside or inside:
        if self.world_model.out_of_bounds or self.world_model.battery_low:
            self.state = NavState.ERROR
            return self.state

        if self.state == NavState.IDLE:
            if self.world_model.target_waypoint:
                self.state = NavState.SEARCH
            
        elif self.state == NavState.SEARCH:
            if not self.world_model.is_lost and self.world_model.rover_pose:
                self.state = NavState.PLAN
                
        elif self.state == NavState.PLAN:
            if self.world_model.target_waypoint:
                self.state = NavState.ALIGN
            else:
                self.state = NavState.IDLE
                
        elif self.state == NavState.ALIGN:
            # Check heading error
            if self.world_model.rover_pose and self.world_model.target_waypoint:
                current_pos = self.world_model.rover_pose
                target_pos = self.world_model.target_waypoint
                if current_pos.heading is not None:
                    desired_angle = math.degrees(math.atan2(target_pos.y - current_pos.y, target_pos.x - current_pos.x)) % 360
                    heading_error = (desired_angle - current_pos.heading + 180) % 360 - 180
                    if abs(heading_error) <= Config.HEADING_TOLERANCE:
                        self.state = NavState.DRIVE
            else:
                self.state = NavState.SEARCH
                
        elif self.state == NavState.DRIVE:
            # After drive command issued, transition to VERIFY
            self.state = NavState.VERIFY
            
        elif self.state == NavState.VERIFY:
            # Check distance to waypoint
            if self.world_model.rover_pose and self.world_model.target_waypoint:
                current_pos = self.world_model.rover_pose
                target_pos = self.world_model.target_waypoint
                dist = math.hypot(target_pos.x - current_pos.x, target_pos.y - current_pos.y)
                if dist <= Config.ARRIVAL_THRESHOLD:
                    self.state = NavState.NEXT_WAYPOINT
                else:
                    # Still need to go to waypoint
                    self.state = NavState.PLAN
            else:
                self.state = NavState.SEARCH
                
        elif self.state == NavState.NEXT_WAYPOINT:
            if self.world_model.target_waypoint:
                self.world_model.target_waypoint.visited = True
                
            # If there are more waypoints, target the next one (logic usually handled by a navigator, but state machine transitions)
            unvisited = [wp for wp in self.world_model.waypoints if not wp.visited]
            if unvisited:
                self.world_model.target_waypoint = unvisited[0]
                self.state = NavState.PLAN
            else:
                self.world_model.target_waypoint = None
                self.state = NavState.IDLE
                
        elif self.state == NavState.ERROR:
            # Requires manual reset
            pass

        return self.state
