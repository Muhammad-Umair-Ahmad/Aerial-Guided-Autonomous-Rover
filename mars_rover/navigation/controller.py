import math
from typing import Tuple
from mars_rover.config import Config
from mars_rover.navigation.world_model import WorldModel
from mars_rover.communication.esp32 import Direction

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-255, 255)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = output_limits
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        # Clamp output
        return max(self.limits[0], min(self.limits[1], output))

class DifferentialSteeringController:
    """
    Decides between turning in place or driving forward based on heading error.
    Consumes WorldModel state to determine PID steering and motor commands.
    """
    def __init__(self):
        self.heading_tolerance = Config.HEADING_TOLERANCE
        self.pid_steer = PIDController(kp=1.5, ki=0.0, kd=0.1, output_limits=(-255, 255))
        self.last_time = 0.0
        self.current_speed = 0.0
        self.max_acceleration = 50.0  # max speed change per second

    @staticmethod
    def angle_diff(target: float, current: float) -> float:
        """Compute shortest angular difference between two angles [-180, 180]."""
        return (target - current + 180) % 360 - 180

    def apply_acceleration_limits(self, target_speed: float, dt: float) -> float:
        max_delta = self.max_acceleration * dt
        if target_speed > self.current_speed + max_delta:
            self.current_speed += max_delta
        elif target_speed < self.current_speed - max_delta:
            self.current_speed -= max_delta
        else:
            self.current_speed = target_speed
        return self.current_speed

    def compute_commands(self, world_state: WorldModel) -> Tuple[Direction, int, float]:
        """
        Returns (Direction, duration_ms, recommended_speed)
        """
        current_time = world_state.rover_pose.timestamp if world_state.rover_pose else 0.0
        dt = current_time - self.last_time if self.last_time > 0 else 0.1
        self.last_time = current_time

        if not world_state.rover_pose or not world_state.target_waypoint or world_state.rover_pose.heading is None:
            return Direction.STOP, 0, 0.0

        target_pos = world_state.target_waypoint
        current_pos = world_state.rover_pose

        # Desired angle
        desired_angle = math.degrees(math.atan2(target_pos.y - current_pos.y, target_pos.x - current_pos.x)) % 360
        heading_error = self.angle_diff(desired_angle, current_pos.heading)

        # PID Steering Output
        steer_output = self.pid_steer.compute(heading_error, dt)
        
        # Turn-then-drive logic
        if abs(heading_error) > self.heading_tolerance:
            turn_duration = 250
            # Could scale turn_duration with steer_output
            turn_speed = self.apply_acceleration_limits(Config.DEFAULT_SPEED, dt)
            if heading_error > 0:
                return Direction.RIGHT, turn_duration, turn_speed
            else:
                return Direction.LEFT, turn_duration, turn_speed
        else:
            drive_duration = 350
            drive_speed = self.apply_acceleration_limits(Config.DEFAULT_SPEED, dt)
            return Direction.FORWARD, drive_duration, drive_speed
