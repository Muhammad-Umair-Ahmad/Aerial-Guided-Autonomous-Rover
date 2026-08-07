import logging
from mars_rover.navigation.world_model import WorldModel

logger = logging.getLogger("MarsRover.Safety")

class FailsafeChecker:
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def check_safety(self, world_model: WorldModel) -> bool:
        """
        Returns True if safe, False if an emergency stop is required.
        Updates world_model state with errors.
        """
        emergency_stop = False

        if world_model.out_of_bounds:
            logger.error("EMERGENCY STOP: Rover is outside the geofence!")
            emergency_stop = True

        if world_model.is_lost:
            logger.warning("EMERGENCY STOP: Tracking is lost!")
            emergency_stop = True

        if world_model.confidence < self.min_confidence and not world_model.is_lost:
            # Only trigger low confidence if not already considered lost, though both could happen
            logger.warning(f"EMERGENCY STOP: Detection confidence ({world_model.confidence:.2f}) is below threshold ({self.min_confidence})!")
            emergency_stop = True

        if world_model.battery_low:
            logger.error("EMERGENCY STOP: Battery is critically low!")
            emergency_stop = True

        return not emergency_stop
