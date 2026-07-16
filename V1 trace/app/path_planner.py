import json
import os

class PathPlanner:
    def __init__(self, grid_cols, grid_rows, cell_size_px=50):
        self.cols = grid_cols
        self.rows = grid_rows
        self.cell_size = cell_size_px

    def generate_lawnmower_path(self, poi_grid_coords=None):
        """
        Generates a Boustrophedon (lawnmower) sweep path.
        If a POI coordinate is provided, the planner modifies actions for that node.
        """
        path = []
        waypoint_id = 0

        for r in range(self.rows):
            # If row is even, move left-to-right. If odd, move right-to-left.
            col_range = range(self.cols) if r % 2 == 0 else range(self.cols - 1, -1, -1)
            
            for c in col_range:
                # Calculate the center pixel coordinates of this grid cell
                px_x = int((c * self.cell_size) + (self.cell_size / 2))
                px_y = int((r * self.cell_size) + (self.cell_size / 2))
                
                # Default action
                action = "SWEEP_SAMPLE"
                dwell_time = 1.0 # seconds
                
                # Check if this cell is the Anomaly Zone (POI)
                if poi_grid_coords and (c, r) == poi_grid_coords:
                    action = "DENSE_DWELL_SAMPLE"
                    dwell_time = 5.0  # Dwell longer to collect intensive sensor data

                waypoint = {
                    "id": waypoint_id,
                    "grid_coords": [c, r],
                    "pixel_coords": [px_x, px_y],
                    "action": action,
                    "dwell_time_sec": dwell_time
                }
                path.append(waypoint)
                waypoint_id += 1
                
        return path

    def export_mission_plan(self, path, output_path):
        """Exports the path list to a standardized JSON file."""
        mission = {
            "metadata": {
                "grid_dimensions": {"cols": self.cols, "rows": self.rows},
                "total_waypoints": len(path)
            },
            "waypoints": path
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mission, f, indent=4)
        print(f"[SUCCESS] Exported mission plan to: {output_path}")
        