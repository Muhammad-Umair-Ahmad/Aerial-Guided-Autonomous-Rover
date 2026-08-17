import cv2
import numpy as np
import os
from path_planner import PathPlanner # Import our new module

class ReconAnalyzer:
    def __init__(self, grid_size=50):
        self.grid_size = grid_size

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not decode image at {image_path}")
        img = cv2.resize(img, (800, 600)) 
        return img

    def draw_coordinate_grid(self, image):
        grid_img = image.copy()
        h, w, _ = grid_img.shape
        for x in range(0, w, self.grid_size):
            cv2.line(grid_img, (x, 0), (x, h), (100, 100, 100), 1, cv2.LINE_AA)
        for y in range(0, h, self.grid_size):
            cv2.line(grid_img, (0, y), (w, y), (100, 100, 100), 1, cv2.LINE_AA)
        return grid_img

    def detect_anomaly_poi(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        if not valid_contours:
            return None, None

        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            grid_x = cX // self.grid_size
            grid_y = cY // self.grid_size
            return (cX, cY), (grid_x, grid_y)
            
        return None, None

    def draw_path_overlay(self, image, path):
        """Draws arrows connecting the planned path waypoints."""
        overlay_img = image.copy()
        for i in range(len(path) - 1):
            pt1 = tuple(path[i]["pixel_coords"])
            pt2 = tuple(path[i + 1]["pixel_coords"])
            
            # Draw lines between waypoints
            color = (0, 255, 0) # Green for normal path
            if path[i]["action"] == "DENSE_DWELL_SAMPLE" or path[i+1]["action"] == "DENSE_DWELL_SAMPLE":
                color = (0, 165, 255) # Orange near/at POI
                
            cv2.arrowedLine(overlay_img, pt1, pt2, color, 2, tipLength=0.2)
            
            # Tiny marker for each node
            cv2.circle(overlay_img, pt1, 3, (255, 0, 0), -1)
            
        return overlay_img

    def process_overhead_scan(self, input_path, output_img_path, output_json_path):
        """Pipeline execution."""
        print(f"[INFO] Ingesting: {input_path}")
        original_img = self.load_image(input_path)
        h, w, _ = original_img.shape
        
        # Calculate grid bounds
        grid_cols = w // self.grid_size
        grid_rows = h // self.grid_size
        
        # 1. Analyze and find POI
        pixel_coords, grid_coords = self.detect_anomaly_poi(original_img)
        
        # 2. Setup path planner and generate path
        planner = PathPlanner(grid_cols, grid_rows, self.grid_size)
        path = planner.generate_lawnmower_path(poi_grid_coords=grid_coords)
        
        # 3. Export API Mission Plan JSON
        planner.export_mission_plan(path, output_json_path)

        # 4. Generate Visualizations
        analyzed_img = self.draw_coordinate_grid(original_img)
        analyzed_img = self.draw_path_overlay(analyzed_img, path)
        
        if pixel_coords:
            print(f"[SUCCESS] Anomaly flagged at Grid Coordinate: {grid_coords}")
            # Draw red marker circle over POI
            cv2.circle(analyzed_img, pixel_coords, 12, (0, 0, 255), 3)
            cv2.putText(analyzed_img, f"POI DWELL ZONE: {grid_coords}", 
                        (pixel_coords[0] - 50, pixel_coords[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print("[WARNING] No anomalies detected in the scene.")

        # Save output image
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, analyzed_img)
        print(f"[SUCCESS] Saved path visualization to: {output_img_path}")

if __name__ == "__main__":
    analyzer = ReconAnalyzer(grid_size=50)
    
    input_file = "/workspace/input/test_recon.jpg"
    output_image = "/workspace/output/analyzed_recon.jpg"
    output_json = "/workspace/output/mission_plan.json"
    
    try:
        analyzer.process_overhead_scan(input_file, output_image, output_json)
    except Exception as e:
        print(f"[ERROR] Run failed: {e}")