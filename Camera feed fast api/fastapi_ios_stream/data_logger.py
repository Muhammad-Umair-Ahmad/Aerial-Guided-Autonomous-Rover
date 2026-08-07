import os
import time
import base64
import csv
from datetime import datetime

class DataLogger:
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.labels_file = os.path.join(dataset_dir, "labels.csv")
        self.is_recording = False
        self.frames_collected = 0
        self.last_record_time = 0
        
        self._setup_directories()

    def _setup_directories(self):
        os.makedirs(self.images_dir, exist_ok=True)
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_filename", "command", "timestamp"])

    def set_recording(self, enabled: bool):
        self.is_recording = enabled
        if enabled:
            print(f"[DATA LOGGER] Recording started. Saving to {self.dataset_dir}")
        else:
            print(f"[DATA LOGGER] Recording stopped. Total frames: {self.frames_collected}")

    def record_frame(self, image_b64: str, command: str) -> bool:
        """
        Record a frame and its label if recording is active.
        Limits recording to ~10 FPS to avoid exploding file sizes.
        Returns True if a frame was saved, False otherwise.
        """
        if not self.is_recording or not image_b64 or not command:
            return False

        now = time.time()
        # Cap at ~10 FPS (100ms interval)
        if now - self.last_record_time < 0.1:
            return False

        self.last_record_time = now
        self.frames_collected += 1

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)

        try:
            # Decode base64 image (assuming it's a data URI like "data:image/jpeg;base64,...")
            if "," in image_b64:
                header, encoded = image_b64.split(",", 1)
            else:
                encoded = image_b64
                
            img_data = base64.b64decode(encoded)
            with open(filepath, "wb") as f:
                f.write(img_data)
                
            # Append to labels.csv
            with open(self.labels_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([filename, command, now])
                
            return True
        except Exception as e:
            print(f"[DATA LOGGER] Error saving frame: {e}")
            return False
