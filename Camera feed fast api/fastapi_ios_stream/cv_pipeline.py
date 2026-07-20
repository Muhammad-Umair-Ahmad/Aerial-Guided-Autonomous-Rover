import os
import cv2
import numpy as np
import base64

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.net = None
        if YOLO_AVAILABLE:
            print(f"[CV] Loading YOLO model {model_name}...")
            # Automatically downloads the nano model if not present locally
            self.net = YOLO(model_name)
            print("[CV] Model loaded successfully.")
        else:
            print("[CV] WARNING: 'ultralytics' not installed. Please run 'pip install ultralytics'")

    def detect_objects(self, base64_img: str, confidence_threshold: float = 0.5):
        """
        Decodes a base64 image, runs YOLOv8 object detection, and returns bounding boxes.
        """
        if self.net is None:
            return {"error": "YOLO not loaded. Did you pip install ultralytics?", "detections": []}

        try:
            # Strip the 'data:image/jpeg;base64,' prefix if present
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            # Decode base64 to numpy array
            img_data = base64.b64decode(base64_img)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Could not decode image", "detections": []}

            (h, w) = img.shape[:2]

            # Run YOLO inference
            # conf parameter filters out predictions below the threshold
            results = self.net(img, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            if len(results) > 0:
                result = results[0]
                # Extract boxes (xyxy format), classes, and confidences
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_idx in zip(boxes, confs, classes):
                    startX, startY, endX, endY = box
                    label = result.names[int(cls_idx)]
                    
                    # Generate deterministic color from hash of label name
                    hash_val = hash(label)
                    color = [
                        (hash_val & 0xFF),
                        ((hash_val >> 8) & 0xFF),
                        ((hash_val >> 16) & 0xFF)
                    ]
                    
                    detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "box": {
                            "x": int(startX),
                            "y": int(startY),
                            "width": int(endX - startX),
                            "height": int(endY - startY)
                        },
                        "color": color
                    })

            return {"detections": detections, "original_size": {"width": w, "height": h}}

        except Exception as e:
            print(f"[CV] Detection error: {e}")
            return {"error": str(e), "detections": []}
