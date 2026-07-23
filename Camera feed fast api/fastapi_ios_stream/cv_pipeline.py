import cv2
import numpy as np
import base64
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectDetector:
    def __init__(self, rover_model_path=None, general_model_path=None):
        """
        Initialize the detection pipeline.

        - rover_model_path:   Path to fine-tuned rover YOLO model (best.pt)
        - general_model_path: Path to general-purpose YOLO model (yolov8n.pt)
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # ── Primary: Fine-tuned rover detector ────────────────────────
        self.rover_net = None
        if rover_model_path is None:
            rover_model_path = os.path.join(script_dir, "best.pt")

        if YOLO_AVAILABLE and os.path.exists(rover_model_path):
            print(f"[CV] Loading fine-tuned rover model: {rover_model_path}")
            self.rover_net = YOLO(rover_model_path)
            print("[CV] Rover model loaded!")
        else:
            if not YOLO_AVAILABLE:
                print("[CV] ERROR: ultralytics not installed!")
            elif not os.path.exists(rover_model_path):
                print(f"[CV] WARNING: Rover model not found at {rover_model_path}")
                print("[CV]          Run train_rover.py first to generate best.pt")

        # ── Optional: General object detector ─────────────────────────
        self.general_net = None
        if general_model_path is None:
            general_model_path = os.path.join(script_dir, "yolov8n.pt")

        if YOLO_AVAILABLE and os.path.exists(general_model_path):
            print(f"[CV] General model available: {general_model_path}")
            # Lazy-load: only actually load when user enables it
            self._general_model_path = general_model_path
        else:
            self._general_model_path = None

        self.general_detection_enabled = False

        # ── Smoothing ─────────────────────────────────────────────────
        self.prev_box = None
        self.miss_count = 0

        print("[CV] Pipeline ready. Fine-tuned YOLO rover detection active.")

    # ══════════════════════════════════════════════════════════════════════
    #  TOGGLE GENERAL DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def set_general_detection(self, enabled: bool):
        """Enable/disable general object detection (bottles, umbrellas, etc.)."""
        self.general_detection_enabled = enabled
        if enabled and self.general_net is None and self._general_model_path:
            print("[CV] Loading general detection model...")
            self.general_net = YOLO(self._general_model_path)
            print("[CV] General model loaded.")

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def detect_objects(self, base64_img: str, confidence_threshold: float = 0.3):
        """
        Detect the rover (and optionally other objects) in a base64-encoded image.

        Returns:
            {
                "detections": [...],
                "general_detections": [...],  # only if general detection is on
                "original_size": {"width": w, "height": h},
            }
        """
        try:
            # ── Decode image ──────────────────────────────────────────
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            img_data = base64.b64decode(base64_img)
            nparr    = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Could not decode image", "detections": []}

            h, w = img.shape[:2]
            rover = None
            general_dets = []

            # ──────────────────────────────────────────────────────────
            #  ROVER DETECTION (Fine-tuned YOLO)
            # ──────────────────────────────────────────────────────────
            if self.rover_net is not None:
                results = self.rover_net(
                    img,
                    conf=max(0.15, confidence_threshold),
                    verbose=False
                )

                if results and len(results) > 0:
                    result = results[0]
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs      = result.boxes.conf.cpu().numpy()
                    classes    = result.boxes.cls.cpu().numpy()

                    best = None
                    best_conf = 0

                    for box, conf, cls_idx in zip(boxes_xyxy, confs, classes):
                        label = result.names[int(cls_idx)]
                        bw = int(box[2] - box[0])
                        bh = int(box[3] - box[1])
                        area_ratio = (bw * bh) / (w * h) if (w * h) > 0 else 0

                        # Filter out impossibly small or huge detections
                        if area_ratio < 0.003 or area_ratio > 0.50:
                            continue

                        if conf > best_conf:
                            best_conf = conf
                            best = {
                                "label":      f"ROVER ({int(conf*100)}%)",
                                "confidence": float(conf),
                                "box":        {
                                    "x": int(box[0]), "y": int(box[1]),
                                    "width": bw, "height": bh
                                },
                                "color":      [0, 255, 100]   # green
                            }

                    rover = best

            # ──────────────────────────────────────────────────────────
            #  GENERAL OBJECT DETECTION (Optional, user-toggled)
            # ──────────────────────────────────────────────────────────
            if self.general_detection_enabled and self.general_net is not None:
                gen_results = self.general_net(
                    img,
                    conf=max(0.25, confidence_threshold),
                    verbose=False
                )

                if gen_results and len(gen_results) > 0:
                    gen_result = gen_results[0]
                    gen_boxes  = gen_result.boxes.xyxy.cpu().numpy()
                    gen_confs  = gen_result.boxes.conf.cpu().numpy()
                    gen_classes = gen_result.boxes.cls.cpu().numpy()

                    for box, conf, cls_idx in zip(gen_boxes, gen_confs, gen_classes):
                        label = gen_result.names[int(cls_idx)]
                        bw = int(box[2] - box[0])
                        bh = int(box[3] - box[1])

                        general_dets.append({
                            "label":      f"{label} ({int(conf*100)}%)",
                            "confidence": float(conf),
                            "box":        {
                                "x": int(box[0]), "y": int(box[1]),
                                "width": bw, "height": bh
                            },
                            "color":      [167, 139, 250]   # purple for general
                        })

            # ── Smooth rover box ──────────────────────────────────────
            if rover is not None:
                rover = self._smooth(rover)
                self.miss_count = 0
            else:
                self.miss_count += 1

            detections = [rover] if rover is not None else []
            result_data = {
                "detections": detections,
                "original_size": {"width": w, "height": h},
            }

            if self.general_detection_enabled:
                result_data["general_detections"] = general_dets

            return result_data

        except Exception as e:
            print(f"[CV] Detection error: {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e), "detections": []}

    # ══════════════════════════════════════════════════════════════════════
    #  EMA SMOOTHING
    # ══════════════════════════════════════════════════════════════════════
    def _smooth(self, det):
        box = det["box"]
        if self.prev_box is None:
            self.prev_box = box
            return det

        a = 0.6
        smoothed = {
            "x":      int(a * box["x"]      + (1-a) * self.prev_box["x"]),
            "y":      int(a * box["y"]      + (1-a) * self.prev_box["y"]),
            "width":  int(a * box["width"]  + (1-a) * self.prev_box["width"]),
            "height": int(a * box["height"] + (1-a) * self.prev_box["height"]),
        }
        self.prev_box = smoothed
        det["box"] = smoothed
        return det

    # ══════════════════════════════════════════════════════════════════════
    #  GEOFENCE CHECK
    # ══════════════════════════════════════════════════════════════════════
    def check_geofence(self, detections, grid):
        if not grid or not detections:
            return "ROVER NOT DETECTED" if not detections else None

        x1, y1 = grid["x1"], grid["y1"]
        x2, y2 = grid["x2"], grid["y2"]

        rover = detections[0]
        box   = rover["box"]
        cx    = box["x"] + box["width"]  / 2
        cy    = box["y"] + box["height"] / 2

        if cx < x1: return "OUT OF BOUNDS: STEER RIGHT"
        if cx > x2: return "OUT OF BOUNDS: STEER LEFT"
        if cy < y1: return "OUT OF BOUNDS: REVERSE"
        if cy > y2: return "OUT OF BOUNDS: MOVE FORWARD"

        return "IN BOUNDS: CONTINUE SWEEP"

    # ══════════════════════════════════════════════════════════════════════
    #  FLOOR GRID DETECTION (Improved)
    # ══════════════════════════════════════════════════════════════════════
    def detect_floor_grid(self, base64_img: str):
        try:
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            img_data = base64.b64decode(base64_img)
            nparr    = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Could not decode image", "lines": []}

            h, w   = img.shape[:2]
            gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ── Improved preprocessing ────────────────────────────────
            # CLAHE for better contrast in varying lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Bilateral filter preserves edges better than Gaussian
            blur = cv2.bilateralFilter(gray, 9, 75, 75)

            # Adaptive threshold + Canny for robust edge detection
            edges_canny = cv2.Canny(blur, 40, 120, apertureSize=3)

            # Also try adaptive threshold for tile edges
            adaptive = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Combine both edge maps
            edges = cv2.bitwise_or(edges_canny, adaptive)

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            edges = cv2.dilate(edges, kernel, iterations=1)

            # ── Hough Lines with better parameters ────────────────────
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=80,
                minLineLength=w // 5,    # shorter minimum for tile lines
                maxLineGap=40
            )

            detected = []
            if lines is not None:
                # Group lines by angle (horizontal vs vertical)
                h_lines = []
                v_lines = []

                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    angle = abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180.0 / np.pi)

                    if angle < 20 or angle > 160:
                        h_lines.append((lx1, ly1, lx2, ly2))
                    elif abs(angle - 90) < 20:
                        v_lines.append((lx1, ly1, lx2, ly2))

                # Merge nearby parallel lines (dedup)
                h_lines = self._merge_lines(h_lines, axis='h', threshold=15)
                v_lines = self._merge_lines(v_lines, axis='v', threshold=15)

                for lx1, ly1, lx2, ly2 in h_lines + v_lines:
                    detected.append({
                        "x1": int(lx1), "y1": int(ly1),
                        "x2": int(lx2), "y2": int(ly2)
                    })

            return {"lines": detected, "original_size": {"width": w, "height": h}}

        except Exception as e:
            print(f"[CV] Grid detection error: {e}")
            return {"error": str(e), "lines": []}

    def _merge_lines(self, lines, axis='h', threshold=15):
        """Merge nearby parallel lines to reduce noise."""
        if not lines:
            return []

        # Sort by position (y for horizontal, x for vertical)
        if axis == 'h':
            lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        else:
            lines.sort(key=lambda l: (l[0] + l[2]) / 2)

        merged = [lines[0]]
        for line in lines[1:]:
            prev = merged[-1]
            if axis == 'h':
                pos_curr = (line[1] + line[3]) / 2
                pos_prev = (prev[1] + prev[3]) / 2
            else:
                pos_curr = (line[0] + line[2]) / 2
                pos_prev = (prev[0] + prev[2]) / 2

            if abs(pos_curr - pos_prev) < threshold:
                # Merge: extend the previous line
                merged[-1] = (
                    min(prev[0], line[0]),
                    min(prev[1], line[1]),
                    max(prev[2], line[2]),
                    max(prev[3], line[3])
                )
            else:
                merged.append(line)

        return merged
