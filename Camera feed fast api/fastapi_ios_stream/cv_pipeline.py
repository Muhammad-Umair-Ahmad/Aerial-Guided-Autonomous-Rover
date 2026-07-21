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

        # ── YOLO ──────────────────────────────────────────────────────────
        self.net = None
        if YOLO_AVAILABLE:
            print(f"[CV] Loading YOLO model {model_name}...")
            self.net = YOLO(model_name)
            print("[CV] YOLO model loaded.")
        else:
            print("[CV] WARNING: ultralytics not installed. pip install ultralytics")

        # ── ArUco (highest priority — print a marker, tape it on) ─────────
        try:
            self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, 'ArucoDetector'):
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            else:
                self.aruco_detector = None
            print("[CV] ArUco engine initialized.")
        except Exception as e:
            print(f"[CV] ArUco init failed: {e}")
            self.aruco_dict = None

        # ── KNN Background Subtractor (PRIMARY — works great for fixed cam) ─
        # history=500  : how many frames it uses to build the background model
        # dist2Threshold: pixel-color distance to call something foreground
        # detectShadows=False: don't mark shadows as foreground (saves noise)
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=500,
            dist2Threshold=400.0,
            detectShadows=False
        )
        self.knn_warmup_frames = 0          # counts how many frames we've fed in
        self.KNN_WARMUP = 30                # frames before we start trusting KNN output
        print("[CV] KNN background subtractor initialized.")

        # ── Template Matching (SECONDARY) ─────────────────────────────────
        self.template       = None
        self.template_gray  = None
        self.template_sizes = []            # multiple scales for multi-scale matching

        # Try to load rover_template.jpg that lives next to this file
        template_path = os.path.join(os.path.dirname(__file__), 'rover_template.jpg')
        if os.path.exists(template_path):
            try:
                print("[CV] Loading rover_template.jpg for template matching...")
                tpl = cv2.imread(template_path)
                if tpl is not None:
                    self.template      = tpl
                    self.template_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                    th, tw             = self.template_gray.shape[:2]
                    # Pre-generate templates at multiple scales so matching works
                    # regardless of the camera height / rover distance
                    for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                        new_w = max(10, int(tw * scale))
                        new_h = max(10, int(th * scale))
                        self.template_sizes.append(
                            cv2.resize(self.template_gray, (new_w, new_h))
                        )
                    print(f"[CV] Template loaded ({tw}x{th}px). {len(self.template_sizes)} scale variants.")
            except Exception as e:
                print(f"[CV] Template load failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN DETECTION ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════
    def detect_objects(self, base64_img: str, confidence_threshold: float = 0.6):
        """
        4-stage pipeline (highest → lowest priority):
          1. ArUco  — deterministic, needs a printed marker taped to rover
          2. KNN    — background subtraction; best for fixed overhead cam
          3. Template — multi-scale normalized cross-correlation vs rover_template.jpg
          4. YOLO   — neural net at the user-set confidence threshold (default 60%)

        Returns a single 'rover' detection (the best one found) plus original_size.
        """
        try:
            # ── Decode image ──────────────────────────────────────────────
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            img_data = base64.b64decode(base64_img)
            nparr    = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Could not decode image", "detections": []}

            h, w     = img.shape[:2]
            gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rover    = None          # will hold the winning detection dict

            # ────────────────────────────────────────────────────────────
            # STAGE 1 — ArUco  (green box, confidence 1.0)
            # ────────────────────────────────────────────────────────────
            if self.aruco_dict is not None:
                if self.aruco_detector is not None:
                    corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params)

                if ids is not None and len(ids) > 0:
                    c    = corners[0][0]
                    mn_x = int(np.min(c[:, 0]));  mx_x = int(np.max(c[:, 0]))
                    mn_y = int(np.min(c[:, 1]));  mx_y = int(np.max(c[:, 1]))
                    rover = {
                        "label":      f"ROVER (ArUco #{ids[0][0]})",
                        "confidence": 1.0,
                        "box":        {"x": mn_x, "y": mn_y,
                                       "width": mx_x - mn_x, "height": mx_y - mn_y},
                        "color":      [0, 255, 0]      # green
                    }

            # ────────────────────────────────────────────────────────────
            # STAGE 2 — KNN Background Subtraction  (cyan box, conf 0.90)
            # Best method for a FIXED overhead camera — camera never moves,
            # so the floor is the stable background and the rover is the
            # only moving / foreground object.
            # ────────────────────────────────────────────────────────────
            if rover is None:
                fg_mask = self.bg_subtractor.apply(img)
                self.knn_warmup_frames += 1

                if self.knn_warmup_frames >= self.KNN_WARMUP:
                    # Clean up the mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

                    # Size guard: rover should be 1%–35% of the frame
                    MIN_A = (w * h) * 0.010
                    MAX_A = (w * h) * 0.35
                    valid = [c for c in contours
                             if MIN_A < cv2.contourArea(c) < MAX_A]

                    if valid:
                        largest   = max(valid, key=cv2.contourArea)
                        rx, ry, rw, rh = cv2.boundingRect(largest)
                        # Small padding so the box doesn't clip the rover edges
                        pad = 8
                        rx  = max(0, rx - pad);  ry  = max(0, ry - pad)
                        rw  = min(w - rx, rw + pad * 2)
                        rh  = min(h - ry, rh + pad * 2)
                        rover = {
                            "label":      "ROVER (KNN)",
                            "confidence": 0.90,
                            "box":        {"x": rx, "y": ry,
                                           "width": rw, "height": rh},
                            "color":      [0, 220, 255]    # cyan
                        }

            # ────────────────────────────────────────────────────────────
            # STAGE 3 — Multi-scale Template Matching  (magenta, conf 0.80)
            # Compares every frame against rover_template.jpg at 7 scales.
            # Works even if the rover is stationary (unlike KNN).
            # ────────────────────────────────────────────────────────────
            if rover is None and self.template_sizes:
                best_val  = -1.0
                best_loc  = None
                best_size = None

                for tpl in self.template_sizes:
                    th, tw = tpl.shape[:2]
                    if tw > w or th > h:
                        continue
                    result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_val:
                        best_val  = max_val
                        best_loc  = max_loc
                        best_size = (tw, th)

                # Only accept if similarity is reasonable (≥ 0.35)
                if best_val >= 0.35 and best_loc is not None:
                    tx, ty = best_loc
                    tw, th = best_size
                    rover = {
                        "label":      f"ROVER (Template {int(best_val*100)}%)",
                        "confidence": float(best_val),
                        "box":        {"x": tx, "y": ty,
                                       "width": tw, "height": th},
                        "color":      [255, 0, 255]    # magenta
                    }

            # ────────────────────────────────────────────────────────────
            # STAGE 4 — YOLO  (orange box, uses user-set threshold ~60%)
            # ────────────────────────────────────────────────────────────
            if rover is None and self.net is not None:
                results = self.net(img, conf=confidence_threshold, verbose=False)

                # Classes a DIY rover might be detected as from overhead
                ROVER_CLASSES = {
                    "car", "truck", "bus", "motorcycle", "bicycle",
                    "remote", "cell phone", "mouse", "keyboard",
                    "book", "clock", "bottle", "cup", "suitcase",
                    "backpack", "skateboard", "sports ball", "frisbee"
                }

                if results and len(results) > 0:
                    result     = results[0]
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs      = result.boxes.conf.cpu().numpy()
                    classes    = result.boxes.cls.cpu().numpy()

                    priority = []
                    generic  = []
                    for box, conf, cls_idx in zip(boxes_xyxy, confs, classes):
                        label = result.names[int(cls_idx)]
                        entry = {
                            "label":      label,
                            "confidence": float(conf),
                            "box": {
                                "x":      int(box[0]),
                                "y":      int(box[1]),
                                "width":  int(box[2] - box[0]),
                                "height": int(box[3] - box[1])
                            },
                            "color": [255, 140, 0]   # orange
                        }
                        (priority if label in ROVER_CLASSES else generic).append(entry)

                    pool = priority if priority else generic
                    if pool:
                        rover = max(pool, key=lambda x: x["confidence"])

            # Return exactly one detection (the winner) or empty list
            detections = [rover] if rover is not None else []
            return {"detections": detections, "original_size": {"width": w, "height": h}}

        except Exception as e:
            print(f"[CV] Detection error: {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e), "detections": []}

    # ══════════════════════════════════════════════════════════════════════
    #  GEOFENCE CHECK
    # ══════════════════════════════════════════════════════════════════════
    def check_geofence(self, detections, grid):
        """
        Returns a direction instruction string.
        Uses CENTER of rover for in/out determination (not full box edges,
        which caused false positives when the rover was clearly inside).
        """
        if not grid or not detections:
            return "ROVER NOT DETECTED" if not detections else None

        x1, y1 = grid["x1"], grid["y1"]
        x2, y2 = grid["x2"], grid["y2"]

        rover = detections[0]   # always the single best detection
        box   = rover["box"]
        cx    = box["x"] + box["width"]  / 2
        cy    = box["y"] + box["height"] / 2

        if cx < x1: return "OUT OF BOUNDS: STEER RIGHT"
        if cx > x2: return "OUT OF BOUNDS: STEER LEFT"
        if cy < y1: return "OUT OF BOUNDS: REVERSE"
        if cy > y2: return "OUT OF BOUNDS: MOVE FORWARD"

        return "IN BOUNDS: CONTINUE SWEEP"

    # ══════════════════════════════════════════════════════════════════════
    #  FLOOR GRID DETECTION (unchanged)
    # ══════════════════════════════════════════════════════════════════════
    def detect_floor_grid(self, base64_img: str):
        """
        Detects tile / grout lines using Canny + Probabilistic Hough Transform.
        """
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
            blur   = cv2.GaussianBlur(gray, (5, 5), 0)
            edges  = cv2.Canny(blur, 50, 150, apertureSize=3)
            kernel = np.ones((3, 3), np.uint8)
            edges  = cv2.dilate(edges, kernel, iterations=1)

            lines  = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                     threshold=100,
                                     minLineLength=w // 4,
                                     maxLineGap=50)
            detected = []
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    angle = abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180.0 / np.pi)
                    if (angle < 15 or angle > 165) or abs(angle - 90) < 15:
                        detected.append({"x1": int(lx1), "y1": int(ly1),
                                         "x2": int(lx2), "y2": int(ly2)})

            return {"lines": detected, "original_size": {"width": w, "height": h}}

        except Exception as e:
            print(f"[CV] Grid detection error: {e}")
            return {"error": str(e), "lines": []}
