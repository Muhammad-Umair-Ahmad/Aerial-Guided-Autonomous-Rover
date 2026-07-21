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

        # ── YOLO (only used as initial hint before calibration) ───────
        self.net = None
        if YOLO_AVAILABLE:
            print(f"[CV] Loading YOLO model {model_name}...")
            self.net = YOLO(model_name)
            print("[CV] YOLO loaded (pre-calibration fallback only).")
        else:
            print("[CV] YOLO not available — calibration mode required.")

        # ══════════════════════════════════════════════════════════════
        #  TRACKER STATE
        # ══════════════════════════════════════════════════════════════
        self.calibrated = False
        self.tracker = None             # OpenCV object tracker
        self.rover_box = None           # (x, y, w, h) in video pixels
        self.rover_features = None      # ORB descriptors for re-acquisition
        self.rover_histogram = None     # Color histogram for re-acquisition
        self.rover_size = None          # (w, h) — calibrated size of rover
        self.track_fail_count = 0       # consecutive tracker failures
        self.orb = cv2.ORB_create(500)  # ORB for re-acquisition

        # ── Smoothing ─────────────────────────────────────────────────
        self.prev_box = None
        self.miss_count = 0

        print("[CV] Tracker pipeline ready. Waiting for calibration...")

    # ══════════════════════════════════════════════════════════════════════
    #  CALIBRATION — user selects the rover once
    # ══════════════════════════════════════════════════════════════════════
    def calibrate(self, base64_img: str, roi: dict):
        """
        User drew a box around the rover. We extract its features
        and initialize a tracker.

        roi = {"x": int, "y": int, "width": int, "height": int}
        """
        try:
            if "," in base64_img:
                base64_img = base64_img.split(",")[1]

            img_data = base64.b64decode(base64_img)
            nparr    = np.frombuffer(img_data, np.uint8)
            img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Could not decode calibration image"}

            h, w = img.shape[:2]
            rx = max(0, int(roi["x"]))
            ry = max(0, int(roi["y"]))
            rw = min(w - rx, int(roi["width"]))
            rh = min(h - ry, int(roi["height"]))

            if rw < 10 or rh < 10:
                return {"error": "ROI too small — draw a bigger box around the rover"}

            # ── Save calibrated size (used to validate future detections) ─
            self.rover_size = (rw, rh)
            self.rover_box  = (rx, ry, rw, rh)

            # ── Extract ORB features for re-acquisition ───────────────
            roi_img  = img[ry:ry+rh, rx:rx+rw]
            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            kp, des  = self.orb.detectAndCompute(roi_gray, None)
            self.rover_features = des

            # ── Extract color histogram for re-acquisition ────────────
            hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            self.rover_histogram = cv2.calcHist(
                [hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256]
            )
            cv2.normalize(self.rover_histogram, self.rover_histogram, 0, 255, cv2.NORM_MINMAX)

            # ── Initialize CSRT tracker ───────────────────────────────
            self._init_tracker(img, (rx, ry, rw, rh))

            self.calibrated = True
            self.track_fail_count = 0
            self.miss_count = 0
            self.prev_box = None

            print(f"[CV] ✅ Calibrated! Rover size: {rw}x{rh}px, "
                  f"ORB features: {len(kp) if kp else 0}")

            return {
                "status": "calibrated",
                "rover_size": {"width": rw, "height": rh},
                "roi": {"x": rx, "y": ry, "width": rw, "height": rh}
            }

        except Exception as e:
            print(f"[CV] Calibration error: {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e)}

    def _init_tracker(self, frame, bbox):
        """Initialize or re-initialize the OpenCV tracker."""
        # OpenCV 4.13 — use the best available deep learning tracker
        tracker_created = False

        # Try TrackerVit first (Vision Transformer — best quality)
        if not tracker_created:
            try:
                self.tracker = cv2.TrackerVit_create()
                tracker_created = True
                print("[CV] Using TrackerVit (Vision Transformer)")
            except (AttributeError, cv2.error):
                pass

        # Try DaSiamRPN (deep Siamese tracker — very good)
        if not tracker_created:
            try:
                self.tracker = cv2.TrackerDaSiamRPN_create()
                tracker_created = True
                print("[CV] Using TrackerDaSiamRPN")
            except (AttributeError, cv2.error):
                pass

        # Fallback to MIL (basic, always available)
        if not tracker_created:
            try:
                self.tracker = cv2.TrackerMIL_create()
                tracker_created = True
                print("[CV] Using TrackerMIL (fallback)")
            except (AttributeError, cv2.error):
                print("[CV] ERROR: No tracker available!")
                return

        self.tracker.init(frame, bbox)

    # ══════════════════════════════════════════════════════════════════════
    #  RE-ACQUISITION — find the rover again when tracker loses it
    # ══════════════════════════════════════════════════════════════════════
    def _reacquire(self, img, frame_w, frame_h):
        """
        Try to find the rover using color histogram backprojection + ORB.
        Returns (x, y, w, h) or None.
        """
        if self.rover_histogram is None:
            return None

        # ── Color histogram backprojection ────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject(
            [hsv], [0, 1], self.rover_histogram, [0, 180, 0, 256], 1
        )

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        backproj = cv2.morphologyEx(backproj, cv2.MORPH_CLOSE, kernel, iterations=2)
        backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel, iterations=1)
        _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(backproj, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # ── Filter by size (must be close to calibrated rover size) ───
        cal_w, cal_h = self.rover_size
        cal_area = cal_w * cal_h
        best = None
        best_score = 0

        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh

            # Size filter: must be 30%–250% of calibrated area
            if area < cal_area * 0.3 or area > cal_area * 2.5:
                continue

            # Score by how close the area matches the calibrated size
            size_score = 1.0 - abs(area - cal_area) / cal_area

            # Bonus for ORB feature matches in this region
            feature_score = 0
            if self.rover_features is not None and len(self.rover_features) > 0:
                roi_gray = cv2.cvtColor(img[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
                kp2, des2 = self.orb.detectAndCompute(roi_gray, None)
                if des2 is not None and len(des2) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(self.rover_features, des2)
                    good = [m for m in matches if m.distance < 60]
                    feature_score = len(good) / max(1, len(self.rover_features))

            total_score = size_score * 0.4 + feature_score * 0.6
            if total_score > best_score:
                best_score = total_score
                best = (rx, ry, rw, rh)

        # Only accept if we have a reasonable match
        if best is not None and best_score > 0.15:
            return best

        return None

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN DETECTION
    # ══════════════════════════════════════════════════════════════════════
    def detect_objects(self, base64_img: str, confidence_threshold: float = 0.3):
        """
        If calibrated: use OpenCV tracker (fast, accurate, knows the rover).
        If not calibrated: fall back to YOLO (hint only — unreliable).
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

            # ──────────────────────────────────────────────────────────
            #  CALIBRATED MODE — OpenCV Tracker
            # ──────────────────────────────────────────────────────────
            if self.calibrated and self.tracker is not None:
                success, bbox = self.tracker.update(img)

                if success:
                    bx, by, bw, bh = [int(v) for v in bbox]

                    # Sanity check: box must be reasonable size
                    cal_w, cal_h = self.rover_size
                    cal_area = cal_w * cal_h
                    box_area = bw * bh

                    if (box_area > cal_area * 0.2 and
                        box_area < cal_area * 3.0 and
                        bx >= 0 and by >= 0 and
                        bx + bw <= w and by + bh <= h):

                        rover = {
                            "label":      "ROVER (tracking)",
                            "confidence": 0.95,
                            "box":        {"x": bx, "y": by,
                                           "width": bw, "height": bh},
                            "color":      [0, 255, 100]   # green = tracked
                        }
                        self.track_fail_count = 0
                    else:
                        success = False  # box went bad

                if not success:
                    self.track_fail_count += 1
                    print(f"[CV] Tracker lost rover (fail #{self.track_fail_count})")

                    # Try re-acquisition after 3 consecutive failures
                    if self.track_fail_count >= 3:
                        print("[CV] Attempting re-acquisition...")
                        reacq = self._reacquire(img, w, h)
                        if reacq is not None:
                            rx, ry, rw, rh = reacq
                            # Re-initialize tracker with new position
                            self._init_tracker(img, (rx, ry, rw, rh))
                            self.track_fail_count = 0
                            rover = {
                                "label":      "ROVER (re-acquired)",
                                "confidence": 0.80,
                                "box":        {"x": rx, "y": ry,
                                               "width": rw, "height": rh},
                                "color":      [255, 165, 0]   # orange = re-acquired
                            }
                            print(f"[CV] ✅ Re-acquired rover at ({rx},{ry})")
                        else:
                            print("[CV] ❌ Re-acquisition failed — need recalibration")

                    # Hold last known position for a few frames
                    if rover is None and self.prev_box is not None and self.track_fail_count <= 5:
                        rover = {
                            "label":      "ROVER (last seen)",
                            "confidence": max(0.2, 0.7 - self.track_fail_count * 0.1),
                            "box":        self.prev_box,
                            "color":      [255, 140, 0]
                        }

            # ──────────────────────────────────────────────────────────
            #  UNCALIBRATED MODE — YOLO hint (unreliable, just a hint)
            # ──────────────────────────────────────────────────────────
            elif self.net is not None:
                results = self.net(img, conf=max(0.20, confidence_threshold),
                                   verbose=False)

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
                        if area_ratio < 0.005 or area_ratio > 0.40:
                            continue
                        if conf > best_conf:
                            best_conf = conf
                            best = {
                                "label":      f"⚠️ UNCALIBRATED ({label} {int(conf*100)}%)",
                                "confidence": float(conf),
                                "box":        {"x": int(box[0]), "y": int(box[1]),
                                               "width": bw, "height": bh},
                                "color":      [255, 60, 60]
                            }
                    rover = best

            # ── Smooth box ────────────────────────────────────────────
            if rover is not None:
                rover = self._smooth(rover)
                self.miss_count = 0
            else:
                self.miss_count += 1

            detections = [rover] if rover is not None else []
            return {
                "detections": detections,
                "original_size": {"width": w, "height": h},
                "calibrated": self.calibrated
            }

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
    #  FLOOR GRID DETECTION
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
