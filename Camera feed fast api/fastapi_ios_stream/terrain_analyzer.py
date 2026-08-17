from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] TERRAIN-V2 | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("terrain_v2")


class PatchAutoencoder:
    """Single-hidden-layer autoencoder trained from scratch on THIS scan's patches."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 16,
                 lr: float = 0.05, momentum: float = 0.9, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, bottleneck_dim))
        self.b1 = np.zeros(bottleneck_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / bottleneck_dim), (bottleneck_dim, input_dim))
        self.b2 = np.zeros(input_dim)
        self.lr = lr
        self.momentum = momentum
        self._vW1 = np.zeros_like(self.W1); self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2); self._vb2 = np.zeros_like(self.b2)

    @staticmethod
    def _relu(x): return np.maximum(0, x)
    @staticmethod
    def _relu_deriv(x): return (x > 0).astype(x.dtype)
    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)
        return z1, a1, z2, a2

    def train(self, X, epochs=250):
        n = X.shape[0]
        for _ in range(epochs):
            z1, a1, z2, a2 = self.forward(X)
            d_a2 = (a2 - X) * (2.0 / n)
            d_z2 = d_a2 * a2 * (1 - a2)
            d_W2 = a1.T @ d_z2
            d_b2 = d_z2.sum(axis=0)
            d_a1 = d_z2 @ self.W2.T
            d_z1 = d_a1 * self._relu_deriv(z1)
            d_W1 = X.T @ d_z1
            d_b1 = d_z1.sum(axis=0)
            self._vW1 = self.momentum * self._vW1 - self.lr * d_W1
            self._vb1 = self.momentum * self._vb1 - self.lr * d_b1
            self._vW2 = self.momentum * self._vW2 - self.lr * d_W2
            self._vb2 = self.momentum * self._vb2 - self.lr * d_b2
            self.W1 += self._vW1; self.b1 += self._vb1
            self.W2 += self._vW2; self.b2 += self._vb2

    def reconstruction_error(self, X):
        _, _, _, recon = self.forward(X)
        return np.mean((X - recon) ** 2, axis=1)


@dataclass
class PatchResult:
    grid_row: int; grid_col: int
    pixel_x: int; pixel_y: int
    norm_x: float; norm_y: float
    world_x_m: float; world_y_m: float
    anomaly_score: float
    surface_cluster: int
    surface_label: str
    life_score: float
    liquid_score: float
    relative_elevation: float
    is_primary_anomaly: bool = False


@dataclass
class MissionScan:
    image_path: str
    image_width: int; image_height: int
    patch_size: int
    grid_shape: Tuple[int, int]
    field_width_m: float; field_height_m: float
    meters_per_patch: float
    surface_class_names: Dict[int, str]
    patches: List[PatchResult] = field(default_factory=list)
    field_life_percent: float = 0.0
    field_liquid_percent: float = 0.0
    processing_time_s: float = 0.0

    def primary_poi(self) -> Optional[PatchResult]:
        return next((p for p in self.patches if p.is_primary_anomaly), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class UnsupervisedTerrainAnalyzer:
    def __init__(self, patch_size=32, max_dim=1000, n_surface_clusters=5,
                 ae_bottleneck=16, ae_epochs=250, top_n_anomalies=8,
                 field_width_m=10.0, field_height_m=10.0, random_state=42):
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.n_surface_clusters = n_surface_clusters
        self.ae_bottleneck = ae_bottleneck
        self.ae_epochs = ae_epochs
        self.top_n_anomalies = top_n_anomalies
        self.field_width_m = field_width_m
        self.field_height_m = field_height_m
        self.random_state = random_state

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement to improve local contrast for aerial/distant captures."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def analyze(self, image_input: Union[str, np.ndarray], enhance_input: bool = True):
        t0 = time.time()
        if isinstance(image_input, (str, Path)):
            image = self._load_image(image_input)
            image_path_str = str(image_input)
        else:
            image = image_input.copy()
            image_path_str = "memory_buffer"
        
        image = self._resize_if_needed(image)
        
        # Apply CLAHE enhancement for aerial/height camera captures
        if enhance_input:
            image = self._apply_clahe(image)
        h, w = image.shape[:2]
        ps = self.patch_size
        rows, cols = h // ps, w // ps
        if rows < 2 or cols < 2:
            raise ValueError(f"Image too small ({w}x{h}) for patch_size={ps}")

        m_per_patch = min(self.field_width_m / cols, self.field_height_m / rows)
        log.info(f"Scanning {w}x{h}px as {rows}x{cols} grid (~{m_per_patch:.2f} m/patch)")

        stat_features, patch_thumbs, centers = self._extract_patches(image, rows, cols)
        anomaly_scores = self._detect_anomalies(patch_thumbs)
        cluster_ids, cluster_names = self._cluster_surfaces(stat_features)
        life_scores, liquid_scores = self._score_life_liquid(image, rows, cols)
        elevation_map = self._estimate_relative_elevation(image)

        patches = self._assemble_patches(rows, cols, centers, w, h, m_per_patch,
            anomaly_scores, cluster_ids, cluster_names, life_scores, liquid_scores, elevation_map, ps)

        scan = MissionScan(
            image_path=image_path_str, image_width=w, image_height=h, patch_size=ps,
            grid_shape=(rows, cols), field_width_m=self.field_width_m, field_height_m=self.field_height_m,
            meters_per_patch=round(m_per_patch, 3), surface_class_names=cluster_names, patches=patches,
            field_life_percent=round(100.0 * np.mean([p.life_score for p in patches]), 2),
            field_liquid_percent=round(100.0 * np.mean([p.liquid_score for p in patches]), 2),
            processing_time_s=round(time.time() - t0, 3),
        )

        images = self._render_all(image, scan, elevation_map, cluster_ids, rows, cols)

        log.info(f"Scan complete in {scan.processing_time_s}s | life~{scan.field_life_percent}% | "
                 f"liquid~{scan.field_liquid_percent}% | primary anomaly at "
                 f"({scan.primary_poi().world_x_m:.2f}m, {scan.primary_poi().world_y_m:.2f}m)")
        return scan, images

    def _load_image(self, image_path):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Survey image not found: {image_path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode '{image_path}'.")
        return image

    def _resize_if_needed(self, image):
        h, w = image.shape[:2]
        if max(h, w) <= self.max_dim:
            return image
        scale = self.max_dim / float(max(h, w))
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _extract_patches(self, image, rows, cols):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ps = self.patch_size
        thumb_size = 12
        stat_features, thumbs, centers = [], [], []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * ps, (r + 1) * ps
                x0, x1 = c * ps, (c + 1) * ps
                patch_bgr = image[y0:y1, x0:x1]
                hsv_patch = hsv[y0:y1, x0:x1]
                gray_patch = gray[y0:y1, x0:x1]
                mean_hsv = hsv_patch.reshape(-1, 3).mean(axis=0)
                std_hsv = hsv_patch.reshape(-1, 3).std(axis=0)
                sx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sx ** 2 + sy ** 2)
                edge_density = float(grad_mag.mean())
                roughness = float(grad_mag.std())
                local_contrast = float(gray_patch.std())
                stat_features.append(np.concatenate([mean_hsv, std_hsv, [edge_density, roughness, local_contrast]]))
                thumb = cv2.resize(patch_bgr, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                thumbs.append((thumb.astype(np.float64) / 255.0).flatten())
                centers.append((x0 + ps // 2, y0 + ps // 2))
        return np.array(stat_features), np.array(thumbs), centers

    def _detect_anomalies(self, patch_thumbs):
        ae = PatchAutoencoder(input_dim=patch_thumbs.shape[1], bottleneck_dim=self.ae_bottleneck, seed=self.random_state)
        ae.train(patch_thumbs, epochs=self.ae_epochs)
        errors = ae.reconstruction_error(patch_thumbs)
        lo, hi = errors.min(), errors.max()
        if hi - lo < 1e-9:
            return np.zeros_like(errors)
        return (errors - lo) / (hi - lo)

    def _cluster_surfaces(self, stat_features):
        k = min(self.n_surface_clusters, len(stat_features))
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(stat_features)
        names = {}
        for cid in range(k):
            h, s, v = km.cluster_centers_[cid][0:3]
            edge_density = km.cluster_centers_[cid][6]
            if s < 40 and v > 150 and edge_density < 15:
                names[cid] = "Smooth / Reflective (possible liquid or hardpan)"
            elif 35 < h < 85 and s > 60:
                names[cid] = "Vegetation-like (green hue, moderate saturation)"
            elif edge_density > 35:
                names[cid] = "Rocky / Rough (high texture gradient)"
            elif v < 90:
                names[cid] = "Shadowed / Low-albedo region"
            else:
                names[cid] = "Sandy / Fine-grained (low texture, warm tone)"
        return labels, names

    def _score_life_liquid(self, image, rows, cols):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sx ** 2 + sy ** 2)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        life_mask = ((hue > 35) & (hue < 85) & (sat > 60) & (val > 50)).astype(np.float64)
        liquid_mask = ((val > 190) & (edge_map < 15) & (hue > 85) & (hue < 140)).astype(np.float64)
        ps = self.patch_size
        life_scores = np.zeros(rows * cols)
        liquid_scores = np.zeros(rows * cols)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * ps, (r + 1) * ps
                x0, x1 = c * ps, (c + 1) * ps
                life_scores[idx] = life_mask[y0:y1, x0:x1].mean()
                liquid_scores[idx] = liquid_mask[y0:y1, x0:x1].mean()
                idx += 1
        return life_scores, liquid_scores

    def _estimate_relative_elevation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0).astype(np.float64)
        sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sx ** 2 + sy ** 2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX)
        blended = 0.7 * macro + 0.3 * micro
        blended = cv2.normalize(blended, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return blended

    def _assemble_patches(self, rows, cols, centers, img_w, img_h, m_per_patch,
                           anomaly_scores, cluster_ids, cluster_names,
                           life_scores, liquid_scores, elevation_map, ps):
        results = []
        for i, (px, py) in enumerate(centers):
            r, c = divmod(i, cols)
            elev_patch = elevation_map[r * ps:(r + 1) * ps, c * ps:(c + 1) * ps]
            results.append(PatchResult(
                grid_row=r, grid_col=c, pixel_x=px, pixel_y=py,
                norm_x=round(px / img_w, 4), norm_y=round(py / img_h, 4),
                world_x_m=round((px / img_w) * self.field_width_m, 3),
                world_y_m=round((py / img_h) * self.field_height_m, 3),
                anomaly_score=round(float(anomaly_scores[i]), 4),
                surface_cluster=int(cluster_ids[i]), surface_label=cluster_names[int(cluster_ids[i])],
                life_score=round(float(life_scores[i]), 4), liquid_score=round(float(liquid_scores[i]), 4),
                relative_elevation=round(float(elev_patch.mean()), 4),
            ))
        top_idx = np.argsort(anomaly_scores)[::-1][: self.top_n_anomalies]
        for rank, idx in enumerate(top_idx):
            results[idx].is_primary_anomaly = (rank == 0)
        return results

    def _render_all(self, image, scan, elevation_map, cluster_ids, rows, cols):
        images = {}
        images["elevation"] = self._render_elevation(elevation_map)
        images["classification"] = self._render_classification(image, cluster_ids, rows, cols, scan)
        images["life_liquid"] = self._render_life_liquid(image, scan)
        images["overlay"] = self._render_anomaly_overlay(image, scan)
        images["dashboard"] = self._render_dashboard(images)
        return images

    def _render_elevation(self, elevation_map):
        elev_u8 = (elevation_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(elev_u8, cv2.COLORMAP_TURBO)
        self._draw_scale_bar(colored, self.field_width_m, colored.shape[1])
        self._draw_colorbar_legend(colored, "REL. ELEVATION", "low", "high", cv2.COLORMAP_TURBO)
        return colored

    def _render_classification(self, image, cluster_ids, rows, cols, scan):
        overlay = image.copy()
        palette = self._distinct_colors(len(scan.surface_class_names))
        ps = self.patch_size
        idx = 0
        for r in range(rows):
            for c in range(cols):
                color = palette[cluster_ids[idx]]
                x0, y0 = c * ps, r * ps
                cv2.rectangle(overlay, (x0, y0), (x0 + ps, y0 + ps), color, -1)
                idx += 1
        blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
        self._draw_class_legend(blended, scan.surface_class_names, palette)
        self._draw_scale_bar(blended, self.field_width_m, blended.shape[1])
        return blended

    def _render_life_liquid(self, image, scan):
        overlay = image.copy()
        for p in scan.patches:
            x0 = p.pixel_x - self.patch_size // 2
            y0 = p.pixel_y - self.patch_size // 2
            if p.liquid_score > 0.4:
                cv2.rectangle(overlay, (x0, y0), (x0 + self.patch_size, y0 + self.patch_size), (255, 120, 0), -1)
            elif p.life_score > 0.4:
                cv2.rectangle(overlay, (x0, y0), (x0 + self.patch_size, y0 + self.patch_size), (0, 200, 0), -1)
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        legend_items = [("Liquid-like (heuristic)", (255, 120, 0)), ("Vegetation-like (heuristic)", (0, 200, 0))]
        self._draw_simple_legend(blended, legend_items)
        self._draw_scale_bar(blended, self.field_width_m, blended.shape[1])
        return blended

    def _render_anomaly_overlay(self, image, scan):
        overlay = image.copy()
        for p in scan.patches:
            if p.anomaly_score < 0.5 and not p.is_primary_anomaly:
                continue
            color = (0, 0, 255) if p.is_primary_anomaly else (0, 140, 255)
            radius = int(10 + 10 * p.anomaly_score)
            cv2.circle(overlay, (p.pixel_x, p.pixel_y), radius, color, 2)
        primary = scan.primary_poi()
        if primary:
            cv2.putText(overlay, "PRIMARY POI", (primary.pixel_x + 14, primary.pixel_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        legend_items = [("Primary anomaly", (0, 0, 255)), ("Secondary anomaly", (0, 140, 255))]
        self._draw_simple_legend(overlay, legend_items)
        self._draw_scale_bar(overlay, self.field_width_m, overlay.shape[1])
        return overlay

    def _render_dashboard(self, images):
        h, w = images["overlay"].shape[:2]
        panels = [images["elevation"], images["classification"], images["life_liquid"], images["overlay"]]
        resized = [cv2.resize(p, (w, h)) for p in panels]
        top = np.hstack([resized[0], resized[1]])
        bottom = np.hstack([resized[2], resized[3]])
        return np.vstack([top, bottom])

    def _distinct_colors(self, n):
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            hsv = np.uint8([[[hue, 200, 230]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors

    def _draw_scale_bar(self, img, field_width_m, img_w_px):
        px_per_m = img_w_px / field_width_m
        bar_len_px = int(px_per_m * 1.0)
        h = img.shape[0]
        x0, y0 = 20, h - 30
        cv2.line(img, (x0, y0), (x0 + bar_len_px, y0), (255, 255, 255), 3)
        cv2.line(img, (x0, y0 - 6), (x0, y0 + 6), (255, 255, 255), 2)
        cv2.line(img, (x0 + bar_len_px, y0 - 6), (x0 + bar_len_px, y0 + 6), (255, 255, 255), 2)
        cv2.putText(img, "1 m", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_colorbar_legend(self, img, title, lo_label, hi_label, colormap):
        bar_h, bar_w = 150, 18
        x0, y0 = img.shape[1] - bar_w - 25, 25
        gradient = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
        gradient = np.repeat(gradient, bar_w, axis=1)
        gradient_colored = cv2.applyColorMap(gradient, colormap)
        img[y0:y0 + bar_h, x0:x0 + bar_w] = gradient_colored
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
        cv2.putText(img, title, (x0 - 90, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, hi_label, (x0 - 30, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, lo_label, (x0 - 30, y0 + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_class_legend(self, img, names, palette):
        x0, y0 = 12, 20
        for cid, label in names.items():
            color = palette[cid]
            cv2.rectangle(img, (x0, y0), (x0 + 14, y0 + 14), color, -1)
            cv2.rectangle(img, (x0, y0), (x0 + 14, y0 + 14), (255, 255, 255), 1)
            cv2.putText(img, label, (x0 + 20, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 20

    def _draw_simple_legend(self, img, items):
        x0, y0 = 12, 20
        for label, color in items:
            cv2.rectangle(img, (x0, y0), (x0 + 14, y0 + 14), color, -1)
            cv2.rectangle(img, (x0, y0), (x0 + 14, y0 + 14), (255, 255, 255), 1)
            cv2.putText(img, label, (x0 + 20, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 20

# print("UnsupervisedTerrainAnalyzer loaded.")
