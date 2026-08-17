import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

class TerrainAnomalyDetector:
    """
    Zero-Shot Anomaly Detection for Rover Terrain Analysis.
    Uses on-the-fly statistical profiling to identify unknown obstacles.
    """
    
    def __init__(self, patch_size=16, contamination=0.03, max_dim=800):
        self.patch_size = patch_size
        self.contamination = contamination
        self.max_dim = max_dim # Prevents rover memory overload

    def _resize_if_needed(self, image):
        """Scales down massive high-res images to save compute."""
        h, w = image.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / float(max(h, w))
            return cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _extract_features(self, patch):
        """
        Extracts robust features using HSV color space to minimize 
        the impact of harsh planetary shadows.
        """
        # Convert to HSV (Hue, Saturation, Value)
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        # 1. Color/Lighting Statistics
        mean_hsv = np.mean(hsv_patch, axis=(0, 1))
        std_hsv = np.std(hsv_patch, axis=(0, 1))
        
        # 2. Structural/Texture Density (Using the 'Value' channel)
        v_channel = hsv_patch[:, :, 2]
        sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        return np.concatenate([mean_hsv, std_hsv, [edge_density]])

    def analyze(self, image_path, output_path="anomaly_scan_result.jpg"):
        """
        Executes the full anomaly detection pipeline on a given image.
        """
        # 1. Load and validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"CRITICAL: Image '{image_path}' not found.")
            
        raw_image = cv2.imread(image_path)
        image = self._resize_if_needed(raw_image)
        h, w, _ = image.shape
        
        # 2. Grid partitioning & Feature Extraction
        features, patch_coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                features.append(self._extract_features(patch))
                patch_coords.append((x, y))
                
        if not features:
            print("Warning: Image too small for patch size.")
            return image, np.zeros((h, w), dtype=np.uint8), image

        # 3. Fit Isolation Forest & Predict on the fly
        clf = IsolationForest(
            n_estimators=150, # Boosted tree count for higher accuracy
            max_samples='auto',
            contamination=self.contamination, 
            random_state=42,
            n_jobs=-1 # Uses all available CPU cores
        )
        predictions = clf.fit_predict(np.array(features))
        
        # 4. Generate Mask & Clean Noise
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y), pred in zip(patch_coords, predictions):
            if pred == -1: # -1 indicates an outlier
                mask[y:y+self.patch_size, x:x+self.patch_size] = 255

        # Morphological operations to merge nearby patches and remove stray pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.patch_size, self.patch_size))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 5. Object Detection & HUD Rendering
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hud_overlay = image.copy()
        
        # Paint the anomalous regions with a red tint
        hud_overlay[mask_clean > 0] = [0, 0, 255] 
        marked_image = cv2.addWeighted(image, 0.6, hud_overlay, 0.4, 0)

        anomaly_count = 0
        min_area = (self.patch_size ** 2) * 2 # Must be at least 2 patches large

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue # Ignore tiny noise
                
            anomaly_count += 1
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # HUD Bounding Box & Target Center
            cv2.rectangle(marked_image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.drawMarker(marked_image, (bx + bw // 2, by + bh // 2), 
                           (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
            
            # HUD Labeling
            label = f"TARGET {anomaly_count}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(marked_image, (bx, by - th - 10), (bx + tw + 10, by), (0, 255, 0), -1)
            cv2.putText(marked_image, label, (bx + 5, by - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 6. Save Data
        cv2.imwrite(output_path, marked_image)
        return image, mask_clean, marked_image


# ==========================================
# SIMULATION / EXECUTION BLOCK
# ==========================================

# 1. Generate a mock image and save it to disk to simulate camera capture
def create_test_image(filename="IMG_0387.jpg"):
    np.random.seed(99)
    # create a dusty, highly textured orange/red terrain
    terrain = np.random.normal(loc=[130, 90, 60], scale=[25, 20, 15], size=(600, 800, 3)).astype(np.uint8)
    
    # Add Anomaly 1: A smooth dark metallic rock
    cv2.ellipse(terrain, (250, 400), (40, 25), 30, 0, 360, (50, 50, 50), -1)
    
    # Add Anomaly 2: A reflective pool of liquid / ice
    cv2.circle(terrain, (600, 150), 45, (180, 190, 210), -1)
    terrain[100:200, 550:650] = cv2.GaussianBlur(terrain[100:200, 550:650], (21, 21), 0)
    
    cv2.imwrite(filename, terrain)
    return filename

# Run the simulation
print("Initializing Rover Diagnostics...")
simulated_image_path = "IMG_0387.jpg"

# Initialize our AI class (Tuned for ~4% terrain anomalies, patch size 16x16)
detector = TerrainAnomalyDetector(patch_size=16, contamination=0.04)

print(f"Scanning {simulated_image_path}...")
# To run this on your real data, change `simulated_image_path` to your actual file string
original, mask, marked = detector.analyze(
    image_path=simulated_image_path, 
    output_path="rover_hud_output.jpg"
)

# Plotting the diagnostic telemetry
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Optical Sensor Feed")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Statistical Isolation Mask")
plt.imshow(mask, cmap='magma')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Anomaly Detection HUD")
plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

print("Scan complete. Outputs saved successfully.")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import os

class PlanetaryTerrainAnalyzer:
    """
    Combined Anomaly Detection and Unsupervised Surface Classifier for Rovers.
    Identifies hazards while segmenting and describing surface terrain types on the fly.
    """
    
    def __init__(self, patch_size=16, n_terrain_classes=3, contamination=0.03, max_dim=800):
        self.patch_size = patch_size
        self.n_terrain_classes = n_terrain_classes
        self.contamination = contamination
        self.max_dim = max_dim

    def _resize_if_needed(self, image):
        """Prevents memory overload on high-resolution rover camera feeds."""
        h, w = image.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / float(max(h, w))
            return cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _extract_patch_features(self, patch):
        """
        Extracts physical texture, specular, and optical features.
        Returns:
            - anomaly_features: Features for Isolation Forest
            - classification_features: Micro-texture vectors for surface profiling
        """
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # 1. Color & Lighting Profiles
        mean_hsv = np.mean(hsv_patch, axis=(0, 1))
        std_hsv = np.std(hsv_patch, axis=(0, 1))
        
        # 2. Micro-Texture Roughness (Sobel Gradient Energy)
        v_channel = hsv_patch[:, :, 2]
        sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        edge_density = np.mean(grad_mag)
        roughness = np.std(grad_mag) # High variance = jagged/uneven terrain
        
        # 3. Surface Reflectivity & Uniformity
        reflectivity = np.max(v_channel) - np.mean(v_channel) # Specular highlight score
        uniformity = 1.0 / (1.0 + np.var(gray_patch)) # Smoothness index
        
        # Feature Vector for Classification
        class_feat = np.array([
            mean_hsv[0], mean_hsv[1], mean_hsv[2], 
            edge_density, roughness, reflectivity, uniformity
        ])
        
        return class_feat

    def _generate_terrain_labels(self, gmm_model):
        """
        Interprets physical cluster metrics to assign human-readable surface descriptors.
        """
        cluster_means = gmm_model.means_
        labels = {}
        
        for k in range(self.n_terrain_classes):
            # Feature indices: 3 = edge_density, 4 = roughness, 5 = reflectivity, 6 = uniformity
            edge_den = cluster_means[k][3]
            roughness = cluster_means[k][4]
            reflectivity = cluster_means[k][5]
            uniformity = cluster_means[k][6]
            
            if roughness > 18.0 or edge_den > 25.0:
                labels[k] = "Rough / Jagged Rock"
            elif reflectivity > 100.0 or uniformity > 0.005:
                labels[k] = "Smooth / Specular (Ice/Liquid)"
            elif roughness > 8.0:
                labels[k] = "Gravel / Loose Rubble"
            else:
                labels[k] = "Fine Regolith / Sand"
                
        return labels

    def process_frame(self, image_path, output_path="rover_telemetry_output.jpg"):
        """
        Runs on-the-fly surface classification and anomaly detection simultaneously.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image '{image_path}' not found.")
            
        raw_image = cv2.imread(image_path)
        image = self._resize_if_needed(raw_image)
        h, w, _ = image.shape
        
        # 1. Grid Extraction
        features, patch_coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                feat = self._extract_patch_features(patch)
                features.append(feat)
                patch_coords.append((x, y))
                
        features = np.array(features)

        # 2. Unsupervised Terrain Classification (Gaussian Mixture Model)
        gmm = GaussianMixture(n_components=self.n_terrain_classes, random_state=42)
        terrain_clusters = gmm.fit_predict(features)
        terrain_descriptors = self._generate_terrain_labels(gmm)

        # 3. Anomaly Detection (Isolation Forest)
        iso_forest = IsolationForest(
            n_estimators=150, 
            contamination=self.contamination, 
            random_state=42, 
            n_jobs=-1
        )
        anomalies = iso_forest.fit_predict(features)

        # 4. Generate Classification Map Overlay
        # Color palette for up to 4 surface classes (BGR format)
        class_colors = [(200, 150, 50), (50, 180, 50), (200, 80, 150), (100, 200, 250)]
        
        segmentation_map = np.zeros_like(image)
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)

        for (x, y), cluster_id, is_anomaly in zip(patch_coords, terrain_clusters, anomalies):
            if is_anomaly == -1:
                anomaly_mask[y:y+self.patch_size, x:x+self.patch_size] = 255
            
            color = class_colors[cluster_id % len(class_colors)]
            segmentation_map[y:y+self.patch_size, x:x+self.patch_size] = color

        # Morphological Mask Cleaning for Anomalies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.patch_size, self.patch_size))
        clean_anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Build Final HUD Visual Output
        classified_hud = cv2.addWeighted(image, 0.5, segmentation_map, 0.5, 0)
        
        # Highlight anomalies with bold red bounding boxes
        contours, _ = cv2.findContours(clean_anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (self.patch_size ** 2) * 2
        
        anomaly_id = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            anomaly_id += 1
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(classified_hud, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.putText(classified_hud, f"HAZARD #{anomaly_id}", (bx, by - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Draw HUD Terrain Legend
        y_offset = 25
        cv2.putText(classified_hud, "TERRAIN CLASSIFICATION LEGEND:", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        
        for k, label_text in terrain_descriptors.items():
            y_offset += 20
            color = class_colors[k % len(class_colors)]
            cv2.rectangle(classified_hud, (10, y_offset - 10), (25, y_offset + 2), color, -1)
            cv2.putText(classified_hud, f"Class {k}: {label_text}", (32, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save output image
        cv2.imwrite(output_path, classified_hud)
        return image, segmentation_map, classified_hud


# ==========================================
# SIMULATION EXECUTION BLOCK
# ==========================================

def create_multi_terrain_sim(filename="planetary_surface_feed.jpg"):
    """Generates a multi-surface synthetic planet frame."""
    np.random.seed(42)
    # Background: Fine Red Sand Regolith
    terrain = np.random.normal(loc=[140, 80, 50], scale=[15, 10, 8], size=(600, 800, 3)).astype(np.uint8)
    
    # Surface Region A: Jagged Dark Rock Outcrop (Rough texture)
    rock_patch = np.random.normal(loc=[50, 50, 50], scale=[40, 40, 40], size=(200, 300, 3)).astype(np.uint8)
    terrain[350:550, 100:400] = rock_patch
    
    # Surface Region B: Reflective/Smooth Ice Sheet
    cv2.ellipse(terrain, (600, 200), (120, 80), 15, 0, 360, (200, 210, 230), -1)
    terrain[120:280, 480:720] = cv2.GaussianBlur(terrain[120:280, 480:720], (25, 25), 0)

    # Isolated Hazard Anomaly: Metallic anomaly structure
    cv2.circle(terrain, (250, 150), 25, (10, 220, 250), -1)

    cv2.imwrite(filename, terrain)
    return filename

# Run System
sim_file = 'IMG_0387.jpg'

analyzer = PlanetaryTerrainAnalyzer(patch_size=16, n_terrain_classes=3, contamination=0.03)
raw, seg_map, hud = analyzer.process_frame(sim_file, output_path="classified_rover_hud.jpg")

# Render Diagnostics
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("1. Raw Optical Sensor")
plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("2. Texture Segmentation Map")
plt.imshow(cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("3. Classified Surface HUD & Hazards")
plt.imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import os

class UltimateRoverAI:
    """
    Comprehensive Planetary Analyzer:
    - Unsupervised Terrain Classification
    - Anomaly / Obstacle Detection
    - Bio-Signature & Liquid Heuristic Detection
    - Pseudo-3D Topographical Mapping
    """
    
    def __init__(self, patch_size=16, max_dim=600):
        self.patch_size = patch_size
        self.max_dim = max_dim

    def _resize(self, image):
        h, w = image.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / float(max(h, w))
            return cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _detect_bio_and_liquids(self, image, hsv_image, edge_map):
        """Uses physical heuristics to explicitly search for life and water."""
        h, w = image.shape[:2]
        analysis_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        hue = hsv_image[:, :, 0]
        sat = hsv_image[:, :, 1]
        val = hsv_image[:, :, 2]

        # 1. Bio-Signature Detection (Searching for non-geological colors like Green/Cyan)
        # Assuming alien plant life uses photosynthesis-like reactions
        bio_mask = ((hue > 30) & (hue < 85) & (sat > 100) & (val > 80)).astype(np.uint8) * 255
        
        # 2. Liquid / Water Detection (High reflectivity + completely flat/smooth)
        liquid_mask = ((val > 200) & (edge_map < 15)).astype(np.uint8) * 255
        
        analysis_map[bio_mask > 0] = [0, 255, 0]    # Green for Biology
        analysis_map[liquid_mask > 0] = [255, 0, 0] # Blue for Liquid
        
        return analysis_map, bio_mask, liquid_mask

    def _generate_3d_elevation(self, image):
        """Approximates a 3D depth map using Shape from Shading and Texture Gradients."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur to simulate large rolling hills and ground contours
        macro_structure = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # Sobel for micro-texture (rocks pop up, flat areas stay low)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        micro_structure = np.sqrt(sobelx**2 + sobely**2)
        micro_structure = cv2.normalize(micro_structure, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Combine macro and micro to create the Z-axis elevation map
        elevation = cv2.addWeighted(macro_structure, 0.7, micro_structure, 0.3, 0)
        
        # Downsample for 3D plotting to prevent matplotlib from crashing
        scale = 0.2
        small_elevation = cv2.resize(elevation, (0,0), fx=scale, fy=scale)
        small_image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        
        return small_elevation, small_image

    def process(self, image_path):
        # 1. Load and prepare data
        image = self._resize(cv2.imread(image_path))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        # 2. Heuristic Scans (Life & Water)
        special_scan_map, bio_m, liquid_m = self._detect_bio_and_liquids(image, hsv, edge_map)

        # 3. Patch Extraction for ML
        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                hsv_patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                
                mean_hsv = np.mean(hsv_patch, axis=(0, 1))
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 4. Unsupervised ML (Anomalies & Terrain)
        clf = IsolationForest(contamination=0.03, random_state=42).fit_predict(features)
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)

        # 5. Build HUD
        hud = image.copy()
        for (x, y), anomaly, terrain_id in zip(coords, clf, gmm):
            if anomaly == -1:
                cv2.rectangle(hud, (x, y), (x+self.patch_size, y+self.patch_size), (0, 0, 255), 2)

        # 6. Generate 3D Elevation
        elevation_map, texture_map = self._generate_3d_elevation(image)
        
        return image, hud, special_scan_map, elevation_map, texture_map


# ==========================================
# SIMULATION & RENDERING
# ==========================================

def create_ultimate_sim():
    np.random.seed(42)
    # Base planet: Dusty orange
    terrain = np.random.normal(loc=[100, 140, 180], scale=[10, 10, 10], size=(500, 600, 3)).astype(np.uint8)
    
    # Add Alien Plant (Green/Cyan high saturation)
    cv2.circle(terrain, (150, 350), 30, (50, 220, 80), -1)
    
    # Add Deep Water Puddle (Smooth, highly reflective blue)
    cv2.ellipse(terrain, (450, 150), (80, 40), 0, 0, 360, (250, 150, 50), -1)
    terrain[100:200, 350:550] = cv2.GaussianBlur(terrain[100:200, 350:550], (25, 25), 0)
    
    # Add a weird geometric monolith (Anomaly)
    cv2.rectangle(terrain, (280, 200), (320, 300), (40, 40, 40), -1)
    
    cv2.imwrite("ultimate_sim.jpg", terrain)
    return "ultimate_sim.jpg"

print("Booting Rover God-Mode Scanners...")
sim_file = 'IMG_0387.jpg'

ai = UltimateRoverAI(patch_size=16)
raw, hud, bio_water, Z, tex = ai.process(sim_file)

# Plotting the 4-way Dashboard
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Raw Feed
ax1 = fig.add_subplot(221)
ax1.set_title("1. Raw Optical Sensor")
ax1.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
ax1.axis('off')

# Subplot 2: ML HUD (Anomalies)
ax2 = fig.add_subplot(222)
ax2.set_title("2. ML Hazard Detection")
ax2.imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
ax2.axis('off')

# Subplot 3: Bio & Liquid Scanners
ax3 = fig.add_subplot(223)
ax3.set_title("3. Heuristic Bio/Liquid Scanner (Green=Bio, Blue=Water)")
overlay = cv2.addWeighted(raw, 0.4, bio_water, 0.6, 0)
ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax3.axis('off')

# Subplot 4: 3D Terrain Topography
ax4 = fig.add_subplot(224, projection='3d')
ax4.set_title("4. Estimated 3D Topography")
X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
# Normalize texture colors for 3D face colors
tex_norm = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB) / 255.0
ax4.plot_surface(X, Y, Z, facecolors=tex_norm, rstride=2, cstride=2, antialiased=True, shade=False)
ax4.view_init(elev=45, azim=-60)
ax4.axis('off')

plt.tight_layout()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

# ---------------------------------------------------------
# 1. MISSION CONTROL TELEMETRY (LOGGING)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] ARES-V TELEMETRY | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

class AresMissionControl:
    """
    Ultimate Aerospace-Grade Rover AI.
    Integrates GMM, Isolation Forests, DBSCAN, and Photometric 3D Modeling.
    """
    
    def __init__(self, patch_size=12, contamination=0.03):
        self.patch_size = patch_size
        self.contamination = contamination
        logging.info(f"System Boot: Patch Size [{patch_size}x{patch_size}], Sensitivity [{contamination}]")

    def _photometric_3d_mesh(self, image):
        """Generates a 3D mesh using Lambertian Shape-from-Shading principles."""
        logging.info("Calculating photometric surface normals...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simulate surface albedo and macro-structure
        macro = cv2.GaussianBlur(gray, (75, 75), 0)
        
        # Scharr operators for high-precision micro-gradients
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        micro = np.sqrt(scharr_x**2 + scharr_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.6, micro, 0.4, 0)
        
        # Downsample for rapid plotting
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale), cv2.resize(image, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating full diagnostic sweep on: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Optical sensor feed offline. Image not found.")
            return None
            
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Edge & Texture Tensors
        logging.info("Extracting spatial feature tensors...")
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        # 2. Extract Patches
        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. Unsupervised Core Models
        logging.info("Running Unsupervised Machine Learning Models...")
        gmm = GaussianMixture(n_components=4, random_state=42).fit_predict(features)
        iso_scores = IsolationForest(contamination=self.contamination, random_state=42).fit_predict(features)

        # 4. Spatial Clustering (DBSCAN) for Anomalies
        logging.info("Isolating target geometries using DBSCAN...")
        anomaly_coords = np.array([coords[i] for i in range(len(iso_scores)) if iso_scores[i] == -1])
        
        hud = image.copy()
        terrain_map = np.zeros_like(image)
        colors = [(30,30,150), (150,100,50), (50,150,50), (100,100,100)]
        
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = colors[tid % 4]

        # Draw precise boxes around clustered anomalies
        if len(anomaly_coords) > 0:
            clustering = DBSCAN(eps=self.patch_size * 2, min_samples=2).fit(anomaly_coords)
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1: continue # Ignore isolated noise
                
                points = anomaly_coords[clustering.labels_ == cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(hud, f"OBJ-{cluster_id}", (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 5. Biological & Chemical Scanners
        logging.info("Scanning for chemical and biological signatures...")
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        bio_mask = ((hue > 35) & (hue < 80) & (sat > 100) & (val > 80)).astype(np.uint8) * 255
        chem_hud = image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0] # Green overlay

        # 6. Topography
        Z, tex = self._photometric_3d_mesh(image)
        
        elapsed = time.time() - start_time
        logging.info(f"Sweep complete in {elapsed:.2f} seconds.")
        
        return image, terrain_map, hud, chem_hud, Z, tex

# ---------------------------------------------------------
# SIMULATION GENERATOR
# ---------------------------------------------------------
def build_martian_proving_ground(filename="ares_proving_ground.jpg"):
    np.random.seed(777)
    terrain = np.random.normal(loc=[100, 130, 190], scale=[15, 12, 10], size=(600, 800, 3)).astype(np.uint8)
    
    # Crater (Darker, smoother)
    cv2.circle(terrain, (650, 450), 90, (80, 110, 160), -1)
    terrain[350:550, 550:750] = cv2.GaussianBlur(terrain[350:550, 550:750], (31, 31), 0)
    
    # Metallic Debris Field (Anomalies)
    cv2.rectangle(terrain, (200, 150), (230, 190), (200, 200, 200), -1)
    cv2.rectangle(terrain, (240, 160), (260, 180), (180, 180, 180), -1)
    
    # Biological / Chemical Outcrop
    cv2.circle(terrain, (400, 300), 25, (40, 190, 60), -1)
    
    cv2.imwrite(filename, terrain)
    return filename

# ---------------------------------------------------------
# MISSION EXECUTION & DASHBOARD
# ---------------------------------------------------------
sim_file = 'IMG_0377.jpg'
rover = AresMissionControl(patch_size=12, contamination=0.02)
raw, terrain, hud, chem, Z, tex = rover.execute_scan(sim_file)

# Build the JPL Mission Control Dashboard
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#111111') # Dark mode for mission control
plt.rcParams['text.color'] = 'white'
gs = gridspec.GridSpec(2, 3, figure=fig)

# Setup subplots
axes = [
    fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
    fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :2]), 
    fig.add_subplot(gs[1, 2], projection='3d')
]

titles = ["1. Optical Sensor (Raw)", "2. Spectral Terrain Clusters", 
          "3. Chemical / Bio Scanners", "4. DBSCAN Target Acquisition"]
images = [raw, terrain, chem, hud]

for ax, title, img in zip(axes[:4], titles, images):
    ax.set_title(title, color='cyan', pad=10, weight='bold')
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

# 3D Plot setup
ax_3d = axes[4]
ax_3d.set_title("5. Photometric Topography", color='cyan', pad=10, weight='bold')
ax_3d.set_facecolor('#111111')
X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
tex_norm = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB) / 255.0
ax_3d.plot_surface(X, Y, Z, facecolors=tex_norm, rstride=2, cstride=2, antialiased=False, shade=False)
ax_3d.view_init(elev=50, azim=-45)
ax_3d.axis('off')

plt.tight_layout()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] ARES-VI | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

class AresQuantificationControl:
    """
    Upgraded Aerospace-Grade Rover AI.
    Features: Surface quantification, hyper-sensitive bio-scanning, and telemetry readouts.
    """
    
    def __init__(self, patch_size=16, contamination=0.02):
        self.patch_size = patch_size
        self.contamination = contamination

    def _generate_smooth_3d(self, image):
        """Generates a smoothed 3D mesh, ignoring micro-noise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge-preserving blur to flatten noise but keep physical structures
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0)
        
        # Less aggressive gradients
        sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sobel_x**2 + sobel_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.7, micro, 0.3, 0)
        
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale), cv2.resize(image, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating ARES-VI sweep on: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Optical sensor feed offline.")
            return None
            
        h, w = image.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Biological & Chemical Quantification (Tuned for Earth-like flora)
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        # Broader range: Catches dead yellow grass up to deep forest green
        bio_mask = ((hue > 25) & (hue < 90) & (sat > 30) & (val > 30)).astype(np.uint8) * 255
        liquid_mask = ((val > 210) & (sat < 40)).astype(np.uint8) * 255 # Bright, low saturation (glare/water)
        
        bio_percent = (np.count_nonzero(bio_mask) / total_pixels) * 100
        liquid_percent = (np.count_nonzero(liquid_mask) / total_pixels) * 100
        
        chem_hud = image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0]
        chem_hud[liquid_mask > 0] = [255, 0, 0]

        # 2. Extract Patches for ML
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. GMM Surface Classification
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)
        
        # Calculate terrain distribution percentages
        terrain_counts = np.bincount(gmm)
        total_patches = len(gmm)
        terrain_percents = [(count / total_patches) * 100 for count in terrain_counts]
        
        terrain_map = np.zeros_like(image)
        colors = [(150,100,50), (50,150,50), (100,100,100)] # Dirt, Vegetation, Rock
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = colors[tid % 3]

        # 4. Isolation Forest & Tuned DBSCAN
        iso_scores = IsolationForest(contamination=self.contamination, random_state=42).fit_predict(features)
        anomaly_coords = np.array([coords[i] for i in range(len(iso_scores)) if iso_scores[i] == -1])
        
        hud = image.copy()
        hazard_count = 0
        
        if len(anomaly_coords) > 0:
            # eps controls clustering distance, min_samples prevents tiny noise boxes
            clustering = DBSCAN(eps=self.patch_size * 2.5, min_samples=4).fit(anomaly_coords)
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1: continue 
                
                points = anomaly_coords[clustering.labels_ == cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                cv2.putText(hud, f"HAZ-{cluster_id}", (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                hazard_count += 1

        # 5. Topography
        Z, tex = self._generate_smooth_3d(image)
        
        # 6. Generate Telemetry Report
        telemetry = {
            "Bio-Signature Match": f"{bio_percent:.2f}%",
            "Liquid Probability": f"{liquid_percent:.2f}%",
            "Identified Hazards": hazard_count,
            "Dominant Surface A": f"{terrain_percents[0]:.1f}%",
            "Dominant Surface B": f"{terrain_percents[1]:.1f}%",
            "Dominant Surface C": f"{terrain_percents[2]:.1f}%"
        }
        
        logging.info("--- TELEMETRY REPORT ---")
        for key, val in telemetry.items():
            logging.info(f"{key}: {val}")
        logging.info("------------------------")
        
        return image, terrain_map, hud, chem_hud, Z, tex, telemetry

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
# Replace this string with the exact file path to your Earth grass image
target_image_path = 'IMG_0377.jpg'

rover = AresQuantificationControl(patch_size=16, contamination=0.015)

try:
    raw, terrain, hud, chem, Z, tex, telemetry = rover.execute_scan(target_image_path)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0a')
    plt.rcParams['text.color'] = 'white'
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2], projection='3d')
    ]

    titles = ["1. Optical Feed", "2. GMM Surface Segments", "3. Bio/Liquid Quantifier", "4. Adaptive Hazard Tracking"]
    images = [raw, terrain, chem, hud]

    for ax, title, img in zip(axes[:4], titles, images):
        ax.set_title(title, color='#00ffcc', pad=10, weight='bold')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Add Telemetry Text Overlay to the Hazard Image
    y_pos = 30
    cv2.putText(hud, "LIVE TELEMETRY:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for key, val in telemetry.items():
        y_pos += 25
        cv2.putText(hud, f"{key}: {val}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    axes[3].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)) # Refresh with text

    ax_3d = axes[4]
    ax_3d.set_title("5. Smoothed Topography", color='#00ffcc', pad=10, weight='bold')
    ax_3d.set_facecolor('#0a0a0a')
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    tex_norm = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB) / 255.0
    ax_3d.plot_surface(X, Y, Z, facecolors=tex_norm, rstride=3, cstride=3, antialiased=False, shade=False)
    ax_3d.view_init(elev=60, azim=-45)
    ax_3d.axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"System Error: Please ensure you replace 'target_image_path' with a valid local image file. Error details: {e}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] ARES-VI | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

class AresQuantificationControl:
    """
    Upgraded Aerospace-Grade Rover AI.
    Features: Surface quantification, hyper-sensitive bio-scanning, telemetry readouts, and HUD Legends.
    """
    
    def __init__(self, patch_size=16, contamination=0.02):
        self.patch_size = patch_size
        self.contamination = contamination

    def _generate_smooth_3d(self, image):
        """Generates a smoothed 3D mesh, ignoring micro-noise."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge-preserving blur to flatten noise but keep physical structures
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0)
        
        # Less aggressive gradients
        sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sobel_x**2 + sobel_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.7, micro, 0.3, 0)
        
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale), cv2.resize(image, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating ARES-VI sweep on: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Optical sensor feed offline.")
            return None
            
        h, w = image.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Biological & Chemical Quantification (Tuned for Earth-like flora)
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        bio_mask = ((hue > 25) & (hue < 90) & (sat > 30) & (val > 30)).astype(np.uint8) * 255
        liquid_mask = ((val > 210) & (sat < 40)).astype(np.uint8) * 255 
        
        bio_percent = (np.count_nonzero(bio_mask) / total_pixels) * 100
        liquid_percent = (np.count_nonzero(liquid_mask) / total_pixels) * 100
        
        chem_hud = image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0]
        chem_hud[liquid_mask > 0] = [255, 0, 0]

        # 2. Extract Patches for ML
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. GMM Surface Classification
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)
        
        terrain_counts = np.bincount(gmm)
        total_patches = len(gmm)
        terrain_percents = [(count / total_patches) * 100 for count in terrain_counts]
        
        terrain_map = np.zeros_like(image)
        colors = [(150,100,50), (50,150,50), (100,100,100)] # BGR format: Blueish, Green, Gray
        
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = colors[tid % 3]

        # --- ADDED: Draw HUD Legend on Terrain Map ---
        legend_x, legend_y = 15, 30
        cv2.putText(terrain_map, "CLUSTER LEGEND:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cluster_labels = [
            f"Surface A ({terrain_percents[0]:.1f}%)", 
            f"Surface B ({terrain_percents[1]:.1f}%)", 
            f"Surface C ({terrain_percents[2]:.1f}%)"
        ]
        
        for i, color in enumerate(colors):
            legend_y += 30
            # Draw color box
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1) # White border
            # Draw label
            cv2.putText(terrain_map, cluster_labels[i], (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # ---------------------------------------------

        # 4. Isolation Forest & Tuned DBSCAN
        iso_scores = IsolationForest(contamination=self.contamination, random_state=42).fit_predict(features)
        anomaly_coords = np.array([coords[i] for i in range(len(iso_scores)) if iso_scores[i] == -1])
        
        hud = image.copy()
        hazard_count = 0
        
        if len(anomaly_coords) > 0:
            clustering = DBSCAN(eps=self.patch_size * 2.5, min_samples=4).fit(anomaly_coords)
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1: continue 
                
                points = anomaly_coords[clustering.labels_ == cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                cv2.putText(hud, f"HAZ-{cluster_id}", (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                hazard_count += 1

        # 5. Topography
        Z, tex = self._generate_smooth_3d(image)
        
        # 6. Generate Telemetry Report
        telemetry = {
            "Bio-Signature Match": f"{bio_percent:.2f}%",
            "Liquid Probability": f"{liquid_percent:.2f}%",
            "Identified Hazards": hazard_count,
            "Dominant Surface A": f"{terrain_percents[0]:.1f}%",
            "Dominant Surface B": f"{terrain_percents[1]:.1f}%",
            "Dominant Surface C": f"{terrain_percents[2]:.1f}%"
        }
        
        return image, terrain_map, hud, chem_hud, Z, tex, telemetry


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
# Replace this string with your exact file path
target_image_path = "IMG_0377.jpg"

rover = AresQuantificationControl(patch_size=16, contamination=0.015)

try:
    raw, terrain, hud, chem, Z, tex, telemetry = rover.execute_scan(target_image_path)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0a')
    plt.rcParams['text.color'] = 'white'
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2], projection='3d')
    ]

    titles = ["1. Optical Feed", "2. GMM Surface Segments", "3. Bio/Liquid Quantifier", "4. Adaptive Hazard Tracking"]
    images = [raw, terrain, chem, hud]

    for ax, title, img in zip(axes[:4], titles, images):
        ax.set_title(title, color='#00ffcc', pad=10, weight='bold')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Add Telemetry Text Overlay to the Hazard Image
    y_pos = 30
    cv2.putText(hud, "LIVE TELEMETRY:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for key, val in telemetry.items():
        y_pos += 25
        cv2.putText(hud, f"{key}: {val}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    axes[3].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)) 

    ax_3d = axes[4]
    ax_3d.set_title("5. Smoothed Topography", color='#00ffcc', pad=10, weight='bold')
    ax_3d.set_facecolor('#0a0a0a')
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    tex_norm = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB) / 255.0
    ax_3d.plot_surface(X, Y, Z, facecolors=tex_norm, rstride=3, cstride=3, antialiased=False, shade=False)
    ax_3d.view_init(elev=60, azim=-45)
    ax_3d.axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"System Error: {e}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] ARES-VII | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

class AresQuantificationControl:
    """
    Upgraded Aerospace-Grade Rover AI.
    Features: Semantic GMM Labeling and Grayscale Photometric DEM 3D rendering.
    """
    
    def __init__(self, patch_size=16, contamination=0.02):
        self.patch_size = patch_size
        self.contamination = contamination

    def _generate_smooth_3d(self, image):
        """Generates a smoothed 3D mesh focused strictly on elevation and physical texture."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge-preserving blur to flatten noise but keep physical structures
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0)
        
        # Micro-gradients for surface roughness mapping
        sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sobel_x**2 + sobel_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.7, micro, 0.3, 0)
        
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating ARES-VII sweep on: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Optical sensor feed offline.")
            return None
            
        h, w = image.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Biological & Chemical Quantification
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        bio_mask = ((hue > 25) & (hue < 90) & (sat > 30) & (val > 30)).astype(np.uint8) * 255
        liquid_mask = ((val > 210) & (sat < 40)).astype(np.uint8) * 255 
        
        bio_percent = (np.count_nonzero(bio_mask) / total_pixels) * 100
        liquid_percent = (np.count_nonzero(liquid_mask) / total_pixels) * 100
        
        chem_hud = image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0]
        chem_hud[liquid_mask > 0] = [255, 0, 0]

        # 2. Extract Patches for ML
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. Semantic GMM Surface Classification
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)
        
        # Analyze clusters to intelligently assign names and colors
        cluster_stats = []
        for i in range(3):
            mask = (gmm == i)
            if np.any(mask):
                cluster_stats.append({
                    'id': i, 
                    'hue': np.mean(features[mask, 0]), 
                    'rough': np.mean(features[mask, 3]), 
                    'pct': (np.sum(mask) / len(gmm)) * 100
                })
                
        # Heuristics for Mapping:
        # 1. Vegetation: Hue closest to OpenCV green (60)
        cluster_stats.sort(key=lambda x: abs(x['hue'] - 60))
        veg_id = cluster_stats[0]['id']
        
        # 2. Smooth/Moisture vs Soil/Rock: Sort remaining by roughness
        remaining = cluster_stats[1:]
        remaining.sort(key=lambda x: x['rough'])
        smooth_id = remaining[0]['id']
        rock_id = remaining[1]['id']
        
        # Map IDs to semantic labels and colors (BGR format)
        mapping = {
            veg_id: {"name": "Vegetation / Flora", "color": (50, 200, 50)},    # Green
            smooth_id: {"name": "Moisture / Smooth", "color": (200, 100, 50)}, # Blue
            rock_id: {"name": "Soil / Bedrock", "color": (120, 120, 120)}      # Gray
        }
        
        terrain_map = np.zeros_like(image)
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = mapping[tid]["color"]

        # Draw Semantic Legend Overlay
        legend_x, legend_y = 15, 30
        cv2.putText(terrain_map, "SURFACE MATERIAL LEGEND:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        display_order = [veg_id, smooth_id, rock_id] # Fixed visual order
        for cid in display_order:
            legend_y += 30
            color = mapping[cid]["color"]
            pct = next(item['pct'] for item in cluster_stats if item["id"] == cid)
            label = f"{mapping[cid]['name']} ({pct:.1f}%)"
            
            # Color Box + White Border
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1) 
            # Text
            cv2.putText(terrain_map, label, (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. Isolation Forest & Tuned DBSCAN
        iso_scores = IsolationForest(contamination=self.contamination, random_state=42).fit_predict(features)
        anomaly_coords = np.array([coords[i] for i in range(len(iso_scores)) if iso_scores[i] == -1])
        
        hud = image.copy()
        hazard_count = 0
        
        if len(anomaly_coords) > 0:
            clustering = DBSCAN(eps=self.patch_size * 2.5, min_samples=4).fit(anomaly_coords)
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1: continue 
                
                points = anomaly_coords[clustering.labels_ == cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                cv2.putText(hud, f"HAZ-{cluster_id}", (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                hazard_count += 1

        # 5. Topography Z-Map Only (Grayscale focus)
        Z = self._generate_smooth_3d(image)
        
        # 6. Generate Telemetry Report
        telemetry = {
            "Bio-Signature Match": f"{bio_percent:.2f}%",
            "Liquid Probability": f"{liquid_percent:.2f}%",
            "Identified Hazards": hazard_count,
            "Vegetation Coverage": f"{next(i['pct'] for i in cluster_stats if i['id'] == veg_id):.1f}%",
            "Exposed Soil/Rock": f"{next(i['pct'] for i in cluster_stats if i['id'] == rock_id):.1f}%"
        }
        
        return image, terrain_map, hud, chem_hud, Z, telemetry


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
# Replace this string with your exact file path
target_image_path = "IMG_0387.jpg"

rover = AresQuantificationControl(patch_size=16, contamination=0.015)

try:
    raw, terrain, hud, chem, Z, telemetry = rover.execute_scan(target_image_path)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0a0a0a')
    plt.rcParams['text.color'] = 'white'
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2], projection='3d')
    ]

    titles = ["1. Optical Feed", "2. Semantic Surface Clusters", "3. Bio/Liquid Quantifier", "4. Adaptive Hazard Tracking"]
    images = [raw, terrain, chem, hud]

    for ax, title, img in zip(axes[:4], titles, images):
        ax.set_title(title, color='#00ffcc', pad=10, weight='bold')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Add Telemetry Text Overlay to the Hazard Image
    y_pos = 30
    cv2.putText(hud, "LIVE TELEMETRY:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for key, val in telemetry.items():
        y_pos += 25
        cv2.putText(hud, f"{key}: {val}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    axes[3].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)) 

    # Plot 5: High-Contrast Grayscale DEM (Digital Elevation Model)
    ax_3d = axes[4]
    ax_3d.set_title("5. Grayscale Topography DEM", color='#00ffcc', pad=10, weight='bold')
    ax_3d.set_facecolor('#0a0a0a')
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    
    # Use LightSource to generate a beautiful shaded relief map in grayscale
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.shade(Z, cmap=plt.cm.gray, blend_mode='soft', vert_exag=1.5)
    
    ax_3d.plot_surface(X, Y, Z, facecolors=shade, rstride=2, cstride=2, antialiased=True, shade=False)
    ax_3d.view_init(elev=55, azim=-45)
    ax_3d.axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"System Error: {e}")
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] ARES-VIII | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

class AresAdvancedQuantification:
    """
    Next-Gen Aerospace Rover AI.
    Features: CLAHE texture enhancement, Anomaly Decision Heatmaps, and DEM Elevation Scales.
    """
    
    def __init__(self, patch_size=16, contamination=0.03):
        self.patch_size = patch_size
        self.contamination = contamination

    def _apply_clahe(self, image):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for texture popping."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _generate_smooth_3d(self, image):
        """Generates a smoothed 3D mesh DEM."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0)
        
        sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sobel_x**2 + sobel_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.7, micro, 0.3, 0)
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating ARES-VIII sweep on: {image_path}")
        
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            logging.error("Optical sensor feed offline.")
            return None
            
        # 0. Apply CLAHE Enhancement
        image = self._apply_clahe(raw_image)
        h, w = image.shape[:2]
        total_pixels = h * w
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Biological & Chemical Quantification
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        bio_mask = ((hue > 25) & (hue < 90) & (sat > 30) & (val > 30)).astype(np.uint8) * 255
        liquid_mask = ((val > 210) & (sat < 40)).astype(np.uint8) * 255 
        
        bio_percent = (np.count_nonzero(bio_mask) / total_pixels) * 100
        liquid_percent = (np.count_nonzero(liquid_mask) / total_pixels) * 100
        
        chem_hud = raw_image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0]
        chem_hud[liquid_mask > 0] = [255, 0, 0]

        # 2. Extract Patches for ML (Using Enhanced Texture)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. Semantic GMM Surface Classification
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)
        
        cluster_stats = []
        for i in range(3):
            mask = (gmm == i)
            if np.any(mask):
                cluster_stats.append({
                    'id': i, 
                    'hue': np.mean(features[mask, 0]), 
                    'rough': np.mean(features[mask, 3]), 
                    'pct': (np.sum(mask) / len(gmm)) * 100
                })
                
        cluster_stats.sort(key=lambda x: abs(x['hue'] - 60))
        veg_id = cluster_stats[0]['id']
        remaining = cluster_stats[1:]
        remaining.sort(key=lambda x: x['rough'])
        smooth_id = remaining[0]['id']
        rock_id = remaining[1]['id']
        
        mapping = {
            veg_id: {"name": "Vegetation / Flora", "color": (50, 200, 50)},
            smooth_id: {"name": "Moisture / Smooth", "color": (200, 100, 50)},
            rock_id: {"name": "Soil / Bedrock", "color": (120, 120, 120)}
        }
        
        terrain_map = np.zeros_like(image)
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = mapping[tid]["color"]

        legend_x, legend_y = 15, 30
        cv2.putText(terrain_map, "SURFACE MATERIAL LEGEND:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for cid in [veg_id, smooth_id, rock_id]:
            legend_y += 30
            color = mapping[cid]["color"]
            pct = next(item['pct'] for item in cluster_stats if item["id"] == cid)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1) 
            cv2.putText(terrain_map, f"{mapping[cid]['name']} ({pct:.1f}%)", (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. Advanced Anomaly Detection (Decision Function Heatmap)
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        iso.fit(features)
        
        # Get raw decision scores (lower means more anomalous)
        decision_scores = iso.decision_function(features)
        # Normalize scores to 0-255 for color mapping (inverted so 255 is most anomalous)
        norm_scores = cv2.normalize(-decision_scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        hud = image.copy()
        heatmap_overlay = np.zeros_like(image)
        
        # Apply colormap to scores
        for (x, y), score in zip(coords, norm_scores):
            # Only colorize areas that pass a baseline threshold to reduce noise
            if score > 150: 
                # Jet colormap approximation: Red/Yellow is high anomaly, blue is low
                color = cv2.applyColorMap(np.array([[score]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0].tolist()
                heatmap_overlay[y:y+self.patch_size, x:x+self.patch_size] = color

        # Blend heatmap with original image
        cv2.addWeighted(heatmap_overlay, 0.6, hud, 0.8, 0, hud)
        
        # Retain DBSCAN Bounding Boxes for severe hazards
        anomaly_coords = np.array([coords[i] for i in range(len(decision_scores)) if decision_scores[i] < np.percentile(decision_scores, self.contamination * 100)])
        hazard_count = 0
        
        if len(anomaly_coords) > 0:
            clustering = DBSCAN(eps=self.patch_size * 2.5, min_samples=4).fit(anomaly_coords)
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1: continue 
                
                points = anomaly_coords[clustering.labels_ == cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                cv2.putText(hud, f"HAZ-{cluster_id} [SEVERE]", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                hazard_count += 1

        # 5. Topography Z-Map (Grayscale focus)
        Z = self._generate_smooth_3d(image)
        
        # 6. Generate Telemetry Report
        telemetry = {
            "Bio-Signature Match": f"{bio_percent:.2f}%",
            "Liquid Probability": f"{liquid_percent:.2f}%",
            "Identified Hazards": hazard_count,
            "Vegetation Coverage": f"{next(i['pct'] for i in cluster_stats if i['id'] == veg_id):.1f}%",
            "Exposed Soil/Rock": f"{next(i['pct'] for i in cluster_stats if i['id'] == rock_id):.1f}%"
        }
        
        return image, terrain_map, hud, chem_hud, Z, telemetry


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
target_image_path = "IMG_0377.jpg"

rover = AresAdvancedQuantification(patch_size=16, contamination=0.02)

try:
    enhanced_raw, terrain, hud, chem, Z, telemetry = rover.execute_scan(target_image_path)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0a0a0a')
    plt.rcParams['text.color'] = 'white'
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2], projection='3d')
    ]

    titles = ["1. CLAHE Enhanced Feed", "2. Semantic Surface Clusters", "3. Bio/Liquid Quantifier", "4. Anomaly Heatmap & Tracking"]
    images = [enhanced_raw, terrain, chem, hud]

    for ax, title, img in zip(axes[:4], titles, images):
        ax.set_title(title, color='#00ffcc', pad=10, weight='bold')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Add Telemetry Text Overlay to the Hazard Image
    y_pos = 30
    cv2.putText(hud, "LIVE TELEMETRY:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for key, val in telemetry.items():
        y_pos += 25
        cv2.putText(hud, f"{key}: {val}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    
    # Add Anomaly Legend to HUD
    cv2.putText(hud, "ANOMALY SEVERITY:", (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(hud, "RED = SEVERE | BLUE = LOW", (10, y_pos + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    axes[3].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)) 

    # Plot 5: High-Contrast Grayscale DEM (Digital Elevation Model) with Colorbar
    ax_3d = axes[4]
    ax_3d.set_title("5. Grayscale Topography DEM", color='#00ffcc', pad=10, weight='bold')
    ax_3d.set_facecolor('#0a0a0a')
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.shade(Z, cmap=plt.cm.gray, blend_mode='soft', vert_exag=1.5)
    
    surf = ax_3d.plot_surface(X, Y, Z, facecolors=shade, rstride=2, cstride=2, antialiased=True, shade=False)
    ax_3d.view_init(elev=55, azim=-45)
    ax_3d.axis('off')
    
    # Generate mapping for Elevation Colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.gray)
    m.set_array(Z)
    cbar = plt.colorbar(m, ax=ax_3d, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Surface Elevation', color='white', weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"System Error: {e}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] ARES-IX | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

class AresAdvancedQuantification:
    """
    Next-Gen Aerospace Rover AI.
    Features: CLAHE texture enhancement, DEM Elevation Scales, and Macro-Density Primary Anomaly Targeting.
    """
    
    def __init__(self, patch_size=16, contamination=0.03):
        self.patch_size = patch_size
        self.contamination = contamination

    def _apply_clahe(self, image):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for texture popping."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _generate_smooth_3d(self, image):
        """Generates a smoothed 3D mesh DEM."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        macro = cv2.GaussianBlur(smooth, (51, 51), 0)
        
        sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        micro = np.sqrt(sobel_x**2 + sobel_y**2)
        micro = cv2.normalize(micro, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        z_map = cv2.addWeighted(macro, 0.7, micro, 0.3, 0)
        scale = 0.2
        return cv2.resize(z_map, (0,0), fx=scale, fy=scale)

    def execute_scan(self, image_path):
        start_time = time.time()
        logging.info(f"Initiating ARES-IX sweep on: {image_path}")
        
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            logging.error("Optical sensor feed offline.")
            return None
            
        # 0. Apply CLAHE Enhancement
        image = self._apply_clahe(raw_image)
        h, w = image.shape[:2]
        total_pixels = h * w
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Biological & Chemical Quantification
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        bio_mask = ((hue > 25) & (hue < 90) & (sat > 30) & (val > 30)).astype(np.uint8) * 255
        liquid_mask = ((val > 210) & (sat < 40)).astype(np.uint8) * 255 
        
        bio_percent = (np.count_nonzero(bio_mask) / total_pixels) * 100
        liquid_percent = (np.count_nonzero(liquid_mask) / total_pixels) * 100
        
        chem_hud = raw_image.copy()
        chem_hud[bio_mask > 0] = [0, 255, 0]
        chem_hud[liquid_mask > 0] = [255, 0, 0]

        # 2. Extract Patches for ML
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobelx**2 + sobely**2)

        features, coords = [], []
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                patch = hsv[y:y+self.patch_size, x:x+self.patch_size]
                rough = np.std(edge_map[y:y+self.patch_size, x:x+self.patch_size])
                mean_hsv = np.mean(patch, axis=(0, 1))
                features.append([mean_hsv[0], mean_hsv[1], mean_hsv[2], rough])
                coords.append((x, y))
                
        features = np.array(features)

        # 3. Semantic GMM Surface Classification
        gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(features)
        
        cluster_stats = []
        for i in range(3):
            mask = (gmm == i)
            if np.any(mask):
                cluster_stats.append({
                    'id': i, 
                    'hue': np.mean(features[mask, 0]), 
                    'rough': np.mean(features[mask, 3]), 
                    'pct': (np.sum(mask) / len(gmm)) * 100
                })
                
        cluster_stats.sort(key=lambda x: abs(x['hue'] - 60))
        veg_id = cluster_stats[0]['id']
        remaining = cluster_stats[1:]
        remaining.sort(key=lambda x: x['rough'])
        smooth_id = remaining[0]['id']
        rock_id = remaining[1]['id']
        
        mapping = {
            veg_id: {"name": "Vegetation / Flora", "color": (50, 200, 50)},
            smooth_id: {"name": "Moisture / Smooth", "color": (200, 100, 50)},
            rock_id: {"name": "Soil / Bedrock", "color": (120, 120, 120)}
        }
        
        terrain_map = np.zeros_like(image)
        for (x, y), tid in zip(coords, gmm):
            terrain_map[y:y+self.patch_size, x:x+self.patch_size] = mapping[tid]["color"]

        legend_x, legend_y = 15, 30
        cv2.putText(terrain_map, "SURFACE MATERIAL LEGEND:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for cid in [veg_id, smooth_id, rock_id]:
            legend_y += 30
            color = mapping[cid]["color"]
            pct = next(item['pct'] for item in cluster_stats if item["id"] == cid)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
            cv2.rectangle(terrain_map, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), (255, 255, 255), 1) 
            cv2.putText(terrain_map, f"{mapping[cid]['name']} ({pct:.1f}%)", (legend_x + 35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. Advanced Anomaly Detection & Majority Zone Targeting
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        iso.fit(features)
        
        decision_scores = iso.decision_function(features)
        norm_scores = cv2.normalize(-decision_scores, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        hud = image.copy()
        heatmap_overlay = np.zeros_like(image)
        
        # Apply Heatmap (visual reference of all anomalies)
        for (x, y), score in zip(coords, norm_scores):
            if score > 160: 
                color = cv2.applyColorMap(np.array([[score]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0].tolist()
                heatmap_overlay[y:y+self.patch_size, x:x+self.patch_size] = color

        cv2.addWeighted(heatmap_overlay, 0.5, hud, 0.8, 0, hud)
        
        # TARGET THE MAJORITY ANOMALY ZONE
        anomaly_threshold = np.percentile(decision_scores, self.contamination * 100)
        anomaly_coords = np.array([coords[i] for i in range(len(decision_scores)) if decision_scores[i] < anomaly_threshold])
        
        primary_anomaly_density = 0
        
        if len(anomaly_coords) > 0:
            clustering = DBSCAN(eps=self.patch_size * 3.0, min_samples=4).fit(anomaly_coords)
            
            # Find the largest cluster (ignoring noise/outliers labeled as -1)
            unique_labels = set(clustering.labels_)
            unique_labels.discard(-1) 
            
            if unique_labels:
                # Count the density of each cluster
                cluster_sizes = {label: np.sum(clustering.labels_ == label) for label in unique_labels}
                
                # Identify the ID of the cluster with the most points
                majority_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
                primary_anomaly_density = cluster_sizes[majority_cluster_id]
                
                # Extract the coordinates specifically for the majority cluster
                points = anomaly_coords[clustering.labels_ == majority_cluster_id]
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0) + self.patch_size
                
                # Draw a prominent targeting box around the majority area
                # Box Color: High-Visibility Red (BGR: 0, 0, 255)
                cv2.rectangle(hud, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                
                # Draw targeting crosshairs on the corners
                crosshair_len = 20
                cv2.line(hud, (x_min, y_min), (x_min + crosshair_len, y_min), (0, 255, 255), 3)
                cv2.line(hud, (x_min, y_min), (x_min, y_min + crosshair_len), (0, 255, 255), 3)
                cv2.line(hud, (x_max, y_max), (x_max - crosshair_len, y_max), (0, 255, 255), 3)
                cv2.line(hud, (x_max, y_max), (x_max, y_max - crosshair_len), (0, 255, 255), 3)

                # Add a bold label
                label_text = f"PRIMARY ANOMALY ZONE (Density: {primary_anomaly_density})"
                cv2.putText(hud, label_text, (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 5. Topography Z-Map (Grayscale focus)
        Z = self._generate_smooth_3d(image)
        
        # 6. Generate Telemetry Report
        telemetry = {
            "Bio-Signature Match": f"{bio_percent:.2f}%",
            "Liquid Probability": f"{liquid_percent:.2f}%",
            "Primary Anomaly Density": f"{primary_anomaly_density} units",
            "Vegetation Coverage": f"{next(i['pct'] for i in cluster_stats if i['id'] == veg_id):.1f}%",
            "Exposed Soil/Rock": f"{next(i['pct'] for i in cluster_stats if i['id'] == rock_id):.1f}%"
        }
        
        return image, terrain_map, hud, chem_hud, Z, telemetry


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
target_image_path = "IMG_0387.jpg"

rover = AresAdvancedQuantification(patch_size=16, contamination=0.02)

try:
    enhanced_raw, terrain, hud, chem, Z, telemetry = rover.execute_scan(target_image_path)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0a0a0a')
    plt.rcParams['text.color'] = 'white'
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0:2]), 
        fig.add_subplot(gs[1, 2], projection='3d')
    ]

    titles = ["1. CLAHE Enhanced Feed", "2. Semantic Surface Clusters", "3. Bio/Liquid Quantifier", "4. Macro-Density Target Lock"]
    images = [enhanced_raw, terrain, chem, hud]

    for ax, title, img in zip(axes[:4], titles, images):
        ax.set_title(title, color='#00ffcc', pad=10, weight='bold')
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Add Telemetry Text Overlay to the Hazard Image
    y_pos = 30
    cv2.putText(hud, "LIVE TELEMETRY:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for key, val in telemetry.items():
        y_pos += 25
        cv2.putText(hud, f"{key}: {val}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    
    # Add Anomaly Legend to HUD
    cv2.putText(hud, "ANOMALY SEVERITY:", (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(hud, "RED = SEVERE | BLUE = LOW", (10, y_pos + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    axes[3].imshow(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)) 

    # Plot 5: High-Contrast Grayscale DEM (Digital Elevation Model) with Colorbar
    ax_3d = axes[4]
    ax_3d.set_title("5. Grayscale Topography DEM", color='#00ffcc', pad=10, weight='bold')
    ax_3d.set_facecolor('#0a0a0a')
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.shade(Z, cmap=plt.cm.gray, blend_mode='soft', vert_exag=1.5)
    
    surf = ax_3d.plot_surface(X, Y, Z, facecolors=shade, rstride=2, cstride=2, antialiased=True, shade=False)
    ax_3d.view_init(elev=55, azim=-45)
    ax_3d.axis('off')
    
    # Generate mapping for Elevation Colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.gray)
    m.set_array(Z)
    cbar = plt.colorbar(m, ax=ax_3d, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Surface Elevation', color='white', weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"System Error: {e}")
"""
aerial_anomaly_detector.py

Aerial-Guided Autonomous Rover — Recon Tower Analysis Module
--------------------------------------------------------------
Runs on the Mac ("brain") node. Consumes the overhead photo captured by
the recon tower iPhone and produces:

  1. A coordinate grid overlaid on the survey area (for the lawnmower
     sweep planner).
  2. A ranked list of Points of Interest (POIs) / anomaly zones, each
     with a normalized (x, y) grid coordinate and an anomaly score.
  3. An annotated JPEG for the mission dashboard.
  4. A JSON payload suitable for publishing on a ROS2 topic (e.g.
     /mission/poi_list) so the rover's path planner can pick it up.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] RECON-TOWER | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recon_tower")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class PointOfInterest:
    """One flagged patch, expressed in both pixel and normalized coords."""
    grid_row: int
    grid_col: int
    pixel_x: int
    pixel_y: int
    norm_x: float          # 0..1 across image width  -> map to survey-area X
    norm_y: float          # 0..1 across image height -> map to survey-area Y
    anomaly_score: float   # higher = more anomalous
    is_primary: bool = False  # the single highest-priority POI


@dataclass
class ScanResult:
    image_path: str
    image_width: int
    image_height: int
    patch_size: int
    grid_shape: Tuple[int, int]      # (rows, cols)
    pois: List[PointOfInterest] = field(default_factory=list)
    processing_time_s: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    def primary_poi(self) -> Optional[PointOfInterest]:
        return next((p for p in self.pois if p.is_primary), None)


# --------------------------------------------------------------------------
# Core detector
# --------------------------------------------------------------------------

class TerrainAnomalyDetector:
    """
    Zero-shot patch-based anomaly detector for an overhead survey image.

    Usage:
        detector = TerrainAnomalyDetector(patch_size=32, contamination=0.03)
        result = detector.analyze("survey_area.jpg")
        result_json = result.to_json()
    """

    def __init__(
        self,
        patch_size: int = 32,
        contamination: float = 0.03,
        max_dim: int = 1200,
        top_n_pois: int = 8,
        random_state: int = 42,
    ):
        if not (0 < contamination < 0.5):
            raise ValueError("contamination must be in (0, 0.5)")
        self.patch_size = patch_size
        self.contamination = contamination
        self.max_dim = max_dim
        self.top_n_pois = top_n_pois
        self.random_state = random_state

    # -- public API --------------------------------------------------------

    def analyze(self, image_path: str, output_overlay_path: Optional[str] = None) -> ScanResult:
        """Full pipeline: load -> resize -> features -> score -> rank -> annotate."""
        t0 = time.time()

        image = self._load_image(image_path)
        image = self._resize_if_needed(image)
        h, w = image.shape[:2]

        rows = h // self.patch_size
        cols = w // self.patch_size
        if rows < 2 or cols < 2:
            raise ValueError(
                f"Image too small ({w}x{h}) for patch_size={self.patch_size}; "
                "reduce patch_size or use a higher-res capture."
            )

        log.info(f"Scanning {w}x{h} image as a {rows}x{cols} patch grid "
                  f"({rows * cols} patches, patch_size={self.patch_size}px)")

        features, centers = self._extract_grid_features(image, rows, cols)
        scores = self._score_anomalies(features)
        pois = self._rank_pois(scores, centers, rows, cols, w, h)

        result = ScanResult(
            image_path=str(image_path),
            image_width=w,
            image_height=h,
            patch_size=self.patch_size,
            grid_shape=(rows, cols),
            pois=pois,
            processing_time_s=round(time.time() - t0, 3),
        )

        if output_overlay_path:
            self._save_overlay(image, result, output_overlay_path)
            log.info(f"Annotated overlay written to {output_overlay_path}")

        log.info(
            f"Scan complete in {result.processing_time_s}s — "
            f"{len(pois)} POI(s) flagged, primary at "
            f"({result.primary_poi().norm_x:.2f}, {result.primary_poi().norm_y:.2f})"
        )
        return result

    # -- pipeline stages -----------------------------------------------------

    def _load_image(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Survey image not found: {image_path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(
                f"OpenCV could not decode '{image_path}'. "
                "Check the file isn't a HEIC/live-photo export — "
                "convert to JPEG/PNG on the iPhone side first."
            )
        return image

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= self.max_dim:
            return image
        scale = self.max_dim / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _extract_grid_features(
        self, image: np.ndarray, rows: int, cols: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        For every grid cell, extract a feature vector robust to shadows and
        lighting variation:
          - HSV mean/std (color + brightness distribution)
          - Sobel gradient energy on grayscale (texture / edge density)
          - Local contrast (roughness proxy)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ps = self.patch_size
        features = []
        centers = []

        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * ps, (r + 1) * ps
                x0, x1 = c * ps, (c + 1) * ps

                hsv_patch = hsv[y0:y1, x0:x1]
                gray_patch = gray[y0:y1, x0:x1]

                mean_hsv = hsv_patch.reshape(-1, 3).mean(axis=0)
                std_hsv = hsv_patch.reshape(-1, 3).std(axis=0)

                sobel_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                edge_density = float(grad_mag.mean())
                roughness = float(grad_mag.std())

                local_contrast = float(gray_patch.std())

                vec = np.concatenate([
                    mean_hsv, std_hsv,
                    [edge_density, roughness, local_contrast],
                ])
                features.append(vec)
                centers.append((x0 + ps // 2, y0 + ps // 2))

        return np.array(features, dtype=np.float64), centers

    def _score_anomalies(self, features: np.ndarray) -> np.ndarray:
        """
        Fit Isolation Forest on this image's own patches. Higher score
        returned = more anomalous (we negate sklearn's raw output, which
        is high = normal, to make the API intuitive downstream).
        """
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        model.fit(features)
        raw_scores = model.decision_function(features)
        return -raw_scores

    def _rank_pois(
        self,
        scores: np.ndarray,
        centers: List[Tuple[int, int]],
        rows: int,
        cols: int,
        img_w: int,
        img_h: int,
    ) -> List[PointOfInterest]:
        order = np.argsort(scores)[::-1]
        top_idx = order[: self.top_n_pois]

        pois = []
        for rank, idx in enumerate(top_idx):
            r, c = divmod(int(idx), cols)
            px, py = centers[idx]
            poi = PointOfInterest(
                grid_row=r,
                grid_col=c,
                pixel_x=px,
                pixel_y=py,
                norm_x=round(px / img_w, 4),
                norm_y=round(py / img_h, 4),
                anomaly_score=round(float(scores[idx]), 4),
                is_primary=(rank == 0),
            )
            pois.append(poi)
        return pois

    def _save_overlay(self, image: np.ndarray, result: ScanResult, out_path: str) -> None:
        """Draw the sweep grid + ranked POI markers for the dashboard."""
        overlay = image.copy()
        ps = result.patch_size
        rows, cols = result.grid_shape

        for r in range(rows + 1):
            y = r * ps
            cv2.line(overlay, (0, y), (cols * ps, y), (60, 60, 60), 1)
        for c in range(cols + 1):
            x = c * ps
            cv2.line(overlay, (x, 0), (x, rows * ps), (60, 60, 60), 1)

        for rank, poi in enumerate(result.pois):
            color = (0, 0, 255) if poi.is_primary else (0, 165, 255)
            radius = 14 if poi.is_primary else 8
            cv2.circle(overlay, (poi.pixel_x, poi.pixel_y), radius, color, 2)
            label = "PRIMARY POI" if poi.is_primary else f"#{rank + 1}"
            cv2.putText(
                overlay, label, (poi.pixel_x + 12, poi.pixel_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        cv2.imwrite(str(out_path), overlay)


print("TerrainAnomalyDetector loaded.")
detector = TerrainAnomalyDetector(patch_size=32, contamination=0.03, top_n_pois=8)
result = detector.analyze(
    "IMG_0387.jpg",                       # <- your image filename here
    output_overlay_path="anomaly_scan_overlay.jpg",
)

print(result.to_json())
from IPython.display import Image, display
display(Image("anomaly_scan_overlay.jpg"))
"""
Unsupervised terrain analysis: NumPy autoencoder anomaly detection,
unsupervised surface clustering, liquid/vegetation heuristic scoring,
and relative elevation — all fit live on the current aerial photo,
no pretrained model, no external dataset.
"""

from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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

    def analyze(self, image_path: str):
        t0 = time.time()
        image = self._load_image(image_path)
        image = self._resize_if_needed(image)
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
            image_path=str(image_path), image_width=w, image_height=h, patch_size=ps,
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

print("UnsupervisedTerrainAnalyzer loaded.")
analyzer = UnsupervisedTerrainAnalyzer(
    patch_size=32,
    field_width_m=10.0,   # your actual test-area width, meters
    field_height_m=8.0,   # your actual test-area height, meters
)

scan, images = analyzer.analyze("IMG_0387.jpg")  # use your real filename

print(scan.to_json())
from IPython.display import Image, display
import cv2

cv2.imwrite("dashboard.jpg", images["dashboard"])
display(Image("dashboard.jpg"))   # 4-panel: elevation / classification / life-liquid / anomaly overlay
 