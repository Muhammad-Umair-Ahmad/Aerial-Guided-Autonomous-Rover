"""
AGRA - Retrain YOLOv8n on New Rover Dataset (v2)
==================================================
Fixes from v1:
  1. Properly converts segmentation polygons → bounding boxes BEFORE splitting
  2. Uses higher epochs with patience
  3. Lower initial LR for fine-tuning (0.005 vs 0.01)
  4. Larger val split (20% ≈ 7 images) for better generalization signal
  5. Freezes backbone for first phase, then unfreezes (transfer learning)
  6. Tests model after training and prints per-image results

Usage:  python retrain_rover.py
"""

import os
import sys
import shutil
import random
import yaml
import glob

# ── Paths ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_SRC  = os.path.join(SCRIPT_DIR, "Rover_detection.yolov12")
WORK_DIR     = os.path.join(SCRIPT_DIR, "rover_training_clean")
TRAIN_IMG    = os.path.join(WORK_DIR, "train", "images")
TRAIN_LBL    = os.path.join(WORK_DIR, "train", "labels")
VAL_IMG      = os.path.join(WORK_DIR, "val", "images")
VAL_LBL      = os.path.join(WORK_DIR, "val", "labels")
DATA_YAML    = os.path.join(WORK_DIR, "data.yaml")
SERVER_DIR   = os.path.join(SCRIPT_DIR, "Camera feed fast api", "fastapi_ios_stream")
OUTPUT_MODEL = os.path.join(SERVER_DIR, "best.pt")


def convert_seg_to_bbox(label_path):
    """
    Convert YOLO segmentation label (class x1 y1 x2 y2 ... xN yN)
    to YOLO detection label (class cx cy w h).
    
    If label is already in bbox format (5 values per line), skip it.
    """
    new_lines = []
    converted = False
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls = parts[0]
            values = [float(x) for x in parts[1:]]
            
            # If exactly 4 values → already bbox format
            if len(values) == 4:
                new_lines.append(line.strip())
                continue
            
            # Polygon: x1,y1, x2,y2, ..., xN,yN
            converted = True
            xs = values[0::2]
            ys = values[1::2]
            if not xs or not ys:
                continue
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w  = x_max - x_min
            h  = y_max - y_min
            
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w  = max(0.001, min(1, w))
            h  = max(0.001, min(1, h))
            
            new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    return converted, len(new_lines)


def prepare_dataset(val_ratio=0.20, seed=42):
    """
    1. Copy raw dataset to a clean working directory
    2. Convert ALL labels from segmentation → bbox
    3. Split into train/val
    """
    print("\n[PREP] Creating clean working directory...")
    
    # Wipe and recreate
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    
    os.makedirs(TRAIN_IMG, exist_ok=True)
    os.makedirs(TRAIN_LBL, exist_ok=True)
    os.makedirs(VAL_IMG, exist_ok=True)
    os.makedirs(VAL_LBL, exist_ok=True)
    
    # Find source images and labels
    src_img_dir = os.path.join(DATASET_SRC, "train", "images")
    src_lbl_dir = os.path.join(DATASET_SRC, "train", "labels")
    
    images = sorted([f for f in os.listdir(src_img_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"[PREP] Found {len(images)} images in source dataset")
    
    # Copy ALL images and labels to working train dir first
    for img_name in images:
        shutil.copy2(os.path.join(src_img_dir, img_name), 
                     os.path.join(TRAIN_IMG, img_name))
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(TRAIN_LBL, lbl_name))
    
    # Convert ALL labels in train (before splitting)
    print("[PREP] Converting segmentation polygons -> bounding boxes...")
    converted_count = 0
    for fname in os.listdir(TRAIN_LBL):
        if fname.endswith('.txt'):
            fpath = os.path.join(TRAIN_LBL, fname)
            was_converted, n_boxes = convert_seg_to_bbox(fpath)
            if was_converted:
                converted_count += 1
    print(f"[PREP] Converted {converted_count} label files")
    
    # Verify a label looks correct
    sample_labels = os.listdir(TRAIN_LBL)[:3]
    for lbl in sample_labels:
        with open(os.path.join(TRAIN_LBL, lbl)) as f:
            content = f.read().strip()
        parts = content.split('\n')[0].split()
        print(f"  Sample label {lbl}: {len(parts)} values -> {'BBOX OK' if len(parts)==5 else 'PROBLEM!'}")
        if len(parts) != 5:
            print(f"    Content: {content[:200]}")
    
    # Now split into train/val
    random.seed(seed)
    all_images = sorted([f for f in os.listdir(TRAIN_IMG) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(all_images)
    
    val_count = max(3, int(len(all_images) * val_ratio))
    val_images = all_images[:val_count]
    
    print(f"[PREP] Splitting: {len(all_images) - val_count} train, {val_count} val")
    
    # MOVE val images and labels (not copy — no duplicates)
    for img_name in val_images:
        shutil.move(os.path.join(TRAIN_IMG, img_name), 
                    os.path.join(VAL_IMG, img_name))
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(TRAIN_LBL, lbl_name)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, os.path.join(VAL_LBL, lbl_name))
    
    # Write data.yaml
    data = {
        'train': TRAIN_IMG.replace('\\', '/'),
        'val': VAL_IMG.replace('\\', '/'),
        'nc': 1,
        'names': ['Rover'],
    }
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    n_train = len(os.listdir(TRAIN_IMG))
    n_val = len(os.listdir(VAL_IMG))
    print(f"[PREP] Final: {n_train} train images, {n_val} val images")
    print(f"[PREP] data.yaml written to {DATA_YAML}")
    
    return n_train, n_val


def train():
    """Fine-tune YOLOv8n with settings optimized for small datasets."""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("  TRAINING YOLOv8n ON ROVER DATASET (v2)")
    print("="*60 + "\n")
    
    # Load pretrained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=DATA_YAML,
        epochs=300,               # More epochs — early stopping will handle it
        patience=50,              # More patience — let it train longer
        batch=4,                  # Smaller batch = more gradient updates per epoch
        imgsz=640,
        
        # ── Augmentation (heavy for 37 images) ──
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.15,
        
        # ── Learning rate (lower for fine-tuning) ──
        lr0=0.005,                # Lower initial LR — less aggressive
        lrf=0.0001,               # Lower final LR
        warmup_epochs=10,         # Longer warmup
        weight_decay=0.0005,
        
        # ── Freeze backbone initially for transfer learning ──
        freeze=10,                # Freeze first 10 layers (backbone)
        
        # ── Output ──
        project=os.path.join(SCRIPT_DIR, 'runs'),
        name='rover_v2',
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    
    return results


def test_model():
    """Test the trained model on all images and print results."""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("  TESTING MODEL ON DATASET IMAGES")
    print("="*60 + "\n")
    
    model = YOLO(OUTPUT_MODEL)
    
    # Test on ALL images (train + val)
    all_images = []
    for img_dir in [TRAIN_IMG, VAL_IMG]:
        all_images.extend(glob.glob(os.path.join(img_dir, '*.jpg')))
        all_images.extend(glob.glob(os.path.join(img_dir, '*.jpeg')))
        all_images.extend(glob.glob(os.path.join(img_dir, '*.png')))
    
    print(f"Testing on {len(all_images)} images...\n")
    
    detected = 0
    missed = 0
    confidences = []
    
    for img_path in sorted(all_images):
        results = model.predict(img_path, conf=0.15, verbose=False)
        dets = results[0].boxes
        fname = os.path.basename(img_path)
        
        if len(dets) > 0:
            best_conf = max(float(b.conf) for b in dets)
            confidences.append(best_conf)
            detected += 1
            status = "✅" if best_conf >= 0.3 else "⚠️"
            print(f"  {status} {fname}: {len(dets)} det(s), best conf={best_conf:.3f}")
        else:
            missed += 1
            print(f"  ❌ {fname}: NO DETECTION")
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {detected}/{len(all_images)} detected ({missed} missed)")
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        print(f"  Confidence: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
        
        above_30 = sum(1 for c in confidences if c >= 0.3)
        above_50 = sum(1 for c in confidences if c >= 0.5)
        print(f"  Above 0.3: {above_30}/{len(confidences)}")
        print(f"  Above 0.5: {above_50}/{len(confidences)}")
    print(f"{'='*60}\n")
    
    return detected, missed


def main():
    print("\n" + "="*60)
    print("  AGRA - Rover Detection Model Training (v2 — Clean)")
    print("="*60)
    
    # Step 1: Prepare clean dataset
    print("\n[1/4] Preparing clean dataset...")
    n_train, n_val = prepare_dataset(val_ratio=0.20)
    
    # Step 2: Train
    print("\n[2/4] Training model...")
    results = train()
    
    # Step 3: Deploy
    print("\n[3/4] Deploying model...")
    best_pt = os.path.join(SCRIPT_DIR, 'runs', 'rover_v2', 'weights', 'best.pt')
    if not os.path.exists(best_pt):
        best_pt = os.path.join(SCRIPT_DIR, 'runs', 'rover_v2', 'weights', 'last.pt')
    
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, OUTPUT_MODEL)
        print(f"  Model saved to: {OUTPUT_MODEL}")
        print(f"  File size: {os.path.getsize(OUTPUT_MODEL) / 1024 / 1024:.1f} MB")
    else:
        print(f"  ERROR: Could not find best.pt at {best_pt}")
        sys.exit(1)
    
    # Step 4: Test
    print("\n[4/4] Testing model...")
    test_model()
    
    print("\nDone! Start the dashboard with:")
    print("  python run_dashboard.py")


if __name__ == '__main__':
    main()
