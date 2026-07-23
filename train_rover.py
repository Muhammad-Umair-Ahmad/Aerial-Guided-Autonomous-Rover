"""
AGRA - Fine-tune YOLOv8n on Rover Detection Dataset
=====================================================
Splits train-only Roboflow export into train/val,
then fine-tunes yolov8n.pt with heavy augmentation.

Usage:  python train_rover.py
Output: best.pt copied to the FastAPI server directory
"""

import os
import sys
import shutil
import random
import yaml

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "Rover_detection.yolov12")
TRAIN_IMG    = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL    = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG      = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL      = os.path.join(DATASET_DIR, "val", "labels")
DATA_YAML    = os.path.join(DATASET_DIR, "data.yaml")
SERVER_DIR   = os.path.join(SCRIPT_DIR, "Camera feed fast api", "fastapi_ios_stream")
OUTPUT_MODEL = os.path.join(SERVER_DIR, "best.pt")


def convert_segmentation_to_bbox(label_path):
    """
    Convert YOLO segmentation labels (class x1 y1 x2 y2 ... xN yN)
    to YOLO detection labels (class cx cy w h).
    Overwrites the file in-place.
    """
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            coords = [float(x) for x in parts[1:]]
            # Polygon points: x1,y1, x2,y2, ...
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            w  = x_max - x_min
            h  = y_max - y_min
            new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    return len(new_lines)


def split_dataset(val_count=4, seed=42):
    """Split train/ into train/ and val/ sets."""
    # Skip if val already exists with images
    if os.path.exists(VAL_IMG) and len(os.listdir(VAL_IMG)) > 0:
        n_train = len([f for f in os.listdir(TRAIN_IMG) if f.endswith(('.jpg', '.jpeg', '.png'))])
        n_val = len([f for f in os.listdir(VAL_IMG) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"[SPLIT] Val already exists ({n_val} images). Skipping split.")
        return n_train, n_val

    random.seed(seed)

    # Get all image files
    images = sorted([f for f in os.listdir(TRAIN_IMG) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[SPLIT] Found {len(images)} images in train/")

    if len(images) <= val_count:
        print(f"[SPLIT] Not enough images to split (need > {val_count}). Using all for train and val.")
        val_count = max(1, len(images) // 4)

    # Shuffle and split
    random.shuffle(images)
    val_images = images[:val_count]
    train_images = images[val_count:]

    print(f"[SPLIT] Train: {len(train_images)}, Val: {len(val_images)}")

    # Create val directories
    os.makedirs(VAL_IMG, exist_ok=True)
    os.makedirs(VAL_LBL, exist_ok=True)

    # Move val images and labels
    for img_name in val_images:
        # Move image
        src_img = os.path.join(TRAIN_IMG, img_name)
        dst_img = os.path.join(VAL_IMG, img_name)
        shutil.copy2(src_img, dst_img)

        # Move label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(TRAIN_LBL, label_name)
        dst_lbl = os.path.join(VAL_LBL, label_name)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"[SPLIT] WARNING: No label for {img_name}")

    return len(train_images), len(val_images)


def convert_all_labels():
    """Convert all segmentation labels to bbox format."""
    print("[CONVERT] Converting segmentation polygons -> bounding boxes...")
    for label_dir in [TRAIN_LBL, VAL_LBL]:
        if not os.path.exists(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith('.txt'):
                fpath = os.path.join(label_dir, fname)
                count = convert_segmentation_to_bbox(fpath)
                # print(f"  {fname}: {count} boxes")
    print("[CONVERT] Done!")


def fix_data_yaml():
    """Update data.yaml with correct absolute paths."""
    data = {
        'train': os.path.join(DATASET_DIR, 'train', 'images').replace('\\', '/'),
        'val':   os.path.join(DATASET_DIR, 'val', 'images').replace('\\', '/'),
        'nc': 1,
        'names': ['Rover'],
    }
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[YAML] Updated {DATA_YAML}")
    print(f"  train: {data['train']}")
    print(f"  val:   {data['val']}")


def train():
    """Fine-tune YOLOv8n on the rover dataset."""
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  TRAINING YOLOv8n ON ROVER DATASET")
    print("="*60 + "\n")

    # Load pretrained YOLOv8n
    model = YOLO('yolov8n.pt')

    # Train with heavy augmentation for small dataset
    results = model.train(
        data=DATA_YAML,
        epochs=150,
        patience=30,          # Early stopping after 30 epochs with no improvement
        batch=8,              # Small batch for small dataset
        imgsz=640,            # Standard YOLO input size
        
        # ── Heavy augmentation for small dataset ──
        augment=True,
        hsv_h=0.02,           # Hue augmentation
        hsv_s=0.7,            # Saturation augmentation
        hsv_v=0.4,            # Value/brightness augmentation
        degrees=15.0,         # Rotation ±15°
        translate=0.15,       # Translation
        scale=0.5,            # Scale augmentation (±50%)
        shear=5.0,            # Shear
        perspective=0.001,    # Perspective warp
        flipud=0.3,           # Vertical flip 30%
        fliplr=0.5,           # Horizontal flip 50%
        mosaic=1.0,           # Mosaic augmentation ON
        mixup=0.15,           # Mixup augmentation
        copy_paste=0.1,       # Copy-paste augmentation
        
        # ── Training params ──
        lr0=0.01,             # Initial learning rate
        lrf=0.001,            # Final learning rate
        warmup_epochs=5,      # Warmup
        weight_decay=0.0005,
        
        # ── Output ──
        project=os.path.join(SCRIPT_DIR, 'runs'),
        name='rover_detection',
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # Find best.pt
    best_pt = os.path.join(SCRIPT_DIR, 'runs', 'rover_detection', 'weights', 'best.pt')
    if not os.path.exists(best_pt):
        # Try last.pt as fallback
        best_pt = os.path.join(SCRIPT_DIR, 'runs', 'rover_detection', 'weights', 'last.pt')

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, OUTPUT_MODEL)
        print(f"\n{'='*60}")
        print(f"  SUCCESS! Model saved to: {OUTPUT_MODEL}")
        print(f"  File size: {os.path.getsize(OUTPUT_MODEL) / 1024 / 1024:.1f} MB")
        print(f"{'='*60}\n")
    else:
        print(f"\nERROR: Could not find best.pt at {best_pt}")
        print("Check the runs/ directory for training output.")
        sys.exit(1)

    return results


def main():
    print("\n" + "="*60)
    print("  AGRA - Rover Detection Model Training")
    print("="*60 + "\n")

    # Step 1: Split dataset
    print("[1/4] Splitting dataset into train/val...")
    n_train, n_val = split_dataset(val_count=4)

    # Step 2: Convert segmentation labels to bounding boxes
    print(f"\n[2/4] Converting labels...")
    convert_all_labels()

    # Step 3: Fix data.yaml
    print(f"\n[3/4] Fixing data.yaml...")
    fix_data_yaml()

    # Step 4: Train
    print(f"\n[4/4] Starting training...")
    train()

    print("\nDone! You can now start the dashboard with:")
    print("  python run_dashboard.py")


if __name__ == '__main__':
    main()
