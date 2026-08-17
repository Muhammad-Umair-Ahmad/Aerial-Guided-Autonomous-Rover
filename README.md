# AGRA Mission Control

**AGRA (Aerial-Guided Rover Autonomy)** is an autonomous physical rover system that combines computer vision, real-time telemetry, and embedded hardware control. An overhead camera feeds a fine-tuned YOLOv8 model that locates the rover in real time, a FastAPI backend streams video and telemetry to a web-based mission control dashboard, and an ESP32 microcontroller executes movement commands on the physical rover.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
  - [1. Machine Learning & Model Training](#1-machine-learning--model-training)
  - [2. Backend & Mission Control Dashboard](#2-backend--mission-control-dashboard)
  - [3. Autonomy Stack](#3-autonomy-stack)
  - [4. Hardware / Physical Layer](#4-hardware--physical-layer)
  - [5. Miscellaneous / Legacy Folders](#5-miscellaneous--legacy-folders)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Detection Model](#training-the-detection-model)
  - [Running the Mission Control Dashboard](#running-the-mission-control-dashboard)
- [Directory Structure](#directory-structure)

---

## Overview

AGRA is designed to let a rover navigate an environment autonomously while being monitored and controlled from a browser-based dashboard. The system is split into four cooperating layers:

1. **Perception** — an overhead camera detects the rover's position using a custom-trained YOLOv8 model.
2. **Mission Control** — a FastAPI server streams live video (via WebRTC) and telemetry (via WebSockets) to a dashboard, and exposes REST/WebSocket endpoints to start and stop autonomous missions.
3. **Autonomy** — a modular stack handles navigation, world modeling, state machines, and safety/failsafe logic.
4. **Hardware** — an ESP32 microcontroller receives movement commands and drives the rover's motors.

---

## Architecture

### 1. Machine Learning & Model Training
Located in the project root, these scripts and assets handle detection model training:

| File / Folder | Purpose |
| :--- | :--- |
| `train_rover.py` | End-to-end training script. Loads the `Rover_detection.yolov12` dataset, splits it into train/val sets, converts YOLO segmentation polygon labels into bounding-box labels, and fine-tunes `yolov8n.pt` with heavy augmentation (rotation, mosaic, mixup, etc.). Automatically copies the resulting `best.pt` into the FastAPI server directory. |
| `retrain_rover.py` | Retraining script for iterating on an existing model with new data. |
| `Rover_detection.yolov12/` | Dataset directory with `train/` and `val/` folders (images + labels) and a `data.yaml` config. |
| `yolov8n.pt` | Base YOLOv8-nano pretrained weights used as the transfer-learning starting point. |

### 2. Backend & Mission Control Dashboard
Located in `Camera feed fast api/fastapi_ios_stream/`, this is the core server running the system.

| File | Purpose |
| :--- | :--- |
| `main.py` | FastAPI entry point. Serves `dashboard.html` and `broadcaster.html`; handles WebRTC signaling over WebSockets (`/ws`) for iOS-to-laptop video streaming; exposes computer vision endpoints (`/ws/cv`) that run YOLO inference and feed the autopilot; exposes autopilot telemetry endpoints and REST routes to start/stop missions. |
| `dashboard.html` | Mission control UI — live stream, rover position, telemetry, mission status. |
| `broadcaster.html` | Broadcaster-side UI used by the streaming device (e.g., an iPhone) to publish video. |
| `cv_pipeline.py` | Computer vision pipeline — object detection, floor-grid detection, and related processing. |
| `terrain_analyzer.py` | Generates terrain heatmaps and surface analysis from vision data. |
| `autopilot.py` | `AutopilotEngine` — orchestrates autonomous missions, manages waypoints, and communicates with the ESP32. |
| `best.pt` | Final trained YOLO weights used for live inference (produced by `train_rover.py`). |

### 3. Autonomy Stack
Located in `mars_rover/`, this holds the modular, higher-level autonomy logic:

| Folder | Contents |
| :--- | :--- |
| `communication/` | `config.py` — communication configuration. |
| `navigation/` | `controller.py`, `state_machine.py`, `world_model.py` — motion control, mission state machine, and the rover's model of its environment. |
| `perception/` | `perception_manager.py` — coordinates perception inputs for the autonomy stack. |
| `safety/` | `failsafe.py` — safety monitoring and failsafe behaviors (e.g., geofence violations). |

### 4. Hardware / Physical Layer
Located in `physical_AI/`:

| File | Purpose |
| :--- | :--- |
| `car_esp32/esp32.py` | Interfaces with the ESP32 microcontroller, translating high-level commands into motor movements. |
| `phase1_calibration.py` | Calibration script for the physical hardware setup (motor speeds, movement sequences, etc.). |

### 5. Miscellaneous / Legacy Folders

| Folder / File | Notes |
| :--- | :--- |
| `archive/V1 trace/` | A Dockerized app (`Dockerfile`, `docker-compose.yml`, `app/`) — likely an earlier iteration or a separate tracing service. |
| `archive/Viz/` | Contains `worlds/`, likely used for simulation or 3D visualization. |
| `archive/side_work/` | Raw image dump (`img1.jpeg`–`img37.jpeg`). |
| `docs/media/` | Debug media documenting the physical grid setup and hardware issues (`issue currently.mp4`, `current grid.jpeg`, `marked+grid.jpeg`, `tire connection.jpeg`). |

---

## Getting Started

### Prerequisites
- Python 3.9+
- `pip` (or a virtual environment tool of your choice)
- An NVIDIA GPU (recommended) for faster training, though CPU training is supported
- An ESP32 board flashed and connected on the same network, if running with live hardware
- A device (e.g., iPhone) capable of broadcasting video via WebRTC for the mission control stream

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics fastapi uvicorn opencv-python numpy websockets
```
> **Note:** Adjust the dependency list to match your actual `requirements.txt` if one exists in the repository.

### Training the Detection Model
The rover-detection model is trained with `train_rover.py`, using the dataset in `Rover_detection.yolov12/`.

```bash
python train_rover.py
```
What this script does:
1. Loads and splits the `Rover_detection.yolov12` dataset into training and validation sets.
2. Converts YOLO segmentation polygon labels into YOLO bounding-box labels.
3. Fine-tunes the pretrained `yolov8n.pt` model with augmentation (rotation, mosaic, mixup, etc.).
4. Saves the best-performing checkpoint as `best.pt`.
5. Automatically copies `best.pt` into `Camera feed fast api/fastapi_ios_stream/`, where the FastAPI server loads it for live inference.

To retrain on new data instead of training from scratch, use:
```bash
python retrain_rover.py
```
*(Training logs are written to `train_output.log` and `retrain_output.log` respectively.)*

### Running the Mission Control Dashboard
The dashboard and backend live in `Camera feed fast api/fastapi_ios_stream/`.

```bash
cd "Camera feed fast api/fastapi_ios_stream"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once the server is running:
1. Open `broadcaster.html` on the streaming device (e.g., an iPhone) to begin publishing the overhead camera feed via WebRTC.
2. Open `dashboard.html` in a browser on your laptop to view the live stream, rover position, and telemetry, and to start or stop autonomous missions.

**The server will:**
- Run YOLO inference on incoming frames (`/ws/cv`) using `best.pt`.
- Feed detections into `autopilot.py`'s `AutopilotEngine` for waypoint navigation and geofence monitoring.
- Relay commands to the ESP32 (`physical_AI/car_esp32/esp32.py`) to drive the rover's motors.

> **Tip:** Run `physical_AI/phase1_calibration.py` first when setting up new hardware, to calibrate motor movement sequences before starting an autonomous mission.

---

## Directory Structure

```
.
├── train_rover.py
├── retrain_rover.py
├── yolov8n.pt
├── README.md
├── Rover_detection.yolov12/
│   ├── train/
│   ├── val/
│   └── data.yaml
├── Camera feed fast api/
│   └── fastapi_ios_stream/
│       ├── main.py
│       ├── dashboard.html
│       ├── broadcaster.html
│       ├── cv_pipeline.py
│       ├── terrain_analyzer.py
│       ├── autopilot.py
│       └── best.pt
├── mars_rover/
│   ├── communication/
│   │   └── config.py
│   ├── navigation/
│   │   ├── controller.py
│   │   ├── state_machine.py
│   │   └── world_model.py
│   ├── perception/
│   │   └── perception_manager.py
│   └── safety/
│       └── failsafe.py
├── physical_AI/
│   ├── car_esp32/
│   │   └── esp32.py
│   └── phase1_calibration.py
├── archive/
│   ├── V1 trace/
│   ├── Viz/
│   └── side_work/
└── docs/
    └── media/
        └── (debug media: issue currently.mp4, current grid.jpeg, marked+grid.jpeg, tire connection.jpeg)
```
