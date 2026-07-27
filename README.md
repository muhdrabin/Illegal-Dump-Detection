# 🗑️ Illegal Waste Dumping Detection System

An AI-powered surveillance pipeline that automatically detects illegal waste dumping events in real-time video. The system tracks persons and vehicles using a pretrained COCO model, detects trash using a custom-trained YOLOv8 model, and confirms dump events using a finite state machine with Kalman-stabilised trash tracking. Evidence clips and actor photos are saved automatically on dump confirmation.

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Running the Tracker](#running-the-tracker)
- [Dump Detection Logic](#dump-detection-logic)
- [Output Files](#output-files)
- [Tunable Parameters](#tunable-parameters)
- [Hardware](#hardware)

---

## How It Works

1. **Dual YOLO inference** — A pretrained YOLOv8 COCO model detects persons and vehicles. A custom-trained YOLOv8 model detects trash. Both run on every frame.
2. **Kalman-stabilised trash tracking** — Raw ByteTrack IDs for trash are wrapped in a custom Kalman manager (`TrashKalmanManager`) that assigns stable IDs regardless of ByteTrack ID switches — critical for stationary objects.
3. **Baseline phase** — First 60 frames catalogue all pre-existing trash as `BASELINE` and permanently exclude them from dump detection.
4. **FSM dump detection** — Each new trash track goes through a 3-stage state machine: `NEW → ASSOCIATED → DUMPED`.
5. **Evidence saving** — On dump confirmation, a 10-second annotated evidence clip (from a RAM ring buffer) and a cropped actor photo are saved automatically.

---

## System Architecture

```
Video / Webcam
      │
      ▼
┌─────────────────────────────────────────┐
│         Frame-by-Frame Loop             │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  COCO Model  │  │  Trash Model     │ │
│  │  (YOLOv8l)   │  │  (YOLOv8m)       │ │
│  │  persons +   │  │  custom trained  │ │
│  │  vehicles    │  │  best.pt         │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         │                   │           │
│         │         ┌─────────▼─────────┐ │
│         │         │ TrashKalmanManager│ │
│         │         │ (stable IDs)      │ │
│         │         └─────────┬─────────┘ │
│         │                   │           │
│         └──────────┬────────┘           │
│                    │                    │
│         ┌──────────▼────────┐           │
│         │  Dump FSM         │           │
│         │  NEW → ASSOCIATED │           │
│         │       → DUMPED    │           │
│         └──────────┬────────┘           │
│                    │                    │
│         ┌──────────▼────────┐           │
│         │  Evidence Saver   │           │
│         │  clip.mp4         │           │
│         │  actor_crop.jpg   │           │
│         └───────────────────┘           │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
├── train_trash.py                  # YOLOv8 training script
├── test_trash.py                   # Inference, evaluation, and testing script
├── person_vehicle_trash_tracker.py # Main tracker and dump detection pipeline
├── trash_dataset.yaml              # Auto-generated dataset config (after training)
├── cctv.mp4                        # Input video
├── runs/
│   └── detect/runs/trash_detection/v1.6/
│                                   ├── weights/
│                                   │   ├── best.pt             # Best checkpoint
│                                   │   └── last.pt             # Final checkpoint
│                                   ├── results.png             # Training curves
│                                   ├── PR_curve.png
│                                   ├── confusion_matrix.png
│                                   └── eval_results.json       # Validation metrics
└── dump_events/                    # Auto-created on first dump confirmation
    ├── dump_<timestamp>_t<id>_clip.mp4
    ├── dump_<timestamp>_t<id>_person_crop.jpg
    └── dump_<timestamp>_t<id>_vehicle_crop.jpg
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (tested on RTX 4060 Laptop 8GB)
- CUDA 11.8+

### Python Dependencies

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
PyYAML>=6.0
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/waste-dump-detection.git
cd waste-dump-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install ultralytics opencv-python pyyaml

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Preparation

Dataset must follow the YOLOv8 folder structure:

```
trash_dataset/
├── train/
│   ├── images/       # .jpg / .png training images
│   └── labels/       # .txt YOLO-format annotations
└── valid/
    ├── images/
    └── labels/
```

Each `.txt` label file follows YOLO format (one object per line):
```
<class_id> <cx> <cy> <width> <height>
```
All values normalised to [0, 1] relative to image dimensions.

### Verify your dataset

```bash
python train_trash.py --dataset ./trash_dataset --verify-only
```

---

## Training

### Basic training (recommended defaults)

```bash
python train_trash.py --dataset ./trash_dataset --model-size m
```

### Full control

```bash
python train_trash.py \
  --dataset ./trash_dataset \
  --classes trash \
  --model-size m \
  --epochs 100 \
  --batch 12 \
  --patience 20 \
  --imgsz 640 \
  --device 0 \
  --workers 6
```

### Validate a saved checkpoint only

```bash
python train_trash.py \
  --dataset ./trash_dataset \
  --validate-only \
  --model runs/trash_detection/weights/best.pt
```

### Safe batch size defaults per model variant

| Model | Batch | VRAM (approx) |
|-------|-------|---------------|
| n     | 32    | ~2 GB         |
| s     | 16    | ~3 GB         |
| m     | 12    | ~5 GB         |
| l     | 8     | ~7 GB         |
| x     | 6     | ~8 GB         |

Training uses mixed precision (`amp=True`) by default — saves ~30% VRAM.

### Training Outputs

After training completes, the following are saved in `runs/trash_detection/<run_name>/`:

| File | Description |
|------|-------------|
| `weights/best.pt` | Best model checkpoint by mAP |
| `weights/last.pt` | Final epoch checkpoint |
| `results.png` | Training and validation loss curves |
| `PR_curve.png` | Precision-Recall curve |
| `F1_curve.png` | F1 vs confidence threshold |
| `confusion_matrix.png` | Normalised confusion matrix |
| `eval_results.json` | mAP, Precision, Recall, F1 |

---

## Testing & Evaluation

### Single image

```bash
python test_trash.py --model best.pt --mode image --input photo.jpg
```

### Batch inference on a folder

```bash
python test_trash.py --model best.pt --mode folder \
  --input ./test_images/ --output ./results/
```

### Video file

```bash
python test_trash.py --model best.pt --mode video \
  --input clip.mp4 --output out.mp4
```

### Live webcam

```bash
python test_trash.py --model best.pt --mode webcam
```

### Full evaluation (mAP / Precision / Recall / F1)

```bash
python test_trash.py --model best.pt --mode evaluate \
  --data trash_dataset.yaml
```

Sample evaluation output:
```
=======================================================
  EVALUATION RESULTS
=======================================================
  mAP @ 0.5        : 0.8731  (87.31%)
  mAP @ 0.5:0.95   : 0.6214  (62.14%)
  Precision        : 0.8902  (89.02%)
  Recall           : 0.8445  (84.45%)
  F1-Score         : 0.8667  (86.67%)
=======================================================
```

---

## Running the Tracker

### Video file

```bash
python person_vehicle_trash_tracker.py \
  --mode video \
  --input road.mp4 \
  --trash-model best.pt \
  --no-display
```

### Live webcam

```bash
python person_vehicle_trash_tracker.py \
  --mode webcam \
  --trash-model best.pt
```

### Single image

```bash
python person_vehicle_trash_tracker.py \
  --mode image \
  --input frame.jpg \
  --trash-model best.pt
```

### Keyboard controls (video / webcam modes)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `T` | Toggle motion trails |
| `R` | Reset tracker (webcam only) |

---

## Dump Detection Logic

Each trash track goes through a finite state machine with 5 states:

```
                  ┌─────────────────────────────────┐
                  │         BASELINE                │
                  │  (pre-existing trash, ignored)  │
                  └─────────────────────────────────┘

NEW ──────────────────────────────────► DISMISSED
 │   actor moves away before           (association
 │   assoc_frames threshold            broken)
 │
 │  actor within PROXIMITY_PX
 │  for ASSOC_MIN_FRAMES frames
 ▼
ASSOCIATED
 │
 │  actor moves > DEPART_PX away
 │  AND trash velocity < STATIONARY_VEL
 │  for STATIONARY_FRAMES frames
 ▼
DUMPED ──► evidence clip + actor crop saved
```

### State colour coding on screen

| State | Colour | Meaning |
|-------|--------|---------|
| NEW | Yellow | Trash appeared, watching |
| ASSOCIATED | Orange | Actor confirmed near trash |
| DUMPED | Red | Dump event confirmed |
| BASELINE | Grey | Pre-existing trash, ignored |

---

## Output Files

Every confirmed dump event saves the following to `./dump_events/`:

| File | Description |
|------|-------------|
| `dump_<ts>_t<id>_clip.mp4` | 10-second annotated evidence video (pre + post event) |
| `dump_<ts>_t<id>_person_crop.jpg` | Cropped photo of the person at highest detection confidence |
| `dump_<ts>_t<id>_vehicle_crop.jpg` | Cropped photo of the vehicle at highest detection confidence |

---

## Tunable Parameters

### FSM / Dump Detection

| Flag | Default | Description |
|------|---------|-------------|
| `--proximity` | 150 | Distance (px) actor must be to trash for association |
| `--assoc-frames` | 8 | Frames actor must stay close to confirm association |
| `--depart` | 180 | Distance (px) actor must move away to start stationary check |
| `--stat-vel` | 5.0 | Max trash velocity (px/frame) to be considered stationary |
| `--stat-frames` | 10 | Stationary frames needed to confirm dump |
| `--baseline` | 60 | Startup frames to catalogue pre-existing trash |
| `--buffer-sec` | 10 | Seconds of raw footage kept in RAM ring buffer |
| `--post-sec` | 3 | Seconds of footage recorded after dump confirmation |

### Kalman Tracker

| Flag | Default | Description |
|------|---------|-------------|
| `--kalman-missed` | 10 | Frames before a trash track is deleted (must be ≥ `--stat-frames`) |
| `--kalman-iou` | 0.08 | IoU threshold for second-pass track-detection matching |

### Detection

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | 0.5 | Confidence threshold for person/vehicle detection |
| `--trash-conf` | 0.25 | Confidence threshold for trash detection (lower = higher recall) |
| `--tracker` | bytetrack.yaml | Underlying tracker (`bytetrack.yaml` or `botsort.yaml`) |
| `--model` | yolov8n.pt | COCO model for person/vehicle tracking |
| `--trash-model` | best.pt | Custom trained trash detection model |

> **Important constraint:** `--kalman-missed` ≥ `--stat-frames` ≥ `--assoc-frames` must hold, otherwise dump events may never be confirmed due to track death during stationary checking.

For Example:
>>>python person_vehicle_trash_tracker.py --mode video  --input cctv.mp4 --model yolov8l.pt --trash-model runs\detect\runs\trash_detection\v1.6\weights\best.pt --trash-conf 0.5 --tracker bytetrack.yaml --kalman-missed 10 --kalman-iou 0.15 --proximity 500 --assoc-frames 2 --no-trail 

---

## Hardware

Developed and tested on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| CPU | Intel Core i7-13650HX |
| Training precision | Mixed (AMP float16/32) |
| Inference device | CUDA GPU (device=0) |

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection and ByteTrack integration
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — multi-object tracking algorithm
- [OpenCV](https://opencv.org/) — video I/O, Kalman filter, drawing utilities
