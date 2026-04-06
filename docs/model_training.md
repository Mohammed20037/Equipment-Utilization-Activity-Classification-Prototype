# Model Training Guide

This document explains how to train a domain-specific construction equipment
detection and segmentation model using the `train_detector.py` pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Sources](#dataset-sources)
3. [Hardware Requirements](#hardware-requirements)
4. [Quick Start](#quick-start)
5. [Training Command Reference](#training-command-reference)
6. [Docker Training Service](#docker-training-service)
7. [Segmentation Training](#segmentation-training)
8. [Model Registry & Runtime Integration](#model-registry--runtime-integration)
9. [Evaluation](#evaluation)
10. [Expected Runtime](#expected-runtime)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The base YOLOv8 model is pretrained on COCO, which lacks construction-site
specificity. Misclassification of static objects (scaffolding, trailers) and
missed detections of active equipment are symptoms of this domain gap.

The training pipeline fine-tunes YOLOv8 on purpose-built construction
equipment datasets, producing `models/equipment_detector.pt`. The CV service
automatically loads this model if it is present.

**Equipment classes detected:**

| ID | Class Name       | Examples                          |
|----|------------------|-----------------------------------|
| 0  | excavator        | tracked excavator, hydraulic arm  |
| 1  | dump_truck       | articulated dump truck, tipper    |
| 2  | wheel_loader     | front loader, bucket loader       |
| 3  | bulldozer        | crawler dozer, blade dozer        |
| 4  | crane            | tower crane, mobile crane         |
| 5  | concrete_mixer   | drum mixer, transit mixer truck   |
| 6  | forklift         | counterbalance, reach truck       |
| 7  | compactor        | road roller, vibratory compactor  |

---

## Dataset Sources

### Roboflow Universe (Recommended)

Roboflow hosts hundreds of labelled construction equipment datasets contributed
by the community. Using Roboflow provides the highest-quality, purpose-specific
training data with correct class labels.

**Setup:**
1. Create a free account at [roboflow.com](https://roboflow.com)
2. Navigate to *Settings → API* and copy your API key
3. Browse [universe.roboflow.com](https://universe.roboflow.com) and fork a
   construction equipment dataset into your workspace, or use the default
   dataset configured via env vars
4. Set env vars:

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
export ROBOFLOW_WORKSPACE="your-workspace-slug"
export ROBOFLOW_PROJECT="your-project-slug"
export ROBOFLOW_VERSION=1
```

**Example datasets to look for on Roboflow Universe:**
- Search: `construction equipment detection`
- Search: `heavy machinery`
- Search: `excavator bulldozer`

---

### Open Images v6 (No API key required)

Google's Open Images v6 validation set is downloaded automatically from public
Google Cloud Storage. Relevant classes: Excavator, Bulldozer, Crane, Forklift,
Truck.

Images are downloaded individually from the public annotation CSV. This
requires a stable internet connection but no credentials.

**Approximate size:** ~2–5 GB for up to 3,000 images per class.

---

### COCO 2017 Vehicles (No API key required)

The COCO 2017 validation annotations (~241 MB zip) are downloaded automatically.
Only images containing vehicle categories (truck) are then downloaded
individually.

Note: COCO does not have dedicated construction equipment categories, so this
source supplements rather than replaces Roboflow/Open Images.

**Approximate size:** ~500 MB for up to 3,000 images.

---

## Hardware Requirements

| Configuration     | Minimum              | Recommended          |
|-------------------|----------------------|----------------------|
| CPU               | 4 cores              | 8+ cores             |
| RAM               | 8 GB                 | 16+ GB               |
| GPU               | None (CPU training)  | NVIDIA GPU ≥ 8 GB VRAM |
| Disk              | 20 GB free           | 50+ GB free          |
| Internet          | Required for download| Required for download|

**GPU note:** CPU training is functional but slow (~10–20× longer).
For GPU training inside Docker, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
and pass `--gpus all` to `docker run`, or set `TRAINING_DEVICE=0` in `.env`.

---

## Quick Start

### Local (without Docker)

```bash
# 1. Install training dependencies
pip install roboflow>=1.1.0 pycocotools>=2.0.7

# 2. (Optional) Set Roboflow credentials
export ROBOFLOW_API_KEY="your_key"
export ROBOFLOW_WORKSPACE="your-workspace"
export ROBOFLOW_PROJECT="construction-equipment-detection"

# 3. Train with Open Images + COCO (no API key needed)
python train_detector.py

# 4. Train with all sources (Roboflow + Open Images + COCO)
python train_detector.py --dataset all
```

The trained model is saved to `models/equipment_detector.pt` automatically.

---

## Training Command Reference

```
python train_detector.py [OPTIONS]

Options:
  --model       {yolov8n,yolov8s,yolov8m,yolov8l,yolov8n-seg,yolov8s-seg,yolov8m-seg}
                YOLOv8 model variant (default: yolov8s)
  --mode        {detect,segment}
                Training mode (default: detect)
  --dataset     Comma-separated sources: roboflow,open_images,coco,all
                (default: open_images,coco)
  --epochs      Max training epochs (default: 100)
  --patience    Early-stopping patience – stops if no improvement for N epochs
                (default: 20)
  --batch       Batch size; -1=auto-detect (default: -1)
  --img-size    Input image resolution (default: 640)
  --device      Compute device: cpu | 0 | 0,1 | mps (default: auto)
  --max-images  Maximum images to download per source (default: 3000)
  --download-only
                Download and validate datasets only; skip training
```

**Environment variables (override CLI defaults):**

| Variable                | Default   | Description                         |
|-------------------------|-----------|-------------------------------------|
| `ROBOFLOW_API_KEY`      | –         | Roboflow API key                    |
| `ROBOFLOW_WORKSPACE`    | construction-equipment-site | Workspace slug   |
| `ROBOFLOW_PROJECT`      | construction-equipment-detection | Project slug |
| `ROBOFLOW_VERSION`      | 1         | Dataset version                     |
| `TRAINING_EPOCHS`       | 100       | Max epochs                          |
| `TRAINING_BATCH_SIZE`   | -1        | Batch size (-1 = auto)              |
| `TRAINING_IMG_SIZE`     | 640       | Image resolution                    |
| `TRAINING_DEVICE`       | auto      | Compute device                      |
| `TRAINING_PATIENCE`     | 20        | Early-stopping patience             |
| `MAX_IMAGES_PER_SOURCE` | 3000      | Per-source download cap             |

---

## Docker Training Service

The `training_service` is defined in `docker-compose.yml` and uses the
dedicated `infra/Dockerfile.training` image which includes build tools for
`pycocotools`.

**Run training via Docker Compose:**

```bash
# Using the docker compose profile (does NOT start other services)
docker compose --profile training up training_service

# Or run directly
docker compose run --rm training_service

# Pass custom arguments
docker compose run --rm training_service \
  --model yolov8m --dataset roboflow --epochs 150

# With Roboflow credentials
ROBOFLOW_API_KEY=your_key docker compose run --rm training_service \
  --dataset all
```

The trained model is written to `./models/equipment_detector.pt` on the host
via the bind-mount volume.

**Build the training image separately:**

```bash
docker build -f infra/Dockerfile.training -t equipment-training .

# Run CPU training
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e ROBOFLOW_API_KEY=your_key \
  equipment-training --dataset all

# Run GPU training (NVIDIA Container Toolkit required)
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e ROBOFLOW_API_KEY=your_key \
  -e TRAINING_DEVICE=0 \
  equipment-training --model yolov8m --dataset all
```

---

## Segmentation Training

Instance segmentation provides pixel-level masks for each detected piece of
equipment. The masks feed directly into the `motion_analyzer.py` for
mask-guided optical flow, improving state estimation accuracy.

```bash
# Train segmentation model
python train_detector.py --model yolov8s-seg --mode segment --dataset roboflow

# Via Docker
docker compose run --rm training_service \
  --model yolov8s-seg --mode segment --dataset roboflow
```

**Note:** Segmentation training requires datasets annotated with instance
masks. Open Images and COCO sources provide bounding boxes only; use Roboflow
with a segmentation-annotated dataset for full segmentation support.

After training, set this environment variable to use the segmentation model
at runtime:

```bash
SEGMENTATION_MODEL_PATH=models/equipment_detector.pt
ENABLE_SEGMENTATION=1
```

---

## Model Registry & Runtime Integration

The CV service (`services/cv_service/detector.py`) resolves the model path
using the following priority:

1. `YOLO_MODEL_PATH` environment variable (explicit override)
2. `models/equipment_detector.pt` (trained model if present)
3. `yolov8n.pt` (pretrained Ultralytics default, auto-downloaded)

When a custom model is loaded, the service logs:

```
Loaded custom equipment model: models/equipment_detector.pt
```

No configuration changes are needed after running the training pipeline—the
service auto-discovers the trained model on the next restart.

---

## Evaluation

After training, evaluate the model's performance:

```bash
# Evaluate against the training dataset validation split
python evaluate_model.py

# Evaluate with a different IoU threshold
python evaluate_model.py --iou 0.6

# Evaluate a specific checkpoint
python evaluate_model.py --model runs/train/equipment_detector/weights/best.pt

# Run inference on video without ground-truth
python evaluate_model.py --predict-only --source data/raw_videos/

# Save annotated frames
python evaluate_model.py --predict-only --source data/raw_videos/ --save-images
```

Metrics produced:
- `mAP@0.50` and `mAP@0.50:0.95`
- Per-class precision and recall
- Normalised confusion matrix (ASCII)
- Per-class accuracy

Results are saved to `models/eval_results/results.json`.

---

## Expected Runtime

| Hardware           | Dataset size | Epochs | Approx. runtime |
|--------------------|--------------|--------|-----------------|
| CPU (8 cores)      | 3,000 images | 100    | 8–12 hours      |
| GPU (RTX 3080)     | 3,000 images | 100    | 45–90 minutes   |
| GPU (RTX 3080)     | 10,000 images| 100    | 2–3 hours       |
| GPU (A100)         | 10,000 images| 100    | 45–60 minutes   |

Early stopping (patience=20) typically halts training around epoch 50–70 when
using a well-annotated dataset, reducing actual runtime by 30–50%.

**Download time:**

| Source       | Size      | Approx. download time (100 Mbps) |
|--------------|-----------|----------------------------------|
| Open Images  | 2–5 GB    | 5–15 minutes                     |
| COCO annots  | 241 MB    | 1–2 minutes                      |
| COCO images  | 0.5–1 GB  | 2–5 minutes                      |
| Roboflow     | 0.1–2 GB  | 1–5 minutes                      |

---

## Troubleshooting

**`roboflow` package not found:**
```bash
pip install roboflow>=1.1.0
```

**`pycocotools` build failure (missing gcc):**
```bash
# Ubuntu/Debian
sudo apt-get install gcc g++ python3-dev
pip install pycocotools>=2.0.7

# macOS
xcode-select --install
pip install pycocotools>=2.0.7
```

**CUDA out of memory:**
- Reduce `--batch` to 8 or 4
- Reduce `--img-size` to 416
- Use a smaller model: `--model yolov8n`

**Open Images download slow / failing:**
- Individual image downloads can be slow; use `--max-images 500` for a quick test
- Network timeouts are handled silently; a summary shows failed images

**No images downloaded:**
- Verify internet connectivity
- For Roboflow, confirm `ROBOFLOW_API_KEY` is set correctly
- Run `--download-only` flag to debug dataset download separately

**Model not loading in CV service:**
- Confirm `models/equipment_detector.pt` exists (relative to the working
  directory where `docker compose up` is run, which is the repo root)
- Check the CV service logs: `docker compose logs cv_service | grep -i model`
