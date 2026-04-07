# Equipment Utilization Activity Classification Prototype

> A professional, interview-ready prototype for heavy-equipment utilization tracking — built using a microservices architecture, real-time streaming, and computer vision-based motion analysis.

This project reflects my ability to design, implement, debug, and stabilize a distributed computer vision pipeline under real engineering constraints — including integration mismatches, environment inconsistencies, and system reliability challenges. It's intentionally structured as a production-style prototype, with a focus on architecture correctness, modularity, validation, and extensibility.

---

## 📌 What This Project Demonstrates

- End-to-end **microservices architecture**
- **Real-time computer vision** inference pipeline
- **Kafka streaming** for inter-service communication
- **PostgreSQL** persistence layer
- **Streamlit** live monitoring dashboard
- **Docker-based** orchestration
- Structured **debugging and validation** workflow
- Modular code organization with production-readiness in mind
- Engineering documentation and clear design reasoning
- **Fault-tolerant** runtime behavior

This isn't just about getting something to run — it's about owning the system end-to-end.

---

## 🏗️ Architecture

```
Video Source
     ↓
cv_service          ← Detection, tracking, motion analysis, activity classification
     ↓
Kafka (equipment.events)
     ↓
analytics_service   ← Utilization engine, validation, DB writes
     ↓
PostgreSQL
     ↓
ui_service          ← Streamlit live dashboard
```

**Core principle:** Event-driven microservices with a deterministic processing flow.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── metadata/
│   │   └── open_source_video_sources.csv
│   ├── raw_videos/
│   ├── processed/
│   └── samples/
│       └── sample_payload.json
│
├── docs/
│   ├── alignment_checklist.md
│   ├── articulated_motion_design.md
│   ├── cv_implementation.md
│   ├── design.md
│   ├── model_data_faq.md
│   ├── motion_analysis_design.md
│   ├── requirement_gap_analysis.md
│   └── stop_duration_logic.md
│
├── infra/
│   └── Dockerfile
│
├── scripts/
│   ├── download_open_source_data.py
│   ├── export_demo_gif.py
│   ├── smoke_test.sh
│   └── validate_short_clip.py
│
├── services/
│   ├── cv_service/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── motion_analyzer.py
│   │   ├── activity_classifier.py
│   │   └── payload_builder.py
│   │
│   ├── analytics_service/
│   │   ├── kafka_consumer.py
│   │   ├── utilization_engine.py
│   │   ├── validation_report.py
│   │   └── db_writer.py
│   │
│   └── ui_service/
│       └── app.py
│
├── shared/
│   ├── db.py
│   └── schemas.py
│
├── sql/
│   └── schema.sql
│
├── tests/
│   ├── test_motion_analyzer.py
│   ├── test_payload_builder.py
│   ├── test_payload_schema.py
│   ├── test_utilization_engine.py
│   ├── test_validation_report.py
│   └── test_video_source_resolution.py
│
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── STATUS.md
├── README.md
└── .env.example
```

This structure was designed with service separation, maintainability, modular debugging, and testability in mind — not just to get the job done, but to make it easy to reason about.

---

## 🤖 Models & Computer Vision

### Detection — YOLO

The primary detection backend uses **Ultralytics YOLO**:

| Model | Purpose |
|---|---|
| `yolov8n.pt` | Default detection model |
| `yolov8n-seg.pt` | Optional segmentation model |

YOLO handles equipment detection, bounding box localization, class identification, and optional mask generation. The segmentation model improves motion precision by isolating equipment movement from background noise.

### Fallback Detection

If YOLO is unavailable, the system automatically falls back to **OpenCV MOG2** (Mixture of Gaussians background subtraction). This ensures graceful degradation and system continuity at runtime.

### Motion Analysis

Motion is evaluated using **Masked Optical Flow (Farnebäck)** — applied inside detected equipment regions only.

**Motion states:** `ACTIVE` | `INACTIVE`

**Activity labels:**

| Label | Description |
|---|---|
| `Digging` | Active digging motion |
| `Swinging-Loading` | Rotational/loading movement |
| `Dumping` | Dumping/discharge motion |
| `Waiting` | Idle/stopped state |

Labels are derived using **rule-based heuristics** rather than a learned classifier — a deliberate choice for interpretability, deterministic behavior, and easier debugging.

### Object Tracking

I built a lightweight custom tracker using a hybrid scoring approach:

- Centroid tracking
- Intersection-over-Union (IoU) matching
- Appearance histogram similarity
- Mask overlap scoring

This combination improves identity stability, tracking continuity, and occlusion handling.

### ⚠️ No Model Training

This project uses **pretrained YOLO models only**. No training pipeline, no dataset required. Videos are runtime/demo input. This was a deliberate design decision for reproducibility, faster deployment, and controlled evaluation.

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Computer Vision | OpenCV, Ultralytics YOLO, NumPy, Farnebäck Optical Flow, MOG2 |
| Streaming | Kafka (confluent-kafka) |
| Storage | PostgreSQL, SQLAlchemy, psycopg2 |
| Dashboard | Streamlit, Pandas |
| Infrastructure | Docker, Docker Compose |
| Testing | Pytest |

---

## ⚙️ How to Run

**Start all services:**
```bash
docker compose up --build
```

**Open the dashboard:**
```
http://localhost:8501
```

**Run pipeline validation:**
```bash
python scripts/validate_short_clip.py
```

**Stop services:**
```bash
docker compose down
```

---

## ✅ Validation Workflow

The primary validation command runs an end-to-end check of the pipeline and generates three artifacts:

| Output | Purpose |
|---|---|
| `equipment_timeline.csv` | Verifies activity classification logic |
| `validation_counts.csv` | Verifies stop duration accuracy |
| `processed_output.mp4` | Visual confirmation of pipeline correctness |

---

## 🐛 Engineering Challenges Solved

Part of what makes this project meaningful is what went wrong during development — and how I dealt with it.

Challenges encountered and resolved:

- Python interpreter mismatches across environments
- Dependency inconsistencies between services
- Kafka readiness timing and consumer race conditions
- Docker service startup ordering
- Missing module imports from environment drift
- Video source resolution issues at runtime
- Environment configuration conflicts across compose services

Each of these required system-level debugging, dependency tracing, and structured validation — not just guessing. I resolved them through environment isolation, deterministic startup sequencing, incremental testing, and structured validation scripts.

---

## 🔭 Optimization Opportunities

### Performance
- GPU acceleration for inference
- Frame sampling to reduce processing load
- Async processing between pipeline stages

### Reliability
- Health check endpoints per service
- Retry logic for Kafka consumers
- Structured logging throughout

### Architecture
- Horizontal scaling per service
- Configuration centralization
- Stricter service isolation

### Computer Vision
- Segmentation-guided motion detection (vs. bounding box only)
- Temporal sequence models for activity smoothing
- Adaptive motion thresholds per equipment class

---

## 🚀 Future Enhancement — Segmentation-Guided Detection

Currently, motion is analyzed within bounding boxes. The next step is switching to **mask-based motion detection**, which would:

- Increase motion precision
- Improve idle state detection
- Reduce false positives from background noise
- Improve overall utilization accuracy

---

## 📊 Demo

> **[ PLACEHOLDER — Demo GIF will be added here ]**

---

## 🙋 My Role

I designed and built this system end-to-end:

- System architecture and service design
- Computer vision pipeline implementation
- Microservice integration
- Validation and debugging workflow
- Code organization and documentation

This project demonstrates system design capability, debugging discipline, production awareness, and engineering ownership — not just the ability to write code that runs.

---

## 📋 Implementation Status

| Component | Status |
|---|---|
| Pipeline | ✅ Working |
| Services | ✅ Stable |
| Validation | ✅ Functional |
| Architecture | ✅ Modular |
| Overall Readiness | ✅ Interview-Ready |
