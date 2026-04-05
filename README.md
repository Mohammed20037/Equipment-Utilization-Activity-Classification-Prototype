# Equipment Utilization Activity Classification Prototype

A professional, interview-ready prototype for heavy-equipment utilization tracking with a **microservices architecture**.

## What this demonstrates

- CV inference service (detection/tracking + motion-based activity/state)
- Kafka streaming between services
- Analytics consumer with DB sink (PostgreSQL)
- Streamlit dashboard for live utilization metrics
- Docker Compose orchestration for local end-to-end run

## Architecture

```text
Video -> cv_service -> Kafka topic (equipment.events) -> analytics_service -> PostgreSQL
                                                         -> ui_service (reads DB)
```

## Repository layout

```text
.
├── docker-compose.yml
├── requirements.txt
├── configs/
├── infra/
├── services/
│   ├── cv_service/
│   ├── analytics_service/
│   └── ui_service/
├── shared/
├── sql/
└── tests/
```

## Quick start

1. Copy env:
   ```bash
   cp .env.example .env
   ```
2. Start services:
   ```bash
   docker compose up --build
   ```
3. Open dashboard:
   - http://localhost:8501


## How to run and test

### 1) Local Python tests (fastest)

```bash
make install
make test
```

Equivalent without Make:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

### 2) Full stack with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

Open UI:
- http://localhost:8501

Tail logs:

```bash
docker compose logs -f --tail=150
```

Stop and clean:

```bash
docker compose down -v
```

### 3) One-command smoke test script

```bash
./scripts/smoke_test.sh
```

This script starts the stack, waits briefly, prints service status, and shows recent logs for `cv_service`, `analytics_service`, and `ui_service`.


### CV model configuration

By default the CV service uses a YOLO model backend (`CV_MODEL_BACKEND=yolo`).
If model loading is unavailable, it automatically falls back to motion-only detection.
You can force fallback mode with:

```bash
CV_MODEL_BACKEND=motion
```

## Service responsibilities

### `services/cv_service`
- Reads video stream (or file)
- Performs detection + tracking
- Computes ACTIVE/INACTIVE using whole-box + articulated-region motion
- Emits Kafka JSON payloads

### `services/analytics_service`
- Consumes Kafka messages
- Maintains active/idle accumulation
- Computes utilization %
- Persists to PostgreSQL

### `services/ui_service`
- Pulls latest rows from PostgreSQL
- Shows processed latest frame with bounding boxes
- Displays live machine state/activity and utilization dashboard in Streamlit

## Kafka payload schema (example)

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "00:00:15.000",
  "timestamp_sec": 15.0,
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "utilization_percent": 83.3
  }
}
```


## Implementation status (important)

This repository is currently a **working MVP scaffold**, **not a fully complete production implementation** yet.

### Already implemented
- End-to-end microservice topology with Docker Compose
- Kafka producer/consumer wiring
- PostgreSQL sink with `frame_events` + `equipment_summary`
- Streamlit dashboard reading live summary data
- Typed event schema and sample payload contract

### Still to complete for a full interview-grade submission
- Real CV inference in `cv_service` (YOLO + tracker) instead of synthetic event generation
- Articulated-part motion analysis (arm/bucket ROI optical-flow or frame-diff)
- Rule engine for DIGGING / SWINGING_LOADING / DUMPING / WAITING using real cues
- Video overlay output + UI playback panel
- Tests beyond schema validation (integration + utilization accuracy + service smoke tests)
- Metrics/logging/health endpoints and basic failure handling

If you present this today, describe it as **Phase-1 foundation complete** with **CV intelligence pending implementation**.

## Day-by-day execution plan

### Day 1 (MVP pipeline)
- Compose stack boots (Kafka, Postgres, 3 services)
- CV producer pushes valid payloads to Kafka
- Analytics consumer writes frame events + summary rows
- Dashboard shows table + utilization chart

### Day 2 (CV quality)
- Implement articulated-motion ROI logic
- Tune motion thresholds for ACTIVE/INACTIVE
- Add rule-based activity classification (DIGGING, SWINGING_LOADING, DUMPING, WAITING)

### Day 3 (polish)
- Add tests for payload schema/utilization engine
- Add design doc with tradeoffs and assumptions
- Record short demo video/GIF + finalize README


## CV implementation confirmation

Yes, computer vision is implemented in this prototype using OpenCV-based detection/tracking/motion analysis.
See `docs/cv_implementation.md` for exact methods and current limitations.

## Notes for interview presentation

- Position this as a **scaled production-style prototype**.
- Be explicit that activity labels are rule-based in v1 and can be replaced with a temporal classifier in v2.
- Emphasize articulated-motion handling as the key requirement solved.


## Demo artifact

Record a 30-60s GIF/video after running the stack to satisfy submission requirements.
Suggested flow: start stack, let CV process frames, open dashboard, and capture updates to state/activity/utilization.
