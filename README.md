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
- Renders utilization dashboard in Streamlit

## Kafka payload schema (example)

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": 15.0,
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

## Notes for interview presentation

- Position this as a **scaled production-style prototype**.
- Be explicit that activity labels are rule-based in v1 and can be replaced with a temporal classifier in v2.
- Emphasize articulated-motion handling as the key requirement solved.
