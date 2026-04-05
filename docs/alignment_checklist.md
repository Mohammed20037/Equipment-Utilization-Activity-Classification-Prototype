# Requirement Alignment Checklist

## 1) Is the current code exactly aligned with the brief?
Short answer: **Yes for assignment scope (scaled-down prototype).**

| Requirement | Status | Notes |
|---|---|---|
| Real-time microservices with Kafka | ✅ | Implemented via `cv_service -> Kafka -> analytics_service -> Postgres -> ui_service`. |
| ACTIVE/INACTIVE utilization tracking | ✅ | Implemented via motion analyzer thresholds. |
| Articulated motion (arm-only ACTIVE) | ✅ | Implemented using region-based ROI differencing in `MotionAnalyzer`. |
| Activity classes (Digging, Swinging/Loading, Dumping, Waiting) | ✅ | Implemented via rule-based `ActivityClassifier`. |
| Working vs idle counters + utilization % | ✅ | Implemented in payload builder and persisted in DB summary/events. |
| Data sink (Postgres/Timescale) | ✅ | PostgreSQL sink implemented (`frame_events`, `equipment_summary`). |
| UI: processed feed + live state/activity + utilization dashboard | ✅ | Streamlit shows latest annotated frame, status table, and utilization metrics/charts. |
| Demo video/GIF artifact | ✅ | `cv_service` saves processed video and `scripts/export_demo_gif.py` exports GIF. |
| "Exactly" production-grade model quality | ⚠️ | Current prototype is rule-based + lightweight CV; accuracy depends on scene and thresholds. |

## 2) Is an AI model used in CV?
**Yes.** The CV service can run with a YOLO model backend (`CV_MODEL_BACKEND=yolo`) via Ultralytics.
A motion-only fallback is available when model loading is unavailable (`CV_MODEL_BACKEND=motion`).

## 3) Does it need data?
**Yes, for meaningful results.**

- For **inference/demo** with pretrained YOLO:
  - You need input video clips (`data/raw_videos/*.mp4`).
  - No training labels are strictly required.
- For **better accuracy and interview-strength validation**:
  - You should add labeled clips/intervals for activities and evaluate predictions.
- For **custom model training/fine-tuning**:
  - You need annotated datasets (bounding boxes / segmentation / activity intervals).

## Practical recommendation
For interview submission, use:
1. Pretrained YOLO inference + tracking + articulated motion logic (already supported).
2. A small curated test video set with activity interval CSV labels for quantitative validation.
3. A 30-60s demo GIF/video proving live pipeline behavior.
