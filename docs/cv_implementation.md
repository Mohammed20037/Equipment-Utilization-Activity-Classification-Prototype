# Computer Vision Implementation Notes

## Segmentation-guided pipeline (current)

The CV service now runs a production-style staged pipeline:

1. detection (`HybridDetector`)
2. strict worksite acceptance (allowed class + include ROI + exclude ROI)
3. optional segmentation enrichment (`ENABLE_SEGMENTATION`)
4. mask-guided motion analysis (`MotionAnalyzer`)
5. articulated-aware ACTIVE/INACTIVE
6. equipment-specific activity classification
7. KPI emission only for accepted tracks

## Segmentation backend

- Default backend: `yolov8_seg` using `ultralytics` with `yolov8n-seg.pt`
- Weights are auto-fetched by Ultralytics (no manual dataset download required)
- Config:
  - `ENABLE_SEGMENTATION=1|0`
  - `SEGMENTATION_BACKEND=yolov8_seg`
  - `SEGMENTATION_MODEL_PATH=yolov8n-seg.pt`
  - `SEGMENTATION_CONFIDENCE_THRESHOLD=0.25`
- If backend init fails, service logs warning and falls back to detection-only mode

## Motion and articulated logic

Primary motion score is computed from optical flow on mask pixels (not full bbox).

Outputs:
- `full_body_score`
- `articulated_score` (productive/local region proxy)
- `productive_score` (temporally smoothed)
- `mask_motion_density` / `persistence_score`

Articulated classes can become `ACTIVE` via local productive motion even if chassis/body movement is low.

Approximation used when part-level segmentation is not available:
- top-right mask region ~ arm/bucket/tool proxy
- bottom-left mask region ~ body/chassis proxy

## Class support policy

This project does **not** overclaim class coverage. Baseline support depends on detector model labels.

- Configure supported classes via `ALLOWED_CLASSES`
- Optional class aliasing via `CLASS_NAME_MAP` (example: `car:truck`)
- Only accepted classes are tracked/emitted to business KPI path

## UI behavior

- Main frame and KPI path show accepted worksite tracks only
- Rejected detections are hidden by default
- Debug overlays are enabled only when `DEBUG_OVERLAY=1`

## Validation outputs

For short-clip comparisons:
- `data/processed/equipment_timeline.csv`: state/activity/stop timeline
- `data/processed/validation_counts.csv`: accepted/rejected/tracked counts per frame
- `python scripts/validate_short_clip.py` prints utilization + transitions + counts summary
