# Computer Vision Implementation Notes

## Did we use computer vision?
Yes. The prototype uses CV in `cv_service` with two detector modes:

1. **Model-based mode (default)**
   - `HybridDetector` loads a YOLO model (`ultralytics`, default `yolov8n.pt`)
   - Produces class-labeled detections and bounding boxes
2. **Fallback mode**
   - OpenCV foreground motion detection (MOG2 + contours) when model is unavailable

Tracking and inference stack:
- **Object tracking** (`CentroidTracker`) for persistent machine IDs
- **Articulated motion handling** (`MotionAnalyzer`):
  - full-body motion via bbox center displacement
  - arm-only proxy motion via ROI frame differencing
  - ACTIVE if either body or articulated motion crosses threshold
- **Activity classification** (`ActivityClassifier`):
  - `DIGGING`, `SWINGING_LOADING`, `DUMPING`, `WAITING`

## Interview framing
This satisfies the requirement to use CV and a CV model while still remaining robust in restricted environments (fallback mode).
