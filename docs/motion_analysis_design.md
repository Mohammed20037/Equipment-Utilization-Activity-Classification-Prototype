# Motion Analysis Design (Primary: Optical Flow + YOLO)

## Why Optical Flow + YOLO is primary
For short fixed-camera worksite clips, this path balances accuracy and implementation speed:
1. YOLO provides machine localization/class.
2. Optical flow inside machine ROI captures motion intensity.
3. Articulated-region flow helps detect productive movement even when chassis translation is small.

## Implemented path
- Detector: YOLO (or fallback) -> class allowlist -> ROI filtering.
- Motion analyzer (`optical_flow_yolo`):
  - Dense Farneback flow in bbox.
  - Articulated ROI score (upper-right region proxy).
  - Chassis ROI score (lower-left proxy).
  - Productive score = `max(articulated, full_body - 0.55*chassis)`.
- State: productive threshold + debounce transitions.

## Optional modes
- `c3d_yolo`: temporal-energy proxy over short clip window.
- `lstm_yolo`: exponentially weighted temporal memory proxy.

## Activity labels
- Digging / Swinging/Loading / Dumping / Waiting
- Rules are equipment-aware (excavator/loader/bulldozer/roller vs truck classes), with temporal history retained for smoothing.

