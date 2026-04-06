import importlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("cv_service.detector")


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "equipment"
    mask: Optional[np.ndarray] = None


@dataclass
class RejectedDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    reason: str


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("_", " ")


def parse_allowlist(raw: str) -> Set[str]:
    return {normalize_label(token) for token in raw.split(",") if token.strip()}


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def nms_detections(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    by_label: Dict[str, List[Detection]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    final: List[Detection] = []
    for _, dets in by_label.items():
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in dets:
            if all(_bbox_iou(det.bbox, k.bbox) < iou_threshold for k in kept):
                kept.append(det)
        final.extend(kept)
    return final


def remove_nested_duplicates(detections: List[Detection], nested_iou_threshold: float = 0.8) -> List[Detection]:
    if len(detections) < 2:
        return detections
    dets = sorted(detections, key=lambda d: ((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])), reverse=True)
    kept: List[Detection] = []
    for det in dets:
        drop = False
        for k in kept:
            iou = _bbox_iou(det.bbox, k.bbox)
            if iou > nested_iou_threshold and (_contains(k.bbox, det.bbox) or _contains(det.bbox, k.bbox)):
                drop = True
                break
        if not drop:
            kept.append(det)
    return kept


def parse_polygon(raw: str) -> Optional[List[Tuple[float, float]]]:
    if not raw.strip():
        return None
    points: List[Tuple[float, float]] = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        x_str, y_str = token.split(",")
        points.append((float(x_str), float(y_str)))
    return points if len(points) >= 3 else None


class MotionDetector:
    def __init__(self, min_area: int = 2500, fallback_label: str = "excavator"):
        self.min_area = min_area
        self.fallback_label = normalize_label(fallback_label)
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        mask = self.bg.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=min(0.99, area / 50000.0),
                    label=self.fallback_label,
                )
            )
        return detections


class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        ultralytics_mod = importlib.import_module("ultralytics")
        self.model = ultralytics_mod.YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = normalize_label(names.get(cls_id, str(cls_id)))
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, label=label))
        return detections


class YoloSegmentationBackend:
    def __init__(self, model_path: str, conf_threshold: float):
        ultralytics_mod = importlib.import_module("ultralytics")
        self.model = ultralytics_mod.YOLO(model_path)
        self.conf_threshold = conf_threshold

    def infer_masks(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        segmented: List[Detection] = []
        for result in results:
            names = result.names
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy()
            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                label = normalize_label(names.get(cls_id, str(cls_id)))
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                full_mask = cv2.resize(masks[idx], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary = (full_mask > 0.5).astype(np.uint8)
                segmented.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, label=label, mask=binary))
        return segmented


class SegmentationManager:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_SEGMENTATION", "1") == "1"
        self.backend_name = os.getenv("SEGMENTATION_BACKEND", "yolov8_seg")
        self.conf_threshold = float(os.getenv("SEGMENTATION_CONFIDENCE_THRESHOLD", "0.25"))
        self._warned_fallback = False
        self.backend = None
        self.status = "disabled"

        if not self.enabled:
            return

        try:
            if self.backend_name == "yolov8_seg":
                model_path = os.getenv("SEGMENTATION_MODEL_PATH", "yolov8n-seg.pt")
                self.backend = YoloSegmentationBackend(model_path=model_path, conf_threshold=self.conf_threshold)
                self.status = "enabled"
            else:
                logger.warning("Unknown SEGMENTATION_BACKEND=%s. Falling back to detection-only mode.", self.backend_name)
                self.status = "unavailable"
        except Exception as exc:
            logger.warning("Segmentation backend init failed (%s). Falling back to detection-only mode.", exc)
            self.status = "unavailable"

    def enrich(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        if not detections or self.backend is None:
            if self.enabled and self.status != "enabled" and not self._warned_fallback:
                logger.warning("Segmentation unavailable; running detection-only mode.")
                self._warned_fallback = True
            return detections

        seg_dets = self.backend.infer_masks(frame)
        if not seg_dets:
            return detections

        enriched: List[Detection] = []
        for det in detections:
            best = None
            best_iou = 0.0
            for seg in seg_dets:
                if seg.label != det.label:
                    continue
                iou = _bbox_iou(det.bbox, seg.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best = seg
            if best is not None and best_iou >= 0.3:
                enriched.append(Detection(bbox=det.bbox, confidence=det.confidence, label=det.label, mask=best.mask))
            else:
                enriched.append(det)
        return enriched


class HybridDetector:
    def __init__(self):
        self.allowlist = parse_allowlist(os.getenv("ALLOWED_CLASSES", "truck,excavator,loader"))
        self.supported_class_map = {
            normalize_label(k): normalize_label(v)
            for k, v in (token.split(":", 1) for token in os.getenv("CLASS_NAME_MAP", "car:truck").split(",") if ":" in token)
        }
        self.include_polygon = parse_polygon(os.getenv("ROI_INCLUDE_POLYGON", ""))
        self.exclude_polygon = parse_polygon(os.getenv("ROI_EXCLUDE_POLYGON", ""))
        self.roi_min_intersection = float(os.getenv("ROI_MIN_INTERSECTION_RATIO", "0.25"))
        self.nms_iou_threshold = float(os.getenv("DET_NMS_IOU", "0.5"))
        self.min_box_area = int(os.getenv("MIN_BOX_AREA", "5000"))
        self.enable_rejection_log = os.getenv("LOG_REJECTED_DETECTIONS", "1") == "1"

        self.last_raw_count = 0
        self.last_filtered_count = 0
        self.last_rejections: List[RejectedDetection] = []

        self.segmentation = SegmentationManager()

        backend = os.getenv("CV_MODEL_BACKEND", "yolo").lower()
        if backend == "yolo":
            model_path = self._resolve_model_path()
            conf = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
            try:
                self.impl = YoloDetector(model_path=model_path, conf_threshold=conf)
                self.backend = "yolo"
            except Exception:
                self.impl = MotionDetector(min_area=int(os.getenv("MIN_DET_AREA", "2200")), fallback_label=os.getenv("MOTION_FALLBACK_LABEL", "excavator"))
                self.backend = "motion_fallback"
        else:
            self.impl = MotionDetector(min_area=int(os.getenv("MIN_DET_AREA", "2200")), fallback_label=os.getenv("MOTION_FALLBACK_LABEL", "excavator"))
            self.backend = "motion_fallback"

    @staticmethod
    def _resolve_model_path() -> str:
        """Return path to the best available YOLO model.

        Priority:
          1. Explicitly set YOLO_MODEL_PATH env var
          2. Domain-specific trained model at models/equipment_detector.pt
          3. Default pretrained yolov8n.pt (downloaded from Ultralytics on first use)
        """
        import pathlib  # local import to avoid circular issues at module level

        env_path = os.getenv("YOLO_MODEL_PATH", "")
        if env_path:
            return env_path

        custom_model = pathlib.Path("models/equipment_detector.pt")
        if custom_model.exists():
            logger.info("Loaded custom equipment model: %s", custom_model)
            return str(custom_model)

        logger.info("No custom model found at models/equipment_detector.pt; using yolov8n.pt")
        return "yolov8n.pt"

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
        px, py = point
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-9) + xi)
            if intersects:
                inside = not inside
            j = i
        return inside

    def _bbox_intersection_with_polygon_ratio(self, frame_shape, bbox: Tuple[int, int, int, int], polygon: Sequence[Tuple[float, float]]) -> float:
        h, w = frame_shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[(int(px * w), int(py * h)) for px, py in polygon]], dtype=np.int32)
        cv2.fillPoly(canvas, pts, 1)

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        intersection = np.logical_and(canvas, bbox_mask).sum()
        box_area = max(1, (x2 - x1) * (y2 - y1))
        return float(intersection / box_area)

    def _spatial_rejection_reason(self, frame_shape, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        h, w = frame_shape[:2]
        cx, cy = self._bbox_center(bbox)
        norm_point = (cx / max(w, 1), cy / max(h, 1))

        if self.include_polygon:
            inside = self._point_in_polygon(norm_point, self.include_polygon)
            overlap_ratio = self._bbox_intersection_with_polygon_ratio(frame_shape, bbox, self.include_polygon)
            if not inside and overlap_ratio < self.roi_min_intersection:
                return "outside_worksite_roi"
        if self.exclude_polygon and self._point_in_polygon(norm_point, self.exclude_polygon):
            return "inside_excluded_zone"
        return None

    def _reject(self, det: Detection, reason: str) -> None:
        rej = RejectedDetection(bbox=det.bbox, confidence=det.confidence, label=det.label, reason=reason)
        self.last_rejections.append(rej)
        if self.enable_rejection_log:
            x1, y1, x2, y2 = det.bbox
            area = max(0, (x2 - x1) * (y2 - y1))
            logger.info(
                "Rejected detection: class=%s confidence=%.3f area=%d reason=%s",
                det.label,
                det.confidence,
                area,
                reason,
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raw_detections = self.impl.detect(frame)
        self.last_raw_count = len(raw_detections)
        self.last_rejections = []

        filtered: List[Detection] = []
        for det in raw_detections:
            label = normalize_label(det.label)
            label = self.supported_class_map.get(label, label)
            det = Detection(bbox=det.bbox, confidence=det.confidence, label=label)
            if label not in self.allowlist:
                self._reject(det, "class_not_allowed")
                continue
            rejection_reason = self._spatial_rejection_reason(frame.shape, det.bbox)
            if rejection_reason:
                self._reject(det, rejection_reason)
                continue
            x1, y1, x2, y2 = det.bbox
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                self._reject(det, "box_too_small")
                continue
            filtered.append(det)

        filtered = nms_detections(filtered, self.nms_iou_threshold)
        filtered = remove_nested_duplicates(filtered, nested_iou_threshold=0.8)
        filtered = self.segmentation.enrich(frame, filtered)
        self.last_filtered_count = len(filtered)
        return filtered
