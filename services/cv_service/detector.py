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
            detections.append(Detection(bbox=(x, y, x + w, y + h), confidence=min(0.99, area / 50000.0), label=self.fallback_label))
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


class HybridDetector:
    def __init__(self):
        self.allowlist = parse_allowlist(os.getenv("ALLOWED_CLASSES", "truck,excavator,loader"))
        self.include_polygon = parse_polygon(os.getenv("ROI_INCLUDE_POLYGON", ""))
        self.exclude_polygon = parse_polygon(os.getenv("ROI_EXCLUDE_POLYGON", ""))
        self.nms_iou_threshold = float(os.getenv("DET_NMS_IOU", "0.5"))
        self.min_box_area = int(os.getenv("MIN_BOX_AREA", "5000"))
        self.enable_rejection_log = os.getenv("LOG_REJECTED_DETECTIONS", "1") == "1"

        self.last_raw_count = 0
        self.last_filtered_count = 0
        self.last_rejections: List[RejectedDetection] = []

        backend = os.getenv("CV_MODEL_BACKEND", "yolo").lower()
        if backend == "yolo":
            model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
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

    def _spatially_allowed(self, frame_shape, bbox: Tuple[int, int, int, int]) -> bool:
        h, w = frame_shape[:2]
        cx, cy = self._bbox_center(bbox)
        norm_point = (cx / max(w, 1), cy / max(h, 1))

        if self.include_polygon and not self._point_in_polygon(norm_point, self.include_polygon):
            return False
        if self.exclude_polygon and self._point_in_polygon(norm_point, self.exclude_polygon):
            return False
        return True

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
            det = Detection(bbox=det.bbox, confidence=det.confidence, label=label)
            if label not in self.allowlist:
                self._reject(det, "class_not_allowed")
                continue
            if not self._spatially_allowed(frame.shape, det.bbox):
                self._reject(det, "outside_roi")
                continue
            x1, y1, x2, y2 = det.bbox
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                self._reject(det, "box_too_small")
                continue
            filtered.append(det)

        filtered = nms_detections(filtered, self.nms_iou_threshold)
        filtered = remove_nested_duplicates(filtered, nested_iou_threshold=0.8)
        self.last_filtered_count = len(filtered)
        return filtered
