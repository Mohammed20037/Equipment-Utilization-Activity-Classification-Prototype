import importlib
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "equipment"


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("_", " ")


def parse_allowlist(raw: str) -> Set[str]:
    return {normalize_label(token) for token in raw.split(",") if token.strip()}


def parse_polygon(raw: str) -> Optional[List[Tuple[float, float]]]:
    """
    Parse normalized polygon string as: "x1,y1;x2,y2;..." with values in [0,1].
    """
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
    """Fallback detector using foreground segmentation for demo/prototype mode."""

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
    """Optional model-based detector for interview requirement alignment."""

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
    """Selects YOLO model or motion fallback via env config and applies class/ROI filtering."""

    def __init__(self):
        self.allowlist = parse_allowlist(
            os.getenv(
                "TARGET_EQUIPMENT_CLASSES",
                "excavator,dump truck,loader,roller,bulldozer,truck",
            )
        )
        self.include_polygon = parse_polygon(os.getenv("ROI_INCLUDE_POLYGON", ""))
        self.exclude_polygon = parse_polygon(os.getenv("ROI_EXCLUDE_POLYGON", ""))

        backend = os.getenv("CV_MODEL_BACKEND", "yolo").lower()
        if backend == "yolo":
            model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
            conf = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
            try:
                self.impl = YoloDetector(model_path=model_path, conf_threshold=conf)
                self.backend = "yolo"
            except Exception:
                self.impl = MotionDetector(
                    min_area=int(os.getenv("MIN_DET_AREA", "2200")),
                    fallback_label=os.getenv("MOTION_FALLBACK_LABEL", "excavator"),
                )
                self.backend = "motion_fallback"
        else:
            self.impl = MotionDetector(
                min_area=int(os.getenv("MIN_DET_AREA", "2200")),
                fallback_label=os.getenv("MOTION_FALLBACK_LABEL", "excavator"),
            )
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
            intersects = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-9) + xi
            )
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

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raw_detections = self.impl.detect(frame)
        filtered: List[Detection] = []
        for det in raw_detections:
            label = normalize_label(det.label)
            if label not in self.allowlist:
                continue
            if not self._spatially_allowed(frame.shape, det.bbox):
                continue
            filtered.append(Detection(bbox=det.bbox, confidence=det.confidence, label=label))
        return filtered
