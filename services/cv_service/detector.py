import importlib
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "equipment"


class MotionDetector:
    """Fallback detector using foreground segmentation for demo/prototype mode."""

    def __init__(self, min_area: int = 2500):
        self.min_area = min_area
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
            detections.append(Detection(bbox=(x, y, x + w, y + h), confidence=min(0.99, area / 50000.0), label="equipment"))
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
                label = names.get(cls_id, str(cls_id)).lower()
                if label not in {"truck", "car", "bus", "train", "excavator", "construction_vehicle", "equipment"}:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, label=label))
        return detections


class HybridDetector:
    """Selects YOLO model or motion fallback via env config."""

    def __init__(self):
        backend = os.getenv("CV_MODEL_BACKEND", "yolo").lower()
        if backend == "yolo":
            model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
            conf = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
            try:
                self.impl = YoloDetector(model_path=model_path, conf_threshold=conf)
                self.backend = "yolo"
            except Exception:
                self.impl = MotionDetector(min_area=int(os.getenv("MIN_DET_AREA", "2200")))
                self.backend = "motion_fallback"
        else:
            self.impl = MotionDetector(min_area=int(os.getenv("MIN_DET_AREA", "2200")))
            self.backend = "motion_fallback"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self.impl.detect(frame)
