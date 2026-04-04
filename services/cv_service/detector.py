from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "excavator"


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
            detections.append(Detection(bbox=(x, y, x + w, y + h), confidence=min(0.99, area / 50000.0)))
        return detections
