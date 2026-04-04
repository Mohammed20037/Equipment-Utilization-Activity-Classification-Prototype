from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class MotionResult:
    state: str
    motion_source: str
    full_body_score: float
    articulated_score: float


class MotionAnalyzer:
    def __init__(self, full_body_threshold: float = 3.0, articulated_threshold: float = 6.0):
        self.full_body_threshold = full_body_threshold
        self.articulated_threshold = articulated_threshold
        self.prev_gray = None
        self.prev_centers: Dict[str, Tuple[float, float]] = {}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def analyze(self, frame: np.ndarray, track_id: str, bbox) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = self._center(bbox)

        full_body_score = 0.0
        if track_id in self.prev_centers:
            px, py = self.prev_centers[track_id]
            full_body_score = float(((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5)
        self.prev_centers[track_id] = center

        articulated_score = 0.0
        if self.prev_gray is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(gray.shape[1], x2)
            y2 = min(gray.shape[0], y2)
            if x2 > x1 and y2 > y1:
                roi_now = gray[y1:y2, x1:x2]
                roi_prev = self.prev_gray[y1:y2, x1:x2]
                h, w = roi_now.shape[:2]
                # Approximate articulated area as upper-right ROI (arm/bucket proxy)
                arm_now = roi_now[: max(1, h // 2), max(0, w // 2):]
                arm_prev = roi_prev[: max(1, h // 2), max(0, w // 2):]
                diff = cv2.absdiff(arm_now, arm_prev)
                articulated_score = float(np.mean(diff)) if diff.size else 0.0

        self.prev_gray = gray

        if full_body_score >= self.full_body_threshold:
            return MotionResult("ACTIVE", "full_body", full_body_score, articulated_score)
        if articulated_score >= self.articulated_threshold:
            return MotionResult("ACTIVE", "arm_only", full_body_score, articulated_score)
        return MotionResult("INACTIVE", "stationary", full_body_score, articulated_score)
