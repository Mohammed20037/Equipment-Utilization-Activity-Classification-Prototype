from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import cv2
import numpy as np


@dataclass
class MotionResult:
    state: str
    motion_source: str
    full_body_score: float
    articulated_score: float


class MotionAnalyzer:
    """
    Motion analyzer with selectable algorithms that can be combined with YOLO detections.

    Supported modes:
      - optical_flow_yolo: dense optical flow inside the detected equipment bbox
      - c3d_yolo: lightweight C3D-style temporal energy over a short clip window
      - lstm_yolo: temporal memory score over recent motion observations
    """

    def __init__(
        self,
        full_body_threshold: float = 3.0,
        articulated_threshold: float = 6.0,
        mode: str = "optical_flow_yolo",
        temporal_window: int = 16,
    ):
        self.full_body_threshold = full_body_threshold
        self.articulated_threshold = articulated_threshold
        self.mode = mode.lower()
        self.temporal_window = max(4, temporal_window)

        self.prev_gray = None
        self.prev_centers: Dict[str, Tuple[float, float]] = {}
        self.track_gray_clips: Dict[str, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self.track_motion_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.temporal_window))

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _clip_bbox(gray: np.ndarray, bbox) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        return x1, y1, x2, y2

    def _compute_flow_scores(self, roi_prev: np.ndarray, roi_now: np.ndarray) -> Tuple[float, float]:
        if roi_prev.size == 0 or roi_now.size == 0:
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev,
            roi_now,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        full_body_score = float(np.mean(mag))

        h, w = roi_now.shape[:2]
        arm_roi = mag[: max(1, h // 2), max(0, w // 2):]
        articulated_score = float(np.mean(arm_roi)) if arm_roi.size else 0.0
        return full_body_score, articulated_score

    def _c3d_style_score(self, clip: Deque[np.ndarray]) -> float:
        """C3D-like temporal energy proxy over a short frame clip."""
        if len(clip) < 3:
            return 0.0
        energies = []
        frames = list(clip)
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            energies.append(float(np.mean(diff)))
        return float(np.mean(energies)) if energies else 0.0

    def _lstm_style_score(self, history: Deque[float]) -> float:
        """LSTM-like temporal memory proxy using exponentially weighted history."""
        if not history:
            return 0.0
        values = np.array(history, dtype=np.float32)
        weights = np.exp(np.linspace(-1.5, 0.0, num=len(values))).astype(np.float32)
        weights = weights / np.sum(weights)
        return float(np.dot(values, weights))

    def analyze(self, frame: np.ndarray, track_id: str, bbox) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = self._center(bbox)

        center_shift = 0.0
        if track_id in self.prev_centers:
            px, py = self.prev_centers[track_id]
            center_shift = float(((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5)
        self.prev_centers[track_id] = center

        full_body_score = center_shift
        articulated_score = 0.0
        motion_source = "stationary"

        x1, y1, x2, y2 = self._clip_bbox(gray, bbox)
        if self.prev_gray is not None and x2 > x1 and y2 > y1:
            roi_now = gray[y1:y2, x1:x2]
            roi_prev = self.prev_gray[y1:y2, x1:x2]

            flow_full, flow_articulated = self._compute_flow_scores(roi_prev, roi_now)
            self.track_motion_history[track_id].append(flow_full)
            self.track_gray_clips[track_id].append(roi_now)

            if self.mode == "c3d_yolo":
                full_body_score = self._c3d_style_score(self.track_gray_clips[track_id])
                articulated_score = flow_articulated
                motion_source = "c3d_yolo"
            elif self.mode == "lstm_yolo":
                full_body_score = self._lstm_style_score(self.track_motion_history[track_id])
                articulated_score = flow_articulated
                motion_source = "lstm_yolo"
            else:
                full_body_score = max(center_shift, flow_full)
                articulated_score = flow_articulated
                motion_source = "optical_flow_yolo"

        self.prev_gray = gray

        if full_body_score >= self.full_body_threshold:
            return MotionResult("ACTIVE", motion_source, full_body_score, articulated_score)
        if articulated_score >= self.articulated_threshold:
            return MotionResult("ACTIVE", f"{motion_source}_arm", full_body_score, articulated_score)
        return MotionResult("INACTIVE", "stationary", full_body_score, articulated_score)
