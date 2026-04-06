from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionResult:
    state: str
    motion_source: str
    full_body_score: float
    articulated_score: float
    productive_score: float
    mask_motion_density: float
    persistence_score: float


class MotionAnalyzer:
    """
    Segmentation-guided motion analysis with articulated-motion-aware ACTIVE inference.
    Primary motion is computed on mask pixels; full-box motion is used only as fallback.
    """

    ARTICULATED_CLASSES = {
        "excavator", "loader", "backhoe loader", "crane", "telehandler", "bulldozer", "skid steer loader"
    }

    def __init__(
        self,
        full_body_threshold: float = 3.0,
        articulated_threshold: float = 6.0,
        productive_threshold: float = 4.0,
        mode: str = "optical_flow_masked",
        temporal_window: int = 16,
    ):
        self.full_body_threshold = full_body_threshold
        self.articulated_threshold = articulated_threshold
        self.productive_threshold = productive_threshold
        self.mode = mode.lower()
        self.temporal_window = max(4, temporal_window)

        self.prev_gray: Optional[np.ndarray] = None
        self.track_motion_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.temporal_window))
        self.track_density_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.temporal_window))

    @staticmethod
    def _clip_bbox(gray: np.ndarray, bbox) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        return x1, y1, x2, y2

    @staticmethod
    def _weighted_temporal_average(values: Deque[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        weights = np.exp(np.linspace(-1.2, 0.0, len(arr))).astype(np.float32)
        weights = weights / np.sum(weights)
        return float(np.dot(arr, weights))

    def _compute_masked_flow(
        self,
        roi_prev: np.ndarray,
        roi_now: np.ndarray,
        roi_mask: Optional[np.ndarray],
        equipment_class: str,
    ) -> Tuple[float, float, float, float]:
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

        if roi_mask is None or roi_mask.size == 0:
            active_pixels = np.ones_like(mag, dtype=bool)
        else:
            active_pixels = roi_mask > 0
            if not np.any(active_pixels):
                active_pixels = np.ones_like(mag, dtype=bool)

        masked_mag = mag[active_pixels]
        full_body = float(np.mean(masked_mag)) if masked_mag.size else 0.0
        motion_density = float(np.mean(masked_mag > max(0.8, self.full_body_threshold * 0.2))) if masked_mag.size else 0.0

        h, w = mag.shape
        top = slice(0, max(1, h // 2))
        bottom = slice(max(0, h // 2), h)
        right = slice(max(0, w // 2), w)
        left = slice(0, max(1, w // 2))

        productive_region = mag[top, right]
        body_region = mag[bottom, left]

        if roi_mask is not None and roi_mask.size > 0:
            productive_mask = roi_mask[top, right] > 0
            body_mask = roi_mask[bottom, left] > 0
            productive_region = productive_region[productive_mask] if np.any(productive_mask) else productive_region
            body_region = body_region[body_mask] if np.any(body_mask) else body_region

        productive_local = float(np.mean(productive_region)) if productive_region.size else 0.0
        body_local = float(np.mean(body_region)) if body_region.size else 0.0

        if equipment_class in self.ARTICULATED_CLASSES:
            articulated = productive_local
            productive = max(productive_local, full_body - 0.45 * body_local)
        else:
            articulated = 0.5 * productive_local
            productive = max(full_body, productive_local * 0.8)

        return full_body, articulated, productive, motion_density

    def analyze(self, frame: np.ndarray, track_id: str, bbox, mask: Optional[np.ndarray], equipment_class: str) -> MotionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self._clip_bbox(gray, bbox)

        full_body_score = 0.0
        articulated_score = 0.0
        productive_score = 0.0
        density = 0.0
        motion_source = "stationary"

        if self.prev_gray is not None and x2 > x1 and y2 > y1:
            roi_now = gray[y1:y2, x1:x2]
            roi_prev = self.prev_gray[y1:y2, x1:x2]
            roi_mask = None
            if mask is not None:
                roi_mask = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)

            full_body_score, articulated_score, productive_score, density = self._compute_masked_flow(
                roi_prev=roi_prev,
                roi_now=roi_now,
                roi_mask=roi_mask,
                equipment_class=equipment_class,
            )
            motion_source = "masked_optical_flow"

        self.track_motion_history[track_id].append(productive_score)
        self.track_density_history[track_id].append(density)
        smooth_productive = self._weighted_temporal_average(self.track_motion_history[track_id])
        persistence = self._weighted_temporal_average(self.track_density_history[track_id])
        self.prev_gray = gray

        is_articulated_class = equipment_class in self.ARTICULATED_CLASSES
        active_by_productive = smooth_productive >= self.productive_threshold
        active_by_articulated = is_articulated_class and articulated_score >= self.articulated_threshold * 0.75 and persistence >= 0.08
        active_by_body = full_body_score >= self.full_body_threshold and persistence >= 0.05

        if active_by_productive or active_by_articulated or active_by_body:
            state = "ACTIVE"
            source = motion_source if active_by_productive else f"{motion_source}_weak"
        else:
            state = "INACTIVE"
            source = "stationary"

        return MotionResult(
            state=state,
            motion_source=source,
            full_body_score=full_body_score,
            articulated_score=articulated_score,
            productive_score=smooth_productive,
            mask_motion_density=persistence,
            persistence_score=persistence,
        )
