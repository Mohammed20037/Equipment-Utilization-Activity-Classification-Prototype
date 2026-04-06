from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from services.cv_service.detector import Detection


@dataclass
class Track:
    track_id: str
    bbox: Tuple[int, int, int, int]
    label: str
    lost_frames: int = 0
    hit_streak: int = 1
    appearance_hist: Optional[np.ndarray] = None


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0, max_lost: int = 25):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[str, Track] = {}

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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

    @staticmethod
    def _appearance_hist(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _hist_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 1.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

    def _create_track(self, det: Detection, frame: np.ndarray) -> Track:
        tid = f"EQ-{self.next_id:03d}"
        self.next_id += 1
        return Track(
            track_id=tid,
            bbox=det.bbox,
            label=det.label,
            appearance_hist=self._appearance_hist(frame, det.bbox),
        )

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for det in detections:
                tr = self._create_track(det, frame)
                self.tracks[tr.track_id] = tr
            return list(self.tracks.values())

        unmatched_tracks = set(self.tracks.keys())
        for det in detections:
            det_center = self._center(det.bbox)
            det_hist = self._appearance_hist(frame, det.bbox)

            best_track_id = None
            best_score = 1e9
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                if tr.label != det.label:
                    continue

                tr_center = self._center(tr.bbox)
                center_dist = ((det_center[0] - tr_center[0]) ** 2 + (det_center[1] - tr_center[1]) ** 2) ** 0.5
                if center_dist > self.max_distance:
                    continue

                iou = self._iou(det.bbox, tr.bbox)
                hist_dist = self._hist_distance(det_hist, tr.appearance_hist)
                lost_penalty = min(1.0, tr.lost_frames / max(1.0, self.max_lost))

                score = center_dist * 0.5 + (1.0 - iou) * 30.0 + hist_dist * 20.0 + lost_penalty * 10.0
                if score < best_score:
                    best_score = score
                    best_track_id = tid

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track.bbox = det.bbox
                track.lost_frames = 0
                track.hit_streak += 1
                track.appearance_hist = det_hist
                unmatched_tracks.discard(best_track_id)
            else:
                tr = self._create_track(det, frame)
                self.tracks[tr.track_id] = tr

        for tid in list(self.tracks.keys()):
            if tid in unmatched_tracks:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > self.max_lost:
                    del self.tracks[tid]

        return list(self.tracks.values())
