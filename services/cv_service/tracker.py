from dataclasses import dataclass
from typing import Dict, List, Tuple

from services.cv_service.detector import Detection


@dataclass
class Track:
    track_id: str
    bbox: Tuple[int, int, int, int]
    label: str
    lost_frames: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_lost: int = 15):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[str, Track] = {}

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections: List[Detection]) -> List[Track]:
        if not self.tracks:
            for det in detections:
                tid = f"EQ-{self.next_id:03d}"
                self.next_id += 1
                self.tracks[tid] = Track(track_id=tid, bbox=det.bbox, label=det.label)
            return list(self.tracks.values())

        unmatched_tracks = set(self.tracks.keys())
        for det in detections:
            det_center = self._center(det.bbox)
            best_track_id = None
            best_distance = 1e9
            for tid in unmatched_tracks:
                tr_center = self._center(self.tracks[tid].bbox)
                distance = ((det_center[0] - tr_center[0]) ** 2 + (det_center[1] - tr_center[1]) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = tid

            if best_track_id and best_distance <= self.max_distance:
                self.tracks[best_track_id].bbox = det.bbox
                self.tracks[best_track_id].lost_frames = 0
                unmatched_tracks.discard(best_track_id)
            else:
                tid = f"EQ-{self.next_id:03d}"
                self.next_id += 1
                self.tracks[tid] = Track(track_id=tid, bbox=det.bbox, label=det.label)

        for tid in list(self.tracks.keys()):
            if tid in unmatched_tracks:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > self.max_lost:
                    del self.tracks[tid]

        return list(self.tracks.values())
