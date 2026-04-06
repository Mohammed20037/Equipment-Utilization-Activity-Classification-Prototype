from collections import defaultdict, deque
from typing import Deque, Dict

from services.cv_service.motion_analyzer import MotionResult


class ActivityClassifier:
    def __init__(self):
        self.motion_hist: Dict[str, Deque[MotionResult]] = defaultdict(lambda: deque(maxlen=24))
        self.activity_votes: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=6))
        self.last_activity = defaultdict(lambda: "Waiting")

    @staticmethod
    def _majority(votes: Deque[str], default: str = "Waiting") -> str:
        if not votes:
            return default
        counts: Dict[str, int] = {}
        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
        return max(counts, key=lambda k: (counts[k], k == default))

    def _excavator_logic(self, motion: MotionResult) -> str:
        if motion.state == "INACTIVE" or motion.productive_score < 2.0:
            return "Waiting"
        if motion.articulated_score >= 5.2 and motion.full_body_score < 3.2:
            return "Digging"
        if motion.full_body_score >= 4.8 and motion.articulated_score >= 3.0:
            return "Swinging/Loading"
        return "Swinging/Loading"

    def _dump_truck_logic(self, motion: MotionResult) -> str:
        if motion.state == "INACTIVE":
            return "Waiting"
        if motion.articulated_score >= 3.6 and motion.full_body_score < 3.0:
            return "Dumping"
        if motion.full_body_score >= 4.5 or motion.productive_score >= 4.5:
            return "Swinging/Loading"
        return "Waiting"

    def _loader_backhoe_logic(self, motion: MotionResult) -> str:
        if motion.state == "INACTIVE":
            return "Waiting"
        if motion.articulated_score >= 4.3:
            return "Swinging/Loading"
        if motion.productive_score >= 4.8:
            return "Digging"
        return "Waiting"

    def classify(self, equipment_id: str, equipment_class: str, motion: MotionResult) -> str:
        equipment_class = equipment_class.lower().replace("_", " ").replace("-", " ")
        if motion.motion_source == "track_gap_hold":
            return self.last_activity[equipment_id]

        self.motion_hist[equipment_id].append(motion)

        if equipment_class == "excavator":
            raw = self._excavator_logic(motion)
        elif equipment_class in {"dump truck", "truck", "concrete mixer"}:
            raw = self._dump_truck_logic(motion)
        elif equipment_class in {"loader", "backhoe loader", "skid steer loader", "telehandler"}:
            raw = self._loader_backhoe_logic(motion)
        else:
            raw = "Waiting" if motion.state == "INACTIVE" else ("Swinging/Loading" if motion.productive_score >= 4.0 else "Waiting")

        self.activity_votes[equipment_id].append(raw)
        stable = self._majority(self.activity_votes[equipment_id], default=self.last_activity[equipment_id])
        self.last_activity[equipment_id] = stable
        return stable
