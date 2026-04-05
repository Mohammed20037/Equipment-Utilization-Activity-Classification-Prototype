from collections import defaultdict, deque

from services.cv_service.motion_analyzer import MotionResult


class ActivityClassifier:
    def __init__(self):
        self.motion_hist = defaultdict(lambda: deque(maxlen=12))

    def classify(self, equipment_id: str, equipment_class: str, motion: MotionResult) -> str:
        self.motion_hist[equipment_id].append((motion.full_body_score, motion.articulated_score, motion.motion_source))

        if motion.state == "INACTIVE":
            return "WAITING"

        # Articulated motion dominated => digging-like behavior
        if motion.motion_source == "arm_only":
            if motion.articulated_score > 14:
                return "DIGGING"
            return "DUMPING"

        # Body motion with active state implies transport/swing phase
        if motion.full_body_score > 8:
            return "SWINGING_LOADING"

        return "DUMPING"
