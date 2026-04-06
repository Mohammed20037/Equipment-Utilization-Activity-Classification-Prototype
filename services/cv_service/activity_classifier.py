from collections import defaultdict, deque

from services.cv_service.motion_analyzer import MotionResult


class ActivityClassifier:
    def __init__(self):
        self.motion_hist = defaultdict(lambda: deque(maxlen=20))
        self.last_activity = defaultdict(lambda: "Waiting")

    @staticmethod
    def _is_arm_dominant(motion: MotionResult) -> bool:
        return "arm" in motion.motion_source or motion.articulated_score > max(4.0, motion.full_body_score * 0.9)

    def classify(self, equipment_id: str, equipment_class: str, motion: MotionResult) -> str:
        equipment_class = equipment_class.lower().replace("_", " ").replace("-", " ")
        self.motion_hist[equipment_id].append((motion.full_body_score, motion.articulated_score, motion.productive_score, motion.motion_source))

        if motion.motion_source == "track_gap_hold":
            return self.last_activity[equipment_id]

        if motion.state == "INACTIVE":
            self.last_activity[equipment_id] = "Waiting"
            return "Waiting"

        if equipment_class in {"excavator", "loader", "bulldozer", "roller"}:
            if self._is_arm_dominant(motion) or motion.productive_score > 7.0:
                label = "Digging"
            elif motion.full_body_score > 6.0:
                label = "Swinging/Loading"
            else:
                label = "Dumping"
            self.last_activity[equipment_id] = label
            return label

        if equipment_class in {"dump truck", "truck"}:
            if motion.full_body_score > 6.5:
                label = "Swinging/Loading"
            elif motion.articulated_score > 4.0:
                label = "Dumping"
            else:
                label = "Waiting"
            self.last_activity[equipment_id] = label
            return label

        if motion.full_body_score > 7.0:
            label = "Swinging/Loading"
        elif self._is_arm_dominant(motion):
            label = "Digging"
        else:
            label = "Dumping"
        self.last_activity[equipment_id] = label
        return label
