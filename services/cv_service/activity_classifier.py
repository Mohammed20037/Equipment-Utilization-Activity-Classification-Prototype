from services.cv_service.motion_analyzer import MotionResult


class ActivityClassifier:
    def classify(self, equipment_class: str, motion: MotionResult) -> str:
        if motion.state == "INACTIVE":
            return "WAITING"
        if motion.motion_source == "arm_only" and motion.articulated_score > 12:
            return "DIGGING"
        if motion.motion_source == "full_body" and motion.full_body_score > 8:
            return "SWINGING_LOADING"
        return "WORKING"
