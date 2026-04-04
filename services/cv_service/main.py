import os
import time

import cv2
from confluent_kafka import Producer

from services.cv_service.activity_classifier import ActivityClassifier
from services.cv_service.detector import MotionDetector
from services.cv_service.motion_analyzer import MotionAnalyzer
from services.cv_service.payload_builder import PayloadBuilder
from services.cv_service.tracker import CentroidTracker


def build_producer() -> Producer:
    return Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")})


def main() -> None:
    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    video_path = os.getenv("VIDEO_SOURCE", "")
    fps_fallback = float(os.getenv("FRAME_RATE_FALLBACK", "10"))

    producer = build_producer()
    detector = MotionDetector(min_area=int(os.getenv("MIN_DET_AREA", "2200")))
    tracker = CentroidTracker()
    motion = MotionAnalyzer(
        full_body_threshold=float(os.getenv("FULL_BODY_THRESHOLD", "3.0")),
        articulated_threshold=float(os.getenv("ARM_THRESHOLD", "6.0")),
    )
    activity = ActivityClassifier()
    payload_builder = PayloadBuilder()

    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = fps_fallback
    delta_t = 1.0 / fps

    frame_id = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_id += 1
        timestamp = time.time() - t0
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for tr in tracks:
            motion_result = motion.analyze(frame, tr.track_id, tr.bbox)
            activity_label = activity.classify(tr.label, motion_result)
            event = payload_builder.build(
                frame_id=frame_id,
                timestamp=round(timestamp, 3),
                equipment_id=tr.track_id,
                equipment_class=tr.label,
                state=motion_result.state,
                activity=activity_label,
                motion_source=motion_result.motion_source,
                delta_t=delta_t,
            )
            producer.produce(topic, event.model_dump_json().encode("utf-8"))

        producer.poll(0)
        time.sleep(delta_t)


if __name__ == "__main__":
    main()
