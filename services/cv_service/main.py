import os
import time
from datetime import timedelta
from pathlib import Path

import cv2
from confluent_kafka import Producer

from services.cv_service.activity_classifier import ActivityClassifier
from services.cv_service.detector import HybridDetector
from services.cv_service.motion_analyzer import MotionAnalyzer
from services.cv_service.payload_builder import PayloadBuilder
from services.cv_service.tracker import CentroidTracker


def build_producer() -> Producer:
    return Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")})




def resolve_video_source(video_path: str) -> str:
    if video_path:
        return video_path
    raw_dir = Path("data/raw_videos")
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(raw_dir.glob(ext))
        if matches:
            return str(matches[0])
    return ""

def format_ts(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    return str(timedelta(seconds=int(seconds))) + f".{millis:03d}"


def annotate_frame(frame, track_id, bbox, state, activity):
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{track_id} {state} {activity}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    video_path = resolve_video_source(os.getenv("VIDEO_SOURCE", ""))
    fps_fallback = float(os.getenv("FRAME_RATE_FALLBACK", "10"))
    output_frame_path = Path(os.getenv("OUTPUT_FRAME_PATH", "data/processed/latest.jpg"))
    processed_video_path = os.getenv("PROCESSED_VIDEO_PATH", "data/processed/processed_output.mp4")
    output_frame_path.parent.mkdir(parents=True, exist_ok=True)

    producer = build_producer()
    detector = HybridDetector()
    tracker = CentroidTracker()
    motion = MotionAnalyzer(
        full_body_threshold=float(os.getenv("FULL_BODY_THRESHOLD", "3.0")),
        articulated_threshold=float(os.getenv("ARM_THRESHOLD", "6.0")),
    )
    activity = ActivityClassifier()
    payload_builder = PayloadBuilder()

    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source. Set VIDEO_SOURCE or place video in data/raw_videos/")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = fps_fallback
    delta_t = 1.0 / fps

    writer = None

    frame_id = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(processed_video_path, fourcc, fps, (w, h))

        frame_id += 1
        timestamp_sec = time.time() - t0
        timestamp = format_ts(timestamp_sec)
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        cv2.putText(frame, f"detector={detector.backend}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        for tr in tracks:
            motion_result = motion.analyze(frame, tr.track_id, tr.bbox)
            activity_label = activity.classify(tr.track_id, tr.label, motion_result)
            event = payload_builder.build(
                frame_id=frame_id,
                timestamp=timestamp,
                timestamp_sec=round(timestamp_sec, 3),
                equipment_id=tr.track_id,
                equipment_class=tr.label,
                state=motion_result.state,
                activity=activity_label,
                motion_source=motion_result.motion_source,
                delta_t=delta_t,
            )
            annotate_frame(frame, tr.track_id, tr.bbox, motion_result.state, activity_label)
            producer.produce(topic, event.model_dump_json().encode("utf-8"))

        cv2.imwrite(str(output_frame_path), frame)
        if writer is not None:
            writer.write(frame)
        producer.poll(0)
        time.sleep(delta_t)


if __name__ == "__main__":
    main()
