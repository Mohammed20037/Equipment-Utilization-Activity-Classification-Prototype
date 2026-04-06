import csv
import logging
import os
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
from confluent_kafka import Producer

from services.cv_service.activity_classifier import ActivityClassifier
from services.cv_service.detector import HybridDetector
from services.cv_service.motion_analyzer import MotionAnalyzer, MotionResult
from services.cv_service.payload_builder import PayloadBuilder
from services.cv_service.tracker import CentroidTracker

logger = logging.getLogger("cv_service")


@dataclass
class DebounceState:
    confirmed_state: str = "INACTIVE"
    candidate_state: str = "INACTIVE"
    candidate_count: int = 0


class TrackStateDebouncer:
    def __init__(self, active_frames: int = 4, inactive_frames: int = 6):
        self.active_frames = active_frames
        self.inactive_frames = inactive_frames
        self.states: Dict[str, DebounceState] = {}

    def get_confirmed(self, track_id: str) -> str:
        return self.states.get(track_id, DebounceState()).confirmed_state

    def update(self, track_id: str, raw_state: str) -> str:
        state = self.states.setdefault(track_id, DebounceState())
        if raw_state == state.confirmed_state:
            state.candidate_state = raw_state
            state.candidate_count = 0
            return state.confirmed_state

        if raw_state != state.candidate_state:
            state.candidate_state = raw_state
            state.candidate_count = 1
        else:
            state.candidate_count += 1

        threshold = self.active_frames if raw_state == "ACTIVE" else self.inactive_frames
        if state.candidate_count >= threshold:
            state.confirmed_state = raw_state
            state.candidate_count = 0
        return state.confirmed_state


def build_producer() -> Producer:
    return Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")})


def wait_for_kafka(bootstrap_servers: str, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    host_port = bootstrap_servers.split(",")[0].strip()
    host, port = host_port.split(":") if ":" in host_port else (host_port, "9092")

    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"Kafka broker not reachable at {bootstrap_servers} after {timeout_sec}s")


def resolve_video_source(video_path: str) -> str:
    if video_path:
        candidate = Path(video_path)
        if candidate.exists():
            return str(candidate)

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


def init_writer(path: str, fps: float, frame_size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], Path]:
    output = Path(path)
    attempts = []
    if output.suffix.lower() == ".mp4":
        attempts = [("mp4v", output), ("XVID", output.with_suffix(".avi"))]
    else:
        attempts = [("XVID", output), ("mp4v", output.with_suffix(".mp4"))]

    for codec, candidate in attempts:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate), fourcc, fps, frame_size)
        if writer.isOpened():
            logger.info("VideoWriter initialized: path=%s codec=%s fps=%.2f size=%s", candidate, codec, fps, frame_size)
            return writer, candidate
        logger.warning("VideoWriter init failed: path=%s codec=%s", candidate, codec)
        writer.release()

    return None, output


def write_timeline_row(csv_path: Path, event) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "frame_id", "equipment_id", "equipment_class", "timestamp_sec", "state", "activity",
                "current_stop_seconds", "last_stop_seconds", "total_downtime_seconds", "utilization_percent",
            ])
        writer.writerow([
            event.frame_id,
            event.equipment_id,
            event.equipment_class,
            event.timestamp_sec,
            event.utilization.current_state,
            event.utilization.current_activity,
            event.time_analytics.current_stop_seconds,
            event.time_analytics.last_stop_seconds,
            event.time_analytics.total_downtime_seconds,
            event.time_analytics.utilization_percent,
        ])


def write_validation_row(csv_path: Path, frame_id: int, accepted: int, rejected: int, tracked: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["frame_id", "accepted_detections", "rejected_detections", "tracked_objects"])
        writer.writerow([frame_id, accepted, rejected, tracked])


def main() -> None:
    logging.basicConfig(level=os.getenv("CV_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    wait_for_kafka(bootstrap_servers, timeout_sec=int(os.getenv("KAFKA_WAIT_TIMEOUT_SEC", "60")))

    video_path = resolve_video_source(os.getenv("VIDEO_SOURCE", ""))
    fps_fallback = float(os.getenv("FRAME_RATE_FALLBACK", "10"))
    output_frame_path = Path(os.getenv("OUTPUT_FRAME_PATH", "data/processed/latest.jpg"))
    output_frame_tmp_path = output_frame_path.with_name("latest.tmp.jpg")
    processed_video_path = os.getenv("PROCESSED_VIDEO_PATH", "data/processed/processed_output.mp4")
    timeline_csv = Path(os.getenv("TIMELINE_OUTPUT_CSV", "data/processed/equipment_timeline.csv"))
    validation_csv = Path(os.getenv("VALIDATION_OUTPUT_CSV", "data/processed/validation_counts.csv"))
    os.makedirs(output_frame_path.parent, exist_ok=True)

    producer = build_producer()
    detector = HybridDetector()
    tracker = CentroidTracker(
        max_distance=float(os.getenv("TRACKER_MAX_DISTANCE", "90")),
        max_age=int(os.getenv("MAX_AGE", "30")),
        iou_match_threshold=float(os.getenv("IOU_MATCH_THRESHOLD", "0.4")),
        min_track_hits=int(os.getenv("MIN_TRACK_HITS", "3")),
    )
    motion = MotionAnalyzer(
        full_body_threshold=float(os.getenv("FULL_BODY_THRESHOLD", "3.0")),
        articulated_threshold=float(os.getenv("ARM_THRESHOLD", "6.0")),
        productive_threshold=float(os.getenv("PRODUCTIVE_MOTION_THRESHOLD", "4.0")),
        mode=os.getenv("MOTION_ANALYSIS_MODE", "optical_flow_masked"),
        temporal_window=int(os.getenv("MOTION_TEMPORAL_WINDOW", "16")),
    )
    activity = ActivityClassifier()
    payload_builder = PayloadBuilder()
    debouncer = TrackStateDebouncer(
        active_frames=int(os.getenv("ACTIVE_DEBOUNCE_FRAMES", "4")),
        inactive_frames=int(os.getenv("INACTIVE_DEBOUNCE_FRAMES", "6")),
    )
    missing_tolerance = int(os.getenv("TRACK_MISSING_TOLERANCE_FRAMES", "3"))
    min_track_hits_for_kpi = int(os.getenv("MIN_TRACK_HITS_FOR_KPI", os.getenv("MIN_TRACK_HITS", "3")))
    debug_overlay = os.getenv("DEBUG_OVERLAY", "0") == "1"

    logger.info(
        "CV pipeline started: detector=%s segmentation_status=%s motion_mode=%s",
        detector.backend,
        detector.segmentation.status,
        motion.mode,
    )

    cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open video source. Set VIDEO_SOURCE to a short fixed-camera clip "
            "or place video in data/raw_videos/."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = fps_fallback
    delta_t = 1.0 / fps

    writer: Optional[cv2.VideoWriter] = None
    writer_path = Path(processed_video_path)
    writer_size: Optional[Tuple[int, int]] = None

    frame_id = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if writer is None:
                h, w = frame.shape[:2]
                writer_size = (w, h)
                writer, writer_path = init_writer(processed_video_path, fps, writer_size)
                if writer is None:
                    raise RuntimeError("Unable to initialize processed video writer for mp4 or avi output")

            frame_id += 1
            timestamp_sec = time.time() - t0
            timestamp = format_ts(timestamp_sec)
            detections = detector.detect(frame)
            tracks = tracker.update(frame, detections)

            kpi_tracks = [t for t in tracks if t.hit_streak >= min_track_hits_for_kpi]
            logger.info(
                "Frame %s stats: raw_detections=%s accepted_detections=%s rejected_detections=%s tracked_objects=%s",
                frame_id,
                detector.last_raw_count,
                detector.last_filtered_count,
                len(detector.last_rejections),
                len(kpi_tracks),
            )
            write_validation_row(validation_csv, frame_id, detector.last_filtered_count, len(detector.last_rejections), len(kpi_tracks))

            cv2.putText(frame, f"detector={detector.backend}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"motion={motion.mode}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            if debug_overlay:
                for rej in detector.last_rejections:
                    x1, y1, x2, y2 = rej.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, rej.reason, (x1, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            for tr in tracks:
                if tr.lost_frames > missing_tolerance:
                    continue
                if tr.hit_streak < min_track_hits_for_kpi and tr.lost_frames == 0:
                    continue

                if tr.lost_frames > 0:
                    confirmed_state = debouncer.get_confirmed(tr.track_id)
                    motion_result = MotionResult(
                        state=confirmed_state,
                        motion_source="track_gap_hold",
                        full_body_score=0.0,
                        articulated_score=0.0,
                        productive_score=0.0,
                        mask_motion_density=0.0,
                        persistence_score=0.0,
                    )
                else:
                    motion_result = motion.analyze(frame, tr.track_id, tr.bbox, tr.mask, tr.label)
                    confirmed_state = debouncer.update(tr.track_id, motion_result.state)
                    motion_result.state = confirmed_state

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
                write_timeline_row(timeline_csv, event)

            try:
                if not cv2.imwrite(str(output_frame_tmp_path), frame):
                    logger.error("Failed to write temporary frame: %s", output_frame_tmp_path)
                else:
                    os.replace(output_frame_tmp_path, output_frame_path)
            except OSError as exc:
                logger.error("Atomic frame write failed: tmp=%s final=%s err=%s", output_frame_tmp_path, output_frame_path, exc)
                if output_frame_tmp_path.exists():
                    output_frame_tmp_path.unlink(missing_ok=True)
            if writer is not None and writer_size is not None:
                out_frame = frame
                if (frame.shape[1], frame.shape[0]) != writer_size:
                    out_frame = cv2.resize(frame, writer_size)
                writer.write(out_frame)
                if frame_id % int(os.getenv("WRITER_LOG_EVERY_N_FRAMES", "30")) == 0:
                    logger.info("Wrote frame_id=%s to %s", frame_id, writer_path)

            producer.poll(0)
            time.sleep(delta_t)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("VideoWriter released: %s", writer_path)
        producer.flush(5)


if __name__ == "__main__":
    main()
