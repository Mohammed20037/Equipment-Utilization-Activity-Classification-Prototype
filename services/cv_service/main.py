import json
import os
import time
from confluent_kafka import Producer

from shared.schemas import EquipmentEvent, TimeAnalyticsBlock, UtilizationBlock


def build_producer() -> Producer:
    return Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")})


def main() -> None:
    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    producer = build_producer()

    frame_id = 0
    active_seconds = 0.0
    idle_seconds = 0.0

    while True:
        frame_id += 1
        timestamp_sec = frame_id * 0.5
        is_active = (frame_id % 8) < 6
        if is_active:
            active_seconds += 0.5
            state = "ACTIVE"
            activity = "DIGGING"
            motion_source = "arm_only"
        else:
            idle_seconds += 0.5
            state = "INACTIVE"
            activity = "WAITING"
            motion_source = "stationary"

        tracked_seconds = active_seconds + idle_seconds
        utilization = (active_seconds / tracked_seconds * 100.0) if tracked_seconds else 0.0

        payload = EquipmentEvent(
            frame_id=frame_id,
            equipment_id="EX-001",
            equipment_class="excavator",
            timestamp=timestamp_sec,
            utilization=UtilizationBlock(
                current_state=state,
                current_activity=activity,
                motion_source=motion_source,
            ),
            time_analytics=TimeAnalyticsBlock(
                total_tracked_seconds=tracked_seconds,
                total_active_seconds=active_seconds,
                total_idle_seconds=idle_seconds,
                utilization_percent=round(utilization, 2),
            ),
        )

        producer.produce(topic, payload.model_dump_json().encode("utf-8"))
        producer.poll(0)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
