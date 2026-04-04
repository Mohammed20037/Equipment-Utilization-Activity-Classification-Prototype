import json
import os
import time

from confluent_kafka import Consumer
from sqlalchemy import text

from shared.db import get_engine


def build_consumer() -> Consumer:
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": os.getenv("KAFKA_GROUP_ID", "analytics-group"),
        "auto.offset.reset": "earliest",
    }
    return Consumer(conf)


def main() -> None:
    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    consumer = build_consumer()
    consumer.subscribe([topic])
    engine = get_engine()

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        event = json.loads(msg.value().decode("utf-8"))
        util = event["utilization"]
        analytics = event["time_analytics"]

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO frame_events (
                      frame_id, equipment_id, equipment_class, timestamp_sec,
                      current_state, current_activity, motion_source,
                      total_tracked_seconds, total_active_seconds, total_idle_seconds, utilization_percent
                    ) VALUES (
                      :frame_id, :equipment_id, :equipment_class, :timestamp_sec,
                      :current_state, :current_activity, :motion_source,
                      :total_tracked_seconds, :total_active_seconds, :total_idle_seconds, :utilization_percent
                    )
                    """
                ),
                {
                    "frame_id": event["frame_id"],
                    "equipment_id": event["equipment_id"],
                    "equipment_class": event["equipment_class"],
                    "timestamp_sec": event["timestamp"],
                    "current_state": util["current_state"],
                    "current_activity": util["current_activity"],
                    "motion_source": util["motion_source"],
                    "total_tracked_seconds": analytics["total_tracked_seconds"],
                    "total_active_seconds": analytics["total_active_seconds"],
                    "total_idle_seconds": analytics["total_idle_seconds"],
                    "utilization_percent": analytics["utilization_percent"],
                },
            )

            conn.execute(
                text(
                    """
                    INSERT INTO equipment_summary (
                      equipment_id, equipment_class, total_tracked_seconds, total_active_seconds,
                      total_idle_seconds, utilization_percent, last_activity, last_state
                    ) VALUES (
                      :equipment_id, :equipment_class, :total_tracked_seconds, :total_active_seconds,
                      :total_idle_seconds, :utilization_percent, :last_activity, :last_state
                    )
                    ON CONFLICT (equipment_id) DO UPDATE SET
                      equipment_class = EXCLUDED.equipment_class,
                      total_tracked_seconds = EXCLUDED.total_tracked_seconds,
                      total_active_seconds = EXCLUDED.total_active_seconds,
                      total_idle_seconds = EXCLUDED.total_idle_seconds,
                      utilization_percent = EXCLUDED.utilization_percent,
                      last_activity = EXCLUDED.last_activity,
                      last_state = EXCLUDED.last_state,
                      updated_at = NOW();
                    """
                ),
                {
                    "equipment_id": event["equipment_id"],
                    "equipment_class": event["equipment_class"],
                    "total_tracked_seconds": analytics["total_tracked_seconds"],
                    "total_active_seconds": analytics["total_active_seconds"],
                    "total_idle_seconds": analytics["total_idle_seconds"],
                    "utilization_percent": analytics["utilization_percent"],
                    "last_activity": util["current_activity"],
                    "last_state": util["current_state"],
                },
            )
        time.sleep(0.01)


if __name__ == "__main__":
    main()
