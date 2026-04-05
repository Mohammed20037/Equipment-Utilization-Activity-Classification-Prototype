import json
import os

from services.analytics_service.db_writer import DbWriter
from services.analytics_service.kafka_consumer import build_consumer
from shared.db import get_engine


def main() -> None:
    topic = os.getenv("KAFKA_TOPIC", "equipment.events")
    consumer = build_consumer()
    consumer.subscribe([topic])
    writer = DbWriter(get_engine())

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        event = json.loads(msg.value().decode("utf-8"))
        writer.write_event_and_summary(event)


if __name__ == "__main__":
    main()
