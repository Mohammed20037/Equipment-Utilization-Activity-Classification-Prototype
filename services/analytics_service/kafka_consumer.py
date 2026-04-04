import os

from confluent_kafka import Consumer


def build_consumer() -> Consumer:
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": os.getenv("KAFKA_GROUP_ID", "analytics-group"),
        "auto.offset.reset": "earliest",
    }
    return Consumer(conf)
