"""
Message transport abstraction.

Why this exists: the sandbox this project was first built in has no Docker
and a network allowlist that blocks every Kafka/Redpanda binary distribution
channel (Apache mirrors, Confluent's package repo, GitHub release assets all
return blocked-by-allowlist -- verified directly, not assumed). There is
real, unrestricted PyPI access, so the actual `confluent-kafka` client
library (the librdkafka C library wrapped in Python -- the real Kafka wire
protocol implementation, not a mock) installs and imports fine; there's just
no broker to point it at inside that sandbox.

Rather than fake this, the producer/consumer/scorer code below is written
against a small `MessageProducer` / `MessageConsumer` interface with two
implementations:

  - KafkaProducer / KafkaConsumer: the REAL implementation, using
    confluent-kafka against an actual broker (see docker-compose.yml).
    This is what runs in production and what you should run locally to see
    it against real Kafka.
  - InMemoryProducer / InMemoryConsumer: a same-process queue used ONLY by
    the test suite (tests/test_pipeline_e2e.py), so the feature engineering,
    scoring, and monitoring logic can be verified end-to-end without a
    live broker.

This is standard dependency-inversion practice for testable streaming
systems (the same reason production Kafka codebases use consumer/producer
interfaces rather than importing the client directly everywhere) -- it is
not a workaround invented to paper over the missing broker, though the
missing broker is what forced writing it now rather than later.
"""
from __future__ import annotations

import json
import queue
from abc import ABC, abstractmethod
from typing import Callable, Optional


class MessageProducer(ABC):
    @abstractmethod
    def send(self, topic: str, key: str, value: dict) -> None:
        ...

    @abstractmethod
    def flush(self, timeout: float = 10.0) -> None:
        ...


class MessageConsumer(ABC):
    @abstractmethod
    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        """Return the next message's value as a dict, or None if no message
        arrived within `timeout` seconds."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Real Kafka implementation (confluent-kafka / librdkafka)
# ---------------------------------------------------------------------------

class KafkaProducer(MessageProducer):
    def __init__(self, bootstrap_servers: str, on_error: Optional[Callable] = None):
        from confluent_kafka import Producer  # deferred import: optional dep
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._on_error = on_error or (lambda err, msg: print(f"[kafka] delivery failed: {err}"))

    def _delivery_report(self, err, msg):
        if err is not None:
            self._on_error(err, msg)

    def send(self, topic: str, key: str, value: dict) -> None:
        self._producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=self._delivery_report,
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)


class KafkaConsumer(MessageConsumer):
    def __init__(self, bootstrap_servers: str, group_id: str, topics: list[str]):
        from confluent_kafka import Consumer  # deferred import: optional dep
        self._consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })
        self._consumer.subscribe(topics)

    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        msg = self._consumer.poll(timeout)
        if msg is None or msg.error():
            return None
        return json.loads(msg.value().decode("utf-8"))

    def close(self) -> None:
        self._consumer.close()


# ---------------------------------------------------------------------------
# In-memory implementation (used by the test suite only)
# ---------------------------------------------------------------------------

class InMemoryBroker:
    """A shared queue standing in for a Kafka topic partition, scoped to a
    single test process. NOT used anywhere outside tests/."""

    def __init__(self):
        self._topics: dict[str, "queue.Queue[dict]"] = {}

    def _topic_queue(self, topic: str) -> "queue.Queue[dict]":
        return self._topics.setdefault(topic, queue.Queue())


class InMemoryProducer(MessageProducer):
    def __init__(self, broker: InMemoryBroker):
        self._broker = broker

    def send(self, topic: str, key: str, value: dict) -> None:
        self._broker._topic_queue(topic).put(value)

    def flush(self, timeout: float = 10.0) -> None:
        pass


class InMemoryConsumer(MessageConsumer):
    def __init__(self, broker: InMemoryBroker, topics: list[str]):
        self._broker = broker
        self._topics = topics

    def poll(self, timeout: float = 1.0) -> Optional[dict]:
        for topic in self._topics:
            q = self._broker._topic_queue(topic)
            if not q.empty():
                return q.get()
        return None

    def close(self) -> None:
        pass
