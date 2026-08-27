# Phase 2 — Kafka Streaming Layer

## The environment constraint (disclosed, not hidden)
This project was first built inside a sandboxed Cowork session with no Docker and a network allowlist that blocks every channel a real Kafka or Kafka-compatible broker binary could come from. Verified directly, not assumed:
- `downloads.apache.org`, `dlcdn.apache.org`, every Apache mirror tried → blocked (`blocked-by-allowlist`)
- `packages.confluent.io` → blocked
- GitHub *release assets* (`release-assets.githubusercontent.com`) → blocked, even though `github.com` itself and `pypi.org` are reachable (confirmed Redpanda only ships its `rpk` CLI via plain GitHub releases anyway — the actual broker requires their own package repo, which is also blocked)
- No `sudo`/root in the sandbox → can't `apt-get install` a JVM-based Kafka distribution even though OpenJDK 11 is present
- `pip install confluent-kafka` → **works fine** (PyPI wheels are unrestricted), confirming the real Kafka client library (librdkafka, the same C library used in production Kafka deployments) is fully available — there's just no broker to point it at inside that specific sandbox.

## What this phase built anyway
`src/transport.py` defines `MessageProducer` / `MessageConsumer` interfaces with two implementations:
- `KafkaProducer` / `KafkaConsumer` — the real implementation, using `confluent-kafka` against an actual broker. This is what `docker-compose.yml` (official `apache/kafka:3.8.0` image, KRaft mode, no Zookeeper) stands up, and what `src/producer/event_generator.py` / `src/consumer/scorer.py` use by default.
- `InMemoryProducer` / `InMemoryConsumer` — a same-process queue used ONLY by the test suite, so the feature engineering / scoring / monitoring logic is verified end-to-end (`tests/test_pipeline_e2e.py`, all passing) without needing a live broker.

This is standard dependency-inversion practice for testable streaming systems, not an invented workaround — but the missing broker is what forced writing it now rather than later, and that's worth being upfront about.

## What's verified where
- **Verified in the sandbox:** the full pipeline logic (event generation → feature engineering → real trained model scoring → monitoring store → drift detection), via the in-memory transport, `pytest tests/` — 6/6 passing.
- **Needs your own machine (real Docker, real Kafka):** `docker compose up -d`, then run the real producer and scorer against `localhost:9092`, then query `data/monitoring.db` for real numbers. `docker-compose.yml` has the exact commands. Report back the output and it gets documented here honestly, same as the Kafka broker constraint itself.
