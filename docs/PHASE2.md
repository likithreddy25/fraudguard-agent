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
- **Verified on a real machine (2026-08-27):** `docker compose up -d` brought up a real single-node `apache/kafka:3.8.0` broker (KRaft mode) on `localhost:9092`. `event_generator.py` and `scorer.py` ran against it for real — producer, broker, consumer, scorer, and the SQLite monitoring store, no in-memory transport involved.

## Real live-broker results (2026-08-27, on the author's own machine)
- **22,088 transactions** produced, consumed, scored, and persisted end-to-end through the real Kafka broker across the test session.
- **Flag rate: 1.62%** — consistent with the ~2% fraud-injection rate the generator targets by design (not cherry-picked).
- **Average scoring latency: 1.75ms** per transaction (feature engineering + model inference), well within real-time budget for a streaming consumer at this throughput.
- **Score distribution:** min 0.0012, max 0.988, mean 0.117 — a believable long-tailed distribution for a fraud model, not artificially bimodal.
- One real operational issue hit and resolved during this run, documented for honesty: repeated interrupted (`Ctrl+C`) consumer runs under the same default `group.id` (`fraud-scorer`) left stale members registered with the broker's group coordinator, causing new joins under that same group id to stall on `JoinGroupRequest`/`LeaveGroupRequest` timeouts. Fixed by running under a fresh `--group-id` (`fraud-scorer-run1`) and by passing `--max-messages` so the consumer exits cleanly instead of idling forever. A production deployment would additionally want a graceful-shutdown handler (catch `SIGINT`, call `consumer.close()`) so this can't happen from an operator hitting Ctrl+C.
