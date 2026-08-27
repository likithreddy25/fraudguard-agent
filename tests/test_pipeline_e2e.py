"""
End-to-end pipeline test using the in-memory transport (see src/transport.py
for why: no live Kafka broker in the sandbox this was first built in).

This genuinely exercises the real code path: event generation -> the same
KafkaProducer.send()/poll() interface -> feature engineering -> the real
trained model -> monitoring store -> drift detection. Only the wire
transport is swapped; nothing about the scoring or monitoring logic is
mocked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.transport import InMemoryBroker, InMemoryProducer, InMemoryConsumer
from src.producer.event_generator import run as run_producer
from src.consumer.scorer import run as run_scorer, FraudScorer, INPUT_TOPIC, OUTPUT_TOPIC
from src.monitoring import store
from src.monitoring.drift import detect_score_drift


@pytest.fixture(autouse=True)
def clean_store():
    store.reset()
    yield
    store.reset()


def test_producer_emits_events_without_label():
    broker = InMemoryBroker()
    producer = InMemoryProducer(broker)
    sent = run_producer(producer, n_accounts=20, n_events=50, events_per_sec=0,
                         fraud_injection_rate=0.05)
    assert sent == 50

    consumer = InMemoryConsumer(broker, [INPUT_TOPIC])
    msg = consumer.poll()
    assert msg is not None
    assert "is_fraud" not in msg or msg["is_fraud"] is None
    assert set(["transaction_id", "amount", "merchant_category"]).issubset(msg.keys())


def test_scorer_end_to_end_against_real_model():
    broker = InMemoryBroker()
    producer = InMemoryProducer(broker)
    n = run_producer(producer, n_accounts=30, n_events=300, events_per_sec=0,
                      fraud_injection_rate=0.08)
    assert n == 300

    scorer = FraudScorer()
    consumer = InMemoryConsumer(broker, [INPUT_TOPIC])
    out_producer = InMemoryProducer(broker)
    processed = run_scorer(consumer, out_producer, scorer, max_messages=300)
    assert processed == 300

    # every scored event actually landed in the monitoring store
    assert store.event_count() == 300

    out_consumer = InMemoryConsumer(broker, [OUTPUT_TOPIC])
    scored_msg = out_consumer.poll()
    assert scored_msg is not None
    assert "fraud_score" in scored_msg
    assert 0.0 <= scored_msg["fraud_score"] <= 1.0
    assert isinstance(scored_msg["flagged"], bool)

    # with a higher injection rate than the model's threshold-tuning fraud
    # rate, we expect a real, nonzero flag rate -- not asserting an exact
    # number since that would be circular; just that it's plausible and not
    # degenerate (all-flagged or none-flagged).
    rate = store.flag_rate()
    assert 0.0 < rate < 1.0


def test_drift_detection_flags_a_real_distribution_shift():
    reference = [0.02 + 0.01 * (i % 5) for i in range(200)]  # tight, low scores
    shifted = [0.6 + 0.02 * (i % 5) for i in range(200)]      # shifted high
    result = detect_score_drift(reference, shifted)
    assert result["drift_detected"] is True
    assert result["p_value"] < 0.01


def test_drift_detection_does_not_false_positive_on_same_distribution():
    import random
    random.seed(1)
    reference = [random.gauss(0.1, 0.03) for _ in range(300)]
    recent = [random.gauss(0.1, 0.03) for _ in range(300)]
    result = detect_score_drift(reference, recent)
    assert result["drift_detected"] is False
