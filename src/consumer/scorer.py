"""
Real-time scoring consumer: reads raw transaction events, computes the same
features used at training time (src/features/engineering.py), scores them
with the trained model (data/fraud_model.joblib), records the result to the
monitoring store, and emits the scored+flagged event downstream.

Run for real:
  python -m src.consumer.scorer --bootstrap-servers localhost:9092
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import transaction_to_features, FEATURE_COLUMNS
from src.monitoring import store
from src.transport import MessageConsumer, MessageProducer, KafkaConsumer, KafkaProducer

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "fraud_model.joblib"
METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "model_metadata.json"
INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "scored_transactions"


class FraudScorer:
    def __init__(self, model_path: Path = MODEL_PATH, threshold: float | None = None):
        self.model = joblib.load(model_path)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        self.model_version = metadata["winner"]
        self.threshold = threshold if threshold is not None else \
            metadata[metadata["winner"]]["chosen_threshold"]

    def score(self, transaction: dict) -> tuple[float, bool, dict, float]:
        t0 = time.perf_counter()
        features = transaction_to_features(transaction)
        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        prob = float(self.model.predict_proba(row)[0, 1])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        flagged = prob >= self.threshold
        return prob, flagged, features, latency_ms


def run(consumer: MessageConsumer, producer: MessageProducer, scorer: FraudScorer,
        max_messages: int | None = None, idle_timeout: float = 1.0) -> int:
    processed = 0
    idle_polls = 0
    max_idle_polls = 5
    while max_messages is None or processed < max_messages:
        txn = consumer.poll(timeout=idle_timeout)
        if txn is None:
            idle_polls += 1
            if max_messages is not None and idle_polls >= max_idle_polls:
                break
            continue
        idle_polls = 0

        prob, flagged, features, latency_ms = scorer.score(txn)
        store.record_scored_event(
            transaction_id=txn["transaction_id"], score=prob, flagged=flagged,
            model_version=scorer.model_version, scoring_latency_ms=latency_ms,
            features=features,
        )
        result = {**txn, "fraud_score": prob, "flagged": bool(flagged),
                  "model_version": scorer.model_version}
        producer.send(OUTPUT_TOPIC, key=txn["account_id"], value=result)
        processed += 1
    producer.flush()
    return processed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:9092")
    ap.add_argument("--group-id", default="fraud-scorer")
    ap.add_argument("--max-messages", type=int, default=None)
    args = ap.parse_args()

    consumer = KafkaConsumer(args.bootstrap_servers, args.group_id, [INPUT_TOPIC])
    producer = KafkaProducer(args.bootstrap_servers)
    scorer = FraudScorer()
    n = run(consumer, producer, scorer, max_messages=args.max_messages)
    print(f"Scored {n} transactions")
    consumer.close()
