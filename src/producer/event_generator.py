"""
Real-time transaction event producer.

Note it emits raw transactions WITHOUT the is_fraud label (unlike the
offline training CSV) -- the whole point of the online system is to predict
that label; shipping it on the stream would be leaking the answer.

Run for real:  python -m src.producer.event_generator --bootstrap-servers localhost:9092
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict, deque

import numpy as np

from src.features.schema import Transaction, MERCHANT_CATEGORIES, COUNTRIES
from src.transport import MessageProducer, KafkaProducer

TOPIC = "transactions"


class AccountState:
    """Tracks rolling per-account counters the same way a real streaming
    system would (a small in-memory/Redis-backed window), so the emitted
    events carry realistic velocity features rather than random numbers."""

    def __init__(self):
        self.country = random.choice(COUNTRIES)
        self.age_days = random.randint(1, 2600)
        self.avg_amount_30d = float(np.random.lognormal(3.5, 0.8))
        self.recent_timestamps: deque[float] = deque(maxlen=200)

    def record_and_count(self, now: float) -> tuple[int, int]:
        self.recent_timestamps.append(now)
        last_1h = sum(1 for t in self.recent_timestamps if now - t <= 3600)
        last_24h = sum(1 for t in self.recent_timestamps if now - t <= 86400)
        return last_1h, last_24h


def make_event(accounts: dict[str, AccountState], fraud_injection_rate: float) -> dict:
    account_id = random.choice(list(accounts.keys()))
    state = accounts[account_id]
    now = time.time()
    last_1h, last_24h = state.record_and_count(now)

    injecting_fraud_pattern = random.random() < fraud_injection_rate
    if injecting_fraud_pattern:
        merchant_category = random.choice(["gambling", "crypto_exchange", "digital_goods"])
        merchant_country = random.choice([c for c in COUNTRIES if c != state.country])
        is_new_device = True
        amount = state.avg_amount_30d * random.uniform(4, 12)
    else:
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        merchant_country = state.country if random.random() > 0.1 else random.choice(COUNTRIES)
        is_new_device = random.random() < 0.05
        amount = max(1.0, float(np.random.lognormal(np.log(state.avg_amount_30d + 1), 0.7)))

    txn = Transaction(
        transaction_id=Transaction.new_id(),
        account_id=account_id,
        timestamp=Transaction.now_iso(),
        amount=round(amount, 2),
        merchant_category=merchant_category,
        merchant_country=merchant_country,
        account_country=state.country,
        device_id=f"dev_{random.randint(0, 50000)}",
        is_new_device=is_new_device,
        account_age_days=state.age_days,
        txn_count_last_1h=last_1h,
        txn_count_last_24h=last_24h,
        avg_amount_last_30d=round(state.avg_amount_30d, 2),
        is_fraud=None,
    )
    return txn.to_dict()


def run(producer: MessageProducer, n_accounts: int, n_events: int | None,
        events_per_sec: float, fraud_injection_rate: float, seed: int = 7) -> int:
    random.seed(seed)
    np.random.seed(seed)
    accounts = {f"acct_{i:06d}": AccountState() for i in range(n_accounts)}

    sent = 0
    delay = 1.0 / events_per_sec if events_per_sec > 0 else 0
    while n_events is None or sent < n_events:
        event = make_event(accounts, fraud_injection_rate)
        producer.send(TOPIC, key=event["account_id"], value=event)
        sent += 1
        if delay:
            time.sleep(delay)
    producer.flush()
    return sent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-servers", default="localhost:9092")
    ap.add_argument("--n-accounts", type=int, default=500)
    ap.add_argument("--n-events", type=int, default=None, help="omit to run forever")
    ap.add_argument("--events-per-sec", type=float, default=20.0)
    ap.add_argument("--fraud-injection-rate", type=float, default=0.02)
    args = ap.parse_args()

    producer = KafkaProducer(args.bootstrap_servers)
    total = run(producer, args.n_accounts, args.n_events, args.events_per_sec,
                args.fraud_injection_rate)
    print(f"Sent {total} events to topic '{TOPIC}'")
