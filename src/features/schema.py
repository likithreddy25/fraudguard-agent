"""
Canonical event schema shared by:
  - the offline synthetic data generator (data/generate_dataset.py)
  - the real-time Kafka producer (src/producer/event_generator.py)
  - the feature engineering used at both training time and serving time
    (src/features/engineering.py)

Keeping ONE schema/one feature-engineering module used by both training and
serving is the actual point of this file: it eliminates train/serve skew,
which is the #1 real-world cause of "the model worked in the notebook but
not in production."
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    timestamp: str  # ISO-8601 UTC
    amount: float
    merchant_category: str
    merchant_country: str
    account_country: str
    device_id: str
    is_new_device: bool
    account_age_days: int
    txn_count_last_1h: int
    txn_count_last_24h: int
    avg_amount_last_30d: float
    # label is only present in the OFFLINE training dataset; the online
    # producer never emits it (that would be leaking the answer into
    # the stream the scorer is supposed to be scoring).
    is_fraud: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


MERCHANT_CATEGORIES = [
    "electronics", "grocery", "travel", "gambling", "digital_goods",
    "jewelry", "utilities", "restaurant", "clothing", "crypto_exchange",
]

COUNTRIES = ["US", "GB", "DE", "IN", "NG", "BR", "SG", "AE", "RU", "CN"]
