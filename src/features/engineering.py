"""
Feature engineering shared by training (data/generate_dataset.py + model/train.py)
and real-time serving (src/consumer/scorer.py). Same function, same output
columns, same order -- this is what prevents train/serve skew.
"""
from __future__ import annotations

import pandas as pd
from .schema import MERCHANT_CATEGORIES, COUNTRIES

# High-risk merchant categories chosen deliberately (gambling, crypto, digital
# goods are the categories real card-network fraud reports consistently flag
# as elevated-risk -- this isn't arbitrary).
HIGH_RISK_CATEGORIES = {"gambling", "crypto_exchange", "digital_goods", "jewelry"}

FEATURE_COLUMNS = [
    "amount",
    "amount_to_avg_ratio",
    "is_cross_border",
    "is_high_risk_category",
    "is_new_device",
    "account_age_days",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "velocity_ratio_1h_24h",
    "merchant_category_idx",
    "merchant_country_idx",
]

_CAT_IDX = {c: i for i, c in enumerate(MERCHANT_CATEGORIES)}
_COUNTRY_IDX = {c: i for i, c in enumerate(COUNTRIES)}


def transaction_to_features(txn: dict) -> dict:
    """Turn one raw transaction dict into a flat feature dict.

    Works on a single dict so the exact same function runs per-message in the
    real-time consumer and per-row (via DataFrame.apply) at training time.
    """
    avg = txn["avg_amount_last_30d"] or 1.0  # avoid div-by-zero on new accounts
    amount_to_avg_ratio = txn["amount"] / avg if avg > 0 else 0.0

    txn_1h = txn["txn_count_last_1h"]
    txn_24h = max(txn["txn_count_last_24h"], 1)
    velocity_ratio_1h_24h = txn_1h / txn_24h

    return {
        "amount": txn["amount"],
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "is_cross_border": int(txn["merchant_country"] != txn["account_country"]),
        "is_high_risk_category": int(txn["merchant_category"] in HIGH_RISK_CATEGORIES),
        "is_new_device": int(txn["is_new_device"]),
        "account_age_days": txn["account_age_days"],
        "txn_count_last_1h": txn_1h,
        "txn_count_last_24h": txn_24h,
        "velocity_ratio_1h_24h": velocity_ratio_1h_24h,
        "merchant_category_idx": _CAT_IDX.get(txn["merchant_category"], -1),
        "merchant_country_idx": _COUNTRY_IDX.get(txn["merchant_country"], -1),
    }


def dataframe_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version used at training time on the full dataset."""
    records = [transaction_to_features(row) for row in df.to_dict("records")]
    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
