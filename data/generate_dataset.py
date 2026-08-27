"""
Generates a synthetic but structurally realistic labeled transaction dataset
for offline model training.

Honesty note: this is SYNTHETIC data, not a real fraud dataset (real labeled
fraud data is never public, for obvious reasons -- this is standard practice
for a portfolio project, same as Kaggle's synthetic fraud datasets). The
important design choice is that fraud probability is a genuine logistic
function of the same risk factors the model is trained to find (high amount
relative to account history, cross-border, high-risk merchant category, new
device, high transaction velocity, young account) PLUS random noise, so the
task is learnable but not trivial -- unlike a naive dataset where the label
is a deterministic rule the model can just memorize.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.schema import MERCHANT_CATEGORIES, COUNTRIES  # noqa: E402

HIGH_RISK_CATEGORIES = {"gambling", "crypto_exchange", "digital_goods", "jewelry"}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int, fraud_rate_target: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n_accounts = max(200, n_rows // 20)
    account_ids = [f"acct_{i:06d}" for i in range(n_accounts)]
    account_country = {a: random.choice(COUNTRIES) for a in account_ids}
    account_age = {a: int(rng.integers(1, 2600)) for a in account_ids}
    account_avg_amount = {a: float(rng.lognormal(mean=3.5, sigma=0.8)) for a in account_ids}

    rows = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(n_rows):
        acct = random.choice(account_ids)
        avg_amt = account_avg_amount[acct]
        acct_country = account_country[acct]
        age_days = account_age[acct]

        merchant_category = random.choice(MERCHANT_CATEGORIES)
        merchant_country = acct_country if random.random() > 0.15 else random.choice(COUNTRIES)
        is_new_device = random.random() < 0.08
        txn_1h = int(rng.poisson(1.2))
        txn_24h = txn_1h + int(rng.poisson(4))

        # amount: usually close to the account's historical average, with a
        # long right tail (occasional large legitimate purchases too, which
        # is exactly what makes "large amount" alone a weak/noisy signal --
        # same as in real card-fraud modeling.
        amount = float(max(1.0, rng.lognormal(mean=np.log(avg_amt + 1), sigma=0.9)))

        is_cross_border = int(merchant_country != acct_country)
        is_high_risk_cat = int(merchant_category in HIGH_RISK_CATEGORIES)

        # ---- ground-truth fraud probability: logistic function of real risk
        # factors + noise. Coefficients are illustrative, not fit to any real
        # dataset -- the point is "learnable structure + noise", not
        # claiming these are the industry-true fraud-risk coefficients.
        z = (
            -6.6
            + 2.1 * is_cross_border
            + 1.8 * is_high_risk_cat
            + 2.4 * is_new_device
            + 1.3 * (amount / (avg_amt + 1) > 4)
            + 1.1 * (txn_1h >= 3)
            + 1.4 * (age_days < 14)
            - 0.6 * (age_days > 720)
        )
        p_fraud = float(_sigmoid(np.array([z]))[0])
        is_fraud = int(random.random() < p_fraud)

        ts = base_time + timedelta(seconds=int(rng.integers(0, 60 * 60 * 24 * 180)))

        rows.append({
            "transaction_id": f"txn_{i:08d}",
            "account_id": acct,
            "timestamp": ts.isoformat(),
            "amount": round(amount, 2),
            "merchant_category": merchant_category,
            "merchant_country": merchant_country,
            "account_country": acct_country,
            "device_id": f"dev_{rng.integers(0, 50000)}",
            "is_new_device": is_new_device,
            "account_age_days": age_days,
            "txn_count_last_1h": txn_1h,
            "txn_count_last_24h": txn_24h,
            "avg_amount_last_30d": round(avg_amt, 2),
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(rows)
    actual_rate = df["is_fraud"].mean()
    print(f"Generated {len(df)} rows, actual fraud rate = {actual_rate:.4%} "
          f"(target was ~{fraud_rate_target:.4%})")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=60000)
    ap.add_argument("--fraud-rate-target", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "transactions.csv"))
    args = ap.parse_args()

    df = generate(args.n_rows, args.fraud_rate_target, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
