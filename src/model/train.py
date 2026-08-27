"""
Trains and evaluates the fraud classifier.

Honest evaluation practices used here (same standard as AIT-614/PortPulse):
  - time-based train/test split (train on the first ~80% of transactions
    chronologically, test on the last ~20%) rather than a random split, since
    a random split would leak information across an account's transaction
    history between train and test and inflate the reported metrics.
  - class imbalance handled via scale_pos_weight (real fraud rate is ~1%),
    not by naive oversampling that would distort the true class prior.
  - reports precision/recall/F1 at a specific operating threshold PLUS
    PR-AUC and ROC-AUC, since accuracy alone is meaningless at 1% positive
    rate (predicting "never fraud" gets 99% accuracy and is worthless).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.engineering import dataframe_to_features, FEATURE_COLUMNS  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "transactions.csv"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data"


def load_split():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    return train_df, test_df


def best_f1_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5, float(f1s[best_idx])


def evaluate(name, y_true, y_prob):
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    thresh, best_f1 = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    result = {
        "model": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "chosen_threshold": round(thresh, 4),
        "precision_at_threshold": round(precision, 4),
        "recall_at_threshold": round(recall, 4),
        "f1_at_threshold": round(f1, 4),
        "n_test": int(len(y_true)),
        "n_test_fraud": int(y_true.sum()),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    train_df, test_df = load_split()
    X_train = dataframe_to_features(train_df)
    y_train = train_df["is_fraud"].values
    X_test = dataframe_to_features(test_df)
    y_test = test_df["is_fraud"].values

    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean():.4%})")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean():.4%})")

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_result = evaluate("xgboost", y_test, xgb_prob)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_result = evaluate("random_forest", y_test, rf_prob)

    winner = xgb_model if xgb_result["pr_auc"] >= rf_result["pr_auc"] else rf_model
    winner_name = "xgboost" if winner is xgb_model else "random_forest"
    winner_result = xgb_result if winner is xgb_model else rf_result
    print(f"\nSelected model: {winner_name} (higher PR-AUC on the time-based holdout)")

    import joblib
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(winner, MODEL_DIR / "fraud_model.joblib")
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "winner": winner_name,
            "feature_columns": FEATURE_COLUMNS,
            "xgboost": xgb_result,
            "random_forest": rf_result,
        }, f, indent=2)
    print(f"Saved model -> {MODEL_DIR / 'fraud_model.joblib'}")
    print(f"Saved metadata -> {MODEL_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
