"""
Real tools the ops agent can call. These have genuine side effects against
the actual monitoring store and the actual model artifact -- not stubs that
just print a message.

  - query_metrics: reads real numbers out of src/monitoring/store.py
  - query_drift: runs the real KS-test drift check (src/monitoring/drift.py)
    comparing a reference window against the most recent window
  - retrain_model: actually re-runs training (src/model/train.py's pipeline)
    and promotes the new model to data/fraud_model.joblib
  - rollback_model: actually swaps back to the previously saved model
    artifact (data/fraud_model.prev.joblib), a real file operation
  - page_human / no_action: logged, no further side effect (correctly
    modeling that not every anomaly should trigger an automated pipeline
    change -- see docs/PHASE4.md on why the "do nothing" scenario matters
    as much as the "do something" ones)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from src.monitoring import store
from src.monitoring.drift import detect_score_drift

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "data" / "fraud_model.joblib"
PREV_MODEL_PATH = REPO_ROOT / "data" / "fraud_model.prev.joblib"


def query_metrics(window_seconds: float = 3600) -> dict:
    return {
        "flag_rate": round(store.flag_rate(window_seconds), 4),
        "avg_latency_ms": round(store.avg_latency_ms(window_seconds), 2),
        "event_count": store.event_count(window_seconds),
        "window_seconds": window_seconds,
    }


def query_drift(reference_window_seconds: float = 86400,
                 recent_window_seconds: float = 3600) -> dict:
    all_scores = store.score_distribution(reference_window_seconds)
    recent_scores = store.score_distribution(recent_window_seconds)
    # reference = everything OUTSIDE the recent window, so the two samples
    # don't overlap (otherwise the KS-test comparison is meaningless).
    reference_scores = all_scores[:max(0, len(all_scores) - len(recent_scores))]
    return detect_score_drift(reference_scores, recent_scores)


def retrain_model() -> dict:
    """Actually re-runs the real training pipeline (src/model/train.py) and
    backs up the previous model first so rollback_model has something real
    to roll back to."""
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, PREV_MODEL_PATH)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "model" / "train.py")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    return {
        "action": "retrain_model",
        "success": result.returncode == 0,
        "stdout_tail": result.stdout[-800:],
        "stderr_tail": result.stderr[-400:] if result.returncode != 0 else "",
    }


def rollback_model() -> dict:
    if not PREV_MODEL_PATH.exists():
        return {"action": "rollback_model", "success": False,
                 "reason": "no previous model artifact available"}
    shutil.copy(PREV_MODEL_PATH, MODEL_PATH)
    return {"action": "rollback_model", "success": True}


def page_human(message: str) -> dict:
    return {"action": "page_human", "message": message, "success": True}


def no_action(reasoning: str) -> dict:
    return {"action": "no_action", "reasoning": reasoning, "success": True}


TOOL_SPECS = [
    {
        "name": "query_metrics",
        "description": "Get current flag rate, average scoring latency, and event count over a time window.",
        "input_schema": {
            "type": "object",
            "properties": {"window_seconds": {"type": "number", "description": "Lookback window in seconds"}},
        },
    },
    {
        "name": "query_drift",
        "description": "Run a statistical (KS-test) check for whether the recent score distribution has drifted from the historical reference window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_window_seconds": {"type": "number"},
                "recent_window_seconds": {"type": "number"},
            },
        },
    },
    {
        "name": "retrain_model",
        "description": "Retrain the fraud model on the current training dataset and promote it to production. Use when the model's predictive quality itself appears to be the problem (e.g. real distribution drift). Has a real cost (takes ~seconds to minutes) so don't call it for issues unrelated to model quality.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "rollback_model",
        "description": "Roll back to the previously deployed model artifact. Use when a recent deploy/retrain appears to have made things worse and there IS a previous version to restore.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "page_human",
        "description": "Escalate to a human on-call engineer. Use for issues that are not a model-quality problem (e.g. infrastructure/latency issues) or that you're not confident enough to resolve autonomously.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "no_action",
        "description": "Explicitly decide that nothing is wrong and no action is needed. Use this rather than staying silent when metrics look healthy -- an agent that never says 'everything is fine' can't be trusted when it says something is wrong.",
        "input_schema": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}},
            "required": ["reasoning"],
        },
    },
]

TOOL_FUNCTIONS = {
    "query_metrics": query_metrics,
    "query_drift": query_drift,
    "retrain_model": retrain_model,
    "rollback_model": rollback_model,
    "page_human": page_human,
    "no_action": no_action,
}
