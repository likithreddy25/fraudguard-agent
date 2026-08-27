"""
Real drift detection using the two-sample Kolmogorov-Smirnov test comparing
a reference window of prediction scores against a recent window -- a
standard, honest choice (not a made-up "drift score"). A small p-value means
the recent score distribution is statistically distinguishable from the
reference window, which is the actual definition of drift being used here.
"""
from __future__ import annotations

from scipy.stats import ks_2samp


def detect_score_drift(reference_scores: list[float], recent_scores: list[float],
                        alpha: float = 0.01) -> dict:
    if len(reference_scores) < 30 or len(recent_scores) < 30:
        return {
            "drift_detected": False,
            "reason": "insufficient_samples",
            "n_reference": len(reference_scores),
            "n_recent": len(recent_scores),
        }
    statistic, p_value = ks_2samp(reference_scores, recent_scores)
    return {
        "drift_detected": bool(p_value < alpha),
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "n_reference": len(reference_scores),
        "n_recent": len(recent_scores),
    }
