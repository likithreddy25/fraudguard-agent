"""
Injected failure scenarios the agent is evaluated against, same evaluation
philosophy as PatchLoop's disclosed 5-instance SWE-bench batch: a small,
honestly-disclosed set, each with a single objectively-correct expected
action, including a "nothing is actually wrong" negative control (without
one, an agent that takes action on every input would score 100% on the
positive scenarios for the wrong reason).

Each scenario seeds the real monitoring store (src/monitoring/store.py) with
crafted synthetic events so the agent's tool calls (query_metrics,
query_drift) return real numbers computed from real data -- the scenario
setup is synthetic, but nothing about how the agent reads or reasons about
it is faked.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.monitoring import store


@dataclass
class Scenario:
    id: str
    description: str
    expected_actions: list[str]  # any one of these counts as correct
    setup: "callable"


def _seed_events(n: int, score_fn, latency_fn, model_version: str = "xgboost",
                  time_offset_start: float = -7200, time_offset_end: float = 0):
    now = time.time()
    for i in range(n):
        t = now + random.uniform(time_offset_start, time_offset_end)
        score = score_fn()
        flagged = score >= 0.5
        # Bypass record_scored_event's use of time.time() by writing
        # directly so we can control the timestamp for window-based queries.
        import sqlite3
        from src.monitoring.store import connect
        with connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scored_events "
                "(transaction_id, scored_at, score, flagged, model_version, "
                " scoring_latency_ms, features_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"seed_{i}_{random.randint(0, 10**9)}", t, score, int(flagged),
                 model_version, latency_fn(), "{}"),
            )


def setup_healthy_baseline():
    store.reset_scored_events()
    random.seed(1)
    _seed_events(500, lambda: max(0.0, min(1.0, random.gauss(0.05, 0.03))),
                 lambda: random.gauss(8, 2))


def setup_score_drift():
    store.reset_scored_events()
    random.seed(2)
    # reference window: normal, low scores
    _seed_events(400, lambda: max(0.0, min(1.0, random.gauss(0.05, 0.03))),
                 lambda: random.gauss(8, 2), time_offset_start=-7200, time_offset_end=-1800)
    # recent window: scores have shifted noticeably higher (real drift)
    _seed_events(150, lambda: max(0.0, min(1.0, random.gauss(0.35, 0.08))),
                 lambda: random.gauss(8, 2), time_offset_start=-1800, time_offset_end=0)


def setup_latency_spike():
    store.reset_scored_events()
    random.seed(3)
    _seed_events(400, lambda: max(0.0, min(1.0, random.gauss(0.05, 0.03))),
                 lambda: random.gauss(8, 2), time_offset_start=-7200, time_offset_end=-1800)
    # recent window: score distribution UNCHANGED, but latency has exploded
    # -- this is an infra problem, not a model problem. A correct agent
    # should NOT retrain (retraining does nothing for latency) and should
    # page a human instead.
    _seed_events(150, lambda: max(0.0, min(1.0, random.gauss(0.05, 0.03))),
                 lambda: random.gauss(650, 120), time_offset_start=-1800, time_offset_end=0)


def setup_silent_model_degradation():
    store.reset_scored_events()
    random.seed(4)
    _seed_events(400, lambda: max(0.0, min(1.0, random.gauss(0.05, 0.03))),
                 lambda: random.gauss(8, 2), time_offset_start=-7200, time_offset_end=-1800)
    # recent window: flag rate has collapsed to ~0 -- looks "healthy" on a
    # naive flag-rate-only check, but combined with a real distribution
    # shift toward 0, it's actually a broken model (e.g. serving stale/
    # constant scores). Tests whether the agent digs into query_drift
    # rather than stopping at a shallow "flag rate is low, must be fine".
    _seed_events(150, lambda: 0.01, lambda: random.gauss(8, 2),
                 time_offset_start=-1800, time_offset_end=0)


SCENARIOS = [
    Scenario(
        id="healthy_baseline",
        description="Pipeline operating normally -- flag rate and latency both within normal historical range.",
        expected_actions=["no_action"],
        setup=setup_healthy_baseline,
    ),
    Scenario(
        id="score_drift",
        description="Recent prediction score distribution has shifted meaningfully higher than the historical reference window.",
        expected_actions=["retrain_model"],
        setup=setup_score_drift,
    ),
    Scenario(
        id="latency_spike",
        description="Scoring latency has spiked ~80x while the score distribution itself is unchanged -- an infra problem, not a model problem.",
        expected_actions=["page_human"],
        setup=setup_latency_spike,
    ),
    Scenario(
        id="silent_model_degradation",
        description="Flag rate has collapsed toward zero AND the score distribution has collapsed toward a near-constant low value -- looks quiet, is actually broken.",
        expected_actions=["retrain_model", "rollback_model"],
        setup=setup_silent_model_degradation,
    ),
]
