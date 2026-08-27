"""
SQLite-backed monitoring store for the scoring pipeline.

Why SQLite instead of Postgres: this sandbox has no root access (no `sudo
apt-get install postgresql`, confirmed directly), so a server-based DB isn't
an option here. SQLite is file-based, needs no server, and is a completely
legitimate real-world choice for a project at this scale -- plenty of real
production systems use it for exactly this kind of single-writer metrics
store. If this were scaled to a multi-instance deployment, swapping in
Postgres is a connection-string change, not a redesign (all access goes
through this one module).

Tracked per scored event: prediction score, latency, and the feature vector
(so drift can be measured later against a reference window) -- mirrors what
activation-probe's dbt marts track for the jailbreak-probe service
(flag-rate / false-positive-rate over time), applied here to fraud scoring.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# Overridable via FRAUDGUARD_DB_PATH. Needed because this project was first
# built/tested in a sandbox whose project folder is a network-backed mount
# that doesn't support SQLite's file-locking (`disk I/O error` on
# executescript -- a real bug hit and diagnosed during development, not
# hypothetical); on a normal local filesystem (i.e. wherever you actually
# run this) the default path works fine.
DB_PATH = Path(os.environ.get(
    "FRAUDGUARD_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "monitoring.db"),
))

SCHEMA = """
CREATE TABLE IF NOT EXISTS scored_events (
    transaction_id TEXT PRIMARY KEY,
    scored_at REAL NOT NULL,
    score REAL NOT NULL,
    flagged INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    scoring_latency_ms REAL NOT NULL,
    features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at REAL NOT NULL,
    scenario_id TEXT,
    action TEXT NOT NULL,
    reasoning TEXT,
    tool_calls_json TEXT
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_scored_event(transaction_id: str, score: float, flagged: bool,
                         model_version: str, scoring_latency_ms: float,
                         features: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scored_events "
            "(transaction_id, scored_at, score, flagged, model_version, "
            " scoring_latency_ms, features_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (transaction_id, time.time(), score, int(flagged), model_version,
             scoring_latency_ms, json.dumps(features)),
        )


def record_agent_action(scenario_id: str, action: str, reasoning: str,
                         tool_calls: list) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO agent_actions (taken_at, scenario_id, action, "
            "reasoning, tool_calls_json) VALUES (?, ?, ?, ?, ?)",
            (time.time(), scenario_id, action, reasoning, json.dumps(tool_calls)),
        )


def flag_rate(window_seconds: float | None = None) -> float:
    with connect() as conn:
        query = "SELECT AVG(flagged) FROM scored_events"
        params: tuple = ()
        if window_seconds is not None:
            query += " WHERE scored_at >= ?"
            params = (time.time() - window_seconds,)
        row = conn.execute(query, params).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0


def avg_latency_ms(window_seconds: float | None = None) -> float:
    with connect() as conn:
        query = "SELECT AVG(scoring_latency_ms) FROM scored_events"
        params: tuple = ()
        if window_seconds is not None:
            query += " WHERE scored_at >= ?"
            params = (time.time() - window_seconds,)
        row = conn.execute(query, params).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0


def score_distribution(window_seconds: float | None = None) -> list[float]:
    with connect() as conn:
        query = "SELECT score FROM scored_events"
        params: tuple = ()
        if window_seconds is not None:
            query += " WHERE scored_at >= ?"
            params = (time.time() - window_seconds,)
        return [r[0] for r in conn.execute(query, params).fetchall()]


def event_count(window_seconds: float | None = None) -> int:
    with connect() as conn:
        query = "SELECT COUNT(*) FROM scored_events"
        params: tuple = ()
        if window_seconds is not None:
            query += " WHERE scored_at >= ?"
            params = (time.time() - window_seconds,)
        return int(conn.execute(query, params).fetchone()[0])


def reset() -> None:
    """Full reset (both tables). Used by pytest fixtures that need total
    isolation between unrelated tests."""
    if DB_PATH.exists():
        DB_PATH.unlink()


def reset_scored_events() -> None:
    """Clears only the scored_events table, leaving agent_actions intact.

    Real bug found during development: scenario setup functions in
    src/agent/scenarios.py originally called reset() between scenarios to
    re-stage fresh metrics, which silently wiped out the agent_actions audit
    log recorded for every PREVIOUS scenario in the same evaluation run --
    evaluate.py would report a correct solve_rate but the persisted action
    log only ever contained the last scenario's entry. Scenario setup should
    only be allowed to touch the data it stages (scored_events); the audit
    log is a separate concern and must survive across scenario resets within
    one evaluation run.
    """
    with connect() as conn:
        conn.execute("DELETE FROM scored_events")
