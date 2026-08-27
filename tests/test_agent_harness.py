"""
Verifies the agent evaluation harness plumbing using the ScriptedTestClient
(deterministic, not an LLM -- see src/agent/llm_client.py docstring). This
proves scenario setup, tool dispatch, and action logging all work; it does
NOT prove anything about real agent intelligence, which needs
ANTHROPIC_API_KEY and is out of scope for this automated test suite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.agent.llm_client import ScriptedTestClient
from src.agent.evaluate import run_all
from src.monitoring import store


@pytest.fixture(autouse=True)
def clean_store():
    store.reset()
    yield
    store.reset()


def test_harness_runs_all_scenarios_and_logs_actions():
    client = ScriptedTestClient()
    summary = run_all(client, verbose=False)
    assert summary["n_total"] == 4
    assert summary["solve_rate"] >= 0.0  # plumbing check, not an intelligence claim

    with store.connect() as conn:
        n_logged = conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0]
    assert n_logged == 4


def test_scripted_client_calls_query_tools_before_deciding():
    client = ScriptedTestClient()
    from src.agent.scenarios import setup_score_drift
    setup_score_drift()
    episode = client.run_episode("test scenario")
    tool_names_called_before_action = [t["tool"] for t in episode["tool_calls"][:-1]]
    assert "query_metrics" in tool_names_called_before_action
    assert "query_drift" in tool_names_called_before_action
