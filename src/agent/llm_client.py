"""
LLM client abstraction for the ops agent's tool-use loop.

Two implementations, and it matters which one produced a given result:

  - AnthropicLLMClient: the REAL thing -- calls the actual Claude API with
    real tool-use, real model reasoning. Requires ANTHROPIC_API_KEY as an
    environment variable on whatever machine runs it. This project does
    NOT read, store, or transmit that key anywhere except the direct
    Anthropic SDK call itself, and it is never hard-coded or logged.
  - ScriptedTestClient: a deterministic, rule-based stand-in with NO model
    reasoning at all -- it exists only so tests/test_agent_harness.py can
    verify the tool-execution plumbing (tool dispatch, action logging,
    scenario setup/teardown) without needing an API key. Any solve-rate
    number produced using this client is NOT a claim about agent
    intelligence and must never be reported as one -- see
    docs/PHASE4.md and evaluate.py's --client flag.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from src.agent.tools import TOOL_SPECS, TOOL_FUNCTIONS

SYSTEM_PROMPT = """You are an autonomous operations agent responsible for a real-time \
fraud-detection scoring pipeline. You have tools to inspect its live metrics and drift \
status, and to take one of these remediation actions: retrain_model, rollback_model, \
page_human, or no_action.

Investigate using query_metrics and query_drift BEFORE deciding on an action. Take \
exactly one final remediation action (retrain_model, rollback_model, page_human, or \
no_action) and briefly explain why. Do not take a remediation action without having \
called at least one query tool first."""


class LLMClient(ABC):
    @abstractmethod
    def run_episode(self, scenario_description: str, max_turns: int = 6) -> dict:
        """Runs the full tool-use loop for one scenario. Returns:
        {"final_action": str, "reasoning": str, "tool_calls": [...]}"""
        ...


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        import anthropic  # deferred import: optional dep
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it in your own environment "
                "before running the real agent evaluation -- this project never "
                "asks you to paste it anywhere else."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def run_episode(self, scenario_description: str, max_turns: int = 6) -> dict:
        messages = [{"role": "user", "content": scenario_description}]
        tool_calls_log = []
        final_action, reasoning = None, ""

        for _ in range(max_turns):
            response = self._client.messages.create(
                model=self._model, max_tokens=1024, system=SYSTEM_PROMPT,
                tools=TOOL_SPECS, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                text = "".join(b.text for b in response.content if b.type == "text")
                reasoning = text
                break

            tool_results = []
            for block in tool_use_blocks:
                fn = TOOL_FUNCTIONS[block.name]
                result = fn(**block.input)
                tool_calls_log.append({"tool": block.name, "input": block.input, "result": result})
                if block.name in ("retrain_model", "rollback_model", "page_human", "no_action"):
                    final_action = block.name
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})

            if final_action is not None:
                break

        return {"final_action": final_action, "reasoning": reasoning, "tool_calls": tool_calls_log}


class ScriptedTestClient(LLMClient):
    """NOT an LLM. Deterministic stand-in used only to test the harness
    plumbing (see module docstring). Uses simple threshold rules on the
    real tool outputs -- this is intentionally dumber than a real agent so
    nobody mistakes its solve rate for a claim about model intelligence."""

    def run_episode(self, scenario_description: str, max_turns: int = 6) -> dict:
        tool_calls_log = []
        metrics = TOOL_FUNCTIONS["query_metrics"]()
        tool_calls_log.append({"tool": "query_metrics", "input": {}, "result": metrics})
        drift = TOOL_FUNCTIONS["query_drift"]()
        tool_calls_log.append({"tool": "query_drift", "input": {}, "result": drift})

        if metrics["avg_latency_ms"] > 100:
            action, args = "page_human", {"message": "Latency far above baseline; likely infra, not model."}
        elif drift.get("drift_detected"):
            action, args = "retrain_model", {}
        else:
            action, args = "no_action", {"reasoning": "Metrics and drift both within normal range."}

        result = TOOL_FUNCTIONS[action](**args)
        tool_calls_log.append({"tool": action, "input": args, "result": result})
        return {"final_action": action, "reasoning": "scripted rule-based decision (not an LLM)",
                "tool_calls": tool_calls_log}
