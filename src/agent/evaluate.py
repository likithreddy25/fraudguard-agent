"""
Runs every scenario in src/agent/scenarios.py through a given LLM client and
reports an honest solve rate -- same format as PatchLoop's disclosed
5-instance batch: every scenario shown, pass/fail explicit, no averaging
across a hidden larger set.

Usage:
  # harness/plumbing check only -- NOT a claim about agent intelligence:
  python -m src.agent.evaluate --client scripted

  # the real evaluation -- requires your own ANTHROPIC_API_KEY:
  ANTHROPIC_API_KEY=sk-... python -m src.agent.evaluate --client anthropic
"""
from __future__ import annotations

import argparse
import json

from src.agent.llm_client import AnthropicLLMClient, ScriptedTestClient
from src.agent.scenarios import SCENARIOS
from src.monitoring import store


def run_all(client, verbose: bool = True) -> dict:
    results = []
    for scenario in SCENARIOS:
        scenario.setup()
        episode = client.run_episode(scenario.description)
        correct = episode["final_action"] in scenario.expected_actions
        store.record_agent_action(
            scenario_id=scenario.id, action=episode["final_action"] or "NONE",
            reasoning=episode["reasoning"], tool_calls=episode["tool_calls"],
        )
        results.append({
            "scenario_id": scenario.id,
            "description": scenario.description,
            "expected_actions": scenario.expected_actions,
            "actual_action": episode["final_action"],
            "correct": correct,
            "n_tool_calls": len(episode["tool_calls"]),
        })
        if verbose:
            status = "PASS" if correct else "FAIL"
            print(f"[{status}] {scenario.id}: expected one of {scenario.expected_actions}, "
                  f"got '{episode['final_action']}'")

    n_correct = sum(r["correct"] for r in results)
    summary = {
        "solve_rate": round(n_correct / len(results), 4),
        "n_correct": n_correct,
        "n_total": len(results),
        "results": results,
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=["scripted", "anthropic"], default="scripted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.client == "scripted":
        print("NOTE: --client scripted uses a deterministic rule-based stand-in, "
              "NOT a real LLM. This checks the harness plumbing only. It is not a "
              "claim about agent intelligence.\n")
        client = ScriptedTestClient()
    else:
        client = AnthropicLLMClient()

    summary = run_all(client)
    print(f"\nSolve rate: {summary['n_correct']}/{summary['n_total']} "
          f"({summary['solve_rate']:.0%}) [client={args.client}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.out}")
