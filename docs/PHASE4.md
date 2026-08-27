# Phase 4 — Autonomous Ops Agent

## What this phase built
- `src/agent/tools.py`: 6 tools with REAL side effects — `query_metrics` and `query_drift` read real numbers out of the monitoring store; `retrain_model` actually re-runs the training pipeline from Phase 1 and promotes the new artifact; `rollback_model` actually swaps back a saved previous model file; `page_human` and `no_action` are logged decisions with no further side effect (deliberately — not every anomaly should trigger an automated pipeline change).
- `src/agent/scenarios.py`: 4 injected failure scenarios, same evaluation philosophy as this portfolio's PatchLoop project — small, fully disclosed, one objectively-correct expected action each, INCLUDING a "nothing is actually wrong" negative control. Without that negative control, an agent that takes action on every input would score 100% on the positive scenarios for the wrong reason.
- `src/agent/llm_client.py`: two implementations —
  - `AnthropicLLMClient`: the real thing. Calls the actual Claude API with real tool-use. Requires your own `ANTHROPIC_API_KEY` as an environment variable — this project never asks for it any other way, and never logs or transmits it anywhere except the direct SDK call.
  - `ScriptedTestClient`: NOT an LLM — a deterministic, rule-based stand-in used only to verify the harness plumbing (tool dispatch, scenario setup/teardown, action logging) without needing an API key.

## A second real bug found and fixed
First harness run: `evaluate.py` scored 4/4 correct, but querying `agent_actions` afterward showed only 1 row instead of 4. Root cause: each scenario's `setup()` called `store.reset()`, which wipes the *entire* database — including the audit-log table for every previous scenario in the same run, not just the metrics being re-staged for the next one. Fixed by splitting `reset()` (full wipe, used by pytest fixtures needing total isolation) from a new `reset_scored_events()` (clears only the staged metrics, leaves the audit log alone) and switching scenario setup to the latter. Re-ran: all 4 actions now persist correctly.

## What's verified where — read this before citing a solve rate
Running `python -m src.agent.evaluate --client scripted` produces a 4/4 "solve rate" — **this is a harness/plumbing check, not a claim about agent intelligence.** The scripted client is simple threshold rules, not a language model. It proves the scenario setup, tool execution, and scoring logic all work correctly; it says nothing about whether an actual LLM can diagnose these situations well.

The real evaluation is `ANTHROPIC_API_KEY=sk-... python -m src.agent.evaluate --client anthropic`, run on your own machine with your own key. Whatever solve rate that produces is the number worth citing (and worth being honest about if it's not 4/4 — see this project's own PatchLoop precedent of disclosing a 60%, not-perfect, real result).
