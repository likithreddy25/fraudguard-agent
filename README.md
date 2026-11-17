# FraudGuard-Agent

![tests](https://github.com/likithreddy25/fraudguard-agent/actions/workflows/tests.yml/badge.svg)

A real-time fraud-detection pipeline (Kafka + a trained ML model) operated by an autonomous LLM agent that watches the pipeline's live health/drift metrics and decides what to do about it — retrain, roll back, page a human, or correctly do nothing.

This project exists to demonstrate three things together, honestly: real Kafka streaming, a real trained-and-evaluated ML pipeline, and a real agentic tool-use system — not three separate demos glued together, but one pipeline where each layer's output feeds the next.

## Architecture

```
event_generator.py --> [Kafka: transactions] --> scorer.py --> [Kafka: scored_transactions]
                                                       |
                                                       v
                                              monitoring/store.py (SQLite)
                                                       |
                                                       v
                                              agent/ops_agent (Claude + tools)
                                        query_metrics / query_drift / retrain_model /
                                        rollback_model / page_human / no_action
```

## What's real vs. what needs your own machine

| Component | Status |
|---|---|
| Synthetic dataset generation, feature engineering, XGBoost/RF training & evaluation | **Fully built and verified here** — real numbers in [docs/PHASE1.md](docs/PHASE1.md) |
| Kafka producer/consumer code (real `confluent-kafka` client) | **Fully built**; verified end-to-end via an in-memory transport (no broker available in the sandbox this was first built in — see [docs/PHASE2.md](docs/PHASE2.md)). Run `docker compose up -d` on your own machine for the real broker. |
| Monitoring store + KS-test drift detection | **Fully built and verified here** — [docs/PHASE3.md](docs/PHASE3.md), including a real SQLite bug found and fixed |
| Autonomous agent (tools, scenarios, harness) | **Fully built**; harness plumbing verified with a non-LLM scripted stand-in. The real LLM-driven evaluation needs your own `ANTHROPIC_API_KEY` — see [docs/PHASE4.md](docs/PHASE4.md), including a second real bug found and fixed |

Two real bugs were found and fixed during development (a SQLite disk-I/O issue and an audit-log-wiping issue) — both documented in the phase docs rather than quietly fixed and forgotten.

## Explainability and fairness (added 2026-10)

- `src/explain/reason_codes.py`: exact TreeSHAP values from XGBoost (`pred_contribs`), plain-language reason codes per flagged transaction, and global importance. Additivity (contributions + bias = raw score) is tested.
- `src/fairness/audit.py`: disparate impact, equal-opportunity (TPR) gap, FPR gap, and a threshold-vs-fairness trade-off table, for a **synthetic** A/B group attribute that is a deliberate proxy for account age (never a model input).
- `python -m src.fairness.run_audit` writes [docs/EXPLAIN_FAIRNESS_RESULTS.md](docs/EXPLAIN_FAIRNESS_RESULTS.md). Finding on this synthetic setup: the model never sees the group, yet flags group B 1.48x as often (FPR +0.7 pts) because account age leaks it. The TPR gap is small and noisy (72 frauds per group). See [docs/PHASE5.md](docs/PHASE5.md).
- Built with Claude's assistance, like the rest of this repo; all numbers come from running the code on the synthetic dataset.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data + train the model (fully self-contained, no Kafka needed)
python data/generate_dataset.py
python -m src.model.train

# 2. Run the test suite (uses an in-memory transport, no broker needed)
pytest tests/ -v

# 3. For the real Kafka integration (needs Docker):
docker compose up -d
python -m src.producer.event_generator --events-per-sec 20 &
python -m src.consumer.scorer

# 4. Agent harness check (no API key needed, NOT a real intelligence eval):
python -m src.agent.evaluate --client scripted

# 5. The real agent evaluation (needs your own Anthropic API key):
ANTHROPIC_API_KEY=sk-... python -m src.agent.evaluate --client anthropic
```

## Repo layout
```
data/               synthetic dataset, generation script, trained model artifact
src/features/       schema + feature engineering shared by training and serving
src/model/          training + evaluation
src/producer/       Kafka event generator
src/consumer/       real-time scorer
src/monitoring/      SQLite metrics store + KS-test drift detection
src/agent/          tools, injected scenarios, LLM client, evaluation harness
src/transport.py     Kafka / in-memory transport abstraction (see docs/PHASE2.md)
tests/               pytest suite, all passing against the in-memory transport
docs/PHASE1-4.md     honest, numbered write-up of each phase
docker-compose.yml   real single-node Apache Kafka (KRaft mode)
```
