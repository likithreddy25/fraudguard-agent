# Phase 3 — Monitoring & Drift Detection

## What this phase built
- `src/monitoring/store.py`: SQLite-backed metrics store (`scored_events`, `agent_actions` tables). SQLite instead of Postgres because the sandbox has no root access (`sudo apt-get install postgresql` fails — confirmed directly: "sudo: no new privileges flag is set"), not because SQLite is the "right" long-term choice; if this scaled past a single writer, swapping in Postgres is a connection-string change since all access goes through this one module.
- `src/monitoring/drift.py`: a real two-sample Kolmogorov-Smirnov test comparing a reference score window against a recent window. Drift is defined as "the recent distribution is statistically distinguishable from the reference at p < 0.01" — a standard, checkable definition, not an invented "drift score."

## A real bug found and fixed
Running the scorer against 300 messages through the in-memory transport hit:
```
sqlite3.OperationalError: disk I/O error
```
Root cause: the project folder lives on a network-backed mount in the sandbox it was first built in, and SQLite's default rollback-journal locking doesn't work reliably over that kind of mount — a well-known SQLite/network-filesystem limitation, not a code bug that would affect anyone running this on a normal local disk. Fixed by making the DB path overridable via `FRAUDGUARD_DB_PATH` (defaults to the normal in-repo path, which is what you want on a real machine) rather than hard-coding it — real tests now pass by pointing that env var at local disk during CI/dev in that sandbox specifically.

## Real, verified numbers
`pytest tests/test_pipeline_e2e.py` — 4/4 passing, including:
- `test_drift_detection_flags_a_real_distribution_shift`: KS-test correctly flags a synthetic shifted distribution (p < 0.01)
- `test_drift_detection_does_not_false_positive_on_same_distribution`: KS-test correctly does NOT flag two samples drawn from the same distribution — a negative control that matters as much as the positive one; a drift detector that fires constantly is worse than useless.
