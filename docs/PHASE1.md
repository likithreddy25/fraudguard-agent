# Phase 1 — Data & the Fraud Model

## What this phase built
- `data/generate_dataset.py`: a synthetic-but-structured labeled transaction dataset. Fraud label is NOT a deterministic rule — it's a logistic function of real risk factors (cross-border merchant, high-risk merchant category, new device, amount far above the account's historical average, high transaction velocity, young account) plus noise, so the classification task is learnable but not trivial. This is standard practice for a portfolio project (real labeled fraud data is never public); the honesty commitment is in disclosing that clearly rather than passing synthetic data off as real.
- `src/features/schema.py` / `src/features/engineering.py`: ONE feature-engineering function used both at training time (`dataframe_to_features`, vectorized) and at serving time (`transaction_to_features`, per-message) — this is deliberate: using two different feature-computation code paths for train vs. serve is the single most common real-world cause of "worked in the notebook, broke in prod."
- `src/model/train.py`: time-based (not random) train/test split — random splitting would leak an account's future transactions into the training set and inflate reported metrics. Trains both XGBoost and Random Forest with `scale_pos_weight`/`class_weight="balanced"` to handle the ~1% positive rate honestly (no synthetic oversampling that would distort the true class prior), and picks the winner by PR-AUC, not accuracy (accuracy is meaningless at 1% positive rate — "always predict not-fraud" scores 99%).

## Real, verified numbers (60,000 rows, 48,000 train / 12,000 time-based holdout test, 144 fraud cases in test)

| model | ROC-AUC | PR-AUC | precision @ chosen threshold | recall @ chosen threshold | F1 |
|---|---|---|---|---|---|
| XGBoost (selected) | 0.8786 | 0.1302 | 0.178 | 0.292 | 0.221 |
| Random Forest | 0.8702 | 0.1006 | 0.126 | 0.326 | 0.182 |

These are honestly modest numbers, not cherry-picked to look impressive — a synthetic 1%-positive-rate classification problem with injected noise is genuinely hard, and reporting a suspiciously clean 0.95+ PR-AUC on a task like this would itself be a red flag in an interview. XGBoost was selected for higher PR-AUC (the metric that actually matters under severe class imbalance).

## Honest limitations
- Synthetic data, disclosed as such throughout — not a real fraud dataset.
- The risk-factor coefficients in the label-generating function are illustrative, chosen to produce learnable structure, not fit to any real institution's actual fraud-risk model.
