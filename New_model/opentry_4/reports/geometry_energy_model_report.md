# Geometry Energy Model Report

Time: 2026-06-18T18:27:56Z

Data split: train = E318 train hash-train split; train-dev = held-out E318 hash-dev split; validation = E424 val512. Test information used: no.

## Final Model

- Type: sklearn `DictVectorizer + StandardScaler(with_mean=False) + LogisticRegression`.
- Checkpoint: `checkpoints/geometry_energy_model_sklearn_e7006.joblib`
- Objective: rows>=7 and hard-negative weighted BCE surrogate via sample weights.
- Blocked feature classes: labels, sample/material ids, candidate hit flags, target W/A/skeleton labels, CIF text.
- Frozen insertion threshold: 0.81776514, selected from train-dev only as 90th percentile of W/A/skeleton hard-negative scores.

## Train-Dev Metrics

| metric | value |
|---|---:|
| AUC | 0.8873 |
| AP | 0.9568 |
| rows>=7 AUC | 0.8134 |
| rows>=7 AP | 0.6549 |
| pairwise accuracy | 0.7298387096774194 |
| rows>=7 pairwise accuracy | 0.7857142857142857 |
| dev precision at frozen threshold | 0.950207468879668 |
| dev rows>=7 precision at frozen threshold | 1.0 |

## Val512 Metrics

| metric | value |
|---|---:|
| AUC | 0.8665 |
| AP | 0.8522 |
| rows>=7 AUC | 0.7016 |
| rows>=7 AP | 0.2127 |
| pairwise accuracy | 0.592433361994841 |
| rows>=7 pairwise accuracy | 0.7114285714285714 |
| positive top-rank rate | 0.5259259259259259 |

## Match Impact Diagnostic

| order | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| baseline order on energy-eval subset | 48.51% | 56.47% | 57.96% | 57.96% | 0.16897342100460178 |
| energy rerank on energy-eval subset | 42.79% | 55.22% | 57.96% | 57.96% | 0.21835542481126627 |

The sklearn model replaces the earlier standard-library hashed logistic checkpoint as the final opentry_4 energy evidence. Full rerank remains diagnostic only; final selector use is the anchor-safe insertion in `e7008_anchor_safe_replacement_report.md`.
