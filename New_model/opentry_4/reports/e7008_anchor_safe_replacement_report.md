# E7008 Anchor-Safe Replacement Report

Time: 2026-06-18T18:27:56Z

Data split: threshold frozen on E318 train-dev; candidate insertion evaluated on E424 val512 plus E700 proposal rows. Test information used: no.

## Frozen Selector

- Energy checkpoint: `checkpoints/geometry_energy_model_sklearn_e7006.joblib`
- Threshold: 0.81776514
- Anchor count: 4
- Max inserted candidates per W/A group: 2
- Top-k evaluated: 50

## Candidate Ceiling

| metric | value |
|---|---:|
| E700 generated candidates | 1721 |
| E700 samples with output | 216 |
| aligned216 new positives vs first pool | 2 |
| full-val512 new positives vs E424 top20 | 2 |
| anchor-safe inserted rows | 19 |
| anchor-safe inserted positive rows | 0 |
| merged full-val512 positive-any | 46.44% |
| merged rows>=7 positive-any | 19.91% |

## Match Metrics

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| E424 baseline val512 order | 33.40% | 41.30% | 46.05% | 46.05% | 0.16897342100460178 |
| anchor-safe E424+E700 val512 | 33.40% | 41.30% | 46.05% | 46.05% | 0.16897342100460178 |
| E424 rows>=7 baseline order | 13.57% | 17.19% | 19.00% | 19.00% | 0.1340250749335289 |
| anchor-safe rows>=7 | 13.57% | 17.19% | 19.00% | 19.00% | 0.1340250749335289 |

Anchor-safe insertion is the only selector-stage use of the new proposal pool. It preserves original top-4 anchors and only inserts candidates above a train-dev frozen energy threshold, so no val/test tuning is used.
