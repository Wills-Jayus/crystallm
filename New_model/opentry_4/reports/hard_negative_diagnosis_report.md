# Hard-Negative Diagnosis Report

Time: 2026-06-18T18:27:56Z

Data split: E318 train/E424 val512 candidate labels plus E423/E718 val512 per-sample top20/top50 reports. Test information used: no.

## Rows>=7 Top20/Top50 Diagnosis

| system | match@20 | match@50 | W/A@50 | missing top20 | missing top50 | missing top50 with W/A/skeleton hit | late selector cases match50 not20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E423 | 18.67% | 20.44% | 67.56% | 183 | 179 | 126 | 4 |
| E718 | 19.11% | 20.44% | 67.56% | 182 | 179 | 126 | 3 |

## Train/Val Hard-Negative Counts

| split | positives | hard negatives | rows>=7 positives | rows>=7 hard negatives |
|---|---:|---:|---:|---:|
| train | 8893 | 3011 | 310 | 755 |
| val512 | 1733 | 2086 | 127 | 1135 |

## Candidate Ceiling And Failure Mode

- val512 W/A-hit but match-fail rate: 64.54%
- val512 rows>=7 W/A-hit but match-fail rate: 91.49%
- val512 top20 candidate ceiling: 46.05%
- val512 rows>=7 top20 candidate ceiling: 19.00%
- rows>=7 failure buckets: `{"collision_or_short_distance": 0.3353819139596137, "free_param_or_site_mapping": 0.49692712906057945, "inter_row_or_source_distance": 0.0781387181738367, "lattice_volume_vpa": 0.08604038630377524, "sg_wyckoff_legality": 0.003511852502194908}`

## Bottleneck Answer

- Rows>=7 missing-top50 samples that still have W/A or skeleton hit@50 are geometry-conversion failures, not pure W/A recall failures.
- Rows>=7 `match@50 true but match@20 false` cases are selector-ordering failures; these are much fewer than missing-top50 geometry failures.
- Therefore match@20 should be raised primarily by generating new valid free-param/lattice candidates, with selector work limited to anchor-safe insertion after ceiling improves.

Detailed JSON: `eval/hard_negative_diagnosis_v2.json`.
