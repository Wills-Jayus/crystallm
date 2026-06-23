# Joint Geometry Generator Report

Time: 2026-06-18T18:45:35Z

Data split: E7009/E7010 train-dev uses E318 train rendered candidates filtered by `blake2b(sample_id) % 5 == 0`; E700/E7008 evaluates val512. Test information used: no.

## Generator / Proposal Method

The evaluated generator is `pairfield_adam_repel`: a constrained SG/Wyckoff-preserving free-parameter plus isotropic-lattice optimizer. It uses only train CIF pair-distance and VPA statistics, then optimizes rendered candidate fractional geometry/lattice before StructureMatcher evaluation. It is not a selector-only reranker.

## Train-Dev Candidate Health

| metric | value |
|---|---:|
| generated train-dev candidates | 208 |
| train-dev samples | 26 |
| readable rate | 100.00% |
| composition valid rate | 100.00% |
| SG/Wyckoff valid rate | 95.19% |
| W/A-hit rate | 23.08% |
| StructureMatcher positive rate | 8.65% |
| rows>=7 positive rate | 8.65% |
| W/A-hit but match-fail rate | 85.42% |
| new positive samples beyond train-dev baseline | 0 |

## Train-Dev Match Metrics

| system | samples | match@1 | match@5 | match@20 | match@50 | positive-any | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E318 baseline same samples | 26 | 34.62% | 38.46% | 42.31% | 42.31% | 42.31% | 0.04879829767169925 |
| pairfield direct train-dev | 26 | 15.38% | 30.77% | 34.62% | 34.62% | 34.62% | 0.17328691056997114 |
| merged train-dev ceiling | 26 | NA | NA | NA | NA | 42.31% | NA |

## Val512 Candidate And Selector Metrics

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| E424 baseline val512 order | 33.40% | 41.30% | 46.05% | 46.05% | 0.16897342100460178 |
| E700 direct proposal aligned216 | 9.26% | 12.04% | 13.43% | 13.43% | 0.2346528518397651 |
| anchor-safe E424+E700 val512 | 33.40% | 41.30% | 46.05% | 46.05% | 0.16897342100460178 |
| anchor-safe rows>=7 | 13.57% | 17.19% | 19.00% | 19.00% | 0.1340250749335289 |

Val512 new-positive gate: aligned216 new positives beyond baseline = 2; full-val512 new positives beyond E424 top20 = 2; gate passed = True.

Conclusion: the same constrained joint free-param/lattice generator now has train-dev and val512 evidence. It does not add new positive samples beyond the E318 train-dev baseline on this held-out train subset, but it does add 2 rows>=7 new positive samples beyond baseline on val512. Direct ordering is weak and anchor-safe insertion remains conservative.
