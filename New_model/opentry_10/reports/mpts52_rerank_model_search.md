# MPTS-52 K20 Rerank-Only OOF Search

Created: 2026-06-25T03:49:59+00:00

This search only reorders the original CrystaLLM GT-SG K20 candidates. The candidate set is unchanged, so match@20 should remain unchanged apart from label incompleteness diagnostics.

## Input

- Complete samples used: 5000
- Coverage complete: True
- Allow partial: False

## Baseline

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 30.020% | 5.323% |
| match@5 | 40.480% | 9.991% |
| match@20 | 48.000% | 14.747% |

## Models

| model | seed | match@1 | delta@1 | match@5 | delta@5 | match@20 | delta@20 | CI95 delta@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logistic | 0 | 29.080% | -0.940 pp | 40.440% | -0.040 pp | 48.000% | 0.000 pp | NA |
| logistic | 1 | 29.080% | -0.940 pp | 40.440% | -0.040 pp | 48.000% | 0.000 pp | NA |
| logistic | 2 | 29.080% | -0.940 pp | 40.440% | -0.040 pp | 48.000% | 0.000 pp | NA |
| hgb | 0 | 30.940% | 0.920 pp | 41.100% | 0.620 pp | 48.000% | 0.000 pp | NA |
| hgb | 1 | 30.980% | 0.960 pp | 41.240% | 0.760 pp | 48.000% | 0.000 pp | NA |
| hgb | 2 | 31.240% | 1.220 pp | 41.160% | 0.680 pp | 48.000% | 0.000 pp | NA |

## Current Best

- Model: hgb seed=2
- match@1 delta: 1.220 pp
- match@5 delta: 0.680 pp
- match@20 delta: 0.000 pp

Formal freezing requires full label coverage and a selected hyperparameter set before official test generation.
