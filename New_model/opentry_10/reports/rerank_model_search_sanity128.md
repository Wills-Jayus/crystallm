# MP-20 K20 Rerank-Only OOF Search

Created: 2026-06-24T10:54:41+00:00

This search only reorders the original CrystaLLM GT-SG K20 candidates. The candidate set is unchanged, so match@20 should remain unchanged apart from label incompleteness diagnostics.

## Input

- Complete samples used: 128
- Coverage complete: False
- Allow partial: True

## Baseline

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 71.094% | 25.000% |
| match@5 | 82.812% | 55.000% |
| match@20 | 88.281% | 65.000% |

## Models

| model | seed | match@1 | delta@1 | match@5 | delta@5 | match@20 | delta@20 | CI95 delta@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logistic | 0 | 74.219% | 3.125 pp | 83.594% | 0.781 pp | 88.281% | 0.000 pp | [-1.973, 9.004] pp |
| hgb | 0 | 73.438% | 2.344 pp | 84.375% | 1.562 pp | 88.281% | 0.000 pp | [-2.344, 8.223] pp |

## Current Best

- Model: logistic seed=0
- match@1 delta: 3.125 pp
- match@5 delta: 0.781 pp
- match@20 delta: 0.000 pp

Formal freezing requires full label coverage and a selected hyperparameter set before official test generation.
