# MP-20 K20 Rerank-Only OOF Search

Created: 2026-06-24T11:25:05+00:00

This search only reorders the original CrystaLLM GT-SG K20 candidates. The candidate set is unchanged, so match@20 should remain unchanged apart from label incompleteness diagnostics.

## Input

- Complete samples used: 9047
- Coverage complete: True
- Allow partial: False

## Baseline

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 72.731% | 29.135% |
| match@5 | 83.088% | 44.291% |
| match@20 | 87.631% | 55.571% |

## Models

| model | seed | match@1 | delta@1 | match@5 | delta@5 | match@20 | delta@20 | CI95 delta@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logistic | 0 | 72.079% | -0.652 pp | 82.381% | -0.707 pp | 87.631% | 0.000 pp | [-1.382, 0.044] pp |
| logistic | 1 | 72.079% | -0.652 pp | 82.381% | -0.707 pp | 87.631% | 0.000 pp | [-1.382, 0.099] pp |
| logistic | 2 | 72.079% | -0.652 pp | 82.381% | -0.707 pp | 87.631% | 0.000 pp | [-1.426, 0.067] pp |
| hgb | 0 | 73.439% | 0.707 pp | 82.967% | -0.122 pp | 87.631% | 0.000 pp | [0.044, 1.349] pp |
| hgb | 1 | 73.151% | 0.420 pp | 83.022% | -0.066 pp | 87.631% | 0.000 pp | [-0.265, 1.138] pp |
| hgb | 2 | 73.019% | 0.287 pp | 82.934% | -0.155 pp | 87.631% | 0.000 pp | [-0.376, 1.006] pp |

## Current Best

- Model: hgb seed=0
- match@1 delta: 0.707 pp
- match@5 delta: -0.122 pp
- match@20 delta: 0.000 pp

Formal freezing requires full label coverage and a selected hyperparameter set before official test generation.
