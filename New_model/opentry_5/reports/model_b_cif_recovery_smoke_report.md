# Model B CIF Recovery Smoke Report

Time: 2026-06-19T03:10:30Z

Scope: grouped dev fold smoke only. This diagnostic uses GT W/A rows and GT free parameters from canonical grouped-dev labels, then applies the trained Model B lattice denoiser to recover lattice from synthetic corruption. It proves the denoiser checkpoint can be connected to OrbitEngine and StructureMatcher under fixed generation_index order. It does not prove formula+SG to W/A generation and is not a terminal opentry_5 success metric.

No test data and no val512 data were read.

Candidate order:

- generation_index 0: deterministic denoiser output.
- generation_index 1-4: fixed seed perturbations.
- No score, confidence, energy, collision, validity, oracle, or log-probability sorting is used.
- Invalid slots are retained.

| fold | samples | match@1 | match@5 | rows>=7 match@1 | readable@1 | composition exact@1 | atom count@1 | SG@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_a | 128 | 93.75% | 93.75% | 88.14% | 100.00% | 100.00% | 100.00% | 96.09% |
| fold_b | 128 | 96.09% | 96.09% | 100.00% | 100.00% | 100.00% | 100.00% | 99.22% |

Artifacts:

- `eval/model_b_cif_recovery_smoke/fold_a_metrics.json`
- `eval/model_b_cif_recovery_smoke/fold_a_generations.jsonl`
- `eval/model_b_cif_recovery_smoke/fold_b_metrics.json`
- `eval/model_b_cif_recovery_smoke/fold_b_generations.jsonl`

Interpretation:

The geometry/render/evaluator connection is now live on both grouped folds. The remaining bottleneck is not lattice recovery under GT W/A/free-param diagnostics; it is replacing GT W/A/free params with OOF/predicted W/A plus learned free-param/site-mapping generation while preserving fixed generation_index order.
