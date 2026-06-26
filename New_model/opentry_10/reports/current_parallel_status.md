# Current Parallel Status

Timestamp: 2026-06-24 14:15 UTC

## MPTS-52 Line

MPTS-52 validation K100 shard generation is running:

- completed shards: 33 / 79;
- active workers: shard0032 on GPU0, shard0033 on GPU1;
- pending shards in plan: 45, plus shard0032 is active but temporarily shown as pending because the old serial controller overwrote its running marker before exiting;
- GPU status at last check: both A800 GPUs active at 100%.

Acceleration change:

- Added `scripts/run_mpts52_parallel_shard_worker.py`.
- Started two per-GPU workers with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.
- Each worker uses per-shard `.shard.lock` and a global plan lock to avoid duplicate shard writes.
- The old serial controller exited after detecting shard0032 was locked, which is expected.

This line should continue until all MPTS-52 validation shards are complete, then assemble/export/evaluate the MPTS-52 validation anchor.

## MP-20 Line

MP-20 validation K100 generation, export, K50 labeling, and first selector analyses are complete.

Completed MP-20 artifacts:

- `candidates/crystallm_gt_sg_mp20_val_k100.jsonl`
- `metrics/crystallm_gt_sg_mp20_val.json`
- `labels/mp20_val_k50_candidate_labels.jsonl`
- `metrics/mp20_val_k50_candidate_label_metrics.json`
- `metrics/rerank_oof_results.json`
- `metrics/mp20_k50_conservative_selector_oof.json`
- `metrics/mp20_k50_anchor_keep_sweep_from_hgb_scores.json`
- `metrics/mp20_k50_residual_route_sweep_from_hgb_scores.json`
- `metrics/mp20_k50_residual_classifier_oof.json`

Key MP-20 conclusion:

- K50 oracle headroom exists: match@20 improves from 0.8763125898 to 0.8991931027.
- rows>=7 headroom is larger: rows>=7 match@20 improves from 0.5557093426 to 0.6207612457.
- rerank-only best result is HGB seed 0: match@1 +0.707pp, match@5 -0.122pp, match@20 unchanged.
- conservative K50 selector best result is HGB seed 0, anchor_keep=14: match@20 +0.343pp, rows>=7 match@20 +0.969pp, match@1/5 unchanged.
- sample-level residual classifier best result: match@20 +0.376pp, rows>=7 match@20 +0.900pp, match@1/5 unchanged.

Decision:

- Do not freeze MP-20 yet.
- MP-20 has useful validation signal but no selector meets the formal >=1pp total-match gate.
- Continue MPTS-52 validation and use MP-20 results to guide stronger residual scoring or new candidate sources.

Fast-finish acceleration update, 2026-06-25 01:39 UTC:

- Added `scripts/run_mpts52_fast_finish.py`.
- Started watcher session `52750`; state is tracked in `state/mpts52_fast_finish_status.json`.
- The watcher waits for all MPTS-52 shards to complete, then immediately runs:
  `generate_validation_anchor_symprec0p1_mpts52_shards`,
  `assemble_validation_anchor_symprec0p1_mpts52`,
  `export_validation_anchor_symprec0p1_mpts52_jsonl`,
  `copy_validation_anchor_symprec0p1_mpts52_metrics`, and
  `validation_anchor_report`.
- The fast evaluation environment is set to `OPENTRY10_BENCH_WORKERS=120` and
  `OPENTRY10_BENCH_WINDOW=120`, with BLAS/OpenMP thread counts pinned to 1 per
  process.
- At launch, MPTS-52 was at 75 completed shards, 2 running shards, and 2 pending
  shards; both GPUs remained active at 100% utilization.
