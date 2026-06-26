## 2026-06-22 15:58:50 UTC opentry_9 start
- Created directory structure.
- Copied opentry_7/opentry_8 evaluator utility scripts into opentry_9/scripts.
- Write scope: opentry_9 only; historical directories read-only.

## 2026-06-22 16:00:24 UTC Phase A source unification
- Generated 9 unified candidate files under candidates/.
- Key files read: opentry_7/opentry_8 candidate JSONL, SymCIF MP-20 validation K<=5 top20_predictions.
- Gate status: source audit complete; CrystaLLM validation K20 anchor still missing.

## 2026-06-22 16:02:04 UTC opentry_9 start
- Created directory structure.
- Copied opentry_7/opentry_8 evaluator utility scripts into opentry_9/scripts.
- Write scope: opentry_9 only; historical directories read-only.

## 2026-06-22 16:02:04 UTC Phase A source unification
- Reused 9 unified candidate files under candidates/.
- Key files read: opentry_7/opentry_8 candidate JSONL, SymCIF MP-20 validation K<=5 top20_predictions.
- Gate status: source audit complete; CrystaLLM validation K20 anchor still missing.

## 2026-06-22 16:02:06 UTC Phase A gate
- Reports: reports/strategy_oracle_union_audit.md, reports/crystallm_internal_rerank_space.md.
- Main metric: validation oracle union against CrystaLLM K20 could not be computed because the required validation K20 anchor was absent.
- Gate: failed/blocked; Phase B selector/ranker and new official test skipped.

## 2026-06-22 16:02:22 UTC Phase C pure diagnosis
- Generated cache/symcif_train.jsonl, cache/symcif_val.jsonl, cache/symcif_test_targets.jsonl.
- Reports: reports/symcif_canonicalization_audit.md, reports/pure_gt_wa_geometry_report.md, reports/pure_wa_decoder_val_report.md.
- Main metrics: MP-20 GT-WA geometry K<=5 match@1/@5 = 77.16% / 82.94%; baseline WA_hit@1/@5 = 38.63% / 65.11%.
- Gate: pure K20 validation gate not passed; official pure test skipped.

## 2026-06-22 16:02:22 UTC final report
- Artifact: final_report.md.
- Conclusion: diagnostic success; no meaningful official exceed claimed; no test-leakage tuning performed.

## 2026-06-22 16:03:17 UTC opentry_9 start
- Created directory structure.
- Copied opentry_7/opentry_8 evaluator utility scripts into opentry_9/scripts.
- Write scope: opentry_9 only; historical directories read-only.

## 2026-06-22 16:03:17 UTC Phase A source unification
- Reused 9 unified candidate files under candidates/.
- Key files read: opentry_7/opentry_8 candidate JSONL, SymCIF MP-20 validation K<=5 top20_predictions.
- Gate status: source audit complete; CrystaLLM validation K20 anchor still missing.

## 2026-06-22 16:03:19 UTC Phase A gate
- Reports: reports/strategy_oracle_union_audit.md, reports/crystallm_internal_rerank_space.md.
- Main metric: validation oracle union against CrystaLLM K20 could not be computed because the required validation K20 anchor was absent.
- Gate: failed/blocked; Phase B selector/ranker and new official test skipped.

## 2026-06-22 16:03:35 UTC Phase C pure diagnosis
- Generated cache/symcif_train.jsonl, cache/symcif_val.jsonl, cache/symcif_test_targets.jsonl.
- Reports: reports/symcif_canonicalization_audit.md, reports/pure_gt_wa_geometry_report.md, reports/pure_wa_decoder_val_report.md.
- Main metrics: MP-20 GT-WA geometry K<=5 match@1/@5 = 77.16% / 82.94%; baseline WA_hit@1/@5 = 38.63% / 65.11%.
- Gate: pure K20 validation gate not passed; official pure test skipped.

## 2026-06-22 16:03:35 UTC final report
- Artifact: final_report.md.
- Conclusion: diagnostic success; no meaningful official exceed claimed; no test-leakage tuning performed.

## 2026-06-22 16:04:00 UTC final report polish
- Added explicit required-answer bullets for meaningful exceed status, gain source, test leakage risk, and paper-usable conclusion.
## 2026-06-22 17:04:58 UTC partial validation anchor audit
- Read completed CrystaLLM GT-SG validation-anchor shards from state/validation_anchor_shards_{mp20,mpts52}.json.
- Wrote partial K20 candidate JSONL, generated/true tarballs, per-sample labels, metrics, and report.
- Gate: diagnostic only; full validation K20 anchor and oracle-union audit remain incomplete, so Phase B and official test remain skipped.

## 2026-06-22 17:06:28 UTC partial validation anchor audit
- Read completed CrystaLLM GT-SG validation-anchor shards from state/validation_anchor_shards_{mp20,mpts52}.json.
- Wrote partial K20 candidate JSONL, generated/true tarballs, per-sample labels, metrics, and report.
- Gate: diagnostic only; full validation K20 anchor and oracle-union audit remain incomplete, so Phase B and official test remain skipped.

## 2026-06-22 17:26:43 UTC validation GT-SG CIF cache repair
- Built symprec=0.1 validation CIF cache under cache/official_benchmark_cifs_symprec0p1.
- Reason: direct CSV extraction exposes P1 in validation prompts and invalidates GT-SG anchor reproduction.
- Artifact: reports/validation_gt_cifs_symprec_audit.md and metrics/validation_gt_cifs_symprec_audit.json.

## 2026-06-22 17:39:30 UTC opentry_10 corrected validation-anchor resume
- Copied prior opentry_9 artifacts into opentry_10 without deleting any files and preserved opentry_10/prompt.md.
- Reinitialized controller state for opentry_10 and switched validation GT-SG prompt preparation to symprec=0.1 conventionalized CIFs.
- Prepared full MP-20 and MPTS-52 validation prompt/run metadata under generations/crystallm_gt_sg_val_anchor_symprec0p1.
- Planned MP-20 142 shards and MPTS-52 79 shards with paths confined to opentry_10.
- Completed one real K100 smoke shard for each dataset: MP-20 shard 141 raw/post 2300/2300; MPTS-52 shard 78 raw/post 800/800.
- Added persistent flock-based shard locking in scripts/run_opentry10.py after detecting duplicate controllers trying to write the same MP-20 shard.

## 2026-06-22 18:04:30 UTC MPTS-52 validation-anchor shard progress
- Ran `scripts/run_opentry10.py --resume --only generate_validation_anchor_symprec0p1_mpts52_shards --max-shards-per-stage 2`.
- Completed MPTS-52 shard 0000 raw/post 6400/6400 and shard 0001 raw/post 6400/6400.
- Current MPTS-52 shard status: 3 completed, 76 pending, 0 failed, 0 running.
- No residual CrystaLLM generation, postprocess, or benchmark process remained after the batch.

## 2026-06-22 18:42:30 UTC validation-anchor shard progress
- Completed MP-20 shard 0000 raw/post 6400/6400.
- Completed MPTS-52 shards 0002, 0003, 0004, and 0005, each raw/post 6400/6400.
- Current MP-20 shard status: 2 completed, 140 pending, 0 failed, 0 running.
- Current MPTS-52 shard status: 7 completed, 72 pending, 0 failed, 0 running.
- Verified no residual CrystaLLM generation, postprocess, or benchmark process remained after the batch.

## 2026-06-22 17:39:27 UTC opentry_10 corrected validation anchor progress
- Reinitialized copied controller state under opentry_10 and quarantined copied opentry_9 state metadata.
- Prepared symprec0p1 GT-SG validation prompts for MP-20 and MPTS-52.
- Completed one real MP-20 K100 validation shard: 2300 raw CIFs and 2300 postprocessed CIFs.
- Completed one real MPTS-52 K100 validation shard: 800 raw CIFs and 800 postprocessed CIFs.
- Remaining validation anchor work: 141 MP-20 shards and 78 MPTS-52 shards before full K100 metrics/export can be assembled.

## 2026-06-22 18:01:34 UTC MPTS-52 validation anchor shard batch
- Ran `scripts/run_opentry10.py --resume --only generate_validation_anchor_symprec0p1_mpts52_shards --max-shards-per-stage 2`.
- Completed MPTS-52 shards 0000 and 0001, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MPTS-52 validation K100 shard status: 3 completed, 76 remaining.

## 2026-06-22 18:45:07 UTC validation anchor shard progress
- Completed one additional MP-20 shard, shard 0000, with 6400 raw CIFs and 6400 postprocessed CIFs.
- Completed MPTS-52 shards 0002 through 0005, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 2 completed, 140 remaining.
- Current MPTS-52 validation K100 shard status: 7 completed, 72 remaining.
- No real CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-22 19:44:46 UTC validation anchor shard progress
- Completed one additional MP-20 shard, shard 0001, with 6400 raw CIFs and 6400 postprocessed CIFs.
- Completed MPTS-52 shards 0006 through 0009, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 3 completed, 139 remaining.
- Current MPTS-52 validation K100 shard status: 11 completed, 68 remaining.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-22 20:31:04 UTC validation anchor shard progress
- Completed MP-20 shards 0002 through 0005, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 7 completed, 135 remaining.
- Current MPTS-52 validation K100 shard status: 11 completed, 68 remaining.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-22 23:33:47 UTC validation anchor shard progress
- Completed MPTS-52 shards 0010 through 0013, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 7 completed, 135 remaining.
- Current MPTS-52 validation K100 shard status: 15 completed, 64 remaining.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-23 00:31:25 UTC validation anchor shard progress
- Completed MP-20 shards 0006 through 0009, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 11 completed, 131 remaining.
- Current MPTS-52 validation K100 shard status: 15 completed, 64 remaining.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-23 01:34:53 UTC validation anchor shard progress
- Completed MP-20 shards 0010 through 0013, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 15 completed, 127 remaining.
- Current MPTS-52 validation K100 shard status: 15 completed, 64 remaining.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this batch.

## 2026-06-23 02:36:36 UTC MP-20-prioritized validation anchor progress
- Completed MP-20 shards 0014 through 0017, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 19 completed, 123 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining, with the already-started shard 0015 batch still running at the checkpoint.
- Updated execution priority: start new large work on MP-20 first and avoid launching new parallel MPTS-52 work until MP-20 reaches the prompt-required target.

## 2026-06-23 02:50:40 UTC MPTS-52 batch pause for MP-20 priority
- Interrupted the already-started MPTS-52 controller after shard 0014 completed, to honor the updated MP-20-prioritized objective.
- Preserved partial MPTS-52 shard 0015 raw outputs: 2860/6400 raw CIFs, 0/6400 postprocessed CIFs.
- Marked MPTS-52 shard 0015 back to pending with a pause reason so later resume can use the generator's missing-output retry behavior without deleting partial files.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained.

## 2026-06-23 04:08:46 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0018 through 0021, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 23 completed, 119 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 04:51:26 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0022 through 0025, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 27 completed, 115 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 05:37:43 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0026 through 0029, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 31 completed, 111 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 06:12:51 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0030 through 0033, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 35 completed, 107 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 06:51:13 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0034 through 0037, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 39 completed, 103 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 07:32:03 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0038 through 0041, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 43 completed, 99 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 08:15:24 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0042 through 0045, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 47 completed, 95 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 09:04:02 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0046 through 0049, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 51 completed, 91 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 09:59:28 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0050 through 0053, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 55 completed, 87 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 10:57:42 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0054 through 0057, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 59 completed, 83 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 11:57:50 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0058 through 0061, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 63 completed, 79 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 12:50:28 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0062 through 0065, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 67 completed, 75 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 13:51:26 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0066 through 0069, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch with `CUDA_VISIBLE_DEVICES=0,1`; `cuda:auto` distributed workers across both GPUs for the same MP-20 dataset.
- Current MP-20 validation K100 shard status: 71 completed, 71 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess/benchmark process remained after this MP-20-only batch.

## 2026-06-23 15:03:08 UTC Periodic runtime assessment
- Current MP-20 validation K100 shard status: 76 completed, 66 incomplete; shard 0075 is actively running and is not counted as completed.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 process is running.
- This step is slow because MP-20 K100 validation requires full coverage for 9047 targets, up to 904700 generated CIFs, plus postprocessing and sample-coverage checks.
- At the current observed pace of roughly 4 MP-20 shards per 45-60 minutes, remaining MP-20 anchor generation is estimated at about 11-16 hours before validation assembly/evaluation and reranker work.
- Continuing MP-20 is still worthwhile because the full validation bank is mandatory evidence for rerank/fusion decisions; starting additional large MPTS-52 work is not yet justified until MP-20 shows plausible headroom.

## 2026-06-23 15:39:37 UTC MP-20 until-done supervisor
- Added `scripts/run_mp20_until_done.py`.
- Purpose: run `generate_validation_anchor_symprec0p1_mp20_shards` autonomously until all MP-20 validation K100 shards are completed, without requiring Codex to continuously monitor logs.
- The supervisor writes compact status to `state/mp20_until_done_status.json` and logs to `logs/mp20_until_done.log` plus `logs/mp20_until_done_controller.log`.
- It uses a flock lock at `state/mp20_until_done.lock` to prevent duplicate supervisors, waits if another matching opentry_10 MP-20 controller is already active, then launches `run_opentry10.py --resume --only generate_validation_anchor_symprec0p1_mp20_shards --max-shards-per-stage 0`.
- Started detached host supervisor with PID 839819.
- Current supervisor status at launch: waiting for existing controller PID 741625 (`--max-shards-per-stage 4`) to finish current batch; after that, the supervisor will continue MP-20 shard generation without Codex monitoring.

## 2026-06-23 15:47:45 UTC MP-20 worker utilization adjustment
- Current observation: both A800 GPUs were already active, with GPU-Util often 70-100%; low memory usage is expected for the small CrystaLLM checkpoint and does not imply idle GPUs.
- Updated `scripts/run_opentry10.py` so future MP-20 shard generation uses configurable worker counts:
  - `OPENTRY10_GEN_WORKERS`, default 8 instead of the previous hard-coded 4.
  - `OPENTRY10_POST_WORKERS`, default 32.
- Added OOM retry protection for shard generation by passing `oom_worker_arg="--workers"` to `run_logged`; if 8 workers OOM, retry logic can halve the worker count.
- The currently running old controller PID 741625 still uses `--workers 4`; the detached supervisor PID 839819 will use the new 8-worker default after it takes over.

## 2026-06-23 15:25:47 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0070 through 0076 since the last stable checkpoint, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this work through the approved direct controller command without an explicit `CUDA_VISIBLE_DEVICES` prefix; `cuda:auto` used available GPUs for MP-20-only generation.
- Current MP-20 validation K100 shard status: 78 completed, 64 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess process remained after this MP-20-only batch.

## 2026-06-23 16:04:43 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0077 through 0080, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch through the approved direct controller command without an explicit `CUDA_VISIBLE_DEVICES` prefix; `cuda:auto` used available GPUs for MP-20-only generation.
- Current MP-20 validation K100 shard status: 82 completed, 60 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess process remained after this MP-20-only batch.

## 2026-06-23 16:40:37 UTC MP-20 validation anchor shard progress
- Completed MP-20 shards 0081 through 0084, each with 6400 raw CIFs and 6400 postprocessed CIFs.
- Ran this batch through the approved direct controller command without an explicit `CUDA_VISIBLE_DEVICES` prefix; `cuda:auto` used available GPUs for MP-20-only generation.
- Current MP-20 validation K100 shard status: 86 completed, 56 remaining.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 remaining; no MPTS-52 shard is running.
- Verified no residual CrystaLLM generation/postprocess process remained after this MP-20-only batch; detached MP-20 supervisor PID 839819 remains alive and polling.

## 2026-06-23 16:43:56 UTC MP-20 until-done supervisor resumed
- Detached supervisor PID 839819 started MP-20-only controller PID 1380377 with `--max-shards-per-stage 0`.
- Current running shard: MP-20 shard 0085.
- Current completed MP-20 validation K100 shard status remains 86 completed, 56 incomplete while shard 0085 is running.
- MPTS-52 remains paused at 16 completed and 63 pending.

## 2026-06-23 16:52:09 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0085 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 87 completed, 55 incomplete including running shard 0086.
- Completed MP-20 raw/postprocessed CIF totals: 552700 / 552700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:01:32 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0086 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 88 completed, 54 incomplete including running shard 0087.
- Completed MP-20 raw/postprocessed CIF totals: 559100 / 559100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:10:14 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0087 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 89 completed, 53 incomplete including running shard 0088.
- Completed MP-20 raw/postprocessed CIF totals: 565500 / 565500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:19:28 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0088 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 90 completed, 52 incomplete including running shard 0089.
- Completed MP-20 raw/postprocessed CIF totals: 571900 / 571900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:28:53 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0089 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 91 completed, 51 incomplete including running shard 0090.
- Completed MP-20 raw/postprocessed CIF totals: 578300 / 578300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:38:09 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0090 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 92 completed, 50 incomplete including running shard 0091.
- Completed MP-20 raw/postprocessed CIF totals: 584700 / 584700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 17:50:33 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0091 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 93 completed, 49 incomplete including running shard 0092.
- Completed MP-20 raw/postprocessed CIF totals: 591100 / 591100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:00:58 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0092 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 94 completed, 48 incomplete including running shard 0093.
- Completed MP-20 raw/postprocessed CIF totals: 597500 / 597500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:11:00 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0093 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 95 completed, 47 incomplete including running shard 0094.
- Completed MP-20 raw/postprocessed CIF totals: 603900 / 603900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:20:34 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0094 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 96 completed, 46 incomplete including running shard 0095.
- Completed MP-20 raw/postprocessed CIF totals: 610300 / 610300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:30:25 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0095 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 97 completed, 45 incomplete including running shard 0096.
- Completed MP-20 raw/postprocessed CIF totals: 616700 / 616700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:40:19 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0096 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 98 completed, 44 incomplete including running shard 0097.
- Completed MP-20 raw/postprocessed CIF totals: 623100 / 623100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:49:31 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0097 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 99 completed, 43 incomplete including running shard 0098.
- Completed MP-20 raw/postprocessed CIF totals: 629500 / 629500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 18:58:58 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0098 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 100 completed, 42 incomplete including running shard 0099.
- Completed MP-20 raw/postprocessed CIF totals: 635900 / 635900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 19:08:13 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0099 with 6400 raw CIFs and 6400 postprocessed CIFs under detached supervisor PID 839819 / controller PID 1380377.
- Current MP-20 validation K100 shard status: 101 completed, 41 incomplete including running shard 0100.
- Completed MP-20 raw/postprocessed CIF totals: 642300 / 642300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 19:17:51 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0100 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 102 completed, 40 incomplete including running shard 0101.
- Completed MP-20 raw/postprocessed CIF totals: 648700 / 648700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.
- Narrow `pgrep`/`ps` checks from the current sandbox did not reliably expose the older launcher PIDs, but state and log timestamps show the MP-20 generation workflow is still advancing into shard 0101.

## 2026-06-23 19:28:06 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0101 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 103 completed, 39 incomplete including running shard 0102.
- Completed MP-20 raw/postprocessed CIF totals: 655100 / 655100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 19:41:47 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0102 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 104 completed, 38 incomplete including running shard 0103.
- Completed MP-20 raw/postprocessed CIF totals: 661500 / 661500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 19:54:45 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0103 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 105 completed, 37 incomplete including running shard 0104.
- Completed MP-20 raw/postprocessed CIF totals: 667900 / 667900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:08:37 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0104 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 106 completed, 36 incomplete including running shard 0105.
- Completed MP-20 raw/postprocessed CIF totals: 674300 / 674300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:16:43 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0105 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 107 completed, 35 incomplete including running shard 0106.
- Completed MP-20 raw/postprocessed CIF totals: 680700 / 680700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:27:16 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0106 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 108 completed, 34 incomplete including running shard 0107.
- Completed MP-20 raw/postprocessed CIF totals: 687100 / 687100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:37:45 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0107 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 109 completed, 33 incomplete including running shard 0108.
- Completed MP-20 raw/postprocessed CIF totals: 693500 / 693500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:48:17 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0108 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 110 completed, 32 incomplete including running shard 0109.
- Completed MP-20 raw/postprocessed CIF totals: 699900 / 699900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 20:57:37 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0109 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 111 completed, 31 incomplete including running shard 0110.
- Completed MP-20 raw/postprocessed CIF totals: 706300 / 706300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 21:08:04 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0110 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 112 completed, 30 incomplete including running shard 0111.
- Completed MP-20 raw/postprocessed CIF totals: 712700 / 712700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 21:22:49 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0111 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 113 completed, 29 incomplete including running shard 0112.
- Completed MP-20 raw/postprocessed CIF totals: 719100 / 719100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 21:34:23 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0112 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 114 completed, 28 incomplete including running shard 0113.
- Completed MP-20 raw/postprocessed CIF totals: 725500 / 725500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 21:46:25 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0113 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 115 completed, 27 incomplete including running shard 0114.
- Completed MP-20 raw/postprocessed CIF totals: 731900 / 731900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 21:58:05 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0114 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 116 completed, 26 incomplete including running shard 0115.
- Completed MP-20 raw/postprocessed CIF totals: 738300 / 738300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 22:13:18 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0115 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 117 completed, 25 incomplete including running shard 0116.
- Completed MP-20 raw/postprocessed CIF totals: 744700 / 744700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 22:31:47 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0116 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 118 completed, 24 incomplete including running shard 0117.
- Completed MP-20 raw/postprocessed CIF totals: 751100 / 751100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 22:43:25 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0117 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 119 completed, 23 incomplete including running shard 0118.
- Completed MP-20 raw/postprocessed CIF totals: 757500 / 757500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 23:00:15 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0118 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 120 completed, 22 incomplete including running shard 0119.
- Completed MP-20 raw/postprocessed CIF totals: 763900 / 763900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 23:16:59 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0119 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 121 completed, 21 incomplete including running shard 0120.
- Completed MP-20 raw/postprocessed CIF totals: 770300 / 770300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 23:34:12 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0120 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 122 completed, 20 incomplete including running shard 0121.
- Completed MP-20 raw/postprocessed CIF totals: 776700 / 776700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-23 23:47:21 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0121 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 123 completed, 19 incomplete including running shard 0122.
- Completed MP-20 raw/postprocessed CIF totals: 783100 / 783100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 00:06:32 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0122 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 124 completed, 18 incomplete including running shard 0123.
- Completed MP-20 raw/postprocessed CIF totals: 789500 / 789500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 00:23:28 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0123 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 125 completed, 17 incomplete including running shard 0124.
- Completed MP-20 raw/postprocessed CIF totals: 795900 / 795900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 00:34:54 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0124 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 126 completed, 16 incomplete including running shard 0125.
- Completed MP-20 raw/postprocessed CIF totals: 802300 / 802300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 00:46:12 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0125 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 127 completed, 15 incomplete including running shard 0126.
- Completed MP-20 raw/postprocessed CIF totals: 808700 / 808700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 00:57:36 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0126 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 128 completed, 14 incomplete including running shard 0127.
- Completed MP-20 raw/postprocessed CIF totals: 815100 / 815100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:05:59 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0127 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 129 completed, 13 incomplete including running shard 0128.
- Completed MP-20 raw/postprocessed CIF totals: 821500 / 821500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:14:25 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0128 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 130 completed, 12 incomplete including running shard 0129.
- Completed MP-20 raw/postprocessed CIF totals: 827900 / 827900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:26:02 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0129 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 131 completed, 11 incomplete including running shard 0130.
- Completed MP-20 raw/postprocessed CIF totals: 834300 / 834300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:33:38 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0130 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 132 completed, 10 incomplete including running shard 0131.
- Completed MP-20 raw/postprocessed CIF totals: 840700 / 840700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:45:09 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0131 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 133 completed, 9 incomplete including running shard 0132.
- Completed MP-20 raw/postprocessed CIF totals: 847100 / 847100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 01:53:03 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0132 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 134 completed, 8 incomplete including running shard 0133.
- Completed MP-20 raw/postprocessed CIF totals: 853500 / 853500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:00:30 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0133 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 135 completed, 7 incomplete including running shard 0134.
- Completed MP-20 raw/postprocessed CIF totals: 859900 / 859900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:11:58 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0134 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 136 completed, 6 incomplete including running shard 0135.
- Completed MP-20 raw/postprocessed CIF totals: 866300 / 866300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:17:53 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0135 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 137 completed, 5 incomplete including running shard 0136.
- Completed MP-20 raw/postprocessed CIF totals: 872700 / 872700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:29:18 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0136 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 138 completed, 4 incomplete including running shard 0137.
- Completed MP-20 raw/postprocessed CIF totals: 879100 / 879100.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:37:49 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0137 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 139 completed, 3 incomplete including running shard 0138.
- Completed MP-20 raw/postprocessed CIF totals: 885500 / 885500.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:48:29 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0138 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 140 completed, 2 incomplete including running shard 0139.
- Completed MP-20 raw/postprocessed CIF totals: 891900 / 891900.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 02:54:28 UTC MP-20 validation anchor shard progress
- Completed MP-20 shard 0139 with 6400 raw CIFs and 6400 postprocessed CIFs.
- Current MP-20 validation K100 shard status: 141 completed, 1 incomplete including running shard 0140.
- Completed MP-20 raw/postprocessed CIF totals: 898300 / 898300.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.

## 2026-06-24 03:10:34 UTC MP-20 validation anchor generation complete
- Completed MP-20 shard 0140 and the full MP-20 validation K100 shard plan.
- Current MP-20 validation K100 shard status: 142 completed, 0 incomplete, 0 failed, no running shard.
- Completed MP-20 raw/postprocessed CIF totals: 904700 / 904700.
- Completion marker present: `state/validation_anchor_symprec0p1_shards_mp20.complete.json` with expected CIF count 904700.
- Current MPTS-52 validation K100 shard status: 16 completed, 63 pending; no MPTS-52 shard is running.
- Next action: run MP-20 validation anchor assemble/export/metrics stages.

## 2026-06-24 10:05:00 UTC MP-20 validation anchor K100 assembled, evaluated, and exported
- Assembled MP-20 validation anchor K100 tar at `generations/crystallm_gt_sg_val_anchor_symprec0p1/mp20_val_data_atomtype_gt_sg_symprec0p1_k100/tars/generated_data_atomtype_gt_sg.tar.gz` with 904700 generated CIF members.
- Completed K1/K5/K20/K50/K100 metrics and copied summary to `metrics/crystallm_gt_sg_mp20_val.json`.
- Exported full unified candidate bank `candidates/crystallm_gt_sg_mp20_val_k100.jsonl` with 904700 lines.
- Final MP-20 validation anchor metrics:
  - K1 match_rate=0.7275339891676799, RMSE=0.05152308474739601.
  - K5 match_rate=0.8311042334475517, RMSE=0.04369731513726273.
  - K20 match_rate=0.8764231236874102, RMSE=0.04048885252915379.
  - K50 match_rate=0.8991931026859733, RMSE=0.040088549702184.
  - K100 match_rate=0.7562727976124682, RMSE=0.048466794708650326.
- The K100 evaluator initially appeared stalled because upstream `benchmark_metrics.py` uses an ordered multiprocessing window of `workers*8`; a hard timeout terminates the whole pool and requeues later in-flight tasks, discarding completed-but-unharvested work. With K100 this caused repeated pool restarts and multi-hour thrashing.
- Added opentry_10-local `scripts/benchmark_metrics_opentry10.py` and pointed `scripts/run_opentry10.py` at it. The local evaluator preserves completed results before pool termination, uses `OPENTRY10_BENCH_WINDOW`, and marks all active tasks past the hard timeout together. No upstream CrystaLLM files were modified.
- K100 was completed with `OPENTRY10_BENCH_WORKERS=48`, `OPENTRY10_BENCH_WINDOW=48`, `rmsd-timeout-seconds=5.0`, and `hard-timeout-seconds=60.0`. It took 53:50 in the comparison loop and reported 1435 hard timeouts, confirming that pathological StructureMatcher calls were the source of the original stall.
- Caveat: the local K100 evaluator enforces the intended per-active-task hard timeout more strictly than the original ordered evaluator. K1/K5/K20/K50 metrics were produced before this local K100-only evaluator change and are copied unchanged.
- Next action: continue mandatory Phase 1 by completing MPTS-52 validation anchor K100 shards, then assemble/export/evaluate the MPTS-52 validation anchor.

## 2026-06-24 10:15:00 UTC MP-20 validation conclusion while MPTS-52 shards continue
- Wrote MP-20 conclusion report: `reports/mp20_validation_anchor_analysis.md`.
- MP-20 reliable validation evidence is K1/K5/K20/K50. K50 improves over K20 by +2.2769978998563145 percentage points and slightly improves RMSE, so MP-20 has enough validation oracle headroom to justify a reranker/selector branch.
- MP-20 K100 is not a valid monotonic oracle because it is timeout-censored: match_rate drops below K50 and 1435 material-level hard timeouts occurred. The K100 candidate bank remains useful raw material, but K100 aggregate metrics should not drive gates until a per-candidate or checkpointed timeout-safe evaluator exists.
- MPTS-52 validation K100 shard generation is running in parallel with CUDA-enabled escalated execution because sandboxed execution cannot see CUDA devices. Writes remain confined to opentry_10 via controller paths and opentry_10 TMP/cache environment variables.
- Immediate MP-20 branch decision: build K50 selector inputs/labels next; do not wait for MPTS-52 completion to analyze MP-20.

## 2026-06-24 10:16:00 UTC MP-20 K50 selector input features built
- Added `scripts/build_mp20_k50_selector_inputs.py`.
- Built `features/mp20_val_k50_candidate_features.jsonl` from the full MP-20 validation K100 candidate bank.
- Coverage: 452350 rows, 9047 samples, 50 candidates per sample, complete rank coverage.
- Feature summary: `metrics/mp20_val_k50_selector_input_summary.json`.
- Report: `reports/mp20_k50_selector_inputs.md`.
- Labels remain pending; this artifact intentionally avoids StructureMatcher work so it can run in parallel with MPTS-52 shard generation without stealing CPU from postprocess.

## 2026-06-24 10:25:00 UTC MP-20 K50 label pass started in parallel with MPTS-52 generation
- Added `scripts/label_mp20_k50_candidates.py`, a timeout-safe per-candidate StructureMatcher labeler for MP-20 K50 validation candidates.
- Sanity run completed on 128 validation samples: 6400 candidate labels, 0 material hard timeouts, 6356 ok labels, 44 parse errors. The sanity result is process validation only and is not used as a conclusion.
- Started full label pass writing `labels/mp20_val_k50_candidate_labels.jsonl` and `metrics/mp20_val_k50_candidate_label_summary.json` with 24 workers, window 24, hard timeout 90 seconds.
- MPTS-52 validation shard generation continues concurrently on CUDA-enabled execution; shard0015 completed and shard0016 started.

## 2026-06-24 10:49:00 UTC Parallel MP-20 conclusion and MPTS-52 shard execution
- Wrote MP-20 decision report: `reports/mp20_validation_decision.md`.
- MP-20 aggregate validation conclusion: K50 gives +2.277pp reliable oracle headroom over K20, so the MP-20 selector/reranker branch should proceed now rather than waiting for MPTS-52 completion.
- K100 remains a timeout-censored diagnostic only because the current aggregate K100 metric is non-monotonic versus K50 and has 1435 material-level hard timeouts.
- MP-20 K50 label pass is still running and had written 230281 / 452350 candidate-label rows at this snapshot.
- MPTS-52 validation K100 shard generation is running in parallel; current shard state at this snapshot is 19 completed, 1 running, 59 pending out of 79, with shard0018 active.

## 2026-06-24 10:51:00 UTC MP-20 K50 label summarizer prepared
- Added `scripts/summarize_mp20_k50_labels.py` to convert finished per-candidate K50 labels into per-sample K1/K5/K20/K50 metrics, RMSE, label status counts, and target rows>=7 buckets.
- The summarizer reads labels plus the validation target tar only; it does not rerun StructureMatcher and is safe to run after the current full label pass finishes.
- Sanity output on the 128-sample label artifact was written to `metrics/mp20_val_k50_candidate_label_metrics_sanity128.json` and `reports/mp20_k50_label_summary_sanity128.md`. This remains process validation only, not a model-selection conclusion.
- Full label progress at this snapshot: 254337 / 452350 candidate-label rows.

## 2026-06-24 10:55:00 UTC MP-20 K20 rerank-only OOF runner prepared
- Added `scripts/run_mp20_k20_rerank_oof.py` for Phase 2A validation-only rerank search over the original CrystaLLM GT-SG K20 candidate set.
- The script uses deterministic GroupKFold by `sample_id`, so candidates from a validation sample never appear in both train and evaluation folds.
- Supported first-pass model families are regularized logistic regression, HistGradientBoosting, and RandomForest using inference-visible features only.
- Sanity OOF run on the 128-sample label artifact completed and wrote `metrics/rerank_oof_results_sanity128.json`, `reports/rerank_model_search_sanity128.md`, and `features/rerank_oof_predictions_sanity128/`.
- Sanity result is not a conclusion, but it verifies the intended invariant: rerank-only leaves match@20 unchanged while allowing K1/K5 to move.

## 2026-06-24 10:57:00 UTC Controller stages registered for MP-20 selector branch
- Updated `scripts/run_opentry10.py` with resumable stages for `build_mp20_k50_selector_inputs`, `label_mp20_k50_candidates`, `summarize_mp20_k50_labels`, and `run_mp20_k20_rerank_oof`.
- The label stage refuses to overwrite an existing incomplete label JSONL if its summary is absent, protecting the currently running full label process.
- Verified `scripts/run_opentry10.py` syntax and `--list-stages` output for the new stage ids.

## 2026-06-24 11:26:00 UTC MP-20 K50 labels and rerank-only OOF complete
- Full MP-20 K50 candidate labels completed: 452350 / 452350 rows, 9047 / 9047 materials, 6 material hard timeouts.
- Wrote full label summary: `metrics/mp20_val_k50_candidate_label_metrics.json` and `reports/mp20_k50_label_summary.md`.
- K50 oracle headroom over K20 is confirmed by candidate labels: match improves from 0.8763125898087764 to 0.8991931026859733, with 207 K50-only rescues. Rows>=7 match improves from 0.5557093425605536 to 0.6207612456747404.
- Full rerank-only OOF search completed for logistic regression and HistGradientBoosting with 3 seeds: `metrics/rerank_oof_results.json` and `reports/rerank_model_search.md`.
- Best rerank-only result is HGB seed 0: match@1 +0.707pp, match@5 -0.122pp, match@20 unchanged. This is not enough to freeze a rerank-only official-test candidate.
- Wrote selector branch status report: `reports/mp20_selector_branch_status.md`.
- Next MP-20 action: run K50 conservative selector/fusion OOF while MPTS-52 shards continue in parallel.

## 2026-06-24 11:50:00 UTC MP-20 K50 conservative selector OOF complete
- Added and ran `scripts/run_mp20_k50_conservative_selector_oof.py`.
- Full logistic+HGB run with bootstrap=1000 was interrupted because the naive per-strategy bootstrap loop was too slow after logistic seed0; this did not affect completed label or rerank artifacts.
- Reran the selector as HGB-only seeds 0/1/2 with bootstrap disabled for fast model search: `metrics/mp20_k50_conservative_selector_oof.json` and `reports/mp20_k50_conservative_selector_oof.md`.
- Evaluated an additional anchor_keep sweep over 0..20 from the saved HGB OOF scores: `metrics/mp20_k50_anchor_keep_sweep_from_hgb_scores.json` and `reports/mp20_k50_anchor_keep_sweep_from_hgb_scores.md`.
- Best fixed-quota K50 selector is HGB seed 0, anchor_keep=14: match@20 +0.343pp, rows>=7 match@20 +0.969pp, match@1/5 unchanged.
- This confirms usable K50 rescue signal but still does not meet the formal >=1pp total-match validation gate. Do not freeze this MP-20 strategy yet; next MP-20 route should be residual/routed fusion or stronger candidate scoring.

## 2026-06-24 11:54:00 UTC MP-20 residual route sweep from HGB scores
- Evaluated a thresholded residual route sweep from saved HGB K50 OOF scores: `metrics/mp20_k50_residual_route_sweep_from_hgb_scores.json` and `reports/mp20_k50_residual_route_sweep_from_hgb_scores.md`.
- The best routed strategy is still HGB seed 0 with keep_if_routed=14 and all samples routed, identical to the fixed keep=14 result: match@20 +0.343pp and rows>=7 match@20 +0.969pp.
- Thresholding by best supplemental score did not improve over fixed quota, so current candidate scores do not reliably identify which samples should receive extra K50 slots.
- MP-20 status: promising validation signal, but no frozen strategy yet. Continue MPTS-52 shard generation and consider stronger residual classifier/new candidates before official test.

## 2026-06-24 11:57:00 UTC MP-20 sample-level residual classifier tested
- Evaluated a sample-level OOF residual classifier from HGB K50 OOF score features: `metrics/mp20_k50_residual_classifier_oof.json` and `reports/mp20_k50_residual_classifier_oof.md`.
- keep14 creates 68 benefit samples and 37 harm samples relative to baseline K20.
- Best routed classifier is an HGB sample classifier routing 4524 samples: match@20 +0.376pp, rows>=7 match@20 +0.900pp, match@1/5 unchanged.
- This improves slightly over fixed keep14 but remains below the formal >=1pp match gate. MP-20 is not ready to freeze; continue MPTS-52 and pursue stronger candidates/scorers.

## 2026-06-24 14:15:00 UTC MPTS-52 shard generation accelerated with per-GPU workers
- The serial MPTS-52 controller was leaving one GPU idle during parts of shard generation and could exit with a stale running marker.
- Added `scripts/run_mpts52_parallel_shard_worker.py`, a lock-safe worker that claims one MPTS-52 shard at a time using a global plan lock and per-shard `.shard.lock`.
- Started one worker bound to GPU0 and one worker bound to GPU1, both with `OPENTRY10_GEN_WORKERS=8` and `OPENTRY10_POST_WORKERS=32`.
- Current active shards at launch: shard0032 on GPU0 and shard0033 on GPU1. Both GPUs are at 100% utilization.
- The old serial controller exited with a PartialProgress message after seeing shard0032 locked by the new worker; this is expected.
- Expected effect: roughly two shards in parallel instead of one serial shard, reducing remaining MPTS generation wall time from roughly 9-11 hours to roughly 5-7 hours if per-GPU shard times remain stable.

## 2026-06-26 06:14:00 UTC MP-20 frozen K50 HGB ensemble official evaluation complete
- Completed MP-20 test K50 supplemental generation and postprocessing for all blocks. The final watchdog log reached `block 41-50: 90460/90460` and `watchdog complete`.
- Built frozen K20 system `mp20_k50_hgb_mean_seed012_margin_route` from validation-only HGB seed0/1/2 ensemble scores and frozen threshold `0.20066793518000028`.
- Final build manifest: 9046 official test samples, exactly 20 slots per sample, 0 placeholder slots, 707 routed samples, 9217 supplemental slots.
- Ran CrystaLLM Table 3 official full-test metrics for K1/K5/K20 and rows>=7.
- Full-test result:
  - match@1 = 70.230% vs MP-20 anchor 71.67%, delta -1.440 pp.
  - match@5 = 81.870% vs MP-20 anchor 83.08%, delta -1.210 pp.
  - match@20 = 87.486% vs MP-20 anchor 87.81%, delta -0.324 pp.
  - RMSE@1/5/20 = 0.049965 / 0.043623 / 0.042699.
- rows>=7 result:
  - match@1 = 27.200% vs rows>=7 anchor 62.37%, delta -35.170 pp.
  - match@5 = 43.273% vs rows>=7 anchor 76.35%, delta -33.077 pp.
  - match@20 = 55.273% vs rows>=7 anchor 82.61%, delta -27.337 pp.
- Success standard met: false. Failure reasons: no overall match metric improves by >=1.0 pp, and match@20 drops by more than 0.2 pp.
- Wrote `metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json` and `reports/mp20_k50_hgb_mean_seed012_margin_route_official_test.md`.
- Replaced the top of `final_report.md` with the opentry_10 final conclusion. The old opentry_9 copied content remains only as archived text below the new report.
