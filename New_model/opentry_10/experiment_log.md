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
