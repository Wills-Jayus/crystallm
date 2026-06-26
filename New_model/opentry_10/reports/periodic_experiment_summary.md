# Periodic Experiment Summary

- Updated at: 2026-06-24T03:10:34Z
- Scope: operational progress and runtime assessment only; not a final result.

## Current status

- MP-20 validation GT-SG K100 shards: 142 completed out of 142.
- Current live work: MP-20 shard generation is complete. The completion marker exists and records 904700 expected CIFs.
- MPTS-52 validation GT-SG K100 shards: 16 completed out of 79; no MPTS-52 process is currently running.
- Current policy: prioritize MP-20 and avoid parallel large MPTS-52 evaluation until MP-20 has enough evidence.

## Why this is taking a long time

- The active prompt requires full validation candidate generation, not a small sanity check.
- MP-20 validation has 9047 target structures. K100 means up to 904700 generated CIFs before postprocessing and evaluation.
- Each shard must generate multiple temperature/top-k/sample-offset tranches, then run postprocess and coverage checks.
- Missing candidates count as failures under the prompt, so the controller must repair gaps rather than skip them.

## Estimated time remaining

- Recent observed throughput is roughly 4 MP-20 shards per 45-60 minutes.
- Remaining incomplete MP-20 shards at this checkpoint: 0.
- MP-20 anchor generation is complete; additional time is now for assembly, K1/K5/K20/K50/K100 validation evaluation, export, and reranker work.
- Additional time will still be needed for full validation assembly, evaluation, rerank-only experiments, and any strategy/fusion or pure-structural follow-up.

## Is this worth the time?

Yes, for the MP-20 anchor stage. The full K100 validation bank is the required substrate for a defensible reranking/fusion decision and for avoiding test leakage. Stopping early would leave only partial diagnostics, which the prompt explicitly disallows as final completion.

The cost is not yet justified for new MPTS-52 large runs in parallel. Keep MPTS-52 paused until MP-20 validation shows whether the candidate bank has enough oracle/rerank headroom to plausibly beat the official CrystaLLM-a GT-SG anchor.

## Next checkpoint

- Run MP-20 anchor assembly/evaluation/export stages from the controller.
- Refresh `reports/opentry10_progress_status.md`, this summary, and `experiment_log.md` after MP-20 metrics and candidate JSONL are complete.
- Continue MP-20-only downstream analysis until validation evidence justifies launching new large MPTS-52 work.
