# Copied opentry_9 Artifact Quarantine

- Created at: 2026-06-22T17:40:00+00:00
- Scope: artifacts copied from `/data/users/xsw/autodlmini/model/New_model/opentry_9` into `opentry_10` at the user's request.
- Constraint: no copied file or directory was deleted.

## Status

The copied artifacts are preserved for continuity, but they are not authoritative for the active `opentry_10` prompt unless regenerated or re-registered by the `opentry_10` controller.

Known stale copied artifacts include:

- `state/validation_anchor_shards_mp20.json`
- `state/validation_anchor_shards_mpts52.json`
- copied metrics/reports that contain `/opentry_9/` paths
- copied partial validation-anchor diagnostics based on the earlier direct CSV P1 prompt cache
- copied `final_report.md`, which is explicitly marked invalid and is not an `opentry_10` final report

## Active opentry_10 Chain

The active validation-anchor chain is:

- controller: `scripts/run_opentry10.py`
- GT CIF cache: `cache/official_benchmark_cifs_symprec0p1`
- run root: `generations/crystallm_gt_sg_val_anchor_symprec0p1`
- shard root: `generations/crystallm_gt_sg_val_anchor_symprec0p1_shards`
- MP-20 shard plan: `state/validation_anchor_symprec0p1_shards_mp20.json`
- MPTS-52 shard plan: `state/validation_anchor_symprec0p1_shards_mpts52.json`

The copied direct-extraction cache `cache/official_benchmark_cifs` must not be used for GT-SG validation-anchor prompts because it exposes P1 for validation rows.
