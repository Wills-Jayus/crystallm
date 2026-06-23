# Validation Anchor Generation Progress

- Updated at: 2026-06-22T18:42:30+00:00
- Active chain: symprec0p1 GT-SG validation-anchor generation
- Controller: `scripts/run_opentry10.py`
- MP-20 shard plan: `state/validation_anchor_symprec0p1_shards_mp20.json`
- MPTS-52 shard plan: `state/validation_anchor_symprec0p1_shards_mpts52.json`

## Current Status

| dataset | completed shards | pending shards | failed shards | running shards |
|---|---:|---:|---:|---:|
| MP-20 | 2 | 140 | 0 | 0 |
| MPTS-52 | 7 | 72 | 0 | 0 |

## Completed Shards Verified

| dataset | shard | raw CIFs | postprocessed CIFs | expected CIFs | manifest |
|---|---:|---:|---:|---:|---|
| MP-20 | 0 | 6400 | 6400 | 6400 | yes |
| MP-20 | 141 | 2300 | 2300 | 2300 | yes |
| MPTS-52 | 0 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 1 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 2 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 3 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 4 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 5 | 6400 | 6400 | 6400 | yes |
| MPTS-52 | 78 | 800 | 800 | 800 | yes |

## Next Work

Continue Phase 1 until all MP-20 and MPTS-52 validation K100 shards are complete, then assemble full tarballs, run K1/K5/K20/K50/K100 metrics, export JSONL candidate banks, and write `reports/crystallm_validation_anchor_report.md`.
