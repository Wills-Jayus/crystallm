# opentry_10 Progress Status

- Updated at: 2026-06-24T03:10:34Z
- Scope: progress checkpoint only; not a final report.
- Write root enforced by controller: `/data/users/xsw/autodlmini/model/New_model/opentry_10`

## Copied artifact handling

- The user requested the mistakenly written `opentry_9` files be copied into `opentry_10`.
- `opentry_10/prompt.md` was preserved.
- The copied `final_report.md` is explicitly marked invalid because the `opentry_10` prompt forbids a real final report before full official test completion.
- The copied controller state was quarantined by the new controller and reinitialized under an `opentry10_*` run id.

## Completed setup

- Phase 0 resource audit refreshed for `opentry_10`.
- Baseline provenance refreshed for the recovered CrystaLLM-a GT-SG anchor.
- Validation GT-SG CIF cache rebuilt with symprec=0.1 under `cache/official_benchmark_cifs_symprec0p1`.
- MP-20 validation cache: 9047 records, 8900 non-P1, 147 P1, 0 conversion errors.
- MPTS-52 validation cache: 5000 records, 4980 non-P1, 20 P1, 0 conversion errors.
- MP-20 and MPTS-52 validation GT-SG prompt directories were prepared from the symprec cache.
- Sample prompt check: MP-20 first prompt includes `Pm-3m`; MPTS-52 first prompt includes `C2/c`.

## Current validation K100 shard status

| dataset | total shards | completed shards | completed raw CIFs | completed post CIFs | remaining shards |
| --- | ---: | ---: | ---: | ---: | ---: |
| MP-20 | 142 | 142 | 904700 | 904700 | 0 |
| MPTS-52 | 79 | 16 | 96800 | 96800 | 63 |

MP-20 validation K100 shard generation is complete at this checkpoint. The completion marker `state/validation_anchor_symprec0p1_shards_mp20.complete.json` exists and records 904700 expected CIFs. No MP-20 generation process is running.

## Execution priority

- New work now prioritizes MP-20.
- Avoid starting parallel large MPTS-52 work while MP-20 remains below the prompt target.
- The MPTS-52 shard batch that was already launched before this priority update was paused after shard 0014 completed; shard 0015 remains pending with partial raw CIFs preserved for missing-output resume.

## Remaining mandatory work

- Assemble, export, and evaluate the complete MP-20 validation K100 anchor.
- Complete all MPTS-52 validation K100 shards.
- Assemble full generated tarballs and run K1/K5/K20/K50/K100 validation metrics.
- Export full validation candidate JSONL files with provenance.
- Run rerank-only OOF model search and subsequent strategy/fusion loops required by `prompt.md`.
- Do not treat any copied `opentry_9` report or partial shard metrics as a formal `opentry_10` result.

## Periodic runtime assessment

- Current long-running step: MP-20 validation GT-SG K100 candidate assembly and evaluation.
- Why it has taken this long: the prompt requires full validation coverage, not a diagnostic subset. MP-20 has 9047 validation targets and K100 generation requires up to 904700 generated CIF candidates, followed by postprocessing and coverage checks. The controller also preserves retry/resume semantics so missing or partial outputs are repaired instead of silently dropped.
- Current observed pace: recent MP-20 batches complete roughly 4 full shards per 45-60 minutes, with variance from generation tranche length and postprocess time.
- MP-20 anchor generation is complete; remaining MP-20 anchor work is assembly/evaluation/export before reranker work.
- Is it worth continuing: yes for MP-20, because full K100 validation candidates are mandatory evidence for the requested rerank/fusion route and prevent misleading partial conclusions. It is not yet worth launching new large MPTS-52 work in parallel; keep MPTS-52 paused until MP-20 full-validation evidence indicates a plausible path to beating the official anchor.
- Review cadence: refresh this assessment after each controller batch or after roughly 1-2 hours of uninterrupted generation.
