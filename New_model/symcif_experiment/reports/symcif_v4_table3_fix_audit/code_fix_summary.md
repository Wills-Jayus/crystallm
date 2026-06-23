# SymCIF-v4 Table 3 Fix Audit: Code Summary

Generated: 2026-05-23 UTC

## Scope

This file records the code-side fixes required before rerunning the SymCIF-v4 Table 3 benchmark. It covers the geometry prototype free-parameter transfer bug, raw candidate truncation before rerank/select, required artifact logging, WA top700 audit support, the CrystaLLM baseline tokenizer mismatch, and the order-sensitive WA key bug found during the rerun.

## Fixed Files

- `scripts/run_symcif_v4_geometry_model_eval.py`
- `scripts/run_symcif_v4_full_pipeline_eval.py`
- `scripts/run_policy_guided_wa_search.py`
- `scripts/run_streaming_wa_search.py`
- `scripts/build_hybrid_wa_search_report.py`
- `scripts/run_crystallm_gt_sg_same_evaluator.py`
- `scripts/audit_wa_multiset_equivalent_hits.py`
- `src/symcif_v4/canonicalize.py`
- `src/symcif_v4/wa_search.py`
- `src/symcif_v4/wa_scorer.py`

## Fix 1: Reference Row Transfer Safety

`run_symcif_v4_geometry_model_eval.py` now enforces one-time consumption of reference rows and validates any transferred free parameters.

Implemented checks:

- Exact element+orbit reference transfer requires reference row element to equal target row element.
- Reference orbit token must equal target orbit token.
- Reference multiplicity must equal target multiplicity.
- Reference free-parameter symbols must exactly match the target orbit free symbols.
- Render input rows must expand to the target formula counts and total atom count before CIF rendering.

Important clarification: same-orbit fallback is allowed only after exact element+orbit matching fails, and only with one-time reference row consumption. This preserves the useful transfer behavior seen in prior audits while preventing the actual bug: reusing one reference row for multiple target rows.

## Fix 2: Raw Candidate Truncation Before Rerank/Select

`run_symcif_v4_geometry_model_eval.py` now reads `max(top_k, full_wa_candidates)` candidates for full mode. This supports:

- raw top200 -> final selected top20
- raw top700 -> final selected top20

The geometry eval path no longer has to truncate to final `top_k=20` before candidate rerank/selection.

`run_symcif_v4_full_pipeline_eval.py` also accepts either `ranked_wa_candidates` or `candidates` in prediction files, so hybrid WA outputs can be evaluated directly.

## Fix 3: Candidate and Evaluation Artifacts

Every geometry full eval now writes:

- `candidates_raw.jsonl`
- `candidates_reranked.jsonl`
- `selected_top20.jsonl`
- `top20_predictions.jsonl`
- `generated_cifs/`
- `generations/baseline.jsonl`
- `failed_cases.jsonl`
- `metrics/baseline_per_generation_metrics.jsonl`
- `full_eval_summary.json`
- `eval_summary.json`

Each sample records:

- `raw_candidate_count`
- `dedup_candidate_count`
- `final_candidate_count`
- `truncated_before_rerank`
- `source_labels` on selected candidates when available

## Fix 4: WA Search top700 Audit Support

The policy, streaming, and hybrid WA audit scripts now report hit rates at:

- @1
- @5
- @20
- @100
- @200
- @700

`build_hybrid_wa_search_report.py` also accepts `--max-records`, so first1000 top700 policy/old/hybrid audits can be generated without accidentally treating missing full-test rows as empty candidates.

## Fix 5: CrystaLLM Baseline Tokenizer Match

During the CrystaLLM+GT-SG baseline rerun, the first implementation used `symcif_experiment/external/CrystaLLM_code/crystallm/_tokenizer.py`, whose vocabulary size is 405. The published CrystaLLM benchmark checkpoints have `vocab_size=371` and match `model/scp_task/CrystaLLM/crystallm/_tokenizer.py`.

Filtering out-of-range token ids prevented a crash but did not fix token-id alignment; it produced malformed generated text and invalid baseline results. `run_crystallm_gt_sg_same_evaluator.py` now loads the tokenizer from `--crystallm-code-root` and asserts that tokenizer vocabulary size equals checkpoint vocabulary size before generation.

The script also supports `--workers-per-device`, enabling high-throughput baseline generation while keeping one dataset per GPU via `CUDA_VISIBLE_DEVICES`.

## Fix 6: Order-Insensitive WA/Skeleton Keys

Additional audit found a third major issue: structured records kept `canonical_wa_key` in extraction/site order, while WA search candidates sort rows by orbit/element. This made identical WA multisets look different whenever row order changed.

Impact:

- `gt_wa_in_topK` and `gt_skeleton_in_topK` were severely undercounted.
- `train_same_wa` and `train_same_skeleton` prototype retrieval almost never triggered for many valid candidates.
- The full geometry stage overused `same_sg`/fallback references, lowering CIF quality and RMSE.
- WA frequency priors built from training records were also order-mismatched against sorted candidate keys.

Implemented fixes:

- Added WA/skeleton multiset keys in `run_symcif_v4_full_pipeline_eval.py` and used them for train prototype indexes and summary hit metrics.
- Updated `run_symcif_v4_geometry_model_eval.py` to use multiset keys for `same_wa`/`same_skeleton` retrieval and candidate deduplication.
- Updated generated/eval artifacts to carry `wa_multiset_key` and `skeleton_multiset_key`.
- Updated `src/symcif_v4/canonicalize.py`, `wa_search.py`, and `wa_scorer.py` so future rebuilds and priors use canonical sorted keys.
- Added `scripts/audit_wa_multiset_equivalent_hits.py` to report ordered exact hit, multiset-equivalent hit, and order-only miss rate.

First1000 top700 audit shows the magnitude of the issue:

| Dataset | Search | ordered exact WA@20 | WA multiset-equiv@20 | ordered exact WA@700 | WA multiset-equiv@700 |
| --- | --- | ---: | ---: | ---: | ---: |
| MP-20 | hybrid | 18.5% | 82.4% | 19.0% | 97.7% |
| MPTS-52 | hybrid | 10.0% | 79.5% | 10.3% | 94.3% |

This changes the interpretation of previous results: the main immediate bug was not low WA candidate recall, but incorrect key semantics and weak prototype retrieval caused by order-sensitive keys.

## Verification

Command:

```bash
python3 -m py_compile \
  model/New_model/symcif_experiment/scripts/run_symcif_v4_geometry_model_eval.py \
  model/New_model/symcif_experiment/scripts/run_symcif_v4_full_pipeline_eval.py \
  model/New_model/symcif_experiment/scripts/run_policy_guided_wa_search.py \
  model/New_model/symcif_experiment/scripts/run_streaming_wa_search.py \
  model/New_model/symcif_experiment/scripts/build_hybrid_wa_search_report.py \
  model/New_model/symcif_experiment/scripts/run_crystallm_gt_sg_same_evaluator.py \
  model/New_model/symcif_experiment/scripts/audit_wa_multiset_equivalent_hits.py \
  model/New_model/symcif_experiment/src/symcif_v4/canonicalize.py \
  model/New_model/symcif_experiment/src/symcif_v4/wa_search.py \
  model/New_model/symcif_experiment/src/symcif_v4/wa_scorer.py
```

Status: passed.
