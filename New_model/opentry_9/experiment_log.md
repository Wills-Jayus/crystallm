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

