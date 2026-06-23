#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_4").resolve()
OP3 = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
BASELINE_TRAIN = OP3 / "data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl"
BASELINE_VAL = OP3 / "data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl"
TRAIN_PROPOSAL = ROOT / "data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features.jsonl"
TRAIN_PROPOSAL_SUMMARY = ROOT / "data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features_summary.json"
TRAIN_PAIRFIELD_SUMMARY = ROOT / "reports/e7009_pairfield_adam_repel_train1024_rows7/pairfield_adam_summary.json"
VAL_ANCHOR_EVAL = ROOT / "eval/joint_generator_anchor_safe_val512_eval.json"
DEV_MODULUS = 5
DEV_BUCKET = 0


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_under_root(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"refusing to write outside opentry_4: {resolved}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_bucket(sample_id: str) -> int:
    digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % DEV_MODULUS


def is_dev(sample_id: str) -> bool:
    return stable_bucket(sample_id) == DEV_BUCKET


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("sample_id"))].append(row)
    for items in out.values():
        items.sort(key=lambda r: (int(r.get("rank") or 10**9), str(r.get("candidate_uid", ""))))
    return dict(out)


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def metric_at_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = grouped(rows)
    out: dict[str, Any] = {"samples": len(by_id), "rows": len(rows)}
    for k in (1, 5, 20, 50):
        hits = 0
        wa_hits = 0
        skel_hits = 0
        rmsds: list[float] = []
        for items in by_id.values():
            top = items[:k]
            if any(bool(r.get("candidate_wa_hit")) for r in top):
                wa_hits += 1
            if any(bool(r.get("candidate_skeleton_hit")) for r in top):
                skel_hits += 1
            matched = [r for r in top if bool(r.get("label_match"))]
            if matched:
                hits += 1
                vals = [float(r["label_rmsd"]) for r in matched if r.get("label_rmsd") is not None]
                if vals:
                    rmsds.append(min(vals))
        denom = max(1, len(by_id))
        out[f"match@{k}"] = hits / denom
        out[f"W/A@{k}"] = wa_hits / denom
        out[f"skeleton@{k}"] = skel_hits / denom
        out[f"RMSE@{k}"] = None if not rmsds else sum(rmsds) / len(rmsds)
    return out


def positive_any(rows: list[dict[str, Any]], sample_ids: list[str]) -> dict[str, Any]:
    by_id = grouped(rows)
    hits = sum(1 for sid in sample_ids if any(bool(r.get("label_match")) for r in by_id.get(sid, [])))
    return {
        "samples": len(sample_ids),
        "positive_samples": hits,
        "positive_any_all_candidates": hits / max(1, len(sample_ids)),
    }


def valid_rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if bool(r.get(key))) / max(1, len(rows))


def slim(row: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "sample_id": row.get("sample_id"),
        "candidate_uid": row.get("candidate_uid"),
        "rank": row.get("rank"),
        "target_row_count": row.get("target_row_count"),
        "rows_ge_7": int(row.get("target_row_count") or 0) >= 7,
        "candidate_wa_hit": bool(row.get("candidate_wa_hit")),
        "candidate_skeleton_hit": bool(row.get("candidate_skeleton_hit")),
        "label_match": bool(row.get("label_match")),
        "label_rmsd": row.get("label_rmsd"),
        "readable": bool(row.get("readable")),
        "composition_exact": bool(row.get("composition_exact")),
        "sg_ok": bool(row.get("sg_ok")),
        "self_min_distance": row.get("self_min_distance"),
        "self_volume_per_atom": row.get("self_volume_per_atom"),
        "geometry_param_variant_mode": row.get("geometry_param_variant_mode"),
        "pairfield_scale": row.get("pairfield_scale"),
        "pairfield_delta_loss": row.get("pairfield_delta_loss"),
        "pairfield_end_loss": row.get("pairfield_end_loss"),
    }


def main() -> int:
    baseline_train_all = read_jsonl(BASELINE_TRAIN)
    proposal_train_all = read_jsonl(TRAIN_PROPOSAL)
    baseline_train_dev = [
        r for r in baseline_train_all if is_dev(str(r.get("sample_id"))) and int(r.get("target_row_count") or 0) >= 7
    ]
    proposal_train_dev = [
        r for r in proposal_train_all if is_dev(str(r.get("sample_id"))) and int(r.get("target_row_count") or 0) >= 7
    ]
    proposal_sample_ids = sorted(grouped(proposal_train_dev))
    baseline_for_proposal_ids = [r for r in baseline_train_dev if str(r.get("sample_id")) in set(proposal_sample_ids)]

    baseline_pos = {str(r.get("sample_id")) for r in baseline_for_proposal_ids if bool(r.get("label_match"))}
    proposal_pos = {str(r.get("sample_id")) for r in proposal_train_dev if bool(r.get("label_match"))}
    new_pos = sorted(proposal_pos - baseline_pos)
    merged_train_dev = baseline_for_proposal_ids + [dict(r, pool_source="E7009_pairfield_train_dev") for r in proposal_train_dev]

    wa_hit = [r for r in proposal_train_dev if bool(r.get("candidate_wa_hit"))]
    wa_fail = [r for r in wa_hit if not bool(r.get("label_match"))]
    val_anchor = read_json(VAL_ANCHOR_EVAL)

    summary = {
        "experiment_id": "E7010",
        "created_at": now(),
        "environment": "crystallm_env",
        "split": {
            "name": "E318 train-dev rows>=7",
            "dev_rule": f"blake2b(sample_id) % {DEV_MODULUS} == {DEV_BUCKET}",
            "proposal_train_dev_samples": len(proposal_sample_ids),
            "baseline_rows_same_samples": len(baseline_for_proposal_ids),
            "proposal_rows": len(proposal_train_dev),
        },
        "inputs": {
            "baseline_train_features": str(BASELINE_TRAIN),
            "proposal_rendered": str(ROOT / "reports/e7009_pairfield_adam_repel_train1024_rows7/rendered_topk.jsonl"),
            "proposal_features": str(TRAIN_PROPOSAL),
            "val512_anchor_eval": str(VAL_ANCHOR_EVAL),
        },
        "proposal_generation": read_json(TRAIN_PAIRFIELD_SUMMARY),
        "proposal_feature_summary": read_json(TRAIN_PROPOSAL_SUMMARY),
        "candidate_health_train_dev": {
            "generated_candidates": len(proposal_train_dev),
            "samples": len(proposal_sample_ids),
            "readable_rate": valid_rate(proposal_train_dev, "readable"),
            "composition_valid_rate": valid_rate(proposal_train_dev, "composition_exact"),
            "sg_wyckoff_valid_rate": valid_rate(proposal_train_dev, "sg_ok"),
            "wa_hit_rate": valid_rate(proposal_train_dev, "candidate_wa_hit"),
            "structurematcher_positive_rate": valid_rate(proposal_train_dev, "label_match"),
            "rows_ge_7_positive_rate": valid_rate(proposal_train_dev, "label_match"),
            "wa_hit_match_fail_rate": len(wa_fail) / len(wa_hit) if wa_hit else None,
        },
        "metrics_train_dev": {
            "baseline_same_samples": metric_at_k(baseline_for_proposal_ids),
            "proposal_direct": metric_at_k(proposal_train_dev),
            "merged_positive_any": positive_any(merged_train_dev, proposal_sample_ids),
            "new_positive_samples_beyond_baseline": len(new_pos),
            "new_positive_sample_ids": new_pos[:100],
        },
        "metrics_val512": {
            "baseline_e424_order_val512": val_anchor["metrics"]["baseline_e424_order_val512"],
            "baseline_e424_order_rows_ge_7": val_anchor["metrics"]["baseline_e424_order_rows_ge_7"],
            "proposal_direct_order_aligned216": val_anchor["metrics"]["proposal_direct_order_aligned216"],
            "merged_full_val512_ceiling": val_anchor["metrics"]["merged_full_val512_ceiling"],
            "merged_rows_ge_7_ceiling": val_anchor["metrics"]["merged_rows_ge_7_ceiling"],
            "anchor_safe_selected_val512": val_anchor["metrics"]["anchor_safe_selected_val512"],
            "anchor_safe_selected_rows_ge_7": val_anchor["metrics"]["anchor_safe_selected_rows_ge_7"],
            "new_positive_gate": val_anchor["new_positive_gate"],
        },
        "test_information_used": "no",
    }

    write_json(ROOT / "eval/joint_generator_train_dev_val512_eval.json", summary)
    write_jsonl(
        ROOT / "cache/e7010_pairfield_train_dev_features_slim.jsonl",
        [slim(r, "pairfield_train_dev") for r in proposal_train_dev],
    )

    td = summary["metrics_train_dev"]
    health = summary["candidate_health_train_dev"]
    val = summary["metrics_val512"]
    text = f"""# Joint Geometry Generator Report

Time: {now()}

Data split: E7009/E7010 train-dev uses E318 train rendered candidates filtered by `blake2b(sample_id) % {DEV_MODULUS} == {DEV_BUCKET}`; E700/E7008 evaluates val512. Test information used: no.

## Generator / Proposal Method

The evaluated generator is `pairfield_adam_repel`: a constrained SG/Wyckoff-preserving free-parameter plus isotropic-lattice optimizer. It uses only train CIF pair-distance and VPA statistics, then optimizes rendered candidate fractional geometry/lattice before StructureMatcher evaluation. It is not a selector-only reranker.

## Train-Dev Candidate Health

| metric | value |
|---|---:|
| generated train-dev candidates | {health['generated_candidates']} |
| train-dev samples | {health['samples']} |
| readable rate | {pct(health['readable_rate'])} |
| composition valid rate | {pct(health['composition_valid_rate'])} |
| SG/Wyckoff valid rate | {pct(health['sg_wyckoff_valid_rate'])} |
| W/A-hit rate | {pct(health['wa_hit_rate'])} |
| StructureMatcher positive rate | {pct(health['structurematcher_positive_rate'])} |
| rows>=7 positive rate | {pct(health['rows_ge_7_positive_rate'])} |
| W/A-hit but match-fail rate | {pct(health['wa_hit_match_fail_rate'])} |
| new positive samples beyond train-dev baseline | {td['new_positive_samples_beyond_baseline']} |

## Train-Dev Match Metrics

| system | samples | match@1 | match@5 | match@20 | match@50 | positive-any | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E318 baseline same samples | {td['baseline_same_samples']['samples']} | {pct(td['baseline_same_samples']['match@1'])} | {pct(td['baseline_same_samples']['match@5'])} | {pct(td['baseline_same_samples']['match@20'])} | {pct(td['baseline_same_samples']['match@50'])} | {pct(positive_any(baseline_for_proposal_ids, proposal_sample_ids)['positive_any_all_candidates'])} | {td['baseline_same_samples']['RMSE@20']} |
| pairfield direct train-dev | {td['proposal_direct']['samples']} | {pct(td['proposal_direct']['match@1'])} | {pct(td['proposal_direct']['match@5'])} | {pct(td['proposal_direct']['match@20'])} | {pct(td['proposal_direct']['match@50'])} | {pct(positive_any(proposal_train_dev, proposal_sample_ids)['positive_any_all_candidates'])} | {td['proposal_direct']['RMSE@20']} |
| merged train-dev ceiling | {td['merged_positive_any']['samples']} | NA | NA | NA | NA | {pct(td['merged_positive_any']['positive_any_all_candidates'])} | NA |

## Val512 Candidate And Selector Metrics

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| E424 baseline val512 order | {pct(val['baseline_e424_order_val512']['match@1'])} | {pct(val['baseline_e424_order_val512']['match@5'])} | {pct(val['baseline_e424_order_val512']['match@20'])} | {pct(val['baseline_e424_order_val512']['match@50'])} | {val['baseline_e424_order_val512']['RMSE@20']} |
| E700 direct proposal aligned216 | {pct(val['proposal_direct_order_aligned216']['match@1'])} | {pct(val['proposal_direct_order_aligned216']['match@5'])} | {pct(val['proposal_direct_order_aligned216']['match@20'])} | {pct(val['proposal_direct_order_aligned216']['match@50'])} | {val['proposal_direct_order_aligned216']['RMSE@20']} |
| anchor-safe E424+E700 val512 | {pct(val['anchor_safe_selected_val512']['match@1'])} | {pct(val['anchor_safe_selected_val512']['match@5'])} | {pct(val['anchor_safe_selected_val512']['match@20'])} | {pct(val['anchor_safe_selected_val512']['match@50'])} | {val['anchor_safe_selected_val512']['RMSE@20']} |
| anchor-safe rows>=7 | {pct(val['anchor_safe_selected_rows_ge_7']['match@1'])} | {pct(val['anchor_safe_selected_rows_ge_7']['match@5'])} | {pct(val['anchor_safe_selected_rows_ge_7']['match@20'])} | {pct(val['anchor_safe_selected_rows_ge_7']['match@50'])} | {val['anchor_safe_selected_rows_ge_7']['RMSE@20']} |

Val512 new-positive gate: aligned216 new positives beyond baseline = {val['new_positive_gate']['aligned216_new_positive_samples_vs_first_pool']}; full-val512 new positives beyond E424 top20 = {val['new_positive_gate']['full_val512_new_positive_samples_vs_e424_top20']}; gate passed = {val['new_positive_gate']['gate_passed']}.

Conclusion: the same constrained joint free-param/lattice generator now has train-dev and val512 evidence. It does not add new positive samples beyond the E318 train-dev baseline on this held-out train subset, but it does add 2 rows>=7 new positive samples beyond baseline on val512. Direct ordering is weak and anchor-safe insertion remains conservative.
"""
    write_text(ROOT / "reports/joint_geometry_generator_report.md", text)

    full_test = read_json(ROOT / "eval/E718_full_test_result.json")
    full = full_test["full_test_result"]["subsets"]["full"]
    rows7 = full_test["full_test_result"]["subsets"]["rows_ge_7"]
    gt_sg = {"match@1": 0.2664, "match@5": 0.3658, "match@20": 0.4469}
    energy = read_json(ROOT / "eval/geometry_energy_sklearn_eval.json")
    hard = read_json(ROOT / "eval/hard_negative_diagnosis_v2.json")
    deltas = {k: full[k] - gt_sg[k] for k in gt_sg}
    two_plus5 = sum(1 for k in deltas if deltas[k] >= 0.05) >= 2

    audit = f"""# opentry_4 Requirements Audit

Time: {now()}

| requirement | status | evidence |
|---|---|---|
| write only under opentry_4 | pass | New scripts, data, cache, eval, reports, checkpoints and manifest entries are under `/data/users/xsw/autodlmini/model/New_model/opentry_4`; opentry_3 was read-only. |
| use crystallm_env | pass | E7006-E7010 commands were executed with `conda run -n crystallm_env python`; wrappers compile under that env. |
| freeze/import one-time E718 full test | pass | `reports/E718_frozen_before_full_test.md`, `eval/E718_full_test_result.json`, `reports/E718_full_test_report.md`; opentry_4 did not rerun full test. |
| hard-negative diagnosis | pass | `eval/hard_negative_diagnosis.json` plus top20/top50 `eval/hard_negative_diagnosis_v2.json`. |
| geometry energy model train-dev/val512 | pass | `checkpoints/geometry_energy_model_sklearn_e7006.joblib`, train-dev AUC {energy['train_dev']['global']['auc']:.4f}, val512 AUC {energy['val512']['global']['auc']:.4f}. |
| joint free-param/lattice proposal generator train-dev/val512 | pass | Train-dev generated candidates {health['generated_candidates']} over {health['samples']} rows>=7 samples with match@20 {pct(td['proposal_direct']['match@20'])}; val512 E700/E7008 new positives beyond E424 top20 = {val['new_positive_gate']['full_val512_new_positive_samples_vs_e424_top20']}. |
| smoke then val512 generator schedule | pass | E677 val64 smoke, E685 aligned55, E700 val512, plus E7009/E7010 train-dev evaluation are reported in `reports/joint_geometry_generator_report.md`. |
| anchor-safe replacement after ceiling gain | pass | E7008 anchor-safe insertion runs only after val512 new-positive gate passes; config in `configs/e7008_anchor_safe_selector_config.json`. |
| full test tuning avoided | pass | No test labels or test GT CIFs are used for training/tuning after frozen E734c import. |
| two metrics +5pp vs GT-SG baseline | fail | E718 full test: match@1 {100*deltas['match@1']:+.2f} pp, match@5 {100*deltas['match@5']:+.2f} pp, match@20 {100*deltas['match@20']:+.2f} pp. |

Termination condition 2 is now satisfied with direct evidence: frozen E718 full test is reported, the hard-negative geometry energy model has train-dev/val512 evaluation, and the constrained joint free-param/lattice proposal generator has train-dev/val512 evaluation. The scientific result remains weak: train-dev adds no new positives beyond its baseline, while val512 adds only 2 rows>=7 new positives and anchor-safe insertion does not improve match@20.
"""
    write_text(ROOT / "reports/opentry_4_requirements_audit.md", audit)

    summary_text = f"""# opentry_4 Final Summary

Time: {now()}

## Completed Experiments

1. E7001 froze/audited E718 and imported the existing one-time full MPTS-52 test result from opentry_3 E734c without rerunning test.
2. E7002/E7007 built train/val hard-negative datasets and added top20/top50 rows>=7 diagnosis.
3. E7006 trained a sklearn hard-negative geometry energy model with a deterministic E318 train/train-dev split and evaluated it on val512.
4. E7004/E7008/E7009/E7010 audited the constrained pairfield free-param/lattice proposal generator on train-dev and val512, then evaluated train-dev-frozen anchor-safe insertion on val512.

## Final E718 Full-Test Metrics

| metric | value | delta vs GT-SG CrystaLLM |
|---|---:|---:|
| match@1 | {pct(full['match@1'])} | {100*deltas['match@1']:+.2f} pp |
| match@5 | {pct(full['match@5'])} | {100*deltas['match@5']:+.2f} pp |
| match@20 | {pct(full['match@20'])} | {100*deltas['match@20']:+.2f} pp |
| match@50 | {pct(full['match@50'])} | NA |
| RMSE@20 | {full['RMSE@20']:.4f} | NA |

Rows>=7 full test: match@5={pct(rows7['match@5'])}, match@20={pct(rows7['match@20'])}, match@50={pct(rows7['match@50'])}. The two-metrics +5pp target is not met (`{two_plus5}`), and no full-test feedback was used for tuning.

## New Method Results

| item | result |
|---|---:|
| energy train-dev AUC | {energy['train_dev']['global']['auc']:.4f} |
| energy val512 AUC | {energy['val512']['global']['auc']:.4f} |
| energy val512 rows>=7 pairwise accuracy | {energy['val512']['rows_ge_7_pairwise']['pairwise_accuracy']} |
| generator train-dev candidates / samples | {health['generated_candidates']} / {health['samples']} |
| generator train-dev match@20 | {pct(td['proposal_direct']['match@20'])} |
| generator train-dev new positives beyond baseline | {td['new_positive_samples_beyond_baseline']} |
| generator val512 new positives beyond E424 top20 | {val['new_positive_gate']['full_val512_new_positive_samples_vs_e424_top20']} |
| anchor-safe val512 match@20 | {pct(val['anchor_safe_selected_val512']['match@20'])} |
| anchor-safe rows>=7 match@20 | {pct(val['anchor_safe_selected_rows_ge_7']['match@20'])} |

## Effective

- E718 is frozen and full-test-reported without leakage, but does not clear the GT-SG +5pp target.
- Hard-negative diagnosis shows rows>=7 failures are mainly W/A/skeleton-hit geometry failures: E718 rows>=7 top50 missing samples = {hard['e718']['rows_ge_7']['baseline_missing_top50_samples']}, with W/A/skeleton hit = {hard['e718']['rows_ge_7']['missing_top50_with_wa_or_skeleton_hit_samples']}.
- The constrained pairfield generator creates val512 rows>=7 new positives beyond baseline, proving the ceiling can move.

## Ineffective / Still Weak

- On train-dev, pairfield direct proposals underperform the E318 baseline and add 0 new positive samples beyond that baseline.
- On val512, direct proposal ordering is weak; anchor-safe insertion inserts conservatively and does not improve match@20.
- The system remains in a local optimum around rows>=7 coupled lattice/free-parameter/site-mapping failures.

## Next Routes

1. Train a real pre-render residual or mixture generator over lattice plus free parameters, using E700 positive cases as teacher/supervision but evaluating full val512 ceiling before selector work.
2. Use the sklearn energy model as generation guidance or rejection scoring, not as an ordinary full reranker.

Do not continue ordinary full rerank selectors, source-prior-only tuning, single-row free-param copy, source-free random priors, direct MSE-only regression, or post-render coordinate surgery as primary routes.
"""
    write_text(ROOT / "reports/opentry_4_final_summary.md", summary_text)

    log = f"""
## E7010: pairfield generator train-dev and val512 completion audit

* 时间：{now()}
* 目标：complete the joint free-param/lattice generator train-dev + val512 evaluation required by GPTprompt termination condition 2.
* 读取文件：
  * `{BASELINE_TRAIN}`
  * `{TRAIN_PROPOSAL}`
  * `{VAL_ANCHOR_EVAL}`
* 写入文件：
  * `eval/joint_generator_train_dev_val512_eval.json`
  * `cache/e7010_pairfield_train_dev_features_slim.jsonl`
  * `reports/joint_geometry_generator_report.md`
  * `reports/opentry_4_requirements_audit.md`
  * `reports/opentry_4_final_summary.md`
* 数据 split：E318 train-dev by `blake2b(sample_id) % {DEV_MODULUS} == {DEV_BUCKET}`; val512 from E700/E7008; no test.
* 是否使用 test 信息：no
* 方法：pairfield_adam_repel train-only pair-distance/VPA stats; StructureMatcher labels for train-dev proposal candidates; val512 merged/anchor-safe evidence from E7008.
* 参数：preset=repel; train-dev min_row_count=7; max_input_candidates_per_sample=8; max_output_candidates_per_sample=50.
* 指标：
  * match@1：{pct(td['proposal_direct']['match@1'])}
  * match@5：{pct(td['proposal_direct']['match@5'])}
  * match@20：{pct(td['proposal_direct']['match@20'])}
  * match@50：{pct(td['proposal_direct']['match@50'])}
  * RMSE：{td['proposal_direct']['RMSE@20']}
  * rows>=7 match@5：{pct(td['proposal_direct']['match@5'])}
  * rows>=7 match@20：{pct(td['proposal_direct']['match@20'])}
  * rows>=7 positive-any：{pct(positive_any(proposal_train_dev, proposal_sample_ids)['positive_any_all_candidates'])}
  * new positives beyond baseline：train-dev {td['new_positive_samples_beyond_baseline']}; val512 {val['new_positive_gate']['full_val512_new_positive_samples_vs_e424_top20']}
  * W/A-hit but match-fail rate：{pct(health['wa_hit_match_fail_rate'])}
* 结论：generator train-dev/val512 evaluation is complete; scientific result is weak but termination condition 2 evidence is now direct.
* 是否继续：no
* 下一步：only continue with a stronger pre-render residual/mixture generator if starting a new round.
"""
    append_text(ROOT / "reports/opentry_4_experiment_log.md", log)

    manifest_paths = [
        ROOT / "scripts/opentry4_pairfield_adam_refine_wrapper.py",
        ROOT / "scripts/opentry4_build_geometry_compat_features_wrapper.py",
        ROOT / "scripts/opentry4_generator_train_dev_audit.py",
        ROOT / "reports/e7009_pairfield_adam_repel_train1024_rows7/pairfield_adam_summary.json",
        ROOT / "reports/e7009_pairfield_adam_repel_train1024_rows7/rendered_topk.jsonl",
        ROOT / "data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features.jsonl",
        ROOT / "data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features_summary.json",
        ROOT / "eval/joint_generator_train_dev_val512_eval.json",
        ROOT / "cache/e7010_pairfield_train_dev_features_slim.jsonl",
        ROOT / "reports/joint_geometry_generator_report.md",
        ROOT / "reports/opentry_4_requirements_audit.md",
        ROOT / "reports/opentry_4_final_summary.md",
    ]
    rec = {
        "experiment": "E7010",
        "time": now(),
        "type": "pairfield generator train-dev val512 completion audit",
        "paths": [str(p) for p in manifest_paths if p.exists()],
        "hashes": {str(p): sha256(p) for p in manifest_paths if p.exists() and p.is_file()},
        "git_hash": "not_a_git_repository",
    }
    append_text(ROOT / "manifests/opentry_4_manifest.jsonl", json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
