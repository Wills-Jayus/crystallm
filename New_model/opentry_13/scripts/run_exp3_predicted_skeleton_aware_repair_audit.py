#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

EXP2 = RESULT_DIR / "experiment_2_predicted_skeleton_renderer_site_mapping.json"
OLD_REPAIR = RESULT_DIR / "experiment_4_predicted_skeleton_geometry_repair.json"
GEOM_SUMMARY = (
    NEW_MODEL
    / "symcif_experiment"
    / "runs"
    / "symcif_v4_geometry_model_no_oversampling"
    / "training_summary.json"
)
OUT_JSON = RESULT_DIR / "experiment_3_predicted_skeleton_aware_geometry_repair_audit.json"
MARKER = "<!-- OPENTRY13_EXP3_PREDICTED_SKELETON_AWARE_REPAIR_AUDIT -->"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        new_text = text[:start].rstrip() + "\n\n" + replacement
    else:
        new_text = text[:start].rstrip() + "\n\n" + replacement + text[next_marker:]
    REPORT_PATH.write_text(new_text, encoding="utf-8")


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def triplet(d: dict[str, Any], prefix: str) -> str:
    return " / ".join(pct(d.get(f"{prefix}@{k}")) for k in (1, 5, 20))


def delta_triplet(d: dict[str, Any]) -> str:
    return " / ".join(pp(d.get(f"delta_match@{k}")) for k in (1, 5, 20))


def conversion_triplet(d: dict[str, Any]) -> str:
    return " / ".join(pct(d.get(f"repair_conversion@{k}")) for k in (1, 5, 20))


def main() -> int:
    exp2 = read_json(EXP2)
    old_repair = read_json(OLD_REPAIR)
    geom_summary = read_json(GEOM_SUMMARY)

    geom_config = dict(geom_summary.get("config") or {})
    geom_data_root = str(geom_config.get("data_root") or "")
    found_predicted_noise_artifact = "predicted" in geom_data_root.lower() or "skeleton_noise" in geom_data_root.lower()

    overall = dict(old_repair.get("overall") or {})
    rows_ge7 = dict(old_repair.get("rows_ge7") or {})
    exp2_gate = dict(exp2.get("gate") or {})

    structure_gate_pass = bool(
        overall.get("valid_rate", 0.0) >= 0.95
        and overall.get("formula_consistency_rate", 0.0) >= 0.95
        and overall.get("sg_consistency_rate", 0.0) >= 0.95
        and overall.get("exact_cover_retained_rate", 0.0) >= 0.95
        and rows_ge7.get("valid_rate", 0.0) >= 0.90
        and rows_ge7.get("formula_consistency_rate", 0.0) >= 0.95
        and rows_ge7.get("sg_consistency_rate", 0.0) >= 0.95
        and rows_ge7.get("exact_cover_retained_rate", 0.0) >= 0.95
    )
    repair_gate_pass = bool(
        all(float(overall.get(f"delta_match@{k}") or 0.0) >= 0.0 for k in (1, 5, 20))
        and (
            float(overall.get("delta_match@5") or 0.0) >= 0.02
            or float(overall.get("delta_match@20") or 0.0) >= 0.02
        )
        and float(rows_ge7.get("delta_match@5") or 0.0) >= 0.05
        and float(rows_ge7.get("delta_match@20") or 0.0) >= 0.05
        and float(rows_ge7.get("repair_conversion@20") or 0.0) > 0.05
    )

    result = {
        "experiment": "opentry_13_exp3_predicted_skeleton_aware_geometry_repair_audit",
        "time": now_iso(),
        "purpose": "Check whether the current learned geometry repair satisfies the required predicted-skeleton-aware training condition.",
        "inputs": {
            "renderer_gate_result": str(EXP2),
            "diagnostic_repair_result": str(OLD_REPAIR),
            "geometry_training_summary": str(GEOM_SUMMARY),
        },
        "renderer_gate": {
            "selected_train_prototype_passed": bool(exp2_gate.get("selected_train_prototype_passed")),
            "raw_train_prototype_passed": bool(exp2_gate.get("train_prototype_passed")),
            "overall_passed": bool(exp2_gate.get("passed")),
        },
        "training_data_audit": {
            "geometry_model_data_root": geom_data_root,
            "geometry_model_best_val_loss": geom_summary.get("best_val_loss"),
            "predicted_skeleton_noise_training_artifact_found": found_predicted_noise_artifact,
            "required_by_goal": "train split predicted skeleton / exact-cover skeleton noise, not GT-WA or GT-skeleton",
            "verdict": "missing_required_training_condition",
        },
        "diagnostic_old_repair": {
            "source_result": str(OLD_REPAIR),
            "geometry_model": old_repair.get("method", {}).get("geometry_model"),
            "data_scale": old_repair.get("data_scale"),
            "overall": overall,
            "rows_ge7": rows_ge7,
        },
        "gates": {
            "structure_gate_pass": structure_gate_pass,
            "repair_gate_pass": repair_gate_pass,
            "experiment_3_claim_allowed": False,
            "failure_reasons": [
                "No trained/optimized geometry repair artifact was found whose training data contains train-split predicted skeleton or exact-cover skeleton noise.",
                "The available diagnostic repair uses the existing GT-WA-style structured geometry training data root, so it cannot satisfy the experiment-3 training condition.",
                "The diagnostic repair also fails the structure gate and repair gate on the predicted-skeleton validation subset.",
            ],
        },
        "decision": {
            "verdict": "fail_and_do_not_claim_as_predicted_skeleton_aware_repair",
            "next_step": "Construct train-split predicted-skeleton/noisy exact-cover geometry pairs or move to rows>=7 multi-geometry proposals with inference-safe structural filtering.",
        },
    }
    write_json(OUT_JSON, result)

    body = f"""## opentry_13 实验 3：predicted-skeleton-aware geometry repair 审计

结果文件：`model/New_model/opentry_13/results/experiment_3_predicted_skeleton_aware_geometry_repair_audit.json`

- 为什么做：目标要求 learned/optimized geometry repair 的训练数据必须包含 train split 上的 predicted skeleton / exact-cover skeleton 噪声，不能用 GT-WA 或 GT-skeleton 冒充。实验 2 的 selected train-prototype renderer 结构 gate 已过，因此需要检查是否真的具备进入 learned repair 的训练条件。
- 核心假设：如果已有 repair 模型不是 predicted-skeleton-aware，即使推理期输入是 predicted skeleton，也不能算实验 3 成功；同时旧 repair 的结构指标和 match conversion 仍要作为诊断记录。
- 数据规模：旧诊断 repair subset 有样本 `{old_repair.get('data_scale', {}).get('evaluated_samples_with_candidates')}`，rows>=7 样本 `{old_repair.get('data_scale', {}).get('evaluated_rows_ge7_samples_with_candidates')}`，candidate records `{old_repair.get('data_scale', {}).get('candidate_records')}`，topK `{old_repair.get('data_scale', {}).get('top_k')}`。
- baseline：before repair 使用同一 predicted skeleton subset 的 hydrated-existing-eval；after repair 使用旧 learned geometry model 重新渲染后的 StructureMatcher。
- 方法变化：本审计不训练新 scorer、不看 official、不使用 test true CIF；只核验训练数据来源和读取旧 repair 诊断结果。几何模型训练数据 root 为 `{geom_data_root}`，不是 predicted-skeleton-noise 训练集。
- 结果 overall：before match@1/5/20 = `{triplet(overall, 'before_match')}`；after match@1/5/20 = `{triplet(overall, 'after_match')}`；delta = `{delta_triplet(overall)}`；repair conversion@1/5/20 = `{conversion_triplet(overall)}`；valid `{pct(overall.get('valid_rate'))}`，formula `{pct(overall.get('formula_consistency_rate'))}`，SG `{pct(overall.get('sg_consistency_rate'))}`，exact-cover `{pct(overall.get('exact_cover_retained_rate'))}`，collision `{pct(overall.get('collision_rate'))}`。
- 结果 rows>=7：before match@1/5/20 = `{triplet(rows_ge7, 'before_match')}`；after match@1/5/20 = `{triplet(rows_ge7, 'after_match')}`；delta = `{delta_triplet(rows_ge7)}`；repair conversion@1/5/20 = `{conversion_triplet(rows_ge7)}`；valid `{pct(rows_ge7.get('valid_rate'))}`，formula `{pct(rows_ge7.get('formula_consistency_rate'))}`，SG `{pct(rows_ge7.get('sg_consistency_rate'))}`，exact-cover `{pct(rows_ge7.get('exact_cover_retained_rate'))}`，collision `{pct(rows_ge7.get('collision_rate'))}`，skeleton-to-match conversion@20 `{pct(rows_ge7.get('skeleton_to_match_conversion@20'))}`。
- 可信度：中高。训练来源审计是确定性的，旧 repair 结果已真实 render/parse/StructureMatcher；限制是这里没有重新构造 predicted-skeleton-noise 训练集，因此结论是“当前 artifact 不满足实验 3 条件”，不是证明所有 repair 方案不可行。
- 和历史实验关系：实验 2 说明 selected renderer/site mapping 可以保结构；旧 opentry_13 repair 说明把 GT-WA-style 几何模型直接接到 predicted skeleton 上会导致 formula/SG/valid 与 conversion 崩掉。
- 最终判决：实验 3 不通过，不能 claim 为 predicted-skeleton-aware learned geometry repair。原因不是 scorer，而是缺少符合目标定义的 predicted-skeleton-noise 训练 artifact，并且旧诊断 repair 的 structure gate 与 repair gate 都失败。
- 下一步：要么构造 train split predicted-skeleton/noisy exact-cover geometry pairs 后重新训练/优化 repair，要么进入实验 4 的 rows>=7 multi-geometry proposal，用 inference-safe structural checks 过滤多几何解。
"""
    append_or_replace_report(MARKER, body)
    print(json.dumps({"output": str(OUT_JSON), "marker": MARKER}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
