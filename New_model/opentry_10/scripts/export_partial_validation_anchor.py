#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/data/users/xsw/autodlmini")
CRYSTALLM = WORKSPACE / "model/scp_task/CrystaLLM"
BENCH_CIF_ROOT = ROOT / "cache/official_benchmark_cifs"
PARTIAL_ROOT = ROOT / "generations/crystallm_gt_sg_val_anchor_partial"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_9: {resolved}")
    return resolved


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def append_log(text: str) -> None:
    path = under_root(ROOT / "experiment_log.md")
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def load_benchmark_module() -> Any:
    bin_dir = str(CRYSTALLM / "bin")
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)
    import benchmark_metrics  # type: ignore  # noqa: PLC0415

    return benchmark_metrics


def dataset_short(dataset: str) -> str:
    return "mp20" if dataset == "mp_20" else "mpts52"


def sample_prefix(dataset: str) -> str:
    return "mp_20" if dataset == "mp_20" else "mpts_52"


def plan_path(dataset: str) -> Path:
    return ROOT / "state" / f"validation_anchor_shards_{dataset_short(dataset)}.json"


def gt_dir(dataset: str) -> Path:
    return BENCH_CIF_ROOT / dataset / "val/cifs"


def atom_site_rows(cif: str) -> int:
    in_loop = False
    saw_atom_header = False
    rows = 0
    for raw in cif.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "loop_":
            if in_loop and saw_atom_header and rows:
                break
            in_loop = True
            saw_atom_header = False
            continue
        if in_loop and stripped.startswith("_"):
            if stripped.startswith("_atom_site_"):
                saw_atom_header = True
                continue
            if saw_atom_header:
                break
        if in_loop and saw_atom_header:
            if stripped.startswith("_") or stripped.startswith("data_"):
                break
            rows += 1
    return rows


def completed_shards(dataset: str) -> list[dict[str, Any]]:
    path = plan_path(dataset)
    if not path.exists():
        return []
    plan = json.loads(path.read_text(encoding="utf-8"))
    return [s for s in plan.get("shards", []) if s.get("status") == "completed"]


def collect_dataset(dataset: str) -> dict[str, Any]:
    short = dataset_short(dataset)
    out_dir = PARTIAL_ROOT / f"{short}_partial_k20"
    gen_tar = out_dir / "tars/generated_data_atomtype_gt_sg.tar.gz"
    true_tar = out_dir / "tars/true.tar.gz"
    candidate_path = ROOT / "candidates" / f"unified_{dataset}_val_crystallm_gt_sg_anchor_partial_k20.jsonl"
    per_sample_path = ROOT / "eval" / f"crystallm_gt_sg_{short}_val_partial_k20_per_sample.jsonl"

    shards = completed_shards(dataset)
    material_ids: list[str] = []
    for shard in sorted(shards, key=lambda s: int(s["shard_index"])):
        material_ids.extend(str(x) for x in shard.get("material_ids", []))
    material_ids = sorted(dict.fromkeys(material_ids))
    if not material_ids:
        return {
            "dataset": dataset,
            "status": "missing",
            "reason": "no completed validation anchor shards",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tars").mkdir(parents=True, exist_ok=True)
    id_to_gen_cifs: dict[str, list[str]] = {}
    id_to_true_cifs: dict[str, str] = {}
    rows_ge7: dict[str, bool] = {}
    missing: list[str] = []
    rows_written = 0
    source_counter: Counter[str] = Counter()

    with under_root(candidate_path).open("w", encoding="utf-8") as cand_f:
        for shard in sorted(shards, key=lambda s: int(s["shard_index"])):
            run_dir = Path(shard["run_dir"])
            post_dir = run_dir / "cifs_post/data_atomtype_gt_sg"
            for mid in shard.get("material_ids", []):
                mid = str(mid)
                true_path = gt_dir(dataset) / f"{mid}.cif"
                if not true_path.is_file():
                    missing.append(str(true_path))
                    continue
                true_cif = true_path.read_text(encoding="utf-8", errors="replace")
                id_to_true_cifs[mid] = true_cif
                rows_ge7[mid] = atom_site_rows(true_cif) >= 7
                id_to_gen_cifs[mid] = []
                sample_id = f"{sample_prefix(dataset)}_val_orig__{mid}"
                for rank in range(1, 21):
                    cif_path = post_dir / f"{mid}__{rank}.cif"
                    if not cif_path.is_file():
                        missing.append(str(cif_path))
                        continue
                    cif = cif_path.read_text(encoding="utf-8", errors="replace")
                    id_to_gen_cifs[mid].append(cif)
                    row = {
                        "dataset": dataset,
                        "split": "val",
                        "sample_id": sample_id,
                        "material_id": mid,
                        "source": "crystallm_gt_sg_anchor_partial_k20",
                        "source_rank": rank - 1,
                        "rank": rank,
                        "candidate_id": f"{sample_id}__crystallm_gt_sg_rank{rank:03d}",
                        "cif": cif,
                        "metadata": {
                            "partial_validation_anchor": True,
                            "source_shard_index": int(shard["shard_index"]),
                            "source_run_dir": str(run_dir),
                            "sampling": {
                                "temperature": 0.8,
                                "top_k": 10,
                                "seed": 1337,
                                "sample_seed_stride": 100000,
                            },
                        },
                    }
                    cand_f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                    rows_written += 1
                    source_counter[str(shard["shard_index"])] += 1

    if missing:
        write_json(
            out_dir / "missing_partial_inputs.json",
            {"dataset": dataset, "missing_count": len(missing), "first_missing": missing[:50]},
        )
        raise RuntimeError(f"{dataset} partial export missing {len(missing)} files; first={missing[:3]}")

    with tarfile.open(under_root(gen_tar), "w:gz") as tar:
        for mid in material_ids:
            shard = next(s for s in shards if mid in {str(x) for x in s.get("material_ids", [])})
            post_dir = Path(shard["run_dir"]) / "cifs_post/data_atomtype_gt_sg"
            for rank in range(1, 21):
                tar.add(str(post_dir / f"{mid}__{rank}.cif"), arcname=f"{mid}__{rank}.cif")
    with tarfile.open(under_root(true_tar), "w:gz") as tar:
        for mid in material_ids:
            tar.add(str(gt_dir(dataset) / f"{mid}.cif"), arcname=f"{mid}.cif")

    bench = load_benchmark_module()
    metrics_by_k: dict[str, Any] = {}
    rms_by_k: dict[int, dict[str, Any]] = {}
    for k in range(1, 21):
        metrics = bench.get_match_rate_and_rms_robust_mp(
            id_to_gen_cifs,
            id_to_true_cifs,
            n_gens=k,
            length_lo=0.5,
            length_hi=1000.0,
            angle_lo=10.0,
            angle_hi=170.0,
            ltol=0.3,
            stol=0.5,
            angle_tol=10.0,
            max_sites=512,
            rmsd_timeout_s=5.0,
            workers=8,
            hard_timeout_s=60.0,
        )
        rms_by_id = dict(getattr(bench, "_BENCH_LAST_RMS_BY_ID", {}) or {})
        rms_by_k[k] = rms_by_id
        metrics_by_k[f"k{k}"] = metrics

    per_sample_rows = []
    first_hit_rank: dict[str, int | None] = {}
    for mid in material_ids:
        first = None
        for k in range(1, 21):
            if rms_by_k[k].get(mid) is not None:
                first = k
                break
        first_hit_rank[mid] = first
        per_sample_rows.append(
            {
                "dataset": dataset,
                "split": "val",
                "sample_id": f"{sample_prefix(dataset)}_val_orig__{mid}",
                "material_id": mid,
                "rows_ge7": rows_ge7[mid],
                "hit@1": rms_by_k[1].get(mid) is not None,
                "hit@5": rms_by_k[5].get(mid) is not None,
                "hit@20": rms_by_k[20].get(mid) is not None,
                "rmsd@1": rms_by_k[1].get(mid),
                "rmsd@5": rms_by_k[5].get(mid),
                "rmsd@20": rms_by_k[20].get(mid),
                "first_hit_rank": first,
            }
        )
    with under_root(per_sample_path).open("w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    def rate(rows: list[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        return sum(1 for r in rows if r[key]) / len(rows)

    rows7 = [r for r in per_sample_rows if r["rows_ge7"]]
    internal = {
        "samples": len(per_sample_rows),
        "rows_ge7_samples": len(rows7),
        "match@1": rate(per_sample_rows, "hit@1"),
        "match@5": rate(per_sample_rows, "hit@5"),
        "match@20": rate(per_sample_rows, "hit@20"),
        "rows>=7_match@1": rate(rows7, "hit@1"),
        "rows>=7_match@5": rate(rows7, "hit@5"),
        "rows>=7_match@20": rate(rows7, "hit@20"),
        "top1_fail_but_top5_hit": sum((not r["hit@1"]) and r["hit@5"] for r in per_sample_rows),
        "top1_fail_but_top20_hit": sum((not r["hit@1"]) and r["hit@20"] for r in per_sample_rows),
        "top5_fail_but_top20_hit": sum((not r["hit@5"]) and r["hit@20"] for r in per_sample_rows),
        "first_hit_rank_histogram": dict(sorted(Counter(str(v) for v in first_hit_rank.values()).items())),
    }

    manifest = {
        "dataset": dataset,
        "status": "partial",
        "created_at": now_iso(),
        "candidate_jsonl": str(candidate_path),
        "per_sample_jsonl": str(per_sample_path),
        "generated_tar": str(gen_tar),
        "true_tar": str(true_tar),
        "completed_shards": [int(s["shard_index"]) for s in shards],
        "material_ids": len(material_ids),
        "candidate_rows": rows_written,
        "candidate_budget": 20,
        "source_rows_by_shard": dict(source_counter),
        "metrics_by_k": metrics_by_k,
        "internal_rerank_space": internal,
        "not_full_validation_gate": True,
    }
    write_json(out_dir / "partial_anchor_manifest.json", manifest)
    write_json(ROOT / "metrics" / f"crystallm_gt_sg_{short}_val_partial_k20.json", manifest)
    return manifest


def update_candidate_manifest(results: dict[str, Any]) -> None:
    json_path = ROOT / "metrics/candidate_source_manifest.json"
    rows = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
    rows = [r for r in rows if not str(r.get("source_name", "")).endswith("_partial_k20")]
    for dataset, result in results.items():
        if result.get("status") != "partial":
            continue
        short = dataset_short(dataset)
        rows.append(
            {
                "source_name": f"crystallm_gt_sg_anchor_{short}_val_partial_k20",
                "dataset": dataset,
                "split": "val",
                "candidate_count": result["candidate_rows"],
                "sample_coverage": result["material_ids"],
                "per_sample_candidate_slots": {"min": 20, "median": 20, "max": 20},
                "official_full_test": False,
                "validation": True,
                "pure_model": False,
                "strategy_retrieval_fusion": False,
                "can_train_selector": False,
                "can_final_test_fusion": False,
                "unified_path": result["candidate_jsonl"],
                "notes": "Real CrystaLLM GT-SG validation anchor candidates from completed opentry_9 shards only; partial subset, not a full validation gate.",
            }
        )
    write_json(json_path, rows)

    md_path = ROOT / "reports/candidate_source_manifest.md"
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Candidate Source Manifest\n"
    marker = "\n## opentry_9 Partial Validation Anchor Progress\n"
    md = md.split(marker)[0].rstrip()
    lines = [md, marker.rstrip(), ""]
    lines.extend(
        [
            "| source name | dataset | split | candidate count | sample coverage | per-sample slots | validation | final test fusion | notes |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for dataset, result in results.items():
        if result.get("status") != "partial":
            continue
        short = dataset_short(dataset)
        lines.append(
            f"| crystallm_gt_sg_anchor_{short}_val_partial_k20 | {dataset} | val | "
            f"{result['candidate_rows']} | {result['material_ids']} | 20/20/20 | yes, partial | no | "
            "Completed real validation-anchor shards only; not a full validation gate. |"
        )
    write_text(md_path, "\n".join(lines))


def update_reports(results: dict[str, Any]) -> None:
    def fmt_rate(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    write_json(ROOT / "metrics/crystallm_validation_anchor_partial.json", results)
    lines = [
        "# CrystaLLM Validation Anchor Partial Report",
        "",
        f"- Created at: {now_iso()}",
        "- Scope: completed opentry_9 validation GT-SG anchor shards only.",
        "- Candidate budget used for this report: K20, ranks 1-20 from the historical CrystaLLM-a GT-SG sampling policy.",
        "- This is not a full validation gate and must not be used to freeze a strategy.",
        "",
        "| dataset | samples | candidates | match@1 | match@5 | match@20 | rows>=7 samples | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | top1 fail but top20 hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, result in results.items():
        if result.get("status") != "partial":
            lines.append(f"| {dataset} | 0 | 0 | NA | NA | NA | 0 | NA | NA | NA | NA |")
            continue
        ir = result["internal_rerank_space"]
        lines.append(
            f"| {dataset} | {ir['samples']} | {result['candidate_rows']} | "
            f"{ir['match@1']:.4f} | {ir['match@5']:.4f} | {ir['match@20']:.4f} | "
            f"{ir['rows_ge7_samples']} | "
            f"{fmt_rate(ir['rows>=7_match@1'])} | "
            f"{fmt_rate(ir['rows>=7_match@5'])} | "
            f"{fmt_rate(ir['rows>=7_match@20'])} | "
            f"{ir['top1_fail_but_top20_hit']} |"
        )
    write_text(ROOT / "reports/crystallm_validation_anchor_partial_report.md", "\n".join(lines))

    strategy_json_path = ROOT / "metrics/strategy_oracle_union_audit.json"
    strategy = json.loads(strategy_json_path.read_text(encoding="utf-8")) if strategy_json_path.exists() else {}
    strategy["partial_validation_anchor"] = results
    strategy["validation_gate_status"] = (
        "not_passed_partial_anchor_only; full CrystaLLM GT-SG validation K20 and validation union audit remain incomplete"
    )
    write_json(strategy_json_path, strategy)

    strategy_md_path = ROOT / "reports/strategy_oracle_union_audit.md"
    strategy_md = strategy_md_path.read_text(encoding="utf-8") if strategy_md_path.exists() else "# Strategy Oracle Union Audit\n"
    marker = "\n## opentry_9 Partial Validation Anchor Progress\n"
    strategy_md = strategy_md.split(marker)[0].rstrip()
    extra = [
        strategy_md,
        marker.rstrip(),
        "",
        "A resumable real CrystaLLM GT-SG validation-anchor generation was started in opentry_9. "
        "The completed shards were evaluated as a partial K20 anchor only; this does not satisfy the full validation oracle-union gate.",
        "",
        "| dataset | completed shards | samples | candidates | match@1 | match@5 | match@20 | gate use |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset, result in results.items():
        if result.get("status") != "partial":
            continue
        ir = result["internal_rerank_space"]
        extra.append(
            f"| {dataset} | {len(result['completed_shards'])} | {ir['samples']} | {result['candidate_rows']} | "
            f"{ir['match@1']:.4f} | {ir['match@5']:.4f} | {ir['match@20']:.4f} | diagnostic only |"
        )
    extra.extend(
        [
            "",
            "Gate decision remains unchanged: do not train/freeze a selector and do not run a new official test until full validation K20 anchor plus validation-source union is complete.",
        ]
    )
    write_text(strategy_md_path, "\n".join(extra))

    rerank_json_path = ROOT / "metrics/crystallm_internal_rerank_space.json"
    rerank = json.loads(rerank_json_path.read_text(encoding="utf-8")) if rerank_json_path.exists() else {}
    rerank["partial_validation_anchor"] = {
        dataset: result.get("internal_rerank_space")
        for dataset, result in results.items()
        if result.get("status") == "partial"
    }
    write_json(rerank_json_path, rerank)

    rerank_md_path = ROOT / "reports/crystallm_internal_rerank_space.md"
    rerank_md = rerank_md_path.read_text(encoding="utf-8") if rerank_md_path.exists() else "# CrystaLLM Internal Rerank Space\n"
    marker2 = "\n## opentry_9 Partial Validation Anchor Exact Rerank Space\n"
    rerank_md = rerank_md.split(marker2)[0].rstrip()
    rr = [
        rerank_md,
        marker2.rstrip(),
        "",
        "The rows below are exact per-sample cumulative K20 labels for completed validation-anchor shards only.",
        "",
        "| dataset | samples | match@1 | match@5 | match@20 | top1 fail but top5 hit | top1 fail but top20 hit | top5 fail but top20 hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, result in results.items():
        if result.get("status") != "partial":
            continue
        ir = result["internal_rerank_space"]
        rr.append(
            f"| {dataset} | {ir['samples']} | {ir['match@1']:.4f} | {ir['match@5']:.4f} | {ir['match@20']:.4f} | "
            f"{ir['top1_fail_but_top5_hit']} | {ir['top1_fail_but_top20_hit']} | {ir['top5_fail_but_top20_hit']} |"
        )
    write_text(rerank_md_path, "\n".join(rr))


def update_final_report(results: dict[str, Any]) -> None:
    path = ROOT / "final_report.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# opentry_9 Final Report\n"
    marker = "\n## 2026-06-22 Validation Anchor Addendum\n"
    text = text.split(marker)[0].rstrip()
    lines = [text, marker.rstrip(), ""]
    lines.extend(
        [
            "After the initial diagnostic report, opentry_9 started real CrystaLLM-a GT-SG validation-anchor generation under a resumable shard controller. "
            "Two completed shards were exported and evaluated as partial K20 evidence.",
            "",
            "| dataset | completed shards | samples | candidates | match@1 | match@5 | match@20 | use in decision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for dataset, result in results.items():
        if result.get("status") != "partial":
            continue
        ir = result["internal_rerank_space"]
        lines.append(
            f"| {dataset} | {len(result['completed_shards'])} | {ir['samples']} | {result['candidate_rows']} | "
            f"{ir['match@1']:.4f} | {ir['match@5']:.4f} | {ir['match@20']:.4f} | diagnostic only, not full validation gate |"
        )
    lines.extend(
        [
            "",
            "This addendum does not change the gate decision: the full validation K20 CrystaLLM anchor and validation-source oracle union are still incomplete, so no selector/ranker was trained or frozen and no new official test was run. The partial candidates are useful only for checking that the reproduction path is working and resumable.",
        ]
    )
    write_text(path, "\n".join(lines))


def main() -> None:
    for rel in ("candidates", "eval", "metrics", "reports", "generations", "logs"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    results = {dataset: collect_dataset(dataset) for dataset in ("mp_20", "mpts_52")}
    update_candidate_manifest(results)
    update_reports(results)
    update_final_report(results)
    append_log(
        f"## {now_iso().replace('T', ' ').replace('+00:00', ' UTC')} partial validation anchor audit\n"
        "- Read completed CrystaLLM GT-SG validation-anchor shards from state/validation_anchor_shards_{mp20,mpts52}.json.\n"
        "- Wrote partial K20 candidate JSONL, generated/true tarballs, per-sample labels, metrics, and report.\n"
        "- Gate: diagnostic only; full validation K20 anchor and oracle-union audit remain incomplete, so Phase B and official test remain skipped."
    )


if __name__ == "__main__":
    main()
