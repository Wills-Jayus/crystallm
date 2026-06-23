#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_wyckoff_sequence_models import (  # noqa: E402
    AssignmentNet,
    Vocab,
    ensure_under_opentry,
    formula_vec,
    hit_at,
    prior_assignment_logprob,
    remaining_vec,
    seq_key_from_pairs,
    vocab_from_jsonable,
    write_json,
    write_jsonl,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_records is not None and len(rows) >= int(max_records):
                    break
    return rows


def skeleton_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "gt_skeleton_diagnostic",
        "score": 0.0,
        "orbits": [str(x["orbit_id"]) for x in record["skeleton_sequence"]],
        "skeleton_key": str(record["canonical_skeleton_key"]),
        "composition_exact": True,
    }


def build_fixed_orbit_set(train_records: list[dict[str, Any]]) -> set[str]:
    fixed: set[str] = set()
    for record in train_records:
        for row in record.get("wa_sequence") or []:
            orbit_id = str(row["orbit_id"])
            free_symbols = list(row.get("free_param_names") or row.get("free_symbols") or [])
            if not free_symbols:
                fixed.add(orbit_id)
    return fixed


def has_duplicate_fixed_orbits(orbit_ids: list[str], fixed_orbits: set[str]) -> bool:
    seen: set[str] = set()
    for orbit_id in orbit_ids:
        if orbit_id not in fixed_orbits:
            continue
        if orbit_id in seen:
            return True
        seen.add(orbit_id)
    return False


class RecordAssignmentScorer:
    def __init__(
        self,
        model: AssignmentNet,
        vocab: Vocab,
        record: dict[str, Any],
        elements: list[str],
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.record = record
        self.elements = elements
        self.device = device
        self.total = int(record["atom_count"])
        self.sg = int(record["sg"])
        self.sg_id = torch.tensor([vocab.sg_to_id.get(self.sg, 0)], dtype=torch.long, device=device)
        self.formula = formula_vec(record, vocab).unsqueeze(0).to(device)
        self.cache: dict[tuple[tuple[int, ...], str, int], torch.Tensor] = {}

    @torch.no_grad()
    def scores_for_states(self, orbit_id: str, remainings: list[tuple[int, ...]], step: int) -> dict[tuple[int, ...], torch.Tensor]:
        orbit_token = self.vocab.orbit_to_id.get(str(orbit_id))
        info = self.vocab.orbit_info.get(str(orbit_id))
        if orbit_token is None or info is None:
            return {}
        normalized = [tuple(int(x) for x in remaining) for remaining in remainings]
        missing = [remaining for remaining in normalized if (remaining, str(orbit_id), int(step)) not in self.cache]
        if missing:
            rem_dicts = [
                {str(e): int(remaining[idx]) for idx, e in enumerate(self.elements)}
                for remaining in missing
            ]
            rvec = torch.stack([remaining_vec(rem, self.total, self.vocab) for rem in rem_dicts]).to(self.device)
            batch = len(missing)
            sg = self.sg_id.expand(batch)
            formula = self.formula.expand(batch, -1)
            orbit = torch.full((batch,), int(orbit_token), dtype=torch.long, device=self.device)
            numeric = torch.tensor(
                [
                    [
                        float(sum(rem.values())) / max(1.0, float(self.total)),
                        float(step) / 64.0,
                        float(info.multiplicity) / 64.0,
                    ]
                    for rem in rem_dicts
                ],
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model(sg, formula, rvec, orbit, numeric)
            log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
            for remaining, row in zip(missing, log_probs):
                self.cache[(remaining, str(orbit_id), int(step))] = row
        return {
            remaining: self.cache[(remaining, str(orbit_id), int(step))]
            for remaining in normalized
            if (remaining, str(orbit_id), int(step)) in self.cache
        }


def load_assignment_scorer(path: Path, device: torch.device) -> tuple[AssignmentNet, Vocab, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "vocab" not in ckpt:
        raise SystemExit(f"invalid assignment checkpoint: {path}")
    vocab = vocab_from_jsonable(ckpt["vocab"])
    hidden = int((ckpt.get("config") or {}).get("hidden", 256))
    model = AssignmentNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements),
        num_orbits=len(vocab.orbits),
        num_elements=len(vocab.elements),
        hidden=hidden,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab, ckpt


def assert_same_vocab(dp_vocab: Vocab, model_vocab: Vocab) -> None:
    problems: list[str] = []
    if dp_vocab.elements != model_vocab.elements:
        problems.append("elements")
    if dp_vocab.sgs != model_vocab.sgs:
        problems.append("sgs")
    if dp_vocab.orbits != model_vocab.orbits:
        problems.append("orbits")
    if problems:
        raise SystemExit(f"assignment model vocab does not match DP vocab: {', '.join(problems)}")


def top_assignments_dp(
    record: dict[str, Any],
    skeleton: dict[str, Any],
    vocab: Vocab,
    priors: dict[str, Any],
    *,
    top_k: int,
    state_beam: int,
    max_active_paths: int,
    fixed_orbits: set[str] | None,
    assignment_scorer: RecordAssignmentScorer | None,
    prior_weight: float,
    assignment_model_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    orbit_ids = [str(x) for x in skeleton.get("orbits", [])]
    if any(oid not in vocab.orbit_info for oid in orbit_ids):
        return [], {"ok": False, "reason": "orbit_not_in_train_vocab"}
    if fixed_orbits is not None and has_duplicate_fixed_orbits(orbit_ids, fixed_orbits):
        return [], {"ok": False, "reason": "duplicate_fixed_orbit_skeleton"}
    orbit_ids = sorted(orbit_ids, key=lambda oid: vocab.orbit_info[oid].sort_key)
    atom_total = int(record["atom_count"])
    skel_total = sum(int(vocab.orbit_info[oid].multiplicity) for oid in orbit_ids)
    if skel_total != atom_total:
        return [], {"ok": False, "reason": "skeleton_atom_total_mismatch", "skeleton_total": skel_total}

    elements = sorted(str(k) for k in record["formula_counts"])
    start_remaining = tuple(int(record["formula_counts"][e]) for e in elements)
    states: dict[tuple[int, ...], list[tuple[float, list[tuple[str, str]]]]] = {start_remaining: [(float(skeleton.get("score", 0.0)), [])]}
    visited_states = 1
    for step, orbit_id in enumerate(orbit_ids):
        mult = int(vocab.orbit_info[orbit_id].multiplicity)
        next_states: dict[tuple[int, ...], list[tuple[float, list[tuple[str, str]]]]] = {}
        neural_scores = (
            assignment_scorer.scores_for_states(orbit_id, list(states.keys()), step)
            if assignment_scorer is not None and float(assignment_model_weight) != 0.0
            else {}
        )
        for remaining, candidates in states.items():
            log_probs = neural_scores.get(remaining)
            for score, pairs in candidates:
                for idx, element in enumerate(elements):
                    if remaining[idx] < mult:
                        continue
                    rem_list = list(remaining)
                    rem_list[idx] -= mult
                    rem_tuple = tuple(rem_list)
                    step_score = float(prior_weight) * prior_assignment_logprob(element, orbit_id, int(record["sg"]), priors)
                    if log_probs is not None and float(assignment_model_weight) != 0.0:
                        element_id = vocab.element_to_id.get(str(element))
                        if element_id is not None:
                            step_score += float(assignment_model_weight) * float(log_probs[int(element_id)])
                    score_next = float(score) + step_score
                    next_states.setdefault(rem_tuple, []).append((score_next, [*pairs, (element, orbit_id)]))
        pruned: dict[tuple[int, ...], list[tuple[float, list[tuple[str, str]]]]] = {}
        for rem_tuple, candidates in next_states.items():
            candidates.sort(key=lambda item: item[0] / max(1, len(item[1])), reverse=True)
            pruned[rem_tuple] = candidates[: int(state_beam)]
        if max_active_paths > 0:
            flat: list[tuple[float, tuple[int, ...], float, list[tuple[str, str]]]] = []
            for rem_tuple, candidates in pruned.items():
                for score, pairs in candidates:
                    flat.append((float(score) / max(1, len(pairs)), rem_tuple, score, pairs))
            if len(flat) > int(max_active_paths):
                flat.sort(key=lambda item: item[0], reverse=True)
                budgeted: dict[tuple[int, ...], list[tuple[float, list[tuple[str, str]]]]] = {}
                for _, rem_tuple, score, pairs in flat[: int(max_active_paths)]:
                    budgeted.setdefault(rem_tuple, []).append((score, pairs))
                pruned = budgeted
        states = pruned
        visited_states += len(states)
        if not states:
            return [], {"ok": False, "reason": "no_assignment_state", "visited_states": visited_states}
    zero = tuple(0 for _ in elements)
    final = states.get(zero, [])
    final.sort(key=lambda item: item[0] / max(1, len(item[1])), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, pairs in final:
        key = seq_key_from_pairs(pairs)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "wa_key": key,
                "pairs": pairs,
                "score": float(score),
                "source": str(skeleton.get("source") or "assignment_dp_prior"),
                "skeleton_key": str(skeleton.get("skeleton_key") or ""),
                "composition_exact": True,
            }
        )
        if len(out) >= int(top_k):
            break
    return out, {
        "ok": bool(out),
        "reason": None if out else "no_final_assignment",
        "visited_states": visited_states,
        "final_candidates": len(final),
    }


def summarize(per_sample: list[dict[str, Any]], prefix: str, ks: tuple[int, ...]) -> dict[str, Any]:
    total = max(1, len(per_sample))
    out: dict[str, Any] = {"samples": len(per_sample)}
    for k in ks:
        out[f"{prefix}@{k}"] = sum(int(row[f"{prefix}_hit@{k}"]) for row in per_sample) / total
    out["candidate_nonempty_rate"] = sum(int(row[f"{prefix}_candidate_count"] > 0) for row in per_sample) / total
    out["unique_mean"] = sum(int(row[f"{prefix}_unique"]) for row in per_sample) / total
    out["eligible_rate"] = sum(int(row[f"{prefix}_eligible"]) for row in per_sample) / total
    return out


def subset(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def run_gt_skeleton(
    records: list[dict[str, Any]],
    vocab: Vocab,
    priors: dict[str, Any],
    *,
    top_k: int,
    state_beam: int,
    max_active_paths: int,
    fixed_orbits: set[str] | None,
    assignment_model: AssignmentNet | None,
    assignment_device: torch.device,
    prior_weight: float,
    assignment_model_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    candidates_out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        skeleton = skeleton_from_record(record)
        elements = sorted(str(k) for k in record["formula_counts"])
        scorer = (
            RecordAssignmentScorer(assignment_model, vocab, record, elements, assignment_device)
            if assignment_model is not None and float(assignment_model_weight) != 0.0
            else None
        )
        cands, stats = top_assignments_dp(
            record,
            skeleton,
            vocab,
            priors,
            top_k=top_k,
            state_beam=state_beam,
            max_active_paths=max_active_paths,
            fixed_orbits=fixed_orbits,
            assignment_scorer=scorer,
            prior_weight=prior_weight,
            assignment_model_weight=assignment_model_weight,
        )
        keys = [str(c["wa_key"]) for c in cands]
        row = {
            "index": idx,
            "sample_id": record["keys"].get("sample_id"),
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "target_wa_key": str(record["canonical_wa_key"]),
            "gt_dp_candidate_count": len(cands),
            "gt_dp_unique": len(set(keys)),
            "gt_dp_eligible": bool(stats.get("ok")),
            "gt_dp_reason": stats.get("reason"),
        }
        for k in (1, 5, 20, 50, 100):
            row[f"gt_dp_hit@{k}"] = hit_at(keys, str(record["canonical_wa_key"]), k)
        per_sample.append(row)
        candidates_out.append({"sample_id": row["sample_id"], "candidates": cands[: min(top_k, 100)], "stats": stats})
    return per_sample, candidates_out


def run_predicted_skeletons(
    records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    vocab: Vocab,
    priors: dict[str, Any],
    *,
    top_k: int,
    state_beam: int,
    max_active_paths: int,
    fixed_orbits: set[str] | None,
    max_skeletons: int,
    per_skeleton: int,
    assignment_model: AssignmentNet | None,
    assignment_device: torch.device,
    prior_weight: float,
    assignment_model_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample: list[dict[str, Any]] = []
    candidates_out: list[dict[str, Any]] = []
    by_sample = {str(row.get("sample_id")): row for row in candidate_rows}
    for idx, record in enumerate(records):
        sample_id = str(record["keys"].get("sample_id"))
        source = by_sample.get(sample_id, {})
        all_cands: dict[str, dict[str, Any]] = {}
        stats_rows: list[dict[str, Any]] = []
        used_skeletons = 0
        skipped_duplicate_fixed = 0
        elements = sorted(str(k) for k in record["formula_counts"])
        scorer = (
            RecordAssignmentScorer(assignment_model, vocab, record, elements, assignment_device)
            if assignment_model is not None and float(assignment_model_weight) != 0.0
            else None
        )
        for skel in list(source.get("skeleton_candidates") or []):
            skel = dict(skel)
            if fixed_orbits is not None and has_duplicate_fixed_orbits([str(x) for x in skel.get("orbits") or []], fixed_orbits):
                skipped_duplicate_fixed += 1
                continue
            if used_skeletons >= int(max_skeletons):
                break
            used_skeletons += 1
            skel["source"] = "predicted_skeleton_dp_prior"
            if scorer is not None:
                skel["source"] = "predicted_skeleton_dp_neural_assignment"
            cands, stats = top_assignments_dp(
                record,
                skel,
                vocab,
                priors,
                top_k=per_skeleton,
                state_beam=state_beam,
                max_active_paths=max_active_paths,
                fixed_orbits=fixed_orbits,
                assignment_scorer=scorer,
                prior_weight=prior_weight,
                assignment_model_weight=assignment_model_weight,
            )
            stats_rows.append({"skeleton_key": skel.get("skeleton_key"), **stats})
            for cand in cands:
                all_cands.setdefault(str(cand["wa_key"]), cand)
        ranked = sorted(all_cands.values(), key=lambda item: item["score"] / max(1, len(item.get("pairs") or [])), reverse=True)[: int(top_k)]
        keys = [str(c["wa_key"]) for c in ranked]
        row = {
            "index": idx,
            "sample_id": sample_id,
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
            "target_wa_key": str(record["canonical_wa_key"]),
            "pred_skeleton_count": len(source.get("skeleton_candidates") or []),
            "pred_skeleton_used": used_skeletons,
            "pred_skeleton_skipped_duplicate_fixed": skipped_duplicate_fixed,
            "pred_skeleton_hit": any(str(s.get("skeleton_key")) == str(record["canonical_skeleton_key"]) for s in source.get("skeleton_candidates") or []),
            "pred_dp_candidate_count": len(ranked),
            "pred_dp_unique": len(set(keys)),
            "pred_dp_eligible": bool(ranked),
            "pred_dp_reason": None if ranked else "no_ranked_assignment",
        }
        for k in (1, 5, 20, 50, 100):
            row[f"pred_dp_hit@{k}"] = hit_at(keys, str(record["canonical_wa_key"]), k)
        per_sample.append(row)
        candidates_out.append({"sample_id": sample_id, "candidates": ranked[: min(top_k, 100)], "stats": stats_rows})
    return per_sample, candidates_out


def make_summary(gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ks = (1, 5, 20, 50, 100)
    return {
        "gt_skeleton_assignment_dp": {
            "full": summarize(gt_rows, "gt_dp", ks),
            "rows_ge_7": summarize(subset(gt_rows, lambda r: int(r["row_count"]) >= 7), "gt_dp", ks),
            "atoms_ge_12": summarize(subset(gt_rows, lambda r: int(r["atom_count"]) >= 12), "gt_dp", ks),
        },
        "predicted_skeleton_assignment_dp": {
            "full": summarize(pred_rows, "pred_dp", ks),
            "rows_ge_7": summarize(subset(pred_rows, lambda r: int(r["row_count"]) >= 7), "pred_dp", ks),
            "atoms_ge_12": summarize(subset(pred_rows, lambda r: int(r["atom_count"]) >= 12), "pred_dp", ks),
            "skeleton_hit_rate_in_source": sum(int(r["pred_skeleton_hit"]) for r in pred_rows) / max(1, len(pred_rows)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose opentry_3 W/A assignment with exact DP over fixed skeletons.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--train-repr", type=Path, default=None)
    parser.add_argument("--vocab-json", type=Path, default=OPENTRY_ROOT / "runs" / "e05_wyckoff_seq_prior_8k_val32" / "vocab.json")
    parser.add_argument("--priors-json", type=Path, default=OPENTRY_ROOT / "runs" / "e05_wyckoff_seq_prior_8k_val32" / "decoder_priors_train_only.json")
    parser.add_argument("--predicted-candidates", type=Path, default=OPENTRY_ROOT / "reports" / "e05_wyckoff_seq_prior_8k_val32" / "val_candidates.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OPENTRY_ROOT / "reports" / "e07_assignment_dp_diagnostic")
    parser.add_argument("--max-val-records", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--state-beam", type=int, default=100)
    parser.add_argument("--max-active-paths", type=int, default=0)
    parser.add_argument("--max-skeletons", type=int, default=5)
    parser.add_argument("--per-skeleton", type=int, default=50)
    parser.add_argument("--skip-gt-skeleton", action="store_true")
    parser.add_argument("--allow-duplicate-fixed-skeletons", action="store_true")
    parser.add_argument("--assignment-model-ckpt", type=Path, default=None)
    parser.add_argument("--assignment-device", default="cpu")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--assignment-model-weight", type=float, default=0.0)
    args = parser.parse_args()

    start = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data_dir / f"{args.split}.jsonl", args.max_val_records)
    vocab = vocab_from_jsonable(read_json(args.vocab_json))
    priors = read_json(args.priors_json)
    candidate_rows = read_jsonl(args.predicted_candidates, args.max_val_records)
    assignment_device = torch.device(args.assignment_device)
    assignment_model: AssignmentNet | None = None
    assignment_ckpt_summary: dict[str, Any] | None = None
    if args.assignment_model_ckpt is not None and float(args.assignment_model_weight) != 0.0:
        assignment_model, assignment_vocab, assignment_ckpt = load_assignment_scorer(args.assignment_model_ckpt, assignment_device)
        assert_same_vocab(vocab, assignment_vocab)
        assignment_ckpt_summary = {
            "path": str(args.assignment_model_ckpt),
            "epoch": assignment_ckpt.get("epoch"),
            "score": assignment_ckpt.get("score"),
            "val": assignment_ckpt.get("val"),
        }
    fixed_orbits = None
    if not args.allow_duplicate_fixed_skeletons:
        train_repr = args.train_repr if args.train_repr is not None else args.data_dir / "train.jsonl"
        fixed_orbits = build_fixed_orbit_set(read_jsonl(train_repr))

    if args.skip_gt_skeleton:
        gt_rows, gt_candidates = [], []
    else:
        gt_rows, gt_candidates = run_gt_skeleton(
            records,
            vocab,
            priors,
            top_k=args.top_k,
            state_beam=args.state_beam,
            max_active_paths=args.max_active_paths,
            fixed_orbits=fixed_orbits,
            assignment_model=assignment_model,
            assignment_device=assignment_device,
            prior_weight=float(args.prior_weight),
            assignment_model_weight=float(args.assignment_model_weight),
        )
    pred_rows, pred_candidates = run_predicted_skeletons(
        records,
        candidate_rows,
        vocab,
        priors,
        top_k=args.top_k,
        state_beam=args.state_beam,
        max_active_paths=args.max_active_paths,
        fixed_orbits=fixed_orbits,
        max_skeletons=args.max_skeletons,
        per_skeleton=args.per_skeleton,
        assignment_model=assignment_model,
        assignment_device=assignment_device,
        prior_weight=float(args.prior_weight),
        assignment_model_weight=float(args.assignment_model_weight),
    )
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": len(records),
        "elapsed_s": time.time() - start,
        "fixed_orbit_constraint": {
            "enabled": fixed_orbits is not None,
            "fixed_orbits": 0 if fixed_orbits is None else len(fixed_orbits),
        },
        "assignment_scoring": {
            "prior_weight": float(args.prior_weight),
            "assignment_model_weight": float(args.assignment_model_weight),
            "assignment_model": assignment_ckpt_summary,
        },
        **make_summary(gt_rows, pred_rows),
        "notes": [
            "GT-skeleton DP is diagnostic only; it uses validation target skeleton to isolate assignment capacity.",
            "Predicted-skeleton DP uses existing E05 skeleton candidates and train-only priors; no StructureMatcher or test labels.",
        ],
    }
    write_json(out_dir / "assignment_dp_summary.json", summary)
    if not args.skip_gt_skeleton:
        write_jsonl(out_dir / "gt_skeleton_assignment_per_sample.jsonl", gt_rows)
        write_jsonl(out_dir / "gt_skeleton_assignment_candidates.jsonl", gt_candidates)
    write_jsonl(out_dir / "predicted_skeleton_assignment_per_sample.jsonl", pred_rows)
    write_jsonl(out_dir / "predicted_skeleton_assignment_candidates.jsonl", pred_candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
