#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.orbit_token import OrbitToken
from symcif_v4.step_policy import NUMERIC_DIM, StepPolicyNet, StepPolicyVocab, encode_action_batch
from symcif_v4.wa_search import StreamingSearchStats, rows_to_candidate

TOP_KS = (1, 5, 20, 100, 200, 700)


class SearchTimeout(Exception):
    pass


_ENGINE: OrbitEngine | None = None
_MODEL: StepPolicyNet | None = None
_VOCAB: StepPolicyVocab | None = None
_PRIORS: dict[str, Counter[str]] | None = None
_ARGS: dict[str, Any] = {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def counters_from_jsonable(raw: dict[str, Any]) -> dict[str, Counter[str]]:
    return {str(name): Counter({str(k): int(v) for k, v in values.items()}) for name, values in raw.items()}


def load_policy(policy_ckpt: str | Path) -> tuple[StepPolicyNet, StepPolicyVocab, dict[str, Counter[str]]]:
    ckpt = torch.load(policy_ckpt, map_location="cpu")
    vocab = StepPolicyVocab.from_jsonable(ckpt["vocab"])
    model = StepPolicyNet(ckpt["vocab_sizes"], numeric_dim=int(ckpt.get("numeric_dim", NUMERIC_DIM)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    priors = counters_from_jsonable(ckpt.get("priors") or {})
    return model, vocab, priors


def init_worker(lookup_json: str, data_root: str, policy_ckpt: str, args_dict: dict[str, Any]) -> None:
    global _ENGINE, _MODEL, _VOCAB, _PRIORS, _ARGS
    torch.set_num_threads(1)
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, data_root)
    _MODEL, _VOCAB, _PRIORS = load_policy(policy_ckpt)
    _ARGS = args_dict


def q(values: list[int | float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    vals = sorted(float(v) for v in values)
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
        "max": vals[-1],
    }


def hit_at(candidates: list[dict[str, Any]], key: str, field: str, k: int) -> bool:
    return any(str(c.get(field)) == str(key) for c in candidates[:k])


def make_actions(sg: int, formula_counts: dict[str, int], orbits: list[OrbitToken]) -> list[tuple[str, OrbitToken]]:
    actions: list[tuple[str, OrbitToken]] = []
    for element, count in sorted(formula_counts.items()):
        for orbit in orbits:
            if int(orbit.sg) != int(sg):
                continue
            if int(count) < int(orbit.multiplicity):
                continue
            actions.append((str(element), orbit))
    return actions


def prior_score(sg: int, element: str, orbit: OrbitToken, priors: dict[str, Counter[str]]) -> float:
    return (
        1.5 * math.log1p(priors.get("action", Counter()).get(f"{int(sg)}|{element}|{orbit.canonical_orbit_id}", 0))
        + 0.4 * math.log1p(priors.get("orbit", Counter()).get(f"{int(sg)}|{orbit.canonical_orbit_id}", 0))
        + 0.3 * math.log1p(priors.get("element_mult", Counter()).get(f"{int(sg)}|{element}|{int(orbit.multiplicity)}", 0))
    )


@torch.no_grad()
def score_actions(
    *,
    model: StepPolicyNet,
    vocab: StepPolicyVocab,
    priors: dict[str, Counter[str]],
    sg: int,
    formula_counts: dict[str, int],
    remaining_counts: dict[str, int],
    actions: list[tuple[int, str, OrbitToken]],
    step_index: int,
    chosen_count: int,
    policy_weight: float,
    prior_weight: float,
) -> list[tuple[float, int, str, OrbitToken]]:
    if not actions:
        return []
    element_orbits = [(element, orbit) for _, element, orbit in actions]
    batch = encode_action_batch(
        sg=sg,
        formula_counts=formula_counts,
        remaining_counts=remaining_counts,
        element_orbits=element_orbits,
        vocab=vocab,
        priors=priors,
        step_index=step_index,
        chosen_count=chosen_count,
    )
    logits = model(*batch).cpu().tolist()
    scored: list[tuple[float, int, str, OrbitToken]] = []
    for logit, (idx, element, orbit) in zip(logits, actions):
        score = float(policy_weight) * float(logit) + float(prior_weight) * prior_score(sg, element, orbit, priors)
        score -= 0.02 * len(orbit.free_symbols)
        scored.append((score, idx, element, orbit))
    scored.sort(key=lambda row: (-row[0], int(row[3].multiplicity), str(row[3].letter), str(row[2]), str(row[3].canonical_orbit_id)))
    return scored


def policy_guided_exact_cover_search(
    sg: int,
    formula_counts: dict[str, int],
    orbits: list[OrbitToken],
    *,
    model: StepPolicyNet,
    vocab: StepPolicyVocab,
    priors: dict[str, Counter[str]],
    beam_size: int,
    top_k: int,
    max_expanded_states: int,
    timeout_s: float,
    branching_factor: int,
    policy_weight: float,
    prior_weight: float,
) -> tuple[list[dict[str, Any]], StreamingSearchStats]:
    start = time.monotonic()
    elements = tuple(sorted(str(k) for k in formula_counts))
    element_idx = {element: i for i, element in enumerate(elements)}
    initial_counts = tuple(int(formula_counts[e]) for e in elements)
    stats = StreamingSearchStats()
    actions = make_actions(int(sg), {e: int(formula_counts[e]) for e in elements}, orbits)
    if not actions and any(v > 0 for v in initial_counts):
        stats.reason = "no_legal_actions"
        stats.elapsed_s = time.monotonic() - start
        return [], stats

    # Sort once using the policy at the initial step. The search only chooses
    # nondecreasing action indexes, which removes permutation duplicates while
    # still allowing the next chosen action to skip directly to any later index.
    initial_remaining = {e: int(formula_counts[e]) for e in elements}
    init_scored = score_actions(
        model=model,
        vocab=vocab,
        priors=priors,
        sg=int(sg),
        formula_counts={e: int(formula_counts[e]) for e in elements},
        remaining_counts=initial_remaining,
        actions=[(i, element, orbit) for i, (element, orbit) in enumerate(actions)],
        step_index=0,
        chosen_count=0,
        policy_weight=policy_weight,
        prior_weight=prior_weight,
    )
    actions = [(element, orbit) for _, _old_idx, element, orbit in init_scored]
    score_cache: dict[
        tuple[int, tuple[int, ...], tuple[str, ...], tuple[int, ...], int, int],
        list[tuple[float, int, str, OrbitToken]],
    ] = {}

    @lru_cache(maxsize=500_000)
    def feasible(action_idx: int, counts: tuple[int, ...], used_fixed: tuple[str, ...]) -> bool:
        if time.monotonic() - start > float(timeout_s):
            raise SearchTimeout
        if all(v == 0 for v in counts):
            return True
        if action_idx >= len(actions):
            return False
        used = set(used_fixed)
        element, orbit = actions[action_idx]
        if feasible(action_idx + 1, counts, used_fixed):
            return True
        if orbit.is_fully_fixed and orbit.canonical_orbit_id in used:
            return False
        idx = element_idx[element]
        mult = int(orbit.multiplicity)
        max_take = int(counts[idx]) // mult
        if orbit.is_fully_fixed:
            max_take = min(max_take, 1)
        for n in range(max_take, 0, -1):
            nxt = list(counts)
            nxt[idx] -= n * mult
            if any(v < 0 for v in nxt):
                continue
            nxt_used = used
            if orbit.is_fully_fixed:
                nxt_used = set(used)
                nxt_used.add(orbit.canonical_orbit_id)
            if feasible(action_idx + 1, tuple(nxt), tuple(sorted(nxt_used))):
                return True
        return False

    try:
        feasible_initial = feasible(0, initial_counts, tuple())
    except SearchTimeout:
        stats.timeout = True
        stats.reason = "timeout"
        stats.elapsed_s = time.monotonic() - start
        return [], stats
    if not feasible_initial:
        stats.reason = "not_exact_cover_feasible"
        stats.elapsed_s = time.monotonic() - start
        return [], stats

    # Heap item: negative normalized priority, sequence id, next action index,
    # remaining counts, used fixed orbit ids, chosen rows, accumulated score.
    heap: list[tuple[float, int, int, tuple[int, ...], tuple[str, ...], tuple[tuple[str, OrbitToken], ...], float]] = []
    seq = 0
    heapq.heappush(heap, (0.0, seq, 0, initial_counts, tuple(), tuple(), 0.0))
    seen_complete: set[str] = set()
    seen_states: set[tuple[int, tuple[int, ...], tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
    candidates: list[dict[str, Any]] = []
    formula_counts_norm = {e: int(formula_counts[e]) for e in elements}
    while heap and len(candidates) < int(top_k):
        if time.monotonic() - start > float(timeout_s):
            stats.timeout = True
            stats.reason = "timeout"
            break
        if stats.expanded_states >= int(max_expanded_states):
            stats.truncated = True
            stats.reason = "max_expanded_states"
            break
        _priority, _seq, action_idx, counts, used_fixed, chosen_tuple, score = heapq.heappop(heap)
        stats.expanded_states += 1
        if all(v == 0 for v in counts):
            rows = list(chosen_tuple)
            cand = rows_to_candidate(int(sg), dict(formula_counts_norm), rows, score=score)
            key = str(cand["canonical_wa_key"])
            if key not in seen_complete:
                seen_complete.add(key)
                candidates.append(cand)
                stats.complete_states += 1
            continue
        remaining_counts = {element: int(counts[element_idx[element]]) for element in elements}
        used_set = set(used_fixed)
        legal: list[tuple[int, str, OrbitToken]] = []
        for j in range(action_idx, len(actions)):
            element, orbit = actions[j]
            if orbit.is_fully_fixed and orbit.canonical_orbit_id in used_set:
                continue
            idx = element_idx[element]
            mult = int(orbit.multiplicity)
            if counts[idx] < mult:
                continue
            nxt_counts = list(counts)
            nxt_counts[idx] -= mult
            nxt_used = used_set
            if orbit.is_fully_fixed:
                nxt_used = set(used_set)
                nxt_used.add(orbit.canonical_orbit_id)
            next_idx = j + 1 if orbit.is_fully_fixed else j
            try:
                feasible_next = feasible(next_idx, tuple(nxt_counts), tuple(sorted(nxt_used)))
            except SearchTimeout:
                stats.timeout = True
                stats.reason = "timeout"
                break
            if feasible_next:
                legal.append((j, element, orbit))
        if stats.timeout:
            break
        if not legal:
            continue
        cache_key = (
            int(sg),
            tuple(counts),
            tuple(used_fixed),
            tuple(int(row[0]) for row in legal),
            len(chosen_tuple),
            len(chosen_tuple),
        )
        scored = score_cache.get(cache_key)
        if scored is None:
            scored = score_actions(
                model=model,
                vocab=vocab,
                priors=priors,
                sg=int(sg),
                formula_counts=formula_counts_norm,
                remaining_counts=remaining_counts,
                actions=legal,
                step_index=len(chosen_tuple),
                chosen_count=len(chosen_tuple),
                policy_weight=policy_weight,
                prior_weight=prior_weight,
            )
            if len(score_cache) < 200_000:
                score_cache[cache_key] = scored
        for action_score, j, element, orbit in scored[: int(branching_factor)]:
            idx = element_idx[element]
            nxt_counts = list(counts)
            nxt_counts[idx] -= int(orbit.multiplicity)
            nxt_used_set = set(used_fixed)
            if orbit.is_fully_fixed:
                nxt_used_set.add(orbit.canonical_orbit_id)
            nxt_used = tuple(sorted(nxt_used_set))
            next_idx = j + 1 if orbit.is_fully_fixed else j
            new_chosen = chosen_tuple + ((element, orbit),)
            state_key = (
                next_idx,
                tuple(nxt_counts),
                nxt_used,
                tuple((e, o.canonical_orbit_id) for e, o in new_chosen),
            )
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            new_score = float(score) + float(action_score)
            normalized = new_score / max(1.0, float(len(new_chosen)) ** 0.65)
            seq += 1
            heapq.heappush(heap, (-normalized, seq, next_idx, tuple(nxt_counts), nxt_used, new_chosen, new_score))
            stats.generated_states += 1
        if len(heap) > int(beam_size) * 20:
            heap = heapq.nsmallest(int(beam_size) * 10, heap)
            heapq.heapify(heap)
    stats.elapsed_s = time.monotonic() - start
    if heap and len(candidates) >= int(top_k):
        stats.truncated = True
        if stats.reason is None:
            stats.reason = "top_k"
    return candidates, stats


def rerank_and_trim_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = sorted(candidates, key=lambda c: (-float(c.get("score", 0.0)), str(c.get("canonical_wa_key"))))
    return scored[: int(top_k)]


def process_record(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, record = payload
    if _ENGINE is None or _MODEL is None or _VOCAB is None or _PRIORS is None:
        raise RuntimeError("worker not initialized")
    args = _ARGS
    raw_top_k = int(args["top_k"]) * int(args.get("candidate_multiplier", 1))
    candidates, stats = policy_guided_exact_cover_search(
        int(record["sg"]),
        {str(k): int(v) for k, v in record["formula_counts"].items()},
        _ENGINE.get_orbits(int(record["sg"])),
        model=_MODEL,
        vocab=_VOCAB,
        priors=_PRIORS,
        beam_size=int(args["beam_size"]),
        top_k=raw_top_k,
        max_expanded_states=int(args["max_expanded_states"]),
        timeout_s=float(args["timeout_per_sample"]),
        branching_factor=int(args["branching_factor"]),
        policy_weight=float(args["policy_weight"]),
        prior_weight=float(args["prior_weight"]),
    )
    candidates = rerank_and_trim_candidates(candidates, int(args["top_k"]))
    gt_skeleton = str(record["canonical_skeleton_key"])
    gt_wa = str(record["canonical_wa_key"])
    summary: dict[str, Any] = {
        "input_index": index,
        "split": record["split"],
        "sample_id": record["sample_id"],
        "formula": record["formula"],
        "formula_counts": record["formula_counts"],
        "sg": int(record["sg"]),
        "n_sites": int(record["n_sites"]),
        "num_elements": int(record["num_elements"]),
        "total_atoms": int(record["atom_count"]),
        "candidate_count": len(candidates),
        "candidate_nonempty": bool(candidates),
        "expanded_states": int(stats.expanded_states),
        "generated_states": int(stats.generated_states),
        "complete_states": int(stats.complete_states),
        "timeout": bool(stats.timeout),
        "truncated": bool(stats.truncated),
        "elapsed_s": float(stats.elapsed_s),
        "reason": stats.reason,
        "gt_skeleton_key": gt_skeleton,
        "gt_wa_key": gt_wa,
    }
    for k in TOP_KS:
        summary[f"gt_skeleton_in_top{k}"] = hit_at(candidates, gt_skeleton, "canonical_skeleton_key", k)
        summary[f"gt_wa_in_top{k}"] = hit_at(candidates, gt_wa, "canonical_wa_key", k)
    return {
        "input_index": index,
        "candidate_row": {
            "split": record["split"],
            "sample_id": record["sample_id"],
            "formula": record["formula"],
            "formula_counts": record["formula_counts"],
            "sg": int(record["sg"]),
            "n_sites": int(record["n_sites"]),
            "num_elements": int(record["num_elements"]),
            "gt_skeleton_key": gt_skeleton,
            "gt_wa_key": gt_wa,
            "candidates": candidates,
        },
        "summary": summary,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "samples": len(rows),
        "candidate_nonempty_rate": sum(bool(r["candidate_nonempty"]) for r in rows) / max(1, len(rows)),
        "timeout_samples": sum(bool(r["timeout"]) for r in rows),
        "truncated_samples": sum(bool(r["truncated"]) for r in rows),
        "candidate_count": q([r["candidate_count"] for r in rows]),
        "expanded_states": q([r["expanded_states"] for r in rows]),
        "elapsed_s": q([r["elapsed_s"] for r in rows]),
    }
    for k in TOP_KS:
        out[f"gt_skeleton_in_top{k}"] = sum(bool(r[f"gt_skeleton_in_top{k}"]) for r in rows) / max(1, len(rows))
        out[f"gt_wa_in_top{k}"] = sum(bool(r[f"gt_wa_in_top{k}"]) for r in rows) / max(1, len(rows))
    out["gate2_search"] = {
        "skeleton_top200_pass": out["gt_skeleton_in_top200"] >= 0.99,
        "wa_top200_pass": out["gt_wa_in_top200"] >= 0.90,
        "timeout_pass": out["timeout_samples"] == 0,
    }
    return out


def group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        s = summarize(items)
        item: dict[str, Any] = {key: value, "samples": len(items)}
        for name in (
            "candidate_nonempty_rate",
            "gt_skeleton_in_top20",
            "gt_skeleton_in_top100",
            "gt_skeleton_in_top200",
            "gt_skeleton_in_top700",
            "gt_wa_in_top20",
            "gt_wa_in_top100",
            "gt_wa_in_top200",
            "gt_wa_in_top700",
            "timeout_samples",
            "truncated_samples",
        ):
            item[name] = s[name]
        item["candidate_count_mean"] = s["candidate_count"]["mean"]
        item["expanded_states_median"] = s["expanded_states"]["median"]
        out.append(item)
    return out


def merge_summary(out_dir: Path, split: str, summary: dict[str, Any]) -> None:
    path = out_dir / "policy_search_summary.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = {"splits": {}}
    raw.setdefault("splits", {})[split] = summary
    path.write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Policy-guided streaming composition-exact SymCIF-v4 WA search.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--policy-ckpt", type=Path, default=PROJECT_ROOT / "runs" / "step_policy_wa_v1" / "ckpt.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--beam-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--max-expanded-states", type=int, default=500_000)
    parser.add_argument("--timeout-per-sample", type=float, default=30.0)
    parser.add_argument("--branching-factor", type=int, default=64)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=max(1, min(48, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_policy_search")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.data_root / f"{args.split}.jsonl")
    if args.max_records is not None:
        rows = rows[: int(args.max_records)]
    payloads = list(enumerate(rows))
    args_dict = {
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "candidate_multiplier": args.candidate_multiplier,
        "max_expanded_states": args.max_expanded_states,
        "timeout_per_sample": args.timeout_per_sample,
        "branching_factor": args.branching_factor,
        "policy_weight": args.policy_weight,
        "prior_weight": args.prior_weight,
    }
    if args.workers <= 1:
        init_worker(str(args.lookup_json), str(args.data_root), str(args.policy_ckpt), args_dict)
        results = [process_record(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(str(args.lookup_json), str(args.data_root), str(args.policy_ckpt), args_dict),
        ) as pool:
            results = list(pool.imap_unordered(process_record, payloads, chunksize=4))
    results.sort(key=lambda r: int(r["input_index"]))
    candidate_rows = [r["candidate_row"] for r in results]
    summary_rows = [r["summary"] for r in results]
    write_jsonl(args.out_dir / f"{args.split}_policy_candidates.jsonl", candidate_rows)
    write_jsonl(args.out_dir / f"{args.split}_policy_per_sample.jsonl", summary_rows)
    split_summary = {
        **summarize(summary_rows),
        "config": {
            "data_root": str(args.data_root),
            "lookup_json": str(args.lookup_json),
            "policy_ckpt": str(args.policy_ckpt),
            "split": args.split,
            "beam_size": args.beam_size,
            "top_k": args.top_k,
            "candidate_multiplier": args.candidate_multiplier,
            "max_expanded_states": args.max_expanded_states,
            "timeout_per_sample": args.timeout_per_sample,
            "branching_factor": args.branching_factor,
            "policy_weight": args.policy_weight,
            "prior_weight": args.prior_weight,
            "workers": args.workers,
            "max_records": args.max_records,
        },
    }
    (args.out_dir / f"policy_search_summary_{args.split}.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    merge_summary(args.out_dir, args.split, split_summary)
    write_csv(args.out_dir / f"policy_search_breakdown_per_sg_{args.split}.csv", group_rows(summary_rows, "sg"))
    write_csv(args.out_dir / f"policy_search_breakdown_per_nsites_{args.split}.csv", group_rows(summary_rows, "n_sites"))
    write_csv(args.out_dir / f"policy_search_breakdown_per_num_elements_{args.split}.csv", group_rows(summary_rows, "num_elements"))
    if args.split == "test":
        write_csv(args.out_dir / "policy_search_breakdown_per_sg.csv", group_rows(summary_rows, "sg"))
        write_csv(args.out_dir / "policy_search_breakdown_per_nsites.csv", group_rows(summary_rows, "n_sites"))
        write_csv(args.out_dir / "policy_search_breakdown_per_num_elements.csv", group_rows(summary_rows, "num_elements"))
    print(json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
