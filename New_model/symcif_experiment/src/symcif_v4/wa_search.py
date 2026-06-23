from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .canonicalize import canonical_skeleton_key, canonical_wa_key
from .orbit_token import OrbitToken


@dataclass
class SearchState:
    sg: int
    remaining_counts: dict[str, int]
    chosen_rows: list[tuple[str, OrbitToken]] = field(default_factory=list)
    used_fixed_orbits: set[str] = field(default_factory=set)
    remaining_total_atoms: int = 0
    score: float = 0.0


@dataclass(frozen=True)
class SearchAction:
    element: str
    orbit: OrbitToken
    score: float
    action_id: str


@dataclass
class StreamingSearchStats:
    expanded_states: int = 0
    generated_states: int = 0
    complete_states: int = 0
    timeout: bool = False
    truncated: bool = False
    elapsed_s: float = 0.0
    reason: str | None = None


def candidate_key(rows: list[tuple[str, OrbitToken]]) -> tuple[tuple[str, str], ...]:
    return tuple((element, orbit.canonical_orbit_id) for element, orbit in rows)


def _canonical_row_sort_key(row: tuple[str, OrbitToken]) -> tuple[Any, ...]:
    element, orbit = row
    return (
        int(orbit.multiplicity),
        str(orbit.letter),
        str(orbit.enumeration),
        str(orbit.site_symmetry),
        str(element),
        str(orbit.canonical_orbit_id),
    )


def is_feasible(remaining_counts: dict[str, int], available_orbits: list[OrbitToken], used_fixed_orbits: set[str] | None = None) -> bool:
    used_fixed_orbits = used_fixed_orbits or set()
    counts = tuple(sorted(int(v) for v in remaining_counts.values() if int(v) > 0))
    if not counts:
        return True
    multiplicities: list[int] = []
    for orbit in available_orbits:
        if orbit.is_fully_fixed and orbit.canonical_orbit_id in used_fixed_orbits:
            continue
        multiplicities.append(int(orbit.multiplicity))
    targets = sorted(counts, reverse=True)

    from functools import lru_cache

    mults = tuple(sorted(multiplicities))

    @lru_cache(maxsize=None)
    def subset_remainders(rem: tuple[int, ...], target: int) -> tuple[tuple[int, ...], ...]:
        out: set[tuple[int, ...]] = set()
        n = len(rem)

        def rec(pos: int, total: int, chosen: list[int]) -> None:
            if total == target:
                chosen_set = set(chosen)
                out.add(tuple(rem[i] for i in range(n) if i not in chosen_set))
                return
            if total > target or pos >= n:
                return
            prev: int | None = None
            for i in range(pos, n):
                value = rem[i]
                if prev == value:
                    continue
                prev = value
                chosen.append(i)
                rec(i + 1, total + value, chosen)
                chosen.pop()

        rec(0, 0, [])
        return tuple(sorted(out))

    @lru_cache(maxsize=None)
    def rec(rem: tuple[int, ...], idx: int) -> bool:
        if idx == len(targets):
            return not rem
        for nxt in subset_remainders(rem, targets[idx]):
            if rec(nxt, idx + 1):
                return True
        return False

    return rec(mults, 0)


def rows_to_candidate(sg: int, formula_counts: dict[str, int], rows: list[tuple[str, OrbitToken]], score: float = 0.0) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=_canonical_row_sort_key)
    return {
        "sg": int(sg),
        "formula_counts": formula_counts,
        "score": float(score),
        "rows": [
            {
                "element": element,
                "orbit_id": orbit.canonical_orbit_id,
                "letter": orbit.letter,
                "multiplicity": orbit.multiplicity,
                "site_symmetry": orbit.site_symmetry,
                "enumeration": orbit.enumeration,
            }
            for element, orbit in ordered_rows
        ],
        "canonical_skeleton_key": "|".join(orbit.canonical_orbit_id for _, orbit in ordered_rows),
        "canonical_wa_key": "|".join(f"{orbit.canonical_orbit_id}:{element}" for element, orbit in ordered_rows),
    }


def build_search_priors(train_records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    action_counts: Counter[str] = Counter()
    orbit_counts: Counter[str] = Counter()
    element_mult_counts: Counter[str] = Counter()
    skeleton_counts: Counter[str] = Counter()
    wa_counts: Counter[str] = Counter()
    for record in train_records:
        sg = int(record["sg"])
        wa_table = list(record.get("wa_table") or [])
        skeleton_counts[canonical_skeleton_key(wa_table)] += 1
        wa_counts[canonical_wa_key(wa_table)] += 1
        for row in record["wa_table"]:
            orbit_id = str(row["orbit_id"])
            element = str(row["element"])
            mult = int(row["multiplicity"])
            action_counts[f"{sg}|{element}|{orbit_id}"] += 1
            orbit_counts[f"{sg}|{orbit_id}"] += 1
            element_mult_counts[f"{sg}|{element}|{mult}"] += 1
    return {
        "action_counts": action_counts,
        "orbit_counts": orbit_counts,
        "element_mult_counts": element_mult_counts,
        "skeleton_counts": skeleton_counts,
        "wa_counts": wa_counts,
    }


def score_action(sg: int, element: str, orbit: OrbitToken, priors: dict[str, Counter[str]] | None = None) -> float:
    priors = priors or {}
    action_counts = priors.get("action_counts", Counter())
    orbit_counts = priors.get("orbit_counts", Counter())
    element_mult_counts = priors.get("element_mult_counts", Counter())
    action = action_counts.get(f"{int(sg)}|{element}|{orbit.canonical_orbit_id}", 0)
    orbit_count = orbit_counts.get(f"{int(sg)}|{orbit.canonical_orbit_id}", 0)
    em = element_mult_counts.get(f"{int(sg)}|{element}|{int(orbit.multiplicity)}", 0)
    fixed_bonus = 0.1 if orbit.is_fully_fixed else 0.0
    dof_penalty = 0.03 * len(orbit.free_symbols)
    return 5.0 * math.log1p(action) + 1.2 * math.log1p(orbit_count) + 1.0 * math.log1p(em) + fixed_bonus - dof_penalty


def make_actions(
    sg: int,
    formula_counts: dict[str, int],
    orbits: list[OrbitToken],
    priors: dict[str, Counter[str]] | None = None,
) -> list[SearchAction]:
    actions: list[SearchAction] = []
    for element, count in sorted(formula_counts.items()):
        for orbit in orbits:
            if int(orbit.sg) != int(sg):
                continue
            if int(count) < int(orbit.multiplicity):
                continue
            score = score_action(int(sg), str(element), orbit, priors)
            actions.append(
                SearchAction(
                    element=str(element),
                    orbit=orbit,
                    score=score,
                    action_id=f"{element}@{orbit.canonical_orbit_id}",
                )
            )
    actions.sort(
        key=lambda a: (
            -float(a.score),
            int(a.orbit.multiplicity),
            str(a.orbit.letter),
            str(a.element),
            str(a.orbit.canonical_orbit_id),
        )
    )
    return actions


def streaming_exact_cover_search(
    sg: int,
    formula_counts: dict[str, int],
    orbits: list[OrbitToken],
    *,
    priors: dict[str, Counter[str]] | None = None,
    beam_size: int = 256,
    top_k: int = 200,
    max_expanded_states: int = 200_000,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], StreamingSearchStats]:
    start = time.monotonic()
    elements = tuple(sorted(str(k) for k in formula_counts))
    initial_counts = tuple(int(formula_counts[e]) for e in elements)
    actions = make_actions(int(sg), {e: int(formula_counts[e]) for e in elements}, orbits, priors)
    stats = StreamingSearchStats()
    if not actions and any(v > 0 for v in initial_counts):
        stats.reason = "no_legal_actions"
        stats.elapsed_s = time.monotonic() - start
        return [], stats

    from functools import lru_cache

    @lru_cache(maxsize=500_000)
    def feasible(action_idx: int, counts: tuple[int, ...]) -> bool:
        if all(v == 0 for v in counts):
            return True
        if action_idx >= len(actions):
            return False
        action = actions[action_idx]
        elem_idx = elements.index(action.element)
        mult = int(action.orbit.multiplicity)
        max_take = counts[elem_idx] // mult
        if action.orbit.is_fully_fixed:
            max_take = min(max_take, 1)
        # Try high repeats first for free orbits; it tightens feasibility for
        # large composition counts without changing exactness.
        for n in range(max_take, -1, -1):
            nxt = list(counts)
            nxt[elem_idx] -= n * mult
            if feasible(action_idx + 1, tuple(nxt)):
                return True
        return False

    if not feasible(0, initial_counts):
        stats.reason = "not_exact_cover_feasible"
        stats.elapsed_s = time.monotonic() - start
        return [], stats

    # Heap item: optimistic negative priority, tie id, action_idx, counts,
    # chosen rows, actual score. The optimistic term favors states that already
    # accumulated high-prior actions while still allowing skip branches.
    heap: list[tuple[float, int, int, tuple[int, ...], tuple[tuple[str, OrbitToken], ...], float]] = []
    seq = 0
    heapq.heappush(heap, (0.0, seq, 0, initial_counts, tuple(), 0.0))
    seen_complete: set[str] = set()
    candidates: list[dict[str, Any]] = []
    while heap and len(candidates) < int(top_k):
        if time.monotonic() - start > float(timeout_s):
            stats.timeout = True
            stats.reason = "timeout"
            break
        if stats.expanded_states >= int(max_expanded_states):
            stats.truncated = True
            stats.reason = "max_expanded_states"
            break
        _priority, _seq, action_idx, counts, chosen_tuple, score = heapq.heappop(heap)
        stats.expanded_states += 1
        if all(v == 0 for v in counts):
            rows = list(chosen_tuple)
            cand = rows_to_candidate(int(sg), dict(formula_counts), rows, score=score)
            key = str(cand["canonical_wa_key"])
            if key not in seen_complete:
                seen_complete.add(key)
                candidates.append(cand)
                stats.complete_states += 1
            continue
        if action_idx >= len(actions):
            continue
        action = actions[action_idx]
        elem_idx = elements.index(action.element)
        mult = int(action.orbit.multiplicity)

        # Skip current action.
        skip_idx = action_idx + 1
        if feasible(skip_idx, counts):
            seq += 1
            heapq.heappush(heap, (-score, seq, skip_idx, counts, chosen_tuple, score))
            stats.generated_states += 1

        # Take current action once. Free orbits remain available; fixed orbits
        # advance past the current action.
        if counts[elem_idx] >= mult:
            nxt_counts = list(counts)
            nxt_counts[elem_idx] -= mult
            nxt_counts_t = tuple(nxt_counts)
            next_idx = action_idx + 1 if action.orbit.is_fully_fixed else action_idx
            if feasible(next_idx, nxt_counts_t):
                new_score = float(score) + float(action.score)
                seq += 1
                heapq.heappush(
                    heap,
                    (-new_score, seq, next_idx, nxt_counts_t, chosen_tuple + ((action.element, action.orbit),), new_score),
                )
                stats.generated_states += 1
        # Keep the frontier bounded. This is beam-like but still streaming:
        # candidates are never materialized beyond the active frontier.
        if len(heap) > int(beam_size) * 20:
            heap = heapq.nsmallest(int(beam_size) * 10, heap)
            heapq.heapify(heap)
    stats.elapsed_s = time.monotonic() - start
    if heap and len(candidates) >= int(top_k):
        stats.truncated = True
        if stats.reason is None:
            stats.reason = "top_k"
    return candidates, stats
