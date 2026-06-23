from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .formula import total_atoms
from .wa_table import WATableCandidate, assignment_key, skeleton_key, wa_key
from .wyckoff_table import WyckoffSiteToken


@dataclass
class SkeletonCandidate:
    sg: int
    sites: list[WyckoffSiteToken]
    multiplicities: list[int]
    skeleton_key: str
    total_atoms: int
    truncated: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "sg": self.sg,
            "sites": [site.to_jsonable() for site in self.sites],
            "multiplicities": self.multiplicities,
            "skeleton_key": self.skeleton_key,
            "total_atoms": self.total_atoms,
            "truncated": self.truncated,
        }


@dataclass
class EnumerationStats:
    truncated: bool = False
    timeout: bool = False
    visited_states: int = 0
    elapsed_s: float = 0.0
    reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def _suffix_reachable(tokens: list[WyckoffSiteToken], target_total: int) -> list[set[int]]:
    reachable: list[set[int]] = [set() for _ in range(len(tokens) + 1)]
    reachable[len(tokens)].add(0)
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        max_repeat = token.max_repeat if token.is_fully_fixed else target_total // token.multiplicity
        vals: set[int] = set()
        for base in reachable[i + 1]:
            for n in range(max_repeat + 1):
                value = base + n * token.multiplicity
                if value <= target_total:
                    vals.add(value)
        reachable[i] = vals
    return reachable


def enumerate_skeleton_candidates(
    target_counts: dict[str, int],
    sg: int,
    wyckoff_sites: list[WyckoffSiteToken],
    max_candidates: int | None = None,
    timeout_s: float | None = None,
) -> tuple[list[SkeletonCandidate], EnumerationStats]:
    start = time.monotonic()
    total = total_atoms(target_counts)
    tokens = sorted(wyckoff_sites, key=lambda t: (t.multiplicity, t.letter, str(t.enumeration), t.site_symmetry))
    suffix = _suffix_reachable(tokens, total)
    out: list[SkeletonCandidate] = []
    stats = EnumerationStats(debug={"target_total": total, "num_wyckoff_tokens": len(tokens)})
    counts_tuple = tuple(sorted((int(v) for v in target_counts.values()), reverse=True))

    def timed_out() -> bool:
        return timeout_s is not None and (time.monotonic() - start) > timeout_s

    def rec(pos: int, remaining_total: int, chosen: list[WyckoffSiteToken]) -> None:
        if stats.truncated or stats.timeout:
            return
        stats.visited_states += 1
        if timed_out():
            stats.timeout = True
            stats.reason = "timeout"
            return
        if remaining_total < 0:
            return
        if remaining_total not in suffix[pos]:
            return
        if pos == len(tokens):
            if remaining_total != 0:
                return
            mults = tuple(sorted((site.multiplicity for site in chosen)))
            if not _multiplicities_compatible_cached(mults, counts_tuple):
                return
            sites = sorted(chosen, key=lambda t: (t.multiplicity, t.letter, str(t.enumeration), t.site_symmetry))
            out.append(
                SkeletonCandidate(
                    sg=int(sg),
                    sites=sites,
                    multiplicities=[site.multiplicity for site in sites],
                    skeleton_key=skeleton_key(sites),
                    total_atoms=total,
                )
            )
            if max_candidates is not None and len(out) >= max_candidates:
                stats.truncated = True
                stats.reason = "max_candidates"
            return
        token = tokens[pos]
        max_repeat = token.max_repeat if token.is_fully_fixed else remaining_total // token.multiplicity
        for n in range(max_repeat + 1):
            next_total = remaining_total - n * token.multiplicity
            if next_total < 0:
                break
            if next_total not in suffix[pos + 1]:
                continue
            if n:
                chosen.extend([token] * n)
            rec(pos + 1, next_total, chosen)
            if n:
                del chosen[-n:]
            if stats.truncated or stats.timeout:
                break

    rec(0, total, [])
    stats.elapsed_s = time.monotonic() - start
    unique: dict[str, SkeletonCandidate] = {}
    for cand in out:
        unique.setdefault(cand.skeleton_key, cand)
    return list(unique.values()), stats


@lru_cache(maxsize=250_000)
def _multiplicities_compatible_cached(multiplicities: tuple[int, ...], counts: tuple[int, ...]) -> bool:
    remaining = tuple(sorted(int(v) for v in multiplicities))
    targets = tuple(sorted((int(c) for c in counts if int(c) > 0), reverse=True))
    if sum(remaining) != sum(targets):
        return False
    if not targets:
        return not remaining

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
                val = rem[i]
                if prev == val:
                    continue
                prev = val
                chosen.append(i)
                rec(i + 1, total + val, chosen)
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

    return rec(remaining, 0)


def multiplicities_compatible(multiplicities: list[int], counts: dict[str, int]) -> bool:
    return _multiplicities_compatible_cached(
        tuple(sorted(int(v) for v in multiplicities)),
        tuple(sorted((int(v) for v in counts.values()), reverse=True)),
    )


def enumerate_wa_tables_for_skeleton(
    target_counts: dict[str, int],
    skeleton: SkeletonCandidate,
    max_assignments: int | None = None,
    timeout_s: float | None = None,
) -> tuple[list[WATableCandidate], EnumerationStats]:
    start = time.monotonic()
    out: list[WATableCandidate] = []
    stats = EnumerationStats()
    sites = list(skeleton.sites)

    def subtract(remaining: dict[str, int], element: str, mult: int) -> dict[str, int]:
        nxt = dict(remaining)
        nxt[element] = int(nxt[element]) - int(mult)
        if nxt[element] == 0:
            nxt.pop(element)
        return nxt

    def rec(pos: int, remaining: dict[str, int], rows: list[tuple[str, WyckoffSiteToken]]) -> None:
        if stats.truncated or stats.timeout:
            return
        if timeout_s is not None and (time.monotonic() - start) > timeout_s:
            stats.timeout = True
            stats.reason = "timeout"
            return
        stats.visited_states += 1
        if pos == len(sites):
            if remaining:
                return
            skey = skeleton.skeleton_key
            akey = assignment_key(rows)
            out.append(
                WATableCandidate(
                    sg=skeleton.sg,
                    formula_counts=dict(target_counts),
                    rows=list(rows),
                    skeleton_key=skey,
                    assignment_key=akey,
                    wa_key=wa_key(rows, skeleton.sg),
                    source="exhaustive",
                )
            )
            if max_assignments is not None and len(out) >= max_assignments:
                stats.truncated = True
                stats.reason = "max_assignments"
            return
        site = sites[pos]
        rest_mults = [s.multiplicity for s in sites[pos + 1 :]]
        for element, count in sorted(remaining.items()):
            if int(count) < site.multiplicity:
                continue
            nxt = subtract(remaining, element, site.multiplicity)
            if not multiplicities_compatible(rest_mults, nxt):
                continue
            rows.append((element, site))
            rec(pos + 1, nxt, rows)
            rows.pop()
            if stats.truncated or stats.timeout:
                break

    rec(0, dict(target_counts), [])
    stats.elapsed_s = time.monotonic() - start
    return out, stats
