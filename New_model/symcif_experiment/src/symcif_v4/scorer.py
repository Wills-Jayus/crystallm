from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


def anonymous_formula_pattern(counts: dict[str, int]) -> str:
    return "-".join(str(v) for v in sorted((int(v) for v in counts.values()), reverse=True))


@dataclass
class BaselineScorer:
    sg_counts: Counter[int]
    skeleton_counts: Counter[str]
    mult_counts: Counter[str]
    site_counts: Counter[str]
    element_site_counts: Counter[str]
    anon_skeleton_counts: Counter[str]
    nsites_counts: Counter[int]

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "BaselineScorer":
        sg_counts: Counter[int] = Counter()
        skeleton_counts: Counter[str] = Counter()
        mult_counts: Counter[str] = Counter()
        site_counts: Counter[str] = Counter()
        element_site_counts: Counter[str] = Counter()
        anon_skeleton_counts: Counter[str] = Counter()
        nsites_counts: Counter[int] = Counter()
        for row in rows:
            sg = int(row["sg"])
            skel = str(row["skeleton_template_key"])
            mult = "|".join(str(int(v)) for v in row["multiplicities"])
            anon = anonymous_formula_pattern({str(k): int(v) for k, v in row["formula_counts"].items()})
            sg_counts[sg] += 1
            skeleton_counts[f"{sg}|{skel}"] += 1
            mult_counts[f"{sg}|{mult}"] += 1
            anon_skeleton_counts[f"{sg}|{anon}|{skel}"] += 1
            nsites_counts[int(row["n_sites"])] += 1
            for site in row["assignment"]:
                site_key = f"{sg}|{int(site['multiplicity'])}{site['letter']}"
                site_counts[site_key] += 1
                element_site_counts[f"{site_key}:{site['element']}"] += 1
        return cls(sg_counts, skeleton_counts, mult_counts, site_counts, element_site_counts, anon_skeleton_counts, nsites_counts)

    def score(self, sample: dict[str, Any], candidate: dict[str, Any]) -> float:
        sg = int(sample["sg"])
        skel = str(candidate["skeleton_key"])
        rows = candidate["rows"]
        mult = "|".join(str(int(row["multiplicity"])) for row in rows)
        anon = anonymous_formula_pattern({str(k): int(v) for k, v in sample["formula_counts"].items()})
        score = 0.0
        score += 1.5 * math.log1p(self.skeleton_counts[f"{sg}|{skel}"])
        score += 0.6 * math.log1p(self.mult_counts[f"{sg}|{mult}"])
        score += 0.8 * math.log1p(self.anon_skeleton_counts[f"{sg}|{anon}|{skel}"])
        score += 0.2 * math.log1p(self.nsites_counts[len(rows)])
        repeated_free = 0
        seen_sites: Counter[str] = Counter()
        for row in rows:
            site_key = f"{sg}|{int(row['multiplicity'])}{row['letter']}"
            seen_sites[site_key] += 1
            score += 0.15 * math.log1p(self.site_counts[site_key])
            score += 0.35 * math.log1p(self.element_site_counts[f"{site_key}:{row['element']}"])
        for key, count in seen_sites.items():
            if count > 1:
                repeated_free += count - 1
        score -= 0.05 * repeated_free
        score -= 0.01 * len(rows)
        return float(score)


def rank_candidates_baseline(scorer: BaselineScorer, sample: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for idx, cand in enumerate(candidates):
        item = dict(cand)
        item["score"] = scorer.score(sample, cand)
        item["rank_source"] = "baseline"
        item["_idx"] = idx
        ranked.append(item)
    ranked.sort(key=lambda row: (-float(row["score"]), row["_idx"]))
    for row in ranked:
        row.pop("_idx", None)
    return ranked

