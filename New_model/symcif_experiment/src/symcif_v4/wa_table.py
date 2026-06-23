from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .wyckoff_table import WyckoffSiteToken


@dataclass
class WATableCandidate:
    sg: int
    formula_counts: dict[str, int]
    rows: list[tuple[str, WyckoffSiteToken]]
    skeleton_key: str
    assignment_key: str
    wa_key: str
    source: str
    score: float | None = None

    def to_jsonable(self, *, include_tokens: bool = True) -> dict[str, Any]:
        rows = []
        for element, token in self.rows:
            item: dict[str, Any] = {
                "element": element,
                "site": token.short_key,
                "letter": token.letter,
                "multiplicity": token.multiplicity,
                "site_symmetry": token.site_symmetry,
                "enumeration": token.enumeration,
                "canonical_key": token.canonical_key,
                "free_mask": list(token.free_mask),
            }
            if include_tokens:
                item["token"] = token.to_jsonable()
            rows.append(item)
        return {
            "sg": self.sg,
            "formula_counts": self.formula_counts,
            "rows": rows,
            "skeleton_key": self.skeleton_key,
            "assignment_key": self.assignment_key,
            "wa_key": self.wa_key,
            "source": self.source,
            "score": self.score,
        }


def skeleton_key(tokens: list[WyckoffSiteToken]) -> str:
    return "|".join(token.short_key for token in tokens)


def assignment_key(rows: list[tuple[str, WyckoffSiteToken]]) -> str:
    return "|".join(f"{token.short_key}:{element}" for element, token in rows)


def wa_key(rows: list[tuple[str, WyckoffSiteToken]], sg: int) -> str:
    return f"sg={int(sg)}|{assignment_key(rows)}"


def gt_skeleton_key(row: dict[str, Any]) -> str:
    return str(row["skeleton_template_key"])


def gt_assignment_key(row: dict[str, Any]) -> str:
    return "|".join(
        f"{int(site['multiplicity'])}{site['letter']}:{site['element']}"
        for site in row["assignment"]
    )


def gt_wa_key(row: dict[str, Any]) -> str:
    return f"sg={int(row['sg'])}|{gt_assignment_key(row)}"

