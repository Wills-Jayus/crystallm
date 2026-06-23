from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from symcif.lookup import WyckoffLookup


@dataclass(frozen=True)
class WyckoffSiteToken:
    sg: int
    letter: str
    multiplicity: int
    site_symmetry: str
    enumeration: int | str | None
    free_mask: tuple[bool, bool, bool]
    fixed_values: tuple[float, float, float]
    representative_expr: str
    canonical_key: str
    is_fully_fixed: bool
    max_repeat: int = 1

    @property
    def short_key(self) -> str:
        return f"{self.multiplicity}{self.letter}"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "sg": self.sg,
            "letter": self.letter,
            "multiplicity": self.multiplicity,
            "site_symmetry": self.site_symmetry,
            "enumeration": self.enumeration,
            "free_mask": list(self.free_mask),
            "fixed_values": [float(v) for v in self.fixed_values],
            "representative_expr": self.representative_expr,
            "canonical_key": self.canonical_key,
            "is_fully_fixed": self.is_fully_fixed,
            "max_repeat": self.max_repeat,
        }


def canonical_site_id(
    sg: int,
    multiplicity: int,
    letter: str,
    enumeration: int | str | None,
    site_symmetry: str | None,
) -> str:
    enum = "None" if enumeration is None else str(enumeration)
    sym = "UNKNOWN" if not site_symmetry else str(site_symmetry)
    return f"sg={int(sg)}|{int(multiplicity)}{letter}|enum={enum}|sym={sym}"


def observed_repeat_limits(structured_rows: list[dict], lookup: WyckoffLookup) -> dict[tuple[int, str], int]:
    limits: dict[tuple[int, str], int] = {}
    for row in structured_rows:
        counts: dict[tuple[int, str], int] = {}
        for site in row["assignment"]:
            key = (int(row["sg"]), str(site["letter"]))
            counts[key] = counts.get(key, 0) + 1
        for key, value in counts.items():
            try:
                template = lookup.get(key[0], key[1])
            except Exception:
                continue
            if not any(template.free_mask):
                limits[key] = max(limits.get(key, 1), int(value))
    return limits


def wyckoff_tokens_for_sg(
    lookup: WyckoffLookup,
    sg: int,
    *,
    fixed_repeat_limits: dict[tuple[int, str], int] | None = None,
) -> list[WyckoffSiteToken]:
    tokens: list[WyckoffSiteToken] = []
    for (sg_number, letter), template in lookup.templates.items():
        if int(sg_number) != int(sg):
            continue
        site_symmetry = template.site_symmetry or "UNKNOWN"
        enum = template.enumeration
        is_fully_fixed = not any(bool(v) for v in template.free_mask)
        max_repeat = 1
        if is_fully_fixed and fixed_repeat_limits is not None:
            max_repeat = int(fixed_repeat_limits.get((int(sg), str(letter)), 1))
        token = WyckoffSiteToken(
            sg=int(sg),
            letter=str(letter),
            multiplicity=int(template.multiplicity),
            site_symmetry=site_symmetry,
            enumeration=enum,
            free_mask=tuple(bool(v) for v in template.free_mask),  # type: ignore[arg-type]
            fixed_values=tuple(float(v) for v in template.fixed_values),  # type: ignore[arg-type]
            representative_expr=str(template.representative_expr),
            canonical_key=canonical_site_id(int(sg), int(template.multiplicity), str(letter), enum, site_symmetry),
            is_fully_fixed=is_fully_fixed,
            max_repeat=max_repeat,
        )
        tokens.append(token)
    return sorted(tokens, key=lambda t: (t.multiplicity, t.letter, str(t.enumeration), t.site_symmetry))


def load_lookup(path: str | Path) -> WyckoffLookup:
    return WyckoffLookup.from_json(path)

