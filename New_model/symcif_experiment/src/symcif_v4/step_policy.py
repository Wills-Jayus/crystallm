from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .orbit_token import OrbitToken


NUMERIC_DIM = 18


@dataclass
class StepPolicyVocab:
    element_to_id: dict[str, int]
    orbit_to_id: dict[str, int]
    sg_to_id: dict[str, int]
    letter_to_id: dict[str, int]
    site_sym_to_id: dict[str, int]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "StepPolicyVocab":
        elements = sorted({str(e) for r in records for e in r["formula_counts"]})
        orbits = sorted({str(w["orbit_id"]) for r in records for w in r["wa_table"]})
        sgs = sorted({str(int(r["sg"])) for r in records})
        letters = sorted({str(w["letter"]) for r in records for w in r["wa_table"]})
        site_syms = sorted({str(w.get("site_symmetry") or "UNKNOWN") for r in records for w in r["wa_table"]})
        return cls(
            element_to_id={v: i + 1 for i, v in enumerate(elements)},
            orbit_to_id={v: i + 1 for i, v in enumerate(orbits)},
            sg_to_id={v: i + 1 for i, v in enumerate(sgs)},
            letter_to_id={v: i + 1 for i, v in enumerate(letters)},
            site_sym_to_id={v: i + 1 for i, v in enumerate(site_syms)},
        )

    def to_jsonable(self) -> dict[str, dict[str, int]]:
        return {
            "element_to_id": self.element_to_id,
            "orbit_to_id": self.orbit_to_id,
            "sg_to_id": self.sg_to_id,
            "letter_to_id": self.letter_to_id,
            "site_sym_to_id": self.site_sym_to_id,
        }

    @classmethod
    def from_jsonable(cls, raw: dict[str, Any]) -> "StepPolicyVocab":
        return cls(
            element_to_id={str(k): int(v) for k, v in raw["element_to_id"].items()},
            orbit_to_id={str(k): int(v) for k, v in raw["orbit_to_id"].items()},
            sg_to_id={str(k): int(v) for k, v in raw["sg_to_id"].items()},
            letter_to_id={str(k): int(v) for k, v in raw["letter_to_id"].items()},
            site_sym_to_id={str(k): int(v) for k, v in raw["site_sym_to_id"].items()},
        )


class StepPolicyNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], numeric_dim: int = NUMERIC_DIM, emb_dim: int = 64, hidden_dim: int = 192):
        super().__init__()
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 4)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 4)
        in_dim = emb_dim * 2 + emb_dim // 2 + emb_dim // 4 + emb_dim // 4 + numeric_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        element_id: torch.Tensor,
        orbit_id: torch.Tensor,
        sg_id: torch.Tensor,
        letter_id: torch.Tensor,
        site_sym_id: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.element_emb(element_id),
                self.orbit_emb(orbit_id),
                self.sg_emb(sg_id),
                self.letter_emb(letter_id),
                self.site_sym_emb(site_sym_id),
                numeric,
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def vocab_sizes(vocab: StepPolicyVocab) -> dict[str, int]:
    return {
        "element": max(vocab.element_to_id.values(), default=0),
        "orbit": max(vocab.orbit_to_id.values(), default=0),
        "sg": max(vocab.sg_to_id.values(), default=0),
        "letter": max(vocab.letter_to_id.values(), default=0),
        "site_sym": max(vocab.site_sym_to_id.values(), default=0),
    }


def build_frequency_priors(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    priors: dict[str, Counter[str]] = {
        "action": Counter(),
        "orbit": Counter(),
        "element_mult": Counter(),
        "element": Counter(),
        "site_sym": Counter(),
        "letter": Counter(),
    }
    for record in records:
        sg = int(record["sg"])
        for row in record["wa_table"]:
            element = str(row["element"])
            orbit_id = str(row["orbit_id"])
            mult = int(row["multiplicity"])
            priors["action"][f"{sg}|{element}|{orbit_id}"] += 1
            priors["orbit"][f"{sg}|{orbit_id}"] += 1
            priors["element_mult"][f"{sg}|{element}|{mult}"] += 1
            priors["element"][element] += 1
            priors["site_sym"][str(row.get("site_symmetry") or "UNKNOWN")] += 1
            priors["letter"][f"{sg}|{row.get('letter')}"] += 1
    return priors


def encode_action_batch(
    *,
    sg: int,
    formula_counts: dict[str, int],
    remaining_counts: dict[str, int],
    element_orbits: list[tuple[str, OrbitToken]],
    vocab: StepPolicyVocab,
    priors: dict[str, Counter[str]] | None,
    step_index: int,
    chosen_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    priors = priors or build_frequency_priors([])
    total_atoms = max(1, sum(int(v) for v in formula_counts.values()))
    remaining_total = max(0, sum(int(v) for v in remaining_counts.values()))
    num_elements = max(1, len(formula_counts))
    max_count = max(int(v) for v in formula_counts.values()) if formula_counts else 0
    min_count = min(int(v) for v in formula_counts.values()) if formula_counts else 0
    element_ids: list[int] = []
    orbit_ids: list[int] = []
    sg_ids: list[int] = []
    letter_ids: list[int] = []
    site_sym_ids: list[int] = []
    numeric: list[list[float]] = []
    for element, orbit in element_orbits:
        mult = int(orbit.multiplicity)
        free_dof = len(orbit.free_symbols)
        action_count = priors.get("action", Counter()).get(f"{int(sg)}|{element}|{orbit.canonical_orbit_id}", 0)
        orbit_count = priors.get("orbit", Counter()).get(f"{int(sg)}|{orbit.canonical_orbit_id}", 0)
        em_count = priors.get("element_mult", Counter()).get(f"{int(sg)}|{element}|{mult}", 0)
        element_count = priors.get("element", Counter()).get(element, 0)
        site_sym_count = priors.get("site_sym", Counter()).get(str(orbit.site_symmetry), 0)
        letter_count = priors.get("letter", Counter()).get(f"{int(sg)}|{orbit.letter}", 0)
        remaining_for_element = int(remaining_counts.get(element, 0))
        original_for_element = int(formula_counts.get(element, 0))
        numeric.append(
            [
                float(int(sg)) / 230.0,
                float(total_atoms) / 256.0,
                float(remaining_total) / float(total_atoms),
                float(remaining_for_element) / float(total_atoms),
                float(original_for_element) / float(total_atoms),
                float(mult) / float(total_atoms),
                float(num_elements) / 10.0,
                float(step_index) / 64.0,
                float(chosen_count) / 64.0,
                float(free_dof) / 3.0,
                1.0 if orbit.is_fully_fixed else 0.0,
                math.log1p(action_count) / 8.0,
                math.log1p(orbit_count) / 8.0,
                math.log1p(em_count) / 8.0,
                math.log1p(element_count) / 10.0,
                math.log1p(site_sym_count) / 10.0,
                float(max_count) / float(total_atoms),
                float(min_count) / float(total_atoms),
            ]
        )
        element_ids.append(vocab.element_to_id.get(element, 0))
        orbit_ids.append(vocab.orbit_to_id.get(orbit.canonical_orbit_id, 0))
        sg_ids.append(vocab.sg_to_id.get(str(int(sg)), 0))
        letter_ids.append(vocab.letter_to_id.get(str(orbit.letter), 0))
        site_sym_ids.append(vocab.site_sym_to_id.get(str(orbit.site_symmetry), 0))
    return (
        torch.tensor(element_ids, dtype=torch.long),
        torch.tensor(orbit_ids, dtype=torch.long),
        torch.tensor(sg_ids, dtype=torch.long),
        torch.tensor(letter_ids, dtype=torch.long),
        torch.tensor(site_sym_ids, dtype=torch.long),
        torch.tensor(numeric, dtype=torch.float32),
    )


def canonical_sequence(record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        record["wa_table"],
        key=lambda row: (
            int(row["multiplicity"]),
            str(row["letter"]),
            str(row.get("enumeration")),
            str(row.get("site_symmetry")),
            str(row["element"]),
            str(row["orbit_id"]),
        ),
    )
