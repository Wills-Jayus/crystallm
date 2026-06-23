from __future__ import annotations

from collections import Counter
from typing import Any

from .canonicalize import canonical_wa_key


class OrbitFrequencyScorer:
    def __init__(self, row_counts: Counter[str], wa_counts: Counter[str]):
        self.row_counts = row_counts
        self.wa_counts = wa_counts

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "OrbitFrequencyScorer":
        row_counts: Counter[str] = Counter()
        wa_counts: Counter[str] = Counter()
        for record in records:
            wa_counts[canonical_wa_key(list(record.get("wa_table") or []))] += 1
            for row in record["wa_table"]:
                row_counts[f"{record['sg']}|{row['element']}@{row['orbit_id']}"] += 1
        return cls(row_counts, wa_counts)

    def score(self, record: dict[str, Any], candidate: dict[str, Any]) -> float:
        score = float(self.wa_counts.get(str(candidate.get("canonical_wa_key")), 0))
        for row in candidate.get("rows", []):
            score += 0.1 * float(self.row_counts.get(f"{record['sg']}|{row['element']}@{row['orbit_id']}", 0))
        return score
