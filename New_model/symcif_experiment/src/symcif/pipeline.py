from __future__ import annotations

import json
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .extract import extract_record_from_cif, sample_id_from_path
from .lookup import WyckoffLookup
from .render import (
    render_baseline,
    render_baseline_minprompt,
    render_cf_like,
    render_standard_cif,
    render_symcif_v1,
    render_symcif_v1_atomprops,
)
from .validate import validate_roundtrip

FormatName = Literal["baseline", "cf_like", "symcif_v1", "baseline_minprompt", "symcif_v1_atomprops"]


REPORT_KEYS = [
    "total",
    "parse_success",
    "standardize_success",
    "wyckoff_extract_success",
    "site_symmetry_extract_success",
    "enumeration_success",
    "render_success",
    "roundtrip_cif_success",
    "pymatgen_readable",
    "formula_consistent",
    "space_group_consistent",
    "multiplicity_consistent",
]


@dataclass
class ConversionOutput:
    ok: bool
    sample_id: str
    source_path: Path
    text: str | None = None
    roundtrip_cif: str | None = None
    validation: dict[str, object] = field(default_factory=dict)
    stage: str | None = None
    reason: str | None = None
    traceback_text: str | None = None


class Report:
    def __init__(self) -> None:
        self.counts = Counter({k: 0 for k in REPORT_KEYS})
        self.failure_reasons: Counter[str] = Counter()

    def add_success(self, validation: dict[str, object] | None = None, *, has_enum: bool = False) -> None:
        self.counts["parse_success"] += 1
        self.counts["standardize_success"] += 1
        self.counts["wyckoff_extract_success"] += 1
        self.counts["site_symmetry_extract_success"] += 1
        if has_enum:
            self.counts["enumeration_success"] += 1
        self.counts["render_success"] += 1
        self.counts["roundtrip_cif_success"] += 1
        if validation:
            for key in (
                "pymatgen_readable",
                "formula_consistent",
                "space_group_consistent",
                "multiplicity_consistent",
            ):
                if validation.get(key):
                    self.counts[key] += 1

    def add_failure(self, stage: str, reason: str) -> None:
        self.failure_reasons[f"{stage}:{reason}"] += 1

    def as_dict(self) -> dict[str, object]:
        out = {k: int(self.counts[k]) for k in REPORT_KEYS}
        out["top_failure_reasons"] = dict(self.failure_reasons.most_common(25))
        return out


def load_or_build_lookup(
    source_csv: str | Path,
    artifact_json: str | Path,
    *,
    wyformer_json: str | Path | None = None,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
    rebuild: bool = False,
) -> WyckoffLookup:
    artifact_json = Path(artifact_json)
    if artifact_json.exists() and not rebuild:
        return WyckoffLookup.from_json(artifact_json)
    lookup = WyckoffLookup.from_crystalformer_csv(
        source_csv,
        wyformer_json=wyformer_json,
        infer_site_symmetry=True,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    lookup.write_json(artifact_json)
    return lookup


def convert_one(
    cif_path: str | Path,
    fmt: FormatName,
    lookup: WyckoffLookup,
    *,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
    roundtrip: bool = True,
) -> ConversionOutput:
    cif_path = Path(cif_path)
    sample_id = sample_id_from_path(cif_path)
    try:
        if fmt in {"baseline", "baseline_minprompt"}:
            raw_text = cif_path.read_text(encoding="utf-8", errors="replace")
            text = render_baseline(raw_text) if fmt == "baseline" else render_baseline_minprompt(raw_text)
            return ConversionOutput(ok=True, sample_id=sample_id, source_path=cif_path, text=text)

        record = extract_record_from_cif(
            cif_path,
            lookup,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            standardize=True,
        )
        if fmt == "cf_like":
            text = render_cf_like(record)
        elif fmt == "symcif_v1_atomprops":
            text = render_symcif_v1_atomprops(record)
        else:
            text = render_symcif_v1(record)
        rt_cif = None
        validation: dict[str, object] = {}
        if roundtrip:
            rt_cif = render_standard_cif(record, symprec=symprec)
            validation = validate_roundtrip(record, rt_cif, symprec=symprec, angle_tolerance=angle_tolerance)
        return ConversionOutput(
            ok=True,
            sample_id=record.sample_id,
            source_path=cif_path,
            text=text,
            roundtrip_cif=rt_cif,
            validation=validation,
        )
    except Exception as exc:  # noqa: BLE001
        return ConversionOutput(
            ok=False,
            sample_id=sample_id,
            source_path=cif_path,
            stage="convert",
            reason=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
