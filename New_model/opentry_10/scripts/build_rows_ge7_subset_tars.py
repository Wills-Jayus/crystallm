#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from build_mp20_k50_selector_inputs import atom_site_rows


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def read_member_text(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    f = tar.extractfile(member)
    if f is None:
        return ""
    return f.read().decode("utf-8", errors="replace")


def generated_material_id(member_name: str) -> str | None:
    name = Path(member_name).name
    if not name.endswith(".cif"):
        return None
    match = re.match(r"^(?P<material>.+)__[0-9]+\.cif$", name)
    return None if match is None else match.group("material")


def true_material_id(member_name: str) -> str | None:
    name = Path(member_name).name
    if not name.endswith(".cif"):
        return None
    return Path(name).stem


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def add_bytes(out: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    out.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rows>=7 official-test subset tars from target CIF atom-site rows.")
    parser.add_argument("--generated-tar", required=True)
    parser.add_argument("--true-tar", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-rows", type=int, default=7)
    args = parser.parse_args()

    out_dir = under_root(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_true = out_dir / "true_rows_ge7.tar.gz"
    out_generated = out_dir / "generated_rows_ge7.tar.gz"
    manifest_path = out_dir / "rows_ge7_manifest.json"

    rows_by_material: dict[str, int] = {}
    true_payloads: dict[str, tuple[str, bytes]] = {}
    with tarfile.open(args.true_tar, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            material = true_material_id(member.name)
            if material is None:
                continue
            text = read_member_text(tar, member)
            rows_by_material[material] = atom_site_rows(text)
            if rows_by_material[material] >= int(args.min_rows):
                true_payloads[material] = (Path(member.name).name, text.encode("utf-8"))

    selected = set(true_payloads)
    generated_count = 0
    with tarfile.open(out_true, "w:gz") as out:
        for material in sorted(true_payloads):
            name, data = true_payloads[material]
            add_bytes(out, name, data)

    with tarfile.open(args.generated_tar, "r:gz") as src, tarfile.open(out_generated, "w:gz") as out:
        for member in src.getmembers():
            if not member.isfile():
                continue
            material = generated_material_id(member.name)
            if material not in selected:
                continue
            f = src.extractfile(member)
            if f is None:
                continue
            add_bytes(out, Path(member.name).name, f.read())
            generated_count += 1

    expected_generated = len(selected) * 20
    manifest = {
        "generated_tar": str(Path(args.generated_tar).resolve()),
        "true_tar": str(Path(args.true_tar).resolve()),
        "out_generated_tar": str(out_generated.resolve()),
        "out_true_tar": str(out_true.resolve()),
        "min_rows": int(args.min_rows),
        "target_samples_total": int(len(rows_by_material)),
        "target_samples_rows_ge7": int(len(selected)),
        "generated_entries": int(generated_count),
        "expected_generated_entries_for_k20": int(expected_generated),
        "complete_generated_coverage": bool(generated_count == expected_generated),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
