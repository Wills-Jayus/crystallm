#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def load_material_ids(structured_jsonl: Path) -> list[str]:
    ids: list[str] = []
    with structured_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mid = row.get("material_id")
            if not mid:
                sid = str(row.get("sample_id", ""))
                mid = sid.rsplit("__", 1)[-1] if "__" in sid else sid
            ids.append(str(mid))
    return ids


def add_file(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(src), arcname=arcname)
    with src.open("rb") as f:
        tar.addfile(info, f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CrystaLLM common-subset benchmark tars under opentry.")
    parser.add_argument("--structured-jsonl", type=Path, required=True)
    parser.add_argument("--crystallm-gen-dir", type=Path, required=True)
    parser.add_argument("--crystallm-gt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-gens", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    material_ids = load_material_ids(args.structured_jsonl)
    unique_ids = list(dict.fromkeys(material_ids))

    missing_gt: list[str] = []
    missing_gen: list[str] = []
    gen_count = 0
    gt_count = 0

    gen_tar_path = args.out_dir / "crystallm_mpts52_gt_sg_common_subset_gen_k20.tar.gz"
    true_tar_path = args.out_dir / "crystallm_mpts52_gt_sg_common_subset_true.tar.gz"

    with tarfile.open(gen_tar_path, "w:gz") as gen_tar, tarfile.open(true_tar_path, "w:gz") as true_tar:
        for mid in unique_ids:
            gt_src = args.crystallm_gt_dir / f"{mid}.cif"
            if not gt_src.exists():
                missing_gt.append(mid)
            else:
                add_file(true_tar, gt_src, f"{mid}.cif")
                gt_count += 1

            present = 0
            for rank in range(1, int(args.max_gens) + 1):
                gen_src = args.crystallm_gen_dir / f"{mid}__{rank}.cif"
                if not gen_src.exists():
                    continue
                add_file(gen_tar, gen_src, f"{mid}__{rank}.cif")
                present += 1
                gen_count += 1
            if present == 0:
                missing_gen.append(mid)

    manifest = {
        "structured_records": len(material_ids),
        "unique_material_ids": len(unique_ids),
        "gt_count": gt_count,
        "gen_count": gen_count,
        "max_gens": int(args.max_gens),
        "missing_gt_count": len(missing_gt),
        "missing_gen_count": len(missing_gen),
        "missing_gt": missing_gt[:50],
        "missing_gen": missing_gen[:50],
        "gen_tar": str(gen_tar_path),
        "true_tar": str(true_tar_path),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
