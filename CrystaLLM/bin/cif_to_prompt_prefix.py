#!/usr/bin/env python3
"""
Extract a "prompt prefix" from a full CIF file:

Keep only the block starting at the first `data_...` line and ending at the first
`_symmetry_space_group_name_H-M ...` line *after* that (inclusive). Everything else
is dropped.

This is useful when you want to preserve the atom_type property block (loop_
_atom_type_symbol/_atom_type_electronegativity/_atom_type_radius/_atom_type_ionic_radius)
while removing cell params / symmetry ops / atom-site tables, etc.

Examples:
  python3 bin/cif_to_prompt_prefix.py --in path/to/a.cif --out-dir prompts_out
  python3 bin/cif_to_prompt_prefix.py --in cifs_dir --glob '*.cif' --out-dir prompts_out --workers 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


_DATA_PREFIX = "data_"
_SG_KEY = "_symmetry_space_group_name_H-M"


class ExtractError(RuntimeError):
    pass


def _extract_prefix(text: str) -> str:
    lines = text.splitlines(keepends=True)

    start: Optional[int] = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(_DATA_PREFIX):
            start = i
            break
    if start is None:
        raise ExtractError("missing data_ line")

    end: Optional[int] = None
    for j in range(start, len(lines)):
        if lines[j].lstrip().startswith(_SG_KEY):
            end = j
            break
    if end is None:
        raise ExtractError("missing _symmetry_space_group_name_H-M line after data_")

    out = "".join(lines[start : end + 1])
    if out and not out.endswith("\n"):
        out += "\n"
    return out


@dataclass(frozen=True)
class JobResult:
    in_path: str
    out_path: str
    status: str  # ok|skipped|error
    message: str = ""


def _process_one(in_path: str, out_dir: str, suffix: str, overwrite: bool) -> JobResult:
    ip = Path(in_path)
    out = Path(out_dir) / f"{ip.stem}{suffix}"
    if out.exists() and not overwrite:
        return JobResult(in_path=str(ip), out_path=str(out), status="skipped", message="exists")

    text = ip.read_text(encoding="utf-8", errors="ignore")
    try:
        prompt = _extract_prefix(text)
    except Exception as exc:  # noqa: BLE001
        return JobResult(in_path=str(ip), out_path=str(out), status="error", message=f"{type(exc).__name__}: {exc}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt, encoding="utf-8")
    return JobResult(in_path=str(ip), out_path=str(out), status="ok", message="")


def _iter_inputs(root: Path, glob_pat: str) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob(glob_pat))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract prompt prefix (data_.. to _symmetry_space_group_name_H-M) from CIF(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="inp", required=True, help="Input CIF file path or a directory containing CIFs.")
    p.add_argument("--glob", default="*.cif", help="Glob used when --in is a directory.")
    p.add_argument("--out-dir", required=True, help="Directory to write extracted prompt files.")
    p.add_argument(
        "--suffix",
        default=".txt",
        help="Output filename suffix (written as <stem><suffix> inside --out-dir).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Number of worker processes (set 1 to disable multiprocessing).",
    )
    p.add_argument("--chunksize", type=int, default=32, help="Task chunk size for multiprocessing.")
    p.add_argument(
        "--runner",
        default="auto",
        choices=["auto", "processpool", "subprocess"],
        help=(
            "Parallel runner implementation. "
            "'processpool' uses multiprocessing.ProcessPoolExecutor (fastest, but may be restricted on some systems). "
            "'subprocess' spawns independent Python processes per file (no shared semaphores). "
            "'auto' tries processpool, then falls back to subprocess on PermissionError."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.inp).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(_iter_inputs(root, args.glob))
    if not inputs:
        raise SystemExit(f"No inputs found under {root} (glob={args.glob!r}).")

    overwrite = bool(args.overwrite)
    suffix = str(args.suffix)
    workers = int(args.workers)

    ok = skipped = err = 0
    if workers <= 1 or len(inputs) == 1:
        results = []
        for p in inputs:
            r = _process_one(str(p), str(out_dir), suffix, overwrite)
            results.append(r)
            if r.status == "ok":
                ok += 1
            elif r.status == "skipped":
                skipped += 1
            else:
                err += 1
        print(f"[cif_to_prompt_prefix] ok={ok} skipped={skipped} error={err} out_dir={out_dir}")
        # Single-file mode: return a status code that can be used by subprocess runner.
        if root.is_file() and len(inputs) == 1:
            st = results[0].status
            raise SystemExit(0 if st == "ok" else (3 if st == "skipped" else 2))
        raise SystemExit(2 if err else 0)

    def _run_processpool() -> Tuple[int, int, int]:
        _ok = _skipped = _err = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one, str(p), str(out_dir), suffix, overwrite) for p in inputs]
            for fut in as_completed(futs):
                r = fut.result()
                if r.status == "ok":
                    _ok += 1
                elif r.status == "skipped":
                    _skipped += 1
                else:
                    _err += 1
        return _ok, _skipped, _err

    def _run_subprocess() -> Tuple[int, int, int]:
        _ok = _skipped = _err = 0
        script = Path(__file__).resolve()
        procs: list[tuple[subprocess.Popen[bytes], str]] = []
        pending = [str(p) for p in inputs]

        def _spawn(path: str) -> subprocess.Popen[bytes]:
            cmd = [
                sys.executable,
                str(script),
                "--in",
                path,
                "--out-dir",
                str(out_dir),
                "--suffix",
                suffix,
                "--workers",
                "1",
                "--runner",
                "subprocess",
            ]
            if overwrite:
                cmd.append("--overwrite")
            return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while pending or procs:
            while pending and len(procs) < workers:
                path = pending.pop()
                ip = Path(path)
                out_path = out_dir / f"{ip.stem}{suffix}"
                if out_path.exists() and not overwrite:
                    _skipped += 1
                    continue
                procs.append((_spawn(path), path))
            # wait for one
            p0, path0 = procs.pop(0)
            rc = p0.wait()
            if rc == 0:
                _ok += 1
            elif rc == 3:
                _skipped += 1
            else:
                _err += 1
        return _ok, _skipped, _err

    runner = str(args.runner)
    if runner in {"auto", "processpool"}:
        try:
            ok, skipped, err = _run_processpool()
        except PermissionError:
            if runner == "processpool":
                raise
            ok, skipped, err = _run_subprocess()
    else:
        ok, skipped, err = _run_subprocess()

    print(f"[cif_to_prompt_prefix] ok={ok} skipped={skipped} error={err} out_dir={out_dir}")
    raise SystemExit(2 if err else 0)


if __name__ == "__main__":
    main()
