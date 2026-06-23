#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/data/users/xsw/autodlmini")
MODEL_ROOT = WORKSPACE / "model"
CRYSTALLM = MODEL_ROOT / "scp_task/CrystaLLM"
CRYSTALLM_PUBLIC = MODEL_ROOT / "CrystaLLM"
PY = WORKSPACE / "miniforge3/envs/crystallm_env/bin/python"
CONDA_PREFIX = PY.parent.parent

STATE_DIR = ROOT / "state"
LOG_DIR = ROOT / "logs"
RUN_ROOT = ROOT / "generations/crystallm_gt_sg_val_anchor"
SHARD_ROOT = ROOT / "generations/crystallm_gt_sg_val_anchor_shards"
BENCH_CIF_ROOT = ROOT / "cache/official_benchmark_cifs"

CONTROLLER_STATE = STATE_DIR / "controller_state.json"
JOBS_JSONL = STATE_DIR / "jobs.jsonl"
ARTIFACT_REGISTRY = STATE_DIR / "artifact_registry.json"
FROZEN_REGISTRY = STATE_DIR / "frozen_registry.json"


ANCHOR_K100_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "anchor_k20",
        "rank_start": 1,
        "count": 20,
        "temperature": 0.8,
        "top_k": 10,
        "seed": 1337,
        "policy": "historical CrystaLLM-a GT-SG anchor order",
    },
    {"name": "temp070_top10", "rank_start": 21, "count": 10, "temperature": 0.7, "top_k": 10, "seed": 7337},
    {"name": "temp085_top10", "rank_start": 31, "count": 10, "temperature": 0.85, "top_k": 10, "seed": 8537},
    {"name": "temp100_top10", "rank_start": 41, "count": 10, "temperature": 1.0, "top_k": 10, "seed": 10037},
    {"name": "temp115_top10", "rank_start": 51, "count": 10, "temperature": 1.15, "top_k": 10, "seed": 11537},
    {"name": "temp085_top5", "rank_start": 61, "count": 10, "temperature": 0.85, "top_k": 5, "seed": 8505},
    {"name": "temp100_top5", "rank_start": 71, "count": 10, "temperature": 1.0, "top_k": 5, "seed": 10005},
    {"name": "temp085_top20", "rank_start": 81, "count": 10, "temperature": 0.85, "top_k": 20, "seed": 8520},
    {"name": "temp100_top20", "rank_start": 91, "count": 10, "temperature": 1.0, "top_k": 20, "seed": 10020},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_9: {resolved}")
    return resolved


def ensure_dir(path: Path) -> None:
    path = under_root(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_path(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "type": "file",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = [p for p in path.rglob("*") if p.is_file()]
    h = hashlib.sha256()
    total_size = 0
    for p in sorted(files):
        st = p.stat()
        total_size += st.st_size
        rel = p.relative_to(path).as_posix()
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(st.st_mtime_ns).encode("ascii"))
    return {
        "path": str(path),
        "exists": True,
        "type": "dir",
        "file_count": len(files),
        "total_size": total_size,
        "tree_sha256": h.hexdigest(),
    }


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    conda_lib = str(CONDA_PREFIX / "lib")
    prior = env.get("LD_LIBRARY_PATH", "")
    parts = [conda_lib] + ([prior] if prior else [])
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    env["CONDA_PREFIX"] = str(CONDA_PREFIX)
    return env


def run_capture(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(WORKSPACE),
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + "\n[TIMEOUT]\n"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}\n"


def run_logged(
    stage_id: str,
    cmd: list[str],
    *,
    max_attempts: int = 2,
    oom_worker_arg: str = "--gen-workers",
) -> None:
    ensure_dir(LOG_DIR)
    attempt = 1
    current_cmd = list(cmd)
    while attempt <= max_attempts:
        started = now_iso()
        log_path = LOG_DIR / f"{stage_id}.attempt{attempt}.log"
        job: dict[str, Any] = {
            "stage_id": stage_id,
            "attempt": attempt,
            "started_at": started,
            "command": current_cmd,
            "log_path": str(log_path),
        }
        with under_root(log_path).open("w", encoding="utf-8") as log:
            log.write(f"[{started}] command: {command_text(current_cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                [str(x) for x in current_cmd],
                cwd=str(WORKSPACE),
                env=child_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            tail: list[str] = []
            for line in proc.stdout:
                log.write(line)
                if len(tail) >= 200:
                    tail.pop(0)
                tail.append(line.rstrip("\n"))
            return_code = proc.wait()
        finished = now_iso()
        output_tail = "\n".join(tail[-80:])
        job.update(
            {
                "finished_at": finished,
                "return_code": return_code,
                "stderr_tail": output_tail[-12000:],
            }
        )
        append_jsonl(JOBS_JSONL, job)
        if return_code == 0:
            return
        lower_tail = output_tail.lower()
        if "out of memory" in lower_tail or "cuda oom" in lower_tail or "cuda error: out of memory" in lower_tail:
            if oom_worker_arg in current_cmd:
                idx = current_cmd.index(oom_worker_arg) + 1
                try:
                    workers = max(1, int(current_cmd[idx]) // 2)
                    current_cmd[idx] = str(workers)
                except Exception:
                    pass
        attempt += 1
    raise RuntimeError(f"stage {stage_id} failed after {max_attempts} attempts; see {LOG_DIR}")


def csv_record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def path_summary(path: Path | None, *, checksum: bool = False) -> dict[str, Any] | None:
    if path is None:
        return None
    obj = {
        "path": str(path),
        "exists": path.exists(),
        "type": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
    }
    if path.exists():
        st = path.stat()
        obj["size"] = st.st_size
        if checksum and path.is_file():
            obj["sha256"] = sha256_file(path)
    return obj


def load_text_if_exists(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n[truncated]\n"
    return text


def collect_resource_info() -> dict[str, Any]:
    csv_paths = {
        "mp_20_train": CRYSTALLM / "resources/benchmarks/mp_20/train.csv",
        "mp_20_val": CRYSTALLM / "resources/benchmarks/mp_20/val.csv",
        "mp_20_test": CRYSTALLM / "resources/benchmarks/mp_20/test.csv",
        "mpts_52_train": CRYSTALLM / "resources/benchmarks/mpts_52/train.csv",
        "mpts_52_val": CRYSTALLM / "resources/benchmarks/mpts_52/val.csv",
        "mpts_52_test": CRYSTALLM / "resources/benchmarks/mpts_52/test.csv",
    }
    public_csv_paths = {
        k: CRYSTALLM_PUBLIC / str(v.relative_to(CRYSTALLM))
        for k, v in csv_paths.items()
    }

    commands = {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
        "python_version": [str(PY), "--version"],
        "torch_cuda": [
            str(PY),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), "
                "'cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count(), "
                "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
            ),
        ],
        "df": ["df", "-h", str(ROOT)],
    }
    command_outputs = {}
    for name, cmd in commands.items():
        rc, out = run_capture(cmd, timeout=30)
        command_outputs[name] = {"return_code": rc, "output": out.strip()}

    csv_counts = {}
    for name, path in csv_paths.items():
        csv_counts[name] = {
            "path": str(path),
            "records": csv_record_count(path) if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    public_csv_counts = {}
    for name, path in public_csv_paths.items():
        public_csv_counts[name] = {
            "path": str(path),
            "records": csv_record_count(path) if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }

    assets = {
        "crystallm_repo": path_summary(CRYSTALLM),
        "public_crystallm_repo": path_summary(CRYSTALLM_PUBLIC),
        "python": path_summary(PY),
        "mp20_benchmark_ckpt": path_summary(CRYSTALLM / "crystallm_benchmarkmodel/cif_model_mp_20_b/ckpt.pt", checksum=True),
        "mpts52_benchmark_ckpt": path_summary(CRYSTALLM / "crystallm_benchmarkmodel/cif_model_mpts_52_b/ckpt.pt", checksum=True),
        "opentry7_mp20_pure_ckpt": path_summary(MODEL_ROOT / "New_model/opentry_7/checkpoints/pure_crystallm_gt_sg_mp_20/ckpt.pt"),
        "opentry7_mpts52_pure_ckpt": path_summary(MODEL_ROOT / "New_model/opentry_7/checkpoints/pure_crystallm_gt_sg_mpts_52/ckpt.pt"),
        "opentry7_mp20_meta": path_summary(MODEL_ROOT / "New_model/opentry_7/cache/tokens_mp_20_official/meta.pkl", checksum=True),
        "opentry7_mpts52_meta": path_summary(MODEL_ROOT / "New_model/opentry_7/cache/tokens_mpts_52_official/meta.pkl", checksum=True),
    }

    return {
        "created_at": now_iso(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "workspace": str(WORKSPACE),
        "opentry_write_root": str(ROOT),
        "command_outputs": command_outputs,
        "csv_counts": csv_counts,
        "public_csv_counts": public_csv_counts,
        "assets": assets,
    }


def action_resource_audit() -> None:
    info = collect_resource_info()
    write_json(ROOT / "metrics/resource_audit.json", info)
    lines = [
        "# Resource audit",
        "",
        f"- Created at: {info['created_at']}",
        f"- Write root: `{ROOT}`",
        f"- CPU cores: {info['cpu_count']}",
        f"- CrystaLLM repo used for reproduction: `{CRYSTALLM}`",
        f"- crystallm_env python: `{PY}`",
        "",
        "## GPU",
        "```",
        info["command_outputs"]["nvidia_smi"]["output"] or "nvidia-smi unavailable",
        "```",
        "",
        "## Torch CUDA",
        "```",
        info["command_outputs"]["torch_cuda"]["output"],
        "```",
        "",
        "## Disk",
        "```",
        info["command_outputs"]["df"]["output"],
        "```",
        "",
        "## Official CSV record counts",
        "",
        "| split | records | sha256 | path |",
        "|---|---:|---|---|",
    ]
    for key, meta in info["csv_counts"].items():
        lines.append(f"| {key} | {meta['records']} | `{meta['sha256']}` | `{meta['path']}` |")
    lines.extend(
        [
            "",
            "## Key assets",
            "",
            "| asset | exists | size | sha256/path |",
            "|---|---:|---:|---|",
        ]
    )
    for key, meta in info["assets"].items():
        if meta is None:
            continue
        lines.append(
            f"| {key} | {meta.get('exists')} | {meta.get('size', '')} | "
            f"`{meta.get('sha256') or meta.get('path')}` |"
        )
    write_text(ROOT / "reports/resource_audit.md", "\n".join(lines) + "\n")


def action_baseline_provenance() -> None:
    mp20_run = CRYSTALLM / "reproduce/crystallm_gt_sg_csp_test_20260531/mp20_test_data_atomtype_gt_sg_k20_20260531"
    report_json = CRYSTALLM / "reproduce/crystallm_gt_sg_csp_test_20260531/reports/crystallm_gt_sg_mp20_mpts52_report.json"
    mpts_run = CRYSTALLM / "reproduce/mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20"
    op7 = MODEL_ROOT / "New_model/opentry_7"
    provenance = {
        "created_at": now_iso(),
        "anchor_summary_from_prompt": {
            "mp_20": {
                "match@1": 71.67,
                "match@5": 83.08,
                "match@20": 87.81,
                "rows>=7_match@1/5/20": [62.37, 76.35, 82.61],
            },
            "mpts_52": {
                "match@1": 25.23,
                "match@5": 36.46,
                "match@20": 43.96,
                "rows>=7_match@1/5/20": [22.49, 33.37, 41.04],
            },
        },
        "historical_gt_sg_report": json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else None,
        "mp20_run_meta": json.loads((mp20_run / "run_meta.json").read_text(encoding="utf-8")) if (mp20_run / "run_meta.json").exists() else None,
        "mpts52_run_meta": json.loads((mpts_run / "run_meta.json").read_text(encoding="utf-8")) if (mpts_run / "run_meta.json").exists() else None,
        "mp20_generation_command": load_text_if_exists(mp20_run / "generation_command.txt"),
        "mp20_postprocess_command": load_text_if_exists(mp20_run / "postprocess_command.txt"),
        "opentry7_baseline_config": json.loads((op7 / "configs/crystallm_a_baselines.json").read_text(encoding="utf-8"))
        if (op7 / "configs/crystallm_a_baselines.json").exists()
        else None,
        "opentry7_eval_summaries": {
            "mp20": json.loads((op7 / "eval/crystallm_a_gt_sg_mp_20_test_k20/summary.json").read_text(encoding="utf-8"))
            if (op7 / "eval/crystallm_a_gt_sg_mp_20_test_k20/summary.json").exists()
            else None,
            "mpts52": json.loads((op7 / "eval/crystallm_a_gt_sg_mpts_52_test_k20/summary.json").read_text(encoding="utf-8"))
            if (op7 / "eval/crystallm_a_gt_sg_mpts_52_test_k20/summary.json").exists()
            else None,
        },
        "tokenizer_note": (
            "The scp_task/CrystaLLM benchmark generator constructs CIFTokenizer() from package code/resources. "
            "It does not load a per-run meta.pkl. opentry_7 pure-model tokenization artifacts are still recorded "
            "in resource_audit for separate pure-model provenance."
        ),
    }
    write_json(ROOT / "metrics/baseline_provenance.json", provenance)

    hist = provenance.get("historical_gt_sg_report") or {}
    protocol = hist.get("protocol") or {}
    lines = [
        "# Baseline provenance",
        "",
        f"- Created at: {provenance['created_at']}",
        "- Historical GT-SG test provenance was traced through:",
        f"  - `{mp20_run}`",
        f"  - `{mpts_run}`",
        f"  - `{report_json}`",
        f"  - `{op7 / 'configs/crystallm_a_baselines.json'}`",
        "",
        "## Recovered generation protocol",
        "",
        f"- Task: {protocol.get('task')}",
        f"- Prompt format: {protocol.get('prompt_format')}",
        f"- Generation script: `{protocol.get('generation_script')}`",
        f"- Postprocess script: `{protocol.get('postprocess_script')}`",
        f"- Metrics script: `{protocol.get('metrics_script')}`",
        f"- Temperature: {protocol.get('temperature')}",
        f"- Top-k: {protocol.get('top_k')}",
        f"- Max new tokens: {protocol.get('max_new_tokens')}",
        f"- Candidate budget: {protocol.get('num_generations')}",
        "",
        "## MP-20 generation command",
        "",
        "```bash",
        provenance.get("mp20_generation_command", "").strip(),
        "```",
        "",
        "## MPTS-52 note",
        "",
        "The historical report states that MPTS-52 reused a verified full K20 GT-SG run from "
        "`mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20`.",
        "",
        "## Tokenizer",
        "",
        provenance["tokenizer_note"],
    ]
    write_text(ROOT / "reports/baseline_provenance.md", "\n".join(lines) + "\n")


def action_prepare_validation_gt_cifs() -> None:
    script = CRYSTALLM / "bin/prepare_benchmark_cifs.py"
    for dataset in ("mp_20", "mpts_52"):
        cmd = [
            str(PY),
            str(script),
            "--benchmarks-root",
            str(CRYSTALLM / "resources/benchmarks"),
            "--out-root",
            str(BENCH_CIF_ROOT),
            "--glob",
            f"{dataset}/val.csv",
            "--overwrite",
        ]
        run_logged(f"prepare_validation_gt_cifs_{dataset}", cmd, max_attempts=1)


def val_run_name(dataset: str) -> str:
    return "mp20_val_data_atomtype_gt_sg_k100" if dataset == "mp_20" else "mpts52_val_data_atomtype_gt_sg_k100"


def val_run_dir(dataset: str) -> Path:
    return RUN_ROOT / val_run_name(dataset)


def val_csv(dataset: str) -> Path:
    return CRYSTALLM / f"resources/benchmarks/{dataset}/val.csv"


def val_model_dir(dataset: str) -> Path:
    suffix = "mp_20" if dataset == "mp_20" else "mpts_52"
    return CRYSTALLM / f"crystallm_benchmarkmodel/cif_model_{suffix}_b"


def val_gt_dir(dataset: str) -> Path:
    return BENCH_CIF_ROOT / dataset / "val/cifs"


def dataset_short(dataset: str) -> str:
    return "mp20" if dataset == "mp_20" else "mpts52"


def dataset_prefix(dataset: str) -> str:
    return "mp_20" if dataset == "mp_20" else "mpts_52"


def shard_plan_path(dataset: str) -> Path:
    return STATE_DIR / f"validation_anchor_shards_{dataset_short(dataset)}.json"


def shard_complete_path(dataset: str) -> Path:
    return STATE_DIR / f"validation_anchor_shards_{dataset_short(dataset)}.complete.json"


def shard_dataset_root(dataset: str) -> Path:
    return SHARD_ROOT / dataset_short(dataset)


def configured_shard_size() -> int:
    text = os.environ.get("OPENTRY10_SHARD_SIZE", "").strip()
    if text:
        return max(1, int(text))
    return 64


def max_shards_per_stage() -> int | None:
    text = os.environ.get("OPENTRY10_MAX_SHARDS_PER_STAGE", "").strip()
    if not text:
        return None
    value = int(text)
    return value if value > 0 else None


class PartialProgress(RuntimeError):
    pass


def anchor_command(dataset: str, *, prepare_only: bool = False) -> list[str]:
    cmd = [
        str(PY),
        str(CRYSTALLM / "bin/run_gt_sg_csp_benchmark.py"),
        "--dataset",
        dataset,
        "--prompt-mode",
        "data_atomtype_gt_sg",
        "--input-csv",
        str(val_csv(dataset)),
        "--model-dir",
        str(val_model_dir(dataset)),
        "--gt-cifs-dir",
        str(val_gt_dir(dataset)),
        "--out-root",
        str(RUN_ROOT),
        "--run-name",
        val_run_name(dataset),
        "--samples",
        "100",
        "--budgets",
        "1,5,20,50,100",
        "--py",
        str(PY),
        "--device",
        "cuda:auto",
        "--dtype",
        "bfloat16",
        "--temperature",
        "0.8",
        "--top-k",
        "10",
        "--max-new-tokens",
        "2048",
        "--seed",
        "1337",
        "--sample-seed-stride",
        "100000",
        "--gen-workers",
        "4",
        "--bench-workers",
        "32",
        "--max-sites",
        "512",
        "--rmsd-timeout-seconds",
        "5.0",
        "--hard-timeout-seconds",
        "60.0",
        "--resume",
    ]
    if prepare_only:
        cmd.append("--prepare-only")
    return cmd


def load_material_ids(dataset: str) -> list[str]:
    path = val_run_dir(dataset) / "material_ids.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_prompt_ids(dataset: str) -> list[str]:
    prompts_dir = val_run_dir(dataset) / "prompts/data_atomtype_gt_sg"
    if not prompts_dir.is_dir():
        raise FileNotFoundError(prompts_dir)
    return [p.stem for p in sorted(prompts_dir.glob("*.txt")) if p.is_file()]


def load_shard_plan(dataset: str) -> dict[str, Any]:
    path = shard_plan_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing shard plan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def action_plan_anchor_shards(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        material_ids = load_material_ids(dataset)
        prompt_ids = load_prompt_ids(dataset)
        if set(material_ids) != set(prompt_ids):
            missing_from_prompts = sorted(set(material_ids) - set(prompt_ids))[:10]
            missing_from_materials = sorted(set(prompt_ids) - set(material_ids))[:10]
            raise RuntimeError(
                "prompt/material_id mismatch: "
                f"missing_from_prompts={missing_from_prompts} missing_from_materials={missing_from_materials}"
            )
        shard_size = configured_shard_size()
        shards: list[dict[str, Any]] = []
        total = len(prompt_ids)
        for shard_index, start in enumerate(range(0, total, shard_size)):
            limit = min(shard_size, total - start)
            run_name = f"{dataset_short(dataset)}_val_gt_sg_k100_shard{shard_index:04d}_{start:05d}_{limit:04d}"
            shard_material_ids = prompt_ids[start : start + limit]
            shards.append(
                {
                    "shard_index": shard_index,
                    "start_index": start,
                    "limit": limit,
                    "material_ids": shard_material_ids,
                    "run_name": run_name,
                    "run_dir": str(shard_dataset_root(dataset) / run_name),
                    "status": "pending",
                    "expected_cifs": int(limit) * 100,
                }
            )
        payload = {
            "created_at": now_iso(),
            "dataset": dataset,
            "dataset_short": dataset_short(dataset),
            "material_ids_path": str(val_run_dir(dataset) / "material_ids.txt"),
            "prompt_order": "sorted prompt filename stems; matches generate_cifs_from_prompts_dir.py slicing",
            "total_material_ids": total,
            "shard_size": shard_size,
            "num_shards": len(shards),
            "samples_per_prompt": 100,
            "generation_configs": ANCHOR_K100_CONFIGS,
            "note": (
                "Ranks 1-20 preserve the historical CrystaLLM-a GT-SG anchor policy. "
                "Ranks 21-100 are expanded candidate pools with multiple temperatures/top-k values. "
                "Normalized token logprob is not emitted by the upstream generator yet and must be backfilled "
                "before rerank feature freezing."
            ),
            "shards": shards,
        }
        write_json(shard_plan_path(dataset), payload)
    return _run


def anchor_shard_generate_command(dataset: str, shard: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    run_dir = Path(shard["run_dir"])
    raw_dir = run_dir / "cifs_raw/data_atomtype_gt_sg"
    return [
        str(PY),
        str(CRYSTALLM / "bin/generate_cifs_from_prompts_dir.py"),
        "--model-dir",
        str(val_model_dir(dataset)),
        "--prompts-dir",
        str(val_run_dir(dataset) / "prompts/data_atomtype_gt_sg"),
        "--out-dir",
        str(raw_dir),
        "--start-index",
        str(int(shard["start_index"])),
        "--num-prompts",
        str(int(shard["limit"])),
        "--num-samples-per-prompt",
        str(int(cfg["count"])),
        "--sample-index-offset",
        str(int(cfg["rank_start"]) - 1),
        "--sample-seed-stride",
        "100000",
        "--seed",
        str(int(cfg["seed"])),
        "--temperature",
        str(float(cfg["temperature"])),
        "--top-k",
        str(int(cfg["top_k"])),
        "--max-new-tokens",
        "2048",
        "--device",
        "cuda:auto",
        "--dtype",
        "bfloat16",
        "--workers",
        "4",
        "--batch-samples",
        "--retry-missing-single-worker",
    ]


def anchor_shard_postprocess_command(shard: dict[str, Any]) -> list[str]:
    run_dir = Path(shard["run_dir"])
    return [
        str(PY),
        str(CRYSTALLM / "bin/postprocess.py"),
        str(run_dir / "cifs_raw/data_atomtype_gt_sg"),
        str(run_dir / "cifs_post/data_atomtype_gt_sg"),
        "--workers",
        "32",
        "--resume",
    ]


def expected_shard_material_ids(dataset: str, shard: dict[str, Any]) -> list[str]:
    if shard.get("material_ids"):
        return [str(x) for x in shard["material_ids"]]
    ids = load_material_ids(dataset)
    start = int(shard["start_index"])
    limit = int(shard["limit"])
    return ids[start : start + limit]


def count_shard_cifs(dataset: str, shard: dict[str, Any], *, postprocessed: bool) -> tuple[int, list[str]]:
    run_dir = Path(shard["run_dir"])
    cif_dir = run_dir / ("cifs_post/data_atomtype_gt_sg" if postprocessed else "cifs_raw/data_atomtype_gt_sg")
    missing: list[str] = []
    count = 0
    for mid in expected_shard_material_ids(dataset, shard):
        for rank in range(1, 101):
            p = cif_dir / f"{mid}__{rank}.cif"
            if p.is_file():
                count += 1
            else:
                missing.append(str(p))
    return count, missing


def write_shard_manifest(dataset: str, shard: dict[str, Any]) -> None:
    run_dir = Path(shard["run_dir"])
    manifest = {
        "dataset": dataset,
        "dataset_short": dataset_short(dataset),
        "shard_index": int(shard["shard_index"]),
        "start_index": int(shard["start_index"]),
        "limit": int(shard["limit"]),
        "material_ids": expected_shard_material_ids(dataset, shard),
        "samples_per_prompt": 100,
        "generation_configs": ANCHOR_K100_CONFIGS,
        "raw_count": count_shard_cifs(dataset, shard, postprocessed=False)[0],
        "post_count": count_shard_cifs(dataset, shard, postprocessed=True)[0],
        "created_at": now_iso(),
    }
    write_json(run_dir / "shard_manifest.json", manifest)


def update_shard_plan(dataset: str, plan: dict[str, Any]) -> None:
    write_json(shard_plan_path(dataset), plan)


def run_anchor_shard(dataset: str, plan: dict[str, Any], shard: dict[str, Any]) -> None:
    run_dir = Path(shard["run_dir"])
    ensure_dir(run_dir / "cifs_raw/data_atomtype_gt_sg")
    ensure_dir(run_dir / "cifs_post/data_atomtype_gt_sg")
    write_json(
        run_dir / "generation_provenance.json",
        {
            "dataset": dataset,
            "source_full_run_dir": str(val_run_dir(dataset)),
            "shard": {k: v for k, v in shard.items() if k != "last_error"},
            "generation_configs": ANCHOR_K100_CONFIGS,
            "created_at": now_iso(),
        },
    )
    for cfg in ANCHOR_K100_CONFIGS:
        cmd = anchor_shard_generate_command(dataset, shard, cfg)
        stage_id = f"generate_validation_anchor_{dataset_short(dataset)}_shard{int(shard['shard_index']):04d}_{cfg['name']}"
        run_logged(stage_id, cmd, max_attempts=3)
    raw_count, raw_missing = count_shard_cifs(dataset, shard, postprocessed=False)
    if raw_missing:
        raise RuntimeError(f"raw shard coverage incomplete: {raw_count}/{shard['expected_cifs']} first_missing={raw_missing[:3]}")
    run_logged(
        f"postprocess_validation_anchor_{dataset_short(dataset)}_shard{int(shard['shard_index']):04d}",
        anchor_shard_postprocess_command(shard),
        max_attempts=2,
        oom_worker_arg="--workers",
    )
    post_count, post_missing = count_shard_cifs(dataset, shard, postprocessed=True)
    if post_missing:
        raise RuntimeError(f"post shard coverage incomplete: {post_count}/{shard['expected_cifs']} first_missing={post_missing[:3]}")
    write_shard_manifest(dataset, shard)
    shard["status"] = "completed"
    shard["completed_at"] = now_iso()
    shard["output_checksum"] = checksum_path(Path(shard["run_dir"]))
    update_shard_plan(dataset, plan)


def action_generate_anchor_shards(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        plan = load_shard_plan(dataset)
        max_to_run = max_shards_per_stage()
        ran = 0
        pending = [s for s in plan["shards"] if s.get("status") != "completed"]
        if max_to_run is not None:
            pending = sorted(pending, key=lambda s: (int(s["limit"]), int(s["shard_index"])))
        for shard in pending:
            try:
                shard["status"] = "running"
                shard["started_at"] = now_iso()
                update_shard_plan(dataset, plan)
                run_anchor_shard(dataset, plan, shard)
                ran += 1
            except Exception as exc:  # noqa: BLE001
                shard["status"] = "failed"
                shard["last_error"] = f"{type(exc).__name__}: {exc}"
                shard["failed_at"] = now_iso()
                update_shard_plan(dataset, plan)
                raise
            remaining = sum(1 for s in plan["shards"] if s.get("status") != "completed")
            if max_to_run is not None and ran >= max_to_run and remaining:
                raise PartialProgress(
                    f"generated {ran} shard(s) for {dataset}; {remaining} shard(s) remain"
                )
        remaining = sum(1 for s in plan["shards"] if s.get("status") != "completed")
        if remaining:
            raise PartialProgress(f"{remaining} shard(s) remain for {dataset}")
        write_json(
            shard_complete_path(dataset),
            {
                "dataset": dataset,
                "dataset_short": dataset_short(dataset),
                "completed_at": now_iso(),
                "num_shards": len(plan["shards"]),
                "samples_per_prompt": 100,
                "expected_cifs": sum(int(s["expected_cifs"]) for s in plan["shards"]),
                "plan": str(shard_plan_path(dataset)),
            },
        )
    return _run


def generated_cif_path_from_shards(dataset: str, material_index: int, material_id: str, rank: int) -> Path:
    plan = load_shard_plan(dataset)
    for shard in plan["shards"]:
        if str(material_id) in {str(x) for x in shard.get("material_ids", [])}:
            return Path(shard["run_dir"]) / "cifs_post/data_atomtype_gt_sg" / f"{material_id}__{rank}.cif"
    raise KeyError(f"material index {material_index} not covered by shard plan for {dataset}")


def parse_metrics_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
    raise RuntimeError("could not parse metrics dict from benchmark_metrics.py output")


def run_capture_job(stage_id: str, cmd: list[str], *, max_attempts: int = 1) -> str:
    attempt = 1
    current_cmd = list(cmd)
    while attempt <= max_attempts:
        started = now_iso()
        log_path = LOG_DIR / f"{stage_id}.attempt{attempt}.log"
        rc, stdout = run_capture(current_cmd)
        write_text(log_path, f"[{started}] command: {command_text(current_cmd)}\n{stdout}")
        append_jsonl(
            JOBS_JSONL,
            {
                "stage_id": stage_id,
                "attempt": attempt,
                "started_at": started,
                "finished_at": now_iso(),
                "command": current_cmd,
                "return_code": rc,
                "log_path": str(log_path),
                "stderr_tail": stdout[-12000:],
            },
        )
        if rc == 0:
            return stdout
        attempt += 1
    raise RuntimeError(f"command failed after {max_attempts} attempts: {command_text(current_cmd)}")


def action_assemble_anchor_from_shards(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        if not shard_complete_path(dataset).exists():
            raise FileNotFoundError(f"shards are not complete: {shard_complete_path(dataset)}")
        run_dir = val_run_dir(dataset)
        ensure_dir(run_dir / "tars")
        ensure_dir(run_dir / "metrics")
        material_ids = load_material_ids(dataset)
        gen_tar = run_dir / "tars/generated_data_atomtype_gt_sg.tar.gz"
        missing: list[str] = []
        with tarfile.open(under_root(gen_tar), "w:gz") as tar:
            for material_index, mid in enumerate(material_ids):
                for rank in range(1, 101):
                    p = generated_cif_path_from_shards(dataset, material_index, mid, rank)
                    if not p.is_file():
                        missing.append(str(p))
                        continue
                    tar.add(str(p), arcname=f"{mid}__{rank}.cif")
        if missing:
            raise RuntimeError(f"cannot assemble tar; missing {len(missing)} CIFs, first={missing[:3]}")

        true_tar = run_dir / "tars/true.tar.gz"
        metrics_json: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []
        for k in (1, 5, 20, 50, 100):
            cmd = [
                str(PY),
                str(CRYSTALLM / "bin/benchmark_metrics.py"),
                str(gen_tar),
                str(true_tar),
                "--num-gens",
                str(k),
                "--workers",
                "32",
                "--unmatched-diagnostics",
                "off",
                "--max-sites",
                "512",
                "--rmsd-timeout-seconds",
                "5.0",
                "--hard-timeout-seconds",
                "60.0",
            ]
            stdout = run_capture_job(f"benchmark_validation_anchor_{dataset_short(dataset)}_k{k}", cmd, max_attempts=1)
            write_text(run_dir / f"metrics/benchmark_k{k}.txt", stdout)
            metrics = parse_metrics_stdout(stdout)
            metrics_json[f"k{k}"] = metrics
            summary_rows.append(
                {
                    "dataset": dataset,
                    "K": k,
                    "match_rate": metrics.get("match_rate"),
                    "RMSE": metrics.get("rms_dist"),
                    "n_ids": metrics.get("n_ids"),
                    "parse_rate_candidate": metrics.get("parse_rate_candidate"),
                    "valid_rate_candidate": metrics.get("valid_rate_candidate"),
                }
            )
        write_json(run_dir / "metrics/metrics.json", metrics_json)
        with under_root(run_dir / "metrics/summary.tsv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(summary_rows)
    return _run


def action_prepare_anchor(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        run_logged(f"prepare_validation_anchor_{dataset.replace('_', '')}", anchor_command(dataset, prepare_only=True), max_attempts=1)
    return _run


def action_generate_anchor(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        run_logged(f"generate_validation_anchor_{dataset.replace('_', '')}", anchor_command(dataset, prepare_only=False), max_attempts=3)
    return _run


def export_anchor_jsonl(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        prefix = "mp_20" if dataset == "mp_20" else "mpts_52"
        run_dir = val_run_dir(dataset)
        meta_path = run_dir / "run_meta.json"
        post_dir = run_dir / "cifs_post/data_atomtype_gt_sg"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        material_ids = (run_dir / "material_ids.txt").read_text(encoding="utf-8").splitlines()
        samples = int(meta["samples"])
        out_path = ROOT / "candidates" / f"crystallm_gt_sg_{'mp20' if dataset == 'mp_20' else 'mpts52'}_val_k100.jsonl"
        missing: list[str] = []
        rows_written = 0
        full_post_available = post_dir.is_dir() and len(list(post_dir.glob("*.cif"))) >= len(material_ids) * samples
        with under_root(out_path).open("w", encoding="utf-8") as f:
            for material_index, material_id in enumerate(material_ids):
                sample_id = f"{prefix}_val_orig__{material_id}"
                for rank in range(1, samples + 1):
                    if full_post_available:
                        cif_path = post_dir / f"{material_id}__{rank}.cif"
                    else:
                        cif_path = generated_cif_path_from_shards(dataset, material_index, material_id, rank)
                    if not cif_path.is_file():
                        missing.append(str(cif_path))
                        continue
                    cfg = next(
                        (
                            item
                            for item in ANCHOR_K100_CONFIGS
                            if int(item["rank_start"]) <= rank < int(item["rank_start"]) + int(item["count"])
                        ),
                        {},
                    )
                    row = {
                        "dataset": prefix,
                        "split": "val",
                        "sample_id": sample_id,
                        "material_id": material_id,
                        "rank": rank,
                        "gen_index": rank - 1,
                        "source": "crystallm_gt_sg_val_k100",
                        "source_run_dir": str(run_dir),
                        "generation_config": cfg,
                        "normalized_token_logprob": None,
                        "logprob_available": False,
                        "generated_text": cif_path.read_text(encoding="utf-8", errors="replace"),
                    }
                    f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                    rows_written += 1
        if missing:
            preview = "\n".join(missing[:20])
            raise RuntimeError(f"anchor export missing {len(missing)} generated CIFs; examples:\n{preview}")
        write_json(
            out_path.with_suffix(out_path.suffix + ".manifest.json"),
            {
                "dataset": prefix,
                "split": "val",
                "samples_per_prompt": samples,
                "material_ids": len(material_ids),
                "rows_written": rows_written,
                "source_run_dir": str(run_dir),
                "sharded_source": not full_post_available,
                "logprob_available": False,
                "logprob_note": "Upstream generate_cifs_from_prompts_dir.py does not emit token logprob; must be backfilled before ranker feature freeze.",
                "out": str(out_path),
            },
        )
    return _run


def action_copy_anchor_metrics(dataset: str) -> Callable[[], None]:
    def _run() -> None:
        short = "mp20" if dataset == "mp_20" else "mpts52"
        run_dir = val_run_dir(dataset)
        metrics_path = run_dir / "metrics/metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        out_json = ROOT / "metrics" / f"crystallm_gt_sg_{short}_val.json"
        write_json(out_json, {"dataset": dataset, "source": str(metrics_path), "metrics": metrics})
    return _run


def action_validation_anchor_report() -> None:
    reports = []
    for dataset, short in (("mp_20", "mp20"), ("mpts_52", "mpts52")):
        metrics_path = ROOT / "metrics" / f"crystallm_gt_sg_{short}_val.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        reports.append((dataset, obj["metrics"]))
    lines = [
        "# CrystaLLM validation anchor report",
        "",
        f"- Created at: {now_iso()}",
        "- Protocol: CrystaLLM-a GT-SG validation K100 using the traced test-anchor model/prompt/sampling settings.",
        "- Budgets evaluated: K1, K5, K20, K50, K100.",
        "",
        "| dataset | K | match_rate | RMSE | n_ids | parse_rate_candidate | valid_rate_candidate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, metrics in reports:
        for key in ("k1", "k5", "k20", "k50", "k100"):
            row = metrics.get(key, {})
            lines.append(
                f"| {dataset} | {key[1:]} | {row.get('match_rate')} | {row.get('rms_dist')} | "
                f"{row.get('n_ids')} | {row.get('parse_rate_candidate')} | {row.get('valid_rate_candidate')} |"
            )
    write_text(ROOT / "reports/crystallm_validation_anchor_report.md", "\n".join(lines) + "\n")


@dataclass(frozen=True)
class Stage:
    stage_id: str
    action: Callable[[], None]
    inputs: list[Path]
    outputs: list[Path]
    long_running: bool = False


def stages() -> list[Stage]:
    return [
        Stage(
            "phase0_resource_audit",
            action_resource_audit,
            [CRYSTALLM / "resources/benchmarks/mp_20/train.csv", CRYSTALLM / "resources/benchmarks/mpts_52/train.csv"],
            [ROOT / "reports/resource_audit.md", ROOT / "metrics/resource_audit.json"],
        ),
        Stage(
            "phase0_baseline_provenance",
            action_baseline_provenance,
            [
                CRYSTALLM / "reproduce/crystallm_gt_sg_csp_test_20260531/reports/crystallm_gt_sg_mp20_mpts52_report.json",
                MODEL_ROOT / "New_model/opentry_7/configs/crystallm_a_baselines.json",
            ],
            [ROOT / "reports/baseline_provenance.md", ROOT / "metrics/baseline_provenance.json"],
        ),
        Stage(
            "prepare_validation_gt_cifs",
            action_prepare_validation_gt_cifs,
            [CRYSTALLM / "resources/benchmarks/mp_20/val.csv", CRYSTALLM / "resources/benchmarks/mpts_52/val.csv"],
            [BENCH_CIF_ROOT / "mp_20/val/manifest.tsv", BENCH_CIF_ROOT / "mpts_52/val/manifest.tsv"],
        ),
        Stage(
            "prepare_validation_anchor_mp20",
            action_prepare_anchor("mp_20"),
            [val_csv("mp_20"), val_gt_dir("mp_20"), val_model_dir("mp_20") / "ckpt.pt"],
            [val_run_dir("mp_20") / "run_meta.json", val_run_dir("mp_20") / "tars/true.tar.gz"],
        ),
        Stage(
            "prepare_validation_anchor_mpts52",
            action_prepare_anchor("mpts_52"),
            [val_csv("mpts_52"), val_gt_dir("mpts_52"), val_model_dir("mpts_52") / "ckpt.pt"],
            [val_run_dir("mpts_52") / "run_meta.json", val_run_dir("mpts_52") / "tars/true.tar.gz"],
        ),
        Stage(
            "plan_validation_anchor_mp20_shards",
            action_plan_anchor_shards("mp_20"),
            [val_run_dir("mp_20") / "run_meta.json", val_run_dir("mp_20") / "tars/true.tar.gz"],
            [shard_plan_path("mp_20")],
        ),
        Stage(
            "generate_validation_anchor_mp20_shards",
            action_generate_anchor_shards("mp_20"),
            [shard_plan_path("mp_20")],
            [shard_complete_path("mp_20")],
            long_running=True,
        ),
        Stage(
            "assemble_validation_anchor_mp20",
            action_assemble_anchor_from_shards("mp_20"),
            [shard_complete_path("mp_20")],
            [val_run_dir("mp_20") / "metrics/metrics.json", val_run_dir("mp_20") / "tars/generated_data_atomtype_gt_sg.tar.gz"],
            long_running=True,
        ),
        Stage(
            "export_validation_anchor_mp20_jsonl",
            export_anchor_jsonl("mp_20"),
            [val_run_dir("mp_20") / "metrics/metrics.json"],
            [ROOT / "candidates/crystallm_gt_sg_mp20_val_k100.jsonl"],
        ),
        Stage(
            "copy_validation_anchor_mp20_metrics",
            action_copy_anchor_metrics("mp_20"),
            [val_run_dir("mp_20") / "metrics/metrics.json"],
            [ROOT / "metrics/crystallm_gt_sg_mp20_val.json"],
        ),
        Stage(
            "plan_validation_anchor_mpts52_shards",
            action_plan_anchor_shards("mpts_52"),
            [val_run_dir("mpts_52") / "run_meta.json", val_run_dir("mpts_52") / "tars/true.tar.gz"],
            [shard_plan_path("mpts_52")],
        ),
        Stage(
            "generate_validation_anchor_mpts52_shards",
            action_generate_anchor_shards("mpts_52"),
            [shard_plan_path("mpts_52")],
            [shard_complete_path("mpts_52")],
            long_running=True,
        ),
        Stage(
            "assemble_validation_anchor_mpts52",
            action_assemble_anchor_from_shards("mpts_52"),
            [shard_complete_path("mpts_52")],
            [val_run_dir("mpts_52") / "metrics/metrics.json", val_run_dir("mpts_52") / "tars/generated_data_atomtype_gt_sg.tar.gz"],
            long_running=True,
        ),
        Stage(
            "export_validation_anchor_mpts52_jsonl",
            export_anchor_jsonl("mpts_52"),
            [val_run_dir("mpts_52") / "metrics/metrics.json"],
            [ROOT / "candidates/crystallm_gt_sg_mpts52_val_k100.jsonl"],
        ),
        Stage(
            "copy_validation_anchor_mpts52_metrics",
            action_copy_anchor_metrics("mpts_52"),
            [val_run_dir("mpts_52") / "metrics/metrics.json"],
            [ROOT / "metrics/crystallm_gt_sg_mpts52_val.json"],
        ),
        Stage(
            "validation_anchor_report",
            action_validation_anchor_report,
            [ROOT / "metrics/crystallm_gt_sg_mp20_val.json", ROOT / "metrics/crystallm_gt_sg_mpts52_val.json"],
            [ROOT / "reports/crystallm_validation_anchor_report.md"],
        ),
    ]


def stage_inputs_checksum(stage: Stage) -> list[dict[str, Any]]:
    return [checksum_path(p) for p in stage.inputs]


def stage_outputs_checksum(stage: Stage) -> list[dict[str, Any]]:
    return [checksum_path(p) for p in stage.outputs]


def stage_outputs_exist(stage: Stage) -> bool:
    return all(p.exists() for p in stage.outputs)


def init_layout() -> dict[str, Any]:
    for rel in (
        "configs",
        "scripts",
        "state",
        "logs",
        "cache",
        "vendor",
        "candidates",
        "generations",
        "features",
        "labels",
        "checkpoints",
        "frozen_strategy",
        "frozen_pure_model",
        "eval",
        "metrics",
        "reports",
    ):
        ensure_dir(ROOT / rel)
    if not FROZEN_REGISTRY.exists():
        write_json(FROZEN_REGISTRY, {"created_at": now_iso(), "frozen_systems": []})
    state = read_json(CONTROLLER_STATE, {})
    if not state:
        state = {
            "run_id": "opentry10_on_opentry9_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "created_at": now_iso(),
            "write_root": str(ROOT),
            "note": "User write-scope override: opentry_10 prompt is executed under opentry_9 only.",
            "stages": {},
        }
        write_json(CONTROLLER_STATE, state)
    return state


def update_state(state: dict[str, Any], stage: Stage, status: str, **extra: Any) -> None:
    state.setdefault("stages", {})[stage.stage_id] = {
        "stage_id": stage.stage_id,
        "status": status,
        "updated_at": now_iso(),
        **extra,
    }
    write_json(CONTROLLER_STATE, state)


def update_artifacts(stage: Stage) -> None:
    registry = read_json(ARTIFACT_REGISTRY, {"created_at": now_iso(), "artifacts": {}})
    for meta in stage_outputs_checksum(stage):
        registry.setdefault("artifacts", {})[meta["path"]] = {
            **meta,
            "stage_id": stage.stage_id,
            "registered_at": now_iso(),
        }
    write_json(ARTIFACT_REGISTRY, registry)


def run_controller(args: argparse.Namespace) -> None:
    state = init_layout()
    selected = stages()
    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [s for s in selected if s.stage_id in allow]
        missing = allow - {s.stage_id for s in selected}
        if missing:
            raise SystemExit(f"unknown --only stage(s): {sorted(missing)}")
    stop_after_seen = False
    for stage in selected:
        if args.no_long and stage.long_running:
            print(f"[stop-before-long] {stage.stage_id}")
            break
        input_checksum = stage_inputs_checksum(stage)
        prior = state.get("stages", {}).get(stage.stage_id, {})
        if args.resume and prior.get("status") == "completed" and stage_outputs_exist(stage):
            print(f"[resume] already completed: {stage.stage_id}")
        else:
            print(f"[stage] {stage.stage_id}")
            update_state(state, stage, "running", input_checksum=input_checksum, outputs=stage_outputs_checksum(stage))
            started = time.time()
            try:
                stage.action()
            except PartialProgress as exc:
                update_state(
                    state,
                    stage,
                    "partial",
                    input_checksum=input_checksum,
                    output_checksum=stage_outputs_checksum(stage),
                    elapsed_seconds=round(time.time() - started, 3),
                    note=str(exc),
                )
                update_artifacts(stage)
                print(f"[partial] {stage.stage_id}: {exc}")
                break
            except Exception as exc:  # noqa: BLE001
                update_state(
                    state,
                    stage,
                    "failed",
                    input_checksum=input_checksum,
                    outputs=stage_outputs_checksum(stage),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            update_state(
                state,
                stage,
                "completed",
                input_checksum=input_checksum,
                output_checksum=stage_outputs_checksum(stage),
                elapsed_seconds=round(time.time() - started, 3),
            )
            update_artifacts(stage)
        if args.stop_after == stage.stage_id:
            stop_after_seen = True
            break
    if args.stop_after and not stop_after_seen:
        raise SystemExit(f"--stop-after stage not reached: {args.stop_after}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resumable opentry_10 controller executed inside the opentry_9 write scope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume", action="store_true", help="Skip completed stages with existing outputs.")
    p.add_argument("--only", default="", help="Comma-separated stage ids to run.")
    p.add_argument("--stop-after", default="", help="Stop after this stage id completes.")
    p.add_argument("--no-long", action="store_true", help="Skip stages marked long-running generation/evaluation.")
    p.add_argument("--shard-size", type=int, default=64, help="Validation K100 prompt shard size for newly planned shard manifests.")
    p.add_argument("--max-shards-per-stage", type=int, default=0, help="Run at most this many incomplete shards in a long shard stage; 0 means all.")
    p.add_argument("--list-stages", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["OPENTRY10_SHARD_SIZE"] = str(max(1, int(args.shard_size)))
    if int(args.max_shards_per_stage) > 0:
        os.environ["OPENTRY10_MAX_SHARDS_PER_STAGE"] = str(int(args.max_shards_per_stage))
    if args.list_stages:
        for s in stages():
            tag = " long" if s.long_running else ""
            print(f"{s.stage_id}{tag}")
        return
    run_controller(args)


if __name__ == "__main__":
    main()
