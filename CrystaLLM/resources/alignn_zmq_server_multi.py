#!/usr/bin/env python3
"""
ALIGNN multi-property ZMQ scoring server.

Protocol (JSON over ZMQ REQ/REP):
  - request: {"cif": "<CIF text>", "properties": ["formation_energy", "bandgap", ...]}
  - reply:   {"ok": bool, "<prop>": float|null, "errors": { "<prop>|atoms": "<err str>" }?}

Notes:
  - Based on JARVIS/ALIGNN pretrained models.
  - We intentionally do NOT pass `device=` to `alignn.pretrained.get_prediction`, since
    its signature varies across versions and caused runtime errors previously.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import zmq

# noinspection PyUnresolvedReferences
from jarvis.core.atoms import Atoms
# noinspection PyUnresolvedReferences
from alignn.pretrained import get_prediction

PROPERTY_MODEL_MAP: Dict[str, str] = {
    # Common JARVIS pretrained model names; override via CLI if needed.
    "formation_energy": "jv_formation_energy_peratom_alignn",
    "bandgap": "jv_optb88vdw_bandgap_alignn",
}


def _env_flag(name: str, *, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _configure_alignn_device() -> str:
    """
    alignn.pretrained defaults to CUDA if torch sees a GPU.

    In our setup `dgl` is often CPU-only, so `g.to(cuda)` fails with:
      "DGLError: Device API cuda is not enabled. Please install the cuda version of dgl."

    Default to CPU to make scoring reliable. Set `ALIGNN_FORCE_CPU=0` to try GPU.
    """
    force_cpu = _env_flag("ALIGNN_FORCE_CPU", default=True)
    if not force_cpu:
        return "cuda_or_auto"

    try:
        import alignn.pretrained as ap  # noqa: WPS433

        ap.device = "cpu"
        return "cpu"
    except Exception as exc:  # noqa: BLE001
        print(f"[alignn_zmq_server_multi] WARN: failed to force CPU device: {type(exc).__name__}: {exc}")
        return "cpu_unverified"


def _now_s() -> float:
    return float(time.time())


def _alignn_zip_path(model_name: str) -> str:
    # alignn.pretrained stores zip files next to itself (same directory).
    import alignn.pretrained as ap  # noqa: WPS433

    return str(os.path.join(os.path.dirname(ap.__file__), f"{model_name}.zip"))


def _is_valid_zip(path: str) -> bool:
    try:
        import zipfile  # noqa: WPS433

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return False
            has_cfg = any(n.endswith("config.json") for n in names)
            has_chk = any(("checkpoint_" in n and n.endswith(".pt")) or n.endswith("best_model.pt") for n in names)
            return bool(has_cfg and has_chk)
    except Exception:  # noqa: BLE001
        return False


def _download_alignn_model_zip(model_name: str, *, out_path: str) -> None:
    # Mirror alignn.pretrained download logic but make it atomic (write -> replace).
    import alignn.pretrained as ap  # noqa: WPS433
    import requests  # noqa: WPS433

    meta = ap.all_models.get(model_name)
    if not meta:
        raise ValueError(f"unknown alignn model: {model_name}")
    url = meta[0]

    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    headers = {
        # Some CDNs behave differently for default python user agents.
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) CrystaLLM/alignn_zmq_server_multi",
        "Accept": "*/*",
    }

    def _attempt_requests_download() -> int:
        total = 0
        with requests.get(url, stream=True, timeout=(10, 300), headers=headers) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
        return total

    def _attempt_curl_download() -> int:
        import subprocess  # noqa: WPS433

        if not shutil.which("curl"):
            raise RuntimeError("curl not found for fallback download")
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--retry-all-errors",
                "--connect-timeout",
                "10",
                "--max-time",
                "600",
                "-o",
                tmp_path,
                url,
            ],
            check=True,
        )
        try:
            return int(os.path.getsize(tmp_path))
        except Exception:  # noqa: BLE001
            return 0

    import shutil  # noqa: WPS433

    last_exc: Exception | None = None
    size = 0
    try:
        size = _attempt_requests_download()
    except Exception as exc:  # noqa: BLE001
        last_exc = exc

    if size < 1024:
        # Try curl fallback (can help in environments where requests/proxy behaves oddly).
        try:
            size = _attempt_curl_download()
            last_exc = None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    if size < 1024:
        # Best-effort peek at what we got (often an HTML error page or empty content).
        preview = ""
        try:
            with open(tmp_path, "rb") as f:
                preview = f.read(256).decode("utf-8", errors="ignore").strip()
        except Exception:  # noqa: BLE001
            preview = ""
        raise RuntimeError(f"downloaded too small ({size} bytes). preview={preview!r}. last_exc={last_exc!r}")

    # Validate before promoting to the real path.
    if not _is_valid_zip(tmp_path):
        preview = ""
        try:
            with open(tmp_path, "rb") as f:
                preview = f.read(256).decode("utf-8", errors="ignore").strip()
        except Exception:  # noqa: BLE001
            preview = ""
        raise RuntimeError(f"downloaded zip is invalid (not a zip / missing config/checkpoint). preview={preview!r}")

    os.replace(tmp_path, out_path)


# Keep per-model cooldown to avoid hammering downloads if network/proxy is misconfigured.
_MODEL_FETCH_STATE: Dict[str, Dict[str, Any]] = {}


def _seed_dir() -> Optional[str]:
    # Optional seed dir to avoid relying on outbound network from this server.
    # Put files like:
    #   <seed_dir>/jv_formation_energy_peratom_alignn.zip
    #   <seed_dir>/jv_optb88vdw_bandgap_alignn.zip
    v = os.environ.get("ALIGNN_MODEL_ZIP_SEED_DIR")
    if not v:
        return None
    v = v.strip()
    return v or None


def _try_seed_zip(model_name: str, *, dest_path: str) -> bool:
    seed = _seed_dir()
    if not seed:
        return False
    src = os.path.join(seed, f"{model_name}.zip")
    if not os.path.isfile(src):
        return False
    if os.path.getsize(src) <= 1024 or (not _is_valid_zip(src)):
        print(f"[alignn_zmq_server_multi] WARN: seed zip invalid: {src}")
        return False
    try:
        import shutil  # noqa: WPS433

        shutil.copyfile(src, dest_path)
        if _is_valid_zip(dest_path):
            print(f"[alignn_zmq_server_multi] using seeded zip for {model_name}: {src}")
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"[alignn_zmq_server_multi] WARN: failed to copy seed zip: {type(exc).__name__}: {exc}")
    return False


def _ensure_model_zip_available(model_name: str, *, cooldown_s: int = 300) -> None:
    path = _alignn_zip_path(model_name)
    st = _MODEL_FETCH_STATE.setdefault(model_name, {"last_try": 0.0, "last_error": None})
    if os.path.isfile(path) and os.path.getsize(path) > 1024 and _is_valid_zip(path):
        st["last_error"] = None
        return

    # Seeded zip support (copy into the expected alignn.pretrained location).
    if _try_seed_zip(model_name, dest_path=path):
        st["last_error"] = None
        return

    now = _now_s()
    if (now - float(st.get("last_try") or 0.0)) < float(cooldown_s):
        return
    st["last_try"] = now

    # Remove known-bad zips so downstream code doesn't "see file exists" and keep failing.
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:  # noqa: BLE001
            pass

    try:
        print(f"[alignn_zmq_server_multi] fetching model zip: {model_name}")
        _download_alignn_model_zip(model_name, out_path=path)
        if not _is_valid_zip(path):
            raise RuntimeError("downloaded zip is invalid (not a zip / missing config/checkpoint)")
        st["last_error"] = None
        print(f"[alignn_zmq_server_multi] ready: {model_name} ({os.path.getsize(path)} bytes)")
    except Exception as exc:  # noqa: BLE001
        st["last_error"] = f"{type(exc).__name__}: {exc}"
        # Ensure we don't leave behind 0-byte or invalid placeholder files.
        try:
            if os.path.exists(path) and (os.path.getsize(path) < 1024 or (not _is_valid_zip(path))):
                os.remove(path)
        except Exception:  # noqa: BLE001
            pass
        print(f"[alignn_zmq_server_multi] WARN: cannot fetch model {model_name}: {st['last_error']}")


def _extract_numeric(prediction: Any) -> float:
    if prediction is None:
        raise ValueError("prediction is None")
    if isinstance(prediction, (int, float)):
        return float(prediction)
    if isinstance(prediction, (list, tuple)) and prediction:
        if isinstance(prediction[0], (int, float)):
            return float(prediction[0])
        raise ValueError(f"unexpected prediction list element type: {type(prediction[0]).__name__}")
    if isinstance(prediction, dict):
        for k in ("prediction", "pred", "value", "y_pred", "y"):
            v = prediction.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                return float(v[0])
    raise ValueError(f"cannot extract numeric from prediction type: {type(prediction).__name__}")


def load_atoms(cif_text: str) -> Atoms:
    try:
        return Atoms.from_cif(from_string=cif_text, get_primitive_atoms=True)
    except Exception:
        return Atoms.from_cif(from_string=cif_text, get_primitive_atoms=False)


def predict_property(
    prop: str,
    atoms: Atoms,
    cutoff: int,
    max_neighbors: int,
    model_override: Optional[str] = None,
) -> float:
    model_name = (model_override or PROPERTY_MODEL_MAP.get(prop))
    if not model_name:
        raise ValueError(f"unknown property: {prop}")
    _ensure_model_zip_available(model_name)
    # If the zip is still missing/invalid, surface a clear error instead of BadZipFile/list index out of range.
    zip_path = _alignn_zip_path(model_name)
    if not (os.path.isfile(zip_path) and os.path.getsize(zip_path) > 1024 and _is_valid_zip(zip_path)):
        st = _MODEL_FETCH_STATE.get(model_name) or {}
        err = st.get("last_error")
        raise RuntimeError(f"model_zip_unavailable: {model_name} zip={zip_path} last_error={err}")
    pred = get_prediction(model_name, atoms, cutoff=cutoff, max_neighbors=max_neighbors)
    return _extract_numeric(pred)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALIGNN multi-property ZMQ scoring server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=5555, help="Bind port")
    p.add_argument("--cutoff", type=int, default=8, help="ALIGNN cutoff")
    p.add_argument("--max-neighbors", type=int, default=12, help="ALIGNN max neighbors")
    p.add_argument(
        "--properties",
        nargs="+",
        default=["formation_energy", "bandgap"],
        help="Allowed properties (request must be subset of this list)",
    )
    p.add_argument("--model-formation-energy", default=None, help="Override model name for formation_energy")
    p.add_argument("--model-bandgap", default=None, help="Override model name for bandgap")
    return p.parse_args()


def _model_override(args: argparse.Namespace, prop: str) -> Optional[str]:
    if prop == "formation_energy":
        return args.model_formation_energy
    if prop == "bandgap":
        return args.model_bandgap
    return None


def _reply(ok: bool, values: Dict[str, Optional[float]], errors: Dict[str, str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": ok}
    payload.update(values)
    if errors:
        payload["errors"] = errors
    return payload


def main() -> None:
    args = _parse_args()
    allowed = set(args.properties)

    device_label = _configure_alignn_device()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"[alignn_zmq_server_multi] listening on tcp://{args.host}:{args.port}")
    print(f"[alignn_zmq_server_multi] allowed properties: {sorted(allowed)}")
    print(
        "[alignn_zmq_server_multi] compute device: "
        f"{device_label} (ALIGNN_FORCE_CPU={os.environ.get('ALIGNN_FORCE_CPU', '<unset>')})"
    )

    # Pre-flight: attempt to make required model zips available.
    for prop in sorted(allowed):
        m = PROPERTY_MODEL_MAP.get(prop)
        if m:
            _ensure_model_zip_available(m, cooldown_s=0)

    while True:
        raw = socket.recv()
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            socket.send_json(_reply(False, {}, {"request": f"invalid_json: {exc}"}))
            continue

        cif_text = req.get("cif")
        props = req.get("properties") or []
        if not isinstance(cif_text, str) or not cif_text.strip():
            socket.send_json(_reply(False, {}, {"request": "missing_or_empty_cif"}))
            continue
        if not isinstance(props, list) or not all(isinstance(p, str) for p in props):
            socket.send_json(_reply(False, {}, {"request": "properties_must_be_list_of_str"}))
            continue
        if not set(props).issubset(allowed):
            socket.send_json(_reply(False, {}, {"request": f"properties_not_allowed: {props}"}))
            continue

        errors: Dict[str, str] = {}
        values: Dict[str, Optional[float]] = {p: None for p in props}

        try:
            atoms = load_atoms(cif_text)
        except Exception as exc:
            socket.send_json(_reply(False, values, {"atoms": f"{type(exc).__name__}: {exc}"}))
            continue

        ok = True
        for prop in props:
            try:
                values[prop] = predict_property(
                    prop=prop,
                    atoms=atoms,
                    cutoff=args.cutoff,
                    max_neighbors=args.max_neighbors,
                    model_override=_model_override(args, prop),
                )
            except Exception as exc:
                ok = False
                errors[prop] = f"{type(exc).__name__}: {exc}"

        socket.send_json(_reply(ok, values, errors))


if __name__ == "__main__":
    main()
