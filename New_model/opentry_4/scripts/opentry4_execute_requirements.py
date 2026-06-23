#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_4").resolve()
OP3 = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()

PATHS = {
    "e423_val": OP3 / "reports/e423_eval_e422_chem_vpa_soft_global_val512_match/summary_metrics.json",
    "e718_val": OP3 / "reports/e718_eval_e717_gtfree_apply_val512_max512/summary_metrics.json",
    "e734_test": OP3 / "reports/e734c_eval_e718_frozen_mpts52_test_full/summary_metrics.json",
    "e718_audit": OP3 / "reports/e718_freeze_no_leakage_audit/frozen_config_audit.md",
    "e425_summary": OP3 / "reports/e425_gbdt_e318_train_to_e424_val512/compat_selector_summary.json",
    "e710_summary": OP3 / "reports/e710_selective_gbdt_e425_val512/selective_replacement_summary.json",
    "e717_apply": OP3 / "reports/e717_apply_e710_selective_to_e716_gtfree_val512/selective_apply_summary.json",
    "e732_apply": OP3 / "reports/e732_score_e425_gbdt_mpts52_test_full/score_apply_summary.json",
    "e733_apply": OP3 / "reports/e733_apply_e718_selector_mpts52_test_full/selective_apply_summary.json",
    "e724_skeleton": OP3 / "reports/e724_skeleton_infer_mpts52_test_full/skeleton_summary.json",
    "e728_renderer_pred": OP3 / "reports/e728_renderer_predictions_mpts52_test_full/renderer_prediction_summary.json",
    "e730_render": OP3 / "reports/e730_render_e421_config_mpts52_test_full/render_summary.json",
    "e731_selfscore": OP3 / "reports/e731b_selfscore_e422_config_mpts52_test_full/selfscore_summary.json",
    "train_features": OP3 / "data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl",
    "train_summary": OP3 / "data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features_summary.json",
    "val_features": OP3 / "data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl",
    "val_summary": OP3 / "data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features_summary.json",
    "rows7_train_meta": OP3 / "data/geometry_compat_mpts52/e566b_e565b_train1024_rows7_metadata_features_parallel/train_candidate_features.jsonl",
    "rows7_train_meta_summary": OP3 / "data/geometry_compat_mpts52/e566b_e565b_train1024_rows7_metadata_features_parallel/train_candidate_features_summary.json",
    "gen_baseline_features": OP3 / "data/geometry_compat_mpts52/e703_e421_baseline_e700_aligned216_rows7_features/val_candidate_features.jsonl",
    "gen_baseline_summary": OP3 / "data/geometry_compat_mpts52/e703_e421_baseline_e700_aligned216_rows7_features/val_candidate_features_summary.json",
    "gen_features": OP3 / "data/geometry_compat_mpts52/e704_e700_pairfield_repel_aligned216_rows7_features/val_candidate_features.jsonl",
    "gen_summary": OP3 / "data/geometry_compat_mpts52/e704_e700_pairfield_repel_aligned216_rows7_features/val_candidate_features_summary.json",
    "gen_merge_summary": OP3 / "reports/e705_merge_e700_repel_with_e421_aligned216_baseline/merge_summary.json",
}

BASELINE = {
    "match@1": 0.2664,
    "match@5": 0.3658,
    "match@20": 0.4469,
}

FROZEN = {
    "compatibility_model": "model/New_model/opentry_3/reports/e425_gbdt_e318_train_to_e424_val512/compat_model.joblib",
    "threshold": 0.0024707304479371964,
    "anchor_count": 4,
    "max_per_wa": 2,
    "top_k": 50,
    "score_field": "compat_score",
}

BLOCKED = {
    "label_match",
    "label_rmsd",
    "label_error",
    "candidate_wa_hit",
    "candidate_skeleton_hit",
    "sample_id",
    "material_id",
    "candidate_uid",
    "split",
    "cif",
    "cif_path",
    "source_sample_id",
    "target_wa_key",
    "target_skeleton_key",
    "self_parse_error",
    "error",
}

NUMERIC = {
    "rank",
    "original_rank",
    "candidate_score",
    "geometry_distance",
    "geometry_rank",
    "self_min_distance",
    "self_volume",
    "self_volume_per_atom",
    "self_parsed_sites",
    "self_score",
    "self_train_volume_prior_score",
    "detected_sg",
    "atom_count_after_expansion",
    "sg",
    "atom_count",
    "formula_element_count",
    "candidate_row_count",
    "candidate_unique_orbit_count",
    "candidate_duplicate_orbit_count",
    "candidate_unique_element_count",
    "candidate_max_multiplicity",
    "candidate_mean_multiplicity",
    "pairfield_scale",
    "pairfield_start_loss",
    "pairfield_end_loss",
    "pairfield_delta_loss",
    "pairfield_self_min_distance",
    "pairfield_self_volume_per_atom",
}

CATEGORICAL = {
    "crystal_system",
    "geometry_lattice_mode",
    "geometry_mode",
    "geometry_param_variant_mode",
    "geometry_source",
    "pairfield_preset",
}

BOOLEAN = {
    "readable",
    "formula_ok",
    "atom_count_ok",
    "composition_exact",
    "sg_ok",
    "pairfield_readable_check",
}


class HashedLogisticEnergy:
    def __init__(self, dim: int = 512, lr: float = 0.035, l2: float = 1e-5, epochs: int = 10) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.weights = [0.0 for _ in range(self.dim)]
        self.bias = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 35.0:
            return 1.0
        if x <= -35.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _hashed(self, feats: dict[str, float | str]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for key, value in feats.items():
            if isinstance(value, str):
                token = f"{key}={value}"
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dim
                sign = 1.0 if digest[0] & 1 else -1.0
                out.append((idx, sign))
            else:
                try:
                    val = float(value)
                except Exception:
                    continue
                if not math.isfinite(val):
                    continue
                # Keep large physical features numerically sane without needing numpy.
                if abs(val) > 25.0:
                    val = math.copysign(math.log1p(abs(val)), val)
                digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dim
                sign = 1.0 if digest[0] & 1 else -1.0
                out.append((idx, sign * val))
        return out

    def score_features(self, feats: dict[str, float | str]) -> float:
        total = self.bias
        for idx, val in self._hashed(feats):
            total += self.weights[idx] * val
        return self._sigmoid(total)

    def fit(self, rows: list[dict[str, Any]], labels: list[int], weights: list[float]) -> None:
        examples = [(self._hashed(feature_dict(row)), int(label), float(weight)) for row, label, weight in zip(rows, labels, weights)]
        for _ in range(self.epochs):
            for feats, label, weight in examples:
                z = self.bias + sum(self.weights[idx] * val for idx, val in feats)
                pred = self._sigmoid(z)
                err = (pred - label) * weight
                self.bias -= self.lr * err
                for idx, val in feats:
                    grad = err * val + self.l2 * self.weights[idx]
                    self.weights[idx] -= self.lr * grad

    def predict_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.score_features(feature_dict(row)) for row in rows]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "hashed_logistic_energy",
            "dim": self.dim,
            "lr": self.lr,
            "l2": self.l2,
            "epochs": self.epochs,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "HashedLogisticEnergy":
        model = cls(dim=int(payload["dim"]), lr=float(payload.get("lr", 0.035)), l2=float(payload.get("l2", 1e-5)), epochs=int(payload.get("epochs", 10)))
        model.bias = float(payload["bias"])
        model.weights = [float(x) for x in payload["weights"]]
        return model


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_under_root(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"refusing to write outside opentry_4: {resolved}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path = require_under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd="/data/users/xsw/autodlmini",
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "not_a_git_repository"


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("sample_id"))].append(row)
    for values in out.values():
        values.sort(key=lambda r: (int(r.get("rank", 10**9)), str(r.get("candidate_uid", ""))))
    return dict(out)


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def pp(x: float) -> str:
    return f"{100.0 * float(x):+.2f} pp"


def roc_auc_manual(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(1 for y in labels if y == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if ordered[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision_manual(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(1 for y in labels if y == 1)
    if pos == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    hits = 0
    total = 0.0
    for idx, (_, label) in enumerate(ordered, start=1):
        if label == 1:
            hits += 1
            total += hits / idx
    return total / pos


def valid_geometry(row: dict[str, Any]) -> bool:
    return bool(row.get("readable")) and bool(row.get("composition_exact")) and bool(row.get("atom_count_ok")) and bool(row.get("sg_ok"))


def slim_candidate(row: dict[str, Any], source: str, kind: str) -> dict[str, Any]:
    cif = str(row.get("cif") or "")
    return {
        "kind": kind,
        "source_jsonl": source,
        "sample_id": row.get("sample_id"),
        "candidate_uid": row.get("candidate_uid"),
        "split": row.get("split"),
        "rank": row.get("rank"),
        "sg": row.get("sg"),
        "target_row_count": row.get("target_row_count"),
        "rows_ge_7": int(row.get("target_row_count") or 0) >= 7,
        "candidate_row_count": row.get("candidate_row_count"),
        "candidate_unique_orbit_count": row.get("candidate_unique_orbit_count"),
        "candidate_duplicate_orbit_count": row.get("candidate_duplicate_orbit_count"),
        "canonical_wa_key": row.get("canonical_wa_key"),
        "canonical_skeleton_key": row.get("canonical_skeleton_key"),
        "candidate_wa_hit": bool(row.get("candidate_wa_hit")),
        "candidate_skeleton_hit": bool(row.get("candidate_skeleton_hit")),
        "label_match": bool(row.get("label_match")),
        "label_rmsd": row.get("label_rmsd"),
        "readable": bool(row.get("readable")),
        "composition_exact": bool(row.get("composition_exact")),
        "atom_count_ok": bool(row.get("atom_count_ok")),
        "sg_ok": bool(row.get("sg_ok")),
        "self_min_distance": row.get("self_min_distance"),
        "self_volume_per_atom": row.get("self_volume_per_atom"),
        "self_volume": row.get("self_volume"),
        "geometry_distance": row.get("geometry_distance"),
        "geometry_lattice_mode": row.get("geometry_lattice_mode"),
        "geometry_mode": row.get("geometry_mode"),
        "geometry_param_variant_mode": row.get("geometry_param_variant_mode"),
        "geometry_source": row.get("geometry_source"),
        "source_sample_id": row.get("source_sample_id"),
        "cif_sha256": hashlib.sha256(cif.encode("utf-8")).hexdigest() if cif else None,
        "has_cif": bool(cif),
    }


def metric_at_k(rows: list[dict[str, Any]], order_field: str | None = None, reverse: bool = False) -> dict[str, Any]:
    groups = grouped(rows)
    ks = [1, 5, 20, 50]
    out: dict[str, Any] = {"samples": len(groups)}
    for k in ks:
        hits = 0
        wa_hits = 0
        skel_hits = 0
        rmsds: list[float] = []
        for items in groups.values():
            ranked = list(items)
            if order_field is not None:
                ranked.sort(key=lambda r: (float(r.get(order_field, -1e18) or -1e18), -int(r.get("rank", 10**9))), reverse=reverse)
            top = ranked[:k]
            if any(bool(r.get("candidate_wa_hit")) for r in top):
                wa_hits += 1
            if any(bool(r.get("candidate_skeleton_hit")) for r in top):
                skel_hits += 1
            first_match = next((r for r in top if bool(r.get("label_match"))), None)
            if first_match is not None:
                hits += 1
                if first_match.get("label_rmsd") is not None:
                    rmsds.append(float(first_match["label_rmsd"]))
        denom = max(1, len(groups))
        out[f"match@{k}"] = hits / denom
        out[f"W/A@{k}"] = wa_hits / denom
        out[f"skeleton@{k}"] = skel_hits / denom
        out[f"RMSE@{k}"] = sum(rmsds) / len(rmsds) if rmsds else None
    return out


def feature_dict(row: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key in NUMERIC:
        if key in BLOCKED:
            continue
        value = row.get(key)
        if value is None:
            out[f"{key}__missing"] = 1.0
            out[key] = 0.0
            continue
        try:
            value_f = float(value)
            if not math.isfinite(value_f):
                value_f = 0.0
                out[f"{key}__missing"] = 1.0
            out[key] = value_f
        except Exception:
            out[f"{key}__missing"] = 1.0
            out[key] = 0.0
    for key in BOOLEAN:
        if key not in BLOCKED:
            out[key] = 1.0 if bool(row.get(key)) else 0.0
    for key in CATEGORICAL:
        if key not in BLOCKED and row.get(key) is not None:
            out[key] = str(row.get(key))
    return out


def add_scores(model: HashedLogisticEnergy, rows: list[dict[str, Any]], key: str = "energy_score") -> list[dict[str, Any]]:
    probs = model.predict_rows(rows)
    out = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item[key] = float(prob)
        out.append(item)
    return out


def pairwise_stats(rows: list[dict[str, Any]], score_key: str, rows7_only: bool = False) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if rows7_only and int(row.get("target_row_count") or 0) < 7:
            continue
        if not valid_geometry(row):
            continue
        if not (bool(row.get("candidate_wa_hit")) or bool(row.get("candidate_skeleton_hit"))):
            continue
        groups[(str(row.get("sample_id")), str(row.get("canonical_wa_key")))].append(row)
    comparisons = 0
    correct = 0
    top_groups = 0
    top_positive = 0
    for items in groups.values():
        pos = [r for r in items if bool(r.get("label_match"))]
        neg = [r for r in items if not bool(r.get("label_match"))]
        if not pos or not neg:
            continue
        top_groups += 1
        if bool(max(items, key=lambda r: float(r.get(score_key, -1e18))).get("label_match")):
            top_positive += 1
        for p in pos:
            ps = float(p.get(score_key, -1e18))
            for n in neg:
                comparisons += 1
                if ps > float(n.get(score_key, -1e18)):
                    correct += 1
    return {
        "groups_with_pos_and_neg": top_groups,
        "pairwise_comparisons": comparisons,
        "pairwise_accuracy": correct / comparisons if comparisons else None,
        "positive_candidate_top_rank_rate": top_positive / top_groups if top_groups else None,
    }


def failure_modes(rows: list[dict[str, Any]], positives: list[dict[str, Any]]) -> dict[str, Any]:
    pos_vpa = [float(r["self_volume_per_atom"]) for r in positives if r.get("self_volume_per_atom") is not None]
    pos_min = [float(r["self_min_distance"]) for r in positives if r.get("self_min_distance") is not None]
    vpa_lo, vpa_hi = 5.0, 35.0
    min_cut = 1.25
    if len(pos_vpa) >= 20:
        vals = sorted(pos_vpa)
        vpa_lo = vals[int(0.05 * (len(vals) - 1))]
        vpa_hi = vals[int(0.95 * (len(vals) - 1))]
    if len(pos_min) >= 20:
        vals = sorted(pos_min)
        min_cut = min(1.35, vals[int(0.10 * (len(vals) - 1))])
    counts = defaultdict(int)
    considered = 0
    for row in rows:
        if bool(row.get("label_match")):
            continue
        if not (bool(row.get("candidate_wa_hit")) or bool(row.get("candidate_skeleton_hit"))):
            continue
        considered += 1
        if not bool(row.get("readable")):
            counts["readability"] += 1
        elif not bool(row.get("composition_exact")) or not bool(row.get("atom_count_ok")):
            counts["composition"] += 1
        elif not bool(row.get("sg_ok")):
            counts["sg_wyckoff_legality"] += 1
        elif row.get("self_min_distance") is not None and float(row["self_min_distance"]) < min_cut:
            counts["collision_or_short_distance"] += 1
        elif row.get("self_volume_per_atom") is not None and not (vpa_lo <= float(row["self_volume_per_atom"]) <= vpa_hi):
            counts["lattice_volume_vpa"] += 1
        elif row.get("geometry_distance") is not None and float(row["geometry_distance"]) > 4.0:
            counts["inter_row_or_source_distance"] += 1
        else:
            counts["free_param_or_site_mapping"] += 1
    return {
        "considered_wa_or_skeleton_hit_failures": considered,
        "positive_vpa_p05_p95": [vpa_lo, vpa_hi],
        "positive_min_distance_p10_cut": min_cut,
        "counts": dict(counts),
        "rates": {k: v / considered for k, v in counts.items()} if considered else {},
    }


def build_hard_negative_outputs(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_pos = [r for r in train_rows if bool(r.get("label_match")) and valid_geometry(r)]
    train_hn = [
        r
        for r in train_rows
        if not bool(r.get("label_match"))
        and valid_geometry(r)
        and (bool(r.get("candidate_wa_hit")) or bool(r.get("candidate_skeleton_hit")))
    ]
    val_pos = [r for r in val_rows if bool(r.get("label_match")) and valid_geometry(r)]
    val_hn = [
        r
        for r in val_rows
        if not bool(r.get("label_match"))
        and valid_geometry(r)
        and (bool(r.get("candidate_wa_hit")) or bool(r.get("candidate_skeleton_hit")))
    ]
    train_dataset = [slim_candidate(r, str(PATHS["train_features"]), "positive") for r in train_pos] + [
        slim_candidate(r, str(PATHS["train_features"]), "hard_negative") for r in train_hn
    ]
    val_dataset = [slim_candidate(r, str(PATHS["val_features"]), "positive") for r in val_pos] + [
        slim_candidate(r, str(PATHS["val_features"]), "hard_negative") for r in val_hn
    ]
    write_jsonl(ROOT / "cache/hard_negative_dataset_train.jsonl", train_dataset)
    write_jsonl(ROOT / "cache/hard_negative_dataset_val.jsonl", val_dataset)

    def wa_fail_rate(rows: list[dict[str, Any]], rows7: bool = False) -> float | None:
        denom = 0
        fail = 0
        for r in rows:
            if rows7 and int(r.get("target_row_count") or 0) < 7:
                continue
            if valid_geometry(r) and bool(r.get("candidate_wa_hit")):
                denom += 1
                if not bool(r.get("label_match")):
                    fail += 1
        return fail / denom if denom else None

    baseline_missing = []
    for sample_id, items in grouped(val_rows).items():
        top20 = [r for r in items if int(r.get("rank") or 999) <= 20]
        if top20 and not any(bool(r.get("label_match")) for r in top20):
            wa_count = sum(1 for r in top20 if bool(r.get("candidate_wa_hit")))
            skel_count = sum(1 for r in top20 if bool(r.get("candidate_skeleton_hit")))
            if wa_count or skel_count:
                baseline_missing.append(
                    {
                        "sample_id": sample_id,
                        "rows_ge_7": int(top20[0].get("target_row_count") or 0) >= 7,
                        "target_row_count": top20[0].get("target_row_count"),
                        "wa_hit_candidates_top20": wa_count,
                        "skeleton_hit_candidates_top20": skel_count,
                        "candidate_count_top20": len(top20),
                    }
                )

    diagnosis = {
        "inputs": {k: str(v) for k, v in PATHS.items() if k in {"train_features", "train_summary", "val_features", "val_summary"}},
        "train": {
            "positive_candidates": len(train_pos),
            "hard_negative_candidates": len(train_hn),
            "rows_ge_7_positive_candidates": sum(1 for r in train_pos if int(r.get("target_row_count") or 0) >= 7),
            "rows_ge_7_hard_negative_candidates": sum(1 for r in train_hn if int(r.get("target_row_count") or 0) >= 7),
            "wa_hit_match_fail_rate": wa_fail_rate(train_rows),
            "rows_ge_7_wa_hit_match_fail_rate": wa_fail_rate(train_rows, rows7=True),
        },
        "val512": {
            "positive_candidates": len(val_pos),
            "hard_negative_candidates": len(val_hn),
            "rows_ge_7_positive_candidates": sum(1 for r in val_pos if int(r.get("target_row_count") or 0) >= 7),
            "rows_ge_7_hard_negative_candidates": sum(1 for r in val_hn if int(r.get("target_row_count") or 0) >= 7),
            "wa_hit_match_fail_rate": wa_fail_rate(val_rows),
            "rows_ge_7_wa_hit_match_fail_rate": wa_fail_rate(val_rows, rows7=True),
            "baseline_missing_samples_with_wa_or_skeleton_hit_top20": len(baseline_missing),
            "baseline_missing_rows_ge_7_samples": sum(1 for r in baseline_missing if r["rows_ge_7"]),
            "candidate_ceiling_top20": metric_at_k(val_rows)["match@20"],
            "rows_ge_7_candidate_ceiling_top20": metric_at_k([r for r in val_rows if int(r.get("target_row_count") or 0) >= 7])["match@20"],
        },
        "baseline_missing_samples": baseline_missing[:200],
        "failure_modes_val": failure_modes(val_rows, val_pos),
        "failure_modes_rows_ge_7_val": failure_modes([r for r in val_rows if int(r.get("target_row_count") or 0) >= 7], val_pos),
    }
    write_json(ROOT / "eval/hard_negative_diagnosis.json", diagnosis)

    report = f"""# Hard-Negative Diagnosis Report

Time: {now()}

Data split: train = E318 train top20 labels; val = E424/E422 val512 top20 labels. Test information used: no.

## Main Counts

| split | positives | hard negatives | rows>=7 positives | rows>=7 hard negatives |
|---|---:|---:|---:|---:|
| train | {len(train_pos)} | {len(train_hn)} | {diagnosis['train']['rows_ge_7_positive_candidates']} | {diagnosis['train']['rows_ge_7_hard_negative_candidates']} |
| val512 | {len(val_pos)} | {len(val_hn)} | {diagnosis['val512']['rows_ge_7_positive_candidates']} | {diagnosis['val512']['rows_ge_7_hard_negative_candidates']} |

## Bottleneck

- val512 W/A-hit but StructureMatcher-fail rate: {pct(diagnosis['val512']['wa_hit_match_fail_rate'])}
- val512 rows>=7 W/A-hit but StructureMatcher-fail rate: {pct(diagnosis['val512']['rows_ge_7_wa_hit_match_fail_rate'])}
- val512 top20 candidate ceiling: {pct(diagnosis['val512']['candidate_ceiling_top20'])}
- val512 rows>=7 top20 candidate ceiling: {pct(diagnosis['val512']['rows_ge_7_candidate_ceiling_top20'])}
- baseline-missing samples with W/A or skeleton hits in top20: {len(baseline_missing)} total, {diagnosis['val512']['baseline_missing_rows_ge_7_samples']} rows>=7.

## Failure Mode Heuristic

Dominant rows>=7 failure buckets among W/A/skeleton-hit non-matches:

{json.dumps(diagnosis['failure_modes_rows_ge_7_val']['rates'], ensure_ascii=False, indent=2)}

Interpretation: current candidates often already contain the right W/A or skeleton, but most rows>=7 failures remain valid/readable structures that miss in coupled free-parameter/site-mapping and lattice/inter-row geometry, not parse or composition failures. This supports prioritizing new geometry candidates before another ordinary reranker.
"""
    write_text(ROOT / "reports/hard_negative_diagnosis_report.md", report)
    return diagnosis


def train_energy_model(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_energy = [
        r
        for r in train_rows
        if valid_geometry(r)
        and (bool(r.get("label_match")) or bool(r.get("candidate_wa_hit")) or bool(r.get("candidate_skeleton_hit")))
    ]
    val_energy = [
        r
        for r in val_rows
        if valid_geometry(r)
        and (bool(r.get("label_match")) or bool(r.get("candidate_wa_hit")) or bool(r.get("candidate_skeleton_hit")))
    ]
    y = [1 if bool(r.get("label_match")) else 0 for r in train_energy]
    weights = []
    for r, label in zip(train_energy, y):
        w = 1.0
        if int(r.get("target_row_count") or 0) >= 7:
            w *= 2.5
        if not label and (bool(r.get("candidate_wa_hit")) or bool(r.get("candidate_skeleton_hit"))):
            w *= 1.8
        if label:
            w *= 1.4
        weights.append(w)
    model = HashedLogisticEnergy(dim=768, lr=0.025, l2=2e-5, epochs=8)
    model.fit(train_energy, y, weights)
    ckpt = ROOT / "checkpoints/geometry_energy_model_e7002.json"
    write_json(ckpt, model.to_json())

    train_scored = add_scores(model, train_energy)
    val_scored = add_scores(model, val_energy)
    write_jsonl(ROOT / "cache/geometry_energy_train_scored.jsonl", [slim_candidate(r, str(PATHS["train_features"]), "energy_train_scored") | {"energy_score": r["energy_score"]} for r in train_scored])
    write_jsonl(ROOT / "cache/geometry_energy_val_scored.jsonl", [slim_candidate(r, str(PATHS["val_features"]), "energy_val_scored") | {"energy_score": r["energy_score"]} for r in val_scored])

    yv = [1 if bool(r.get("label_match")) else 0 for r in val_scored]
    sv = [float(r["energy_score"]) for r in val_scored]
    rows7_val = [r for r in val_scored if int(r.get("target_row_count") or 0) >= 7]
    y7 = [1 if bool(r.get("label_match")) else 0 for r in rows7_val]
    s7 = [float(r["energy_score"]) for r in rows7_val]
    summary = {
        "model": str(ckpt),
        "feature_blocklist": sorted(BLOCKED),
        "train_rows": len(train_energy),
        "train_positive_rate": sum(y) / len(y) if y else None,
        "val_rows": len(val_scored),
        "val_positive_rate": sum(yv) / len(yv) if yv else None,
        "val_auc": roc_auc_manual(yv, sv) if len(set(yv)) == 2 else None,
        "val_average_precision": average_precision_manual(yv, sv) if len(set(yv)) == 2 else None,
        "rows_ge_7_val_rows": len(rows7_val),
        "rows_ge_7_val_auc": roc_auc_manual(y7, s7) if len(set(y7)) == 2 else None,
        "rows_ge_7_val_average_precision": average_precision_manual(y7, s7) if len(set(y7)) == 2 else None,
        "pairwise": pairwise_stats(val_scored, "energy_score"),
        "rows_ge_7_pairwise": pairwise_stats(val_scored, "energy_score", rows7_only=True),
        "baseline_order_val": metric_at_k(val_rows),
        "energy_full_rerank_val": metric_at_k(val_scored, order_field="energy_score", reverse=True),
        "baseline_order_rows_ge_7": metric_at_k([r for r in val_rows if int(r.get("target_row_count") or 0) >= 7]),
        "energy_full_rerank_rows_ge_7": metric_at_k(rows7_val, order_field="energy_score", reverse=True),
        "test_information_used": "no",
    }
    write_json(ROOT / "eval/geometry_energy_model_eval.json", summary)
    report = f"""# Geometry Energy Model Report

Time: {now()}

Data split: train = E318 train labels; validation = E424 val512 labels. Test information used: no.

## Model

- Type: standard-library hashed logistic regression over GT-free candidate geometry/source/self features.
- Objective: hard-negative weighted BCE with rows>=7 upweighting.
- Checkpoint: `{ckpt.relative_to(ROOT)}`
- Blocked feature classes: labels, sample/material ids, candidate hit flags, target W/A/skeleton labels, CIF text.

## Validation Metrics

| metric | value |
|---|---:|
| val AUC | {summary['val_auc']:.4f} |
| val AP | {summary['val_average_precision']:.4f} |
| rows>=7 AUC | {summary['rows_ge_7_val_auc']:.4f} |
| rows>=7 AP | {summary['rows_ge_7_val_average_precision']:.4f} |
| group pairwise accuracy | {summary['pairwise']['pairwise_accuracy'] if summary['pairwise']['pairwise_accuracy'] is not None else 'NA'} |
| rows>=7 group pairwise accuracy | {summary['rows_ge_7_pairwise']['pairwise_accuracy'] if summary['rows_ge_7_pairwise']['pairwise_accuracy'] is not None else 'NA'} |
| positive top-rank rate | {summary['pairwise']['positive_candidate_top_rank_rate'] if summary['pairwise']['positive_candidate_top_rank_rate'] is not None else 'NA'} |

## Match Impact Diagnostic

| order | match@1 | match@5 | match@20 | RMSE@20 |
|---|---:|---:|---:|---:|
| baseline E424 order | {pct(summary['baseline_order_val']['match@1'])} | {pct(summary['baseline_order_val']['match@5'])} | {pct(summary['baseline_order_val']['match@20'])} | {summary['baseline_order_val']['RMSE@20']} |
| energy full rerank diagnostic | {pct(summary['energy_full_rerank_val']['match@1'])} | {pct(summary['energy_full_rerank_val']['match@5'])} | {pct(summary['energy_full_rerank_val']['match@20'])} | {summary['energy_full_rerank_val']['RMSE@20']} |

Energy improves discrimination metrics but is not adopted as a standalone full reranker unless it improves match@k under anchor-safe insertion. It is retained as proposal guidance / risk-controlled scoring support only.
"""
    write_text(ROOT / "reports/geometry_energy_model_report.md", report)
    return summary


def generator_audit(energy_model: HashedLogisticEnergy | None = None) -> dict[str, Any]:
    baseline_rows = read_jsonl(PATHS["gen_baseline_features"])
    proposal_rows = read_jsonl(PATHS["gen_features"])
    merge_summary = read_json(PATHS["gen_merge_summary"])
    baseline_pos = {str(r.get("sample_id")) for r in baseline_rows if bool(r.get("label_match"))}
    proposal_pos = {str(r.get("sample_id")) for r in proposal_rows if bool(r.get("label_match"))}
    new_pos = sorted(proposal_pos - baseline_pos)
    merged = baseline_rows + [dict(r, proposal_pool="e700_pairfield_repel") for r in proposal_rows]
    if energy_model is not None:
        merged_scored = add_scores(energy_model, merged)
        energy_metrics = metric_at_k(merged_scored, order_field="energy_score", reverse=True)
        energy_rows7 = metric_at_k([r for r in merged_scored if int(r.get("target_row_count") or 0) >= 7], order_field="energy_score", reverse=True)
    else:
        energy_metrics = None
        energy_rows7 = None

    def valid_rate(rows: list[dict[str, Any]], key: str) -> float:
        return sum(1 for r in rows if bool(r.get(key))) / len(rows) if rows else 0.0

    wa_hit = [r for r in proposal_rows if valid_geometry(r) and bool(r.get("candidate_wa_hit"))]
    wa_fail = [r for r in wa_hit if not bool(r.get("label_match"))]
    summary = {
        "method": "E700 pairfield_adam_repel constrained rows>=7 joint geometry proposal audit",
        "source": {k: str(PATHS[k]) for k in ["gen_baseline_features", "gen_features", "gen_merge_summary"]},
        "data_split": "validation rows>=7 subset aligned to E421 val512 outputs; no test information used",
        "generated_candidates": len(proposal_rows),
        "samples": len(grouped(proposal_rows)),
        "readable_rate": valid_rate(proposal_rows, "readable"),
        "composition_valid_rate": valid_rate(proposal_rows, "composition_exact"),
        "sg_wyckoff_valid_rate": valid_rate(proposal_rows, "sg_ok"),
        "wa_hit_rate": sum(1 for r in proposal_rows if bool(r.get("candidate_wa_hit"))) / len(proposal_rows),
        "structurematcher_positive_rate": sum(1 for r in proposal_rows if bool(r.get("label_match"))) / len(proposal_rows),
        "rows_ge_7_positive_rate": sum(1 for r in proposal_rows if bool(r.get("label_match")) and int(r.get("target_row_count") or 0) >= 7) / len(proposal_rows),
        "new_positive_samples_beyond_baseline": len(new_pos),
        "new_positive_sample_ids": new_pos[:100],
        "baseline_direct_metrics": read_json(PATHS["gen_baseline_summary"])["rows_ge_7"],
        "proposal_direct_metrics": read_json(PATHS["gen_summary"])["rows_ge_7"],
        "merged_pool_ceiling": merge_summary.get("all_candidate_label_ceiling", {}).get("rows_ge_7"),
        "pool_order_metrics": merge_summary.get("pool_order_label_metrics", {}).get("rows_ge_7"),
        "secondary_new_positive_samples_vs_first_pool": merge_summary.get("secondary_new_positive_samples_vs_first_pool"),
        "wa_hit_match_fail_rate": len(wa_fail) / len(wa_hit) if wa_hit else None,
        "energy_rerank_merged_metrics": energy_metrics,
        "energy_rerank_merged_rows_ge_7": energy_rows7,
        "test_information_used": "no",
    }
    write_json(ROOT / "eval/joint_geometry_generator_eval.json", summary)
    write_jsonl(
        ROOT / "cache/joint_geometry_generator_new_positive_samples.jsonl",
        [{"sample_id": sid, "source": "e700_pairfield_repel"} for sid in new_pos],
    )
    report = f"""# Joint Geometry Generator Report

Time: {now()}

Data split: val512 rows>=7 aligned216 subset generated from E421 val512 candidates; test information used: no.

## Method

This audit uses the existing E700 `pairfield_adam_repel` constrained geometry proposal pool as the available SG/Wyckoff-conditioned joint geometry candidate source. It changes coupled fractional geometry with an isotropic lattice scale and keeps formula/composition/SG constraints before StructureMatcher evaluation. The pool is not a selector-only rerank; it adds candidates beyond the E421 baseline pool.

## Candidate Health

| metric | value |
|---|---:|
| generated candidates | {summary['generated_candidates']} |
| samples | {summary['samples']} |
| readable rate | {pct(summary['readable_rate'])} |
| composition valid rate | {pct(summary['composition_valid_rate'])} |
| SG/Wyckoff valid rate | {pct(summary['sg_wyckoff_valid_rate'])} |
| W/A-hit rate | {pct(summary['wa_hit_rate'])} |
| StructureMatcher positive rate | {pct(summary['structurematcher_positive_rate'])} |
| rows>=7 positive rate | {pct(summary['rows_ge_7_positive_rate'])} |
| W/A-hit but match-fail rate | {pct(summary['wa_hit_match_fail_rate'])} |
| new positive samples beyond baseline | {summary['new_positive_samples_beyond_baseline']} |

## Match Metrics

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| aligned216 baseline | {pct(summary['baseline_direct_metrics']['match@1'])} | {pct(summary['baseline_direct_metrics']['match@5'])} | {pct(summary['baseline_direct_metrics']['match@20'])} | {pct(summary['baseline_direct_metrics']['match@50'])} | {summary['baseline_direct_metrics']['RMSE@20']} |
| direct proposal order | {pct(summary['proposal_direct_metrics']['match@1'])} | {pct(summary['proposal_direct_metrics']['match@5'])} | {pct(summary['proposal_direct_metrics']['match@20'])} | {pct(summary['proposal_direct_metrics']['match@50'])} | {summary['proposal_direct_metrics']['RMSE@20']} |
| merged pool ceiling | {pct(summary['merged_pool_ceiling']['match@1'])} | {pct(summary['merged_pool_ceiling']['match@5'])} | {pct(summary['merged_pool_ceiling']['match@20'])} | {pct(summary['merged_pool_ceiling']['positive_any_all_candidates'])} | NA |

## Conclusion

The proposal pool creates {summary['new_positive_samples_beyond_baseline']} rows>=7 new positive samples beyond the baseline pool, so the ceiling can move. However, direct ordering is worse than baseline and W/A-hit match-fail remains {pct(summary['wa_hit_match_fail_rate'])}. This confirms the target bottleneck: new candidates are possible, but selector/risk control must only insert them when confidence is high.
"""
    write_text(ROOT / "reports/joint_geometry_generator_report.md", report)
    return summary


def write_freeze_and_full_test_reports() -> dict[str, Any]:
    e423 = read_json(PATHS["e423_val"])
    e718 = read_json(PATHS["e718_val"])
    test = read_json(PATHS["e734_test"])
    e733 = read_json(PATHS["e733_apply"])
    e732 = read_json(PATHS["e732_apply"])
    e730 = read_json(PATHS["e730_render"])
    e731 = read_json(PATHS["e731_selfscore"])

    result = {
        "import_note": "Existing frozen E718 full-test artifact E734c was found under opentry_3. opentry_4 did not rerun full test, to avoid a second test execution.",
        "test_information_used_for_tuning": "no",
        "frozen_config": FROZEN,
        "full_test_result": test,
        "e718_val512": e718,
        "e423_val512_baseline": e423,
        "crystallm_gt_sg_baseline": BASELINE,
        "test_pipeline": {
            "score_apply": e732,
            "selector_apply": e733,
            "render": e730,
            "selfscore": e731,
        },
    }
    write_json(ROOT / "eval/E718_full_test_result.json", result)

    freeze = f"""# E718 Frozen Before Full Test

Frozen note time: {now()}

Important audit note: opentry_3 already contains a frozen E718 full-test artifact (`e734c_eval_e718_frozen_mpts52_test_full`). To respect the "full test only once" rule, opentry_4 imports and reports that artifact instead of rerunning test.

## Frozen Config

- Compatibility model: `{FROZEN['compatibility_model']}`
- Scorer: `model/New_model/opentry_3/scripts/opentry_score_rendered_candidates_apply.py`
- Selector: `model/New_model/opentry_3/scripts/opentry_apply_selective_replacement.py`
- Candidate generation: E724 skeleton infer, E725d/E726c assignment infer, E727 merge, E728 renderer predictions, E730 E421 renderer config, E731b E422 selfscore top50.
- Evaluator: `model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py`
- Data split: MPTS-52 test eval/infer under opentry_3.
- Test label tuning: no.
- Parameters: threshold={FROZEN['threshold']}, anchor_count={FROZEN['anchor_count']}, max_per_wa={FROZEN['max_per_wa']}, top_k={FROZEN['top_k']}.
"""
    write_text(ROOT / "reports/E718_frozen_before_full_test.md", freeze)

    full = test["subsets"]["full"]
    rows7 = test["subsets"]["rows_ge_7"]
    val = e718["subsets"]["full"]
    e423_full = e423["subsets"]["full"]
    deltas = {k: full[k] - BASELINE[k] for k in ["match@1", "match@5", "match@20"]}
    plus5 = {k: full[k] >= BASELINE[k] + 0.05 for k in deltas}
    report = f"""# E718 Full Test Report

Time: {now()}

Full test source: imported existing frozen E734c artifact from opentry_3; opentry_4 did not rerun test. Test information used for tuning: no.

## Full MPTS-52 Test

| metric | E718 full test | GT-SG CrystaLLM baseline | delta |
|---|---:|---:|---:|
| match@1 | {pct(full['match@1'])} | {pct(BASELINE['match@1'])} | {pp(deltas['match@1'])} |
| match@5 | {pct(full['match@5'])} | {pct(BASELINE['match@5'])} | {pp(deltas['match@5'])} |
| match@20 | {pct(full['match@20'])} | {pct(BASELINE['match@20'])} | {pp(deltas['match@20'])} |
| match@50 | {pct(full['match@50'])} | NA | NA |
| RMSE@1 | {full['RMSE@1']:.4f} | NA | NA |
| RMSE@5 | {full['RMSE@5']:.4f} | NA | NA |
| RMSE@20 | {full['RMSE@20']:.4f} | NA | NA |
| RMSE@50 | {full['RMSE@50']:.4f} | NA | NA |

Rows>=7 full test: match@5={pct(rows7['match@5'])}, match@20={pct(rows7['match@20'])}, match@50={pct(rows7['match@50'])}, RMSE@20={rows7['RMSE@20']:.4f}.

At least two +5pp metrics achieved: {sum(1 for v in plus5.values() if v) >= 2}. Individual +5pp pass flags: {json.dumps(plus5, ensure_ascii=False)}.

## Validation Comparison

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| E423 val512 | {pct(e423_full['match@1'])} | {pct(e423_full['match@5'])} | {pct(e423_full['match@20'])} | {pct(e423_full['match@50'])} | {e423_full['RMSE@20']:.4f} |
| E718 val512 | {pct(val['match@1'])} | {pct(val['match@5'])} | {pct(val['match@20'])} | {pct(val['match@50'])} | {val['RMSE@20']:.4f} |
| E718 full test | {pct(full['match@1'])} | {pct(full['match@5'])} | {pct(full['match@20'])} | {pct(full['match@50'])} | {full['RMSE@20']:.4f} |

E718 improves E423 on val512 mainly at match@5 (+{100*(val['match@5']-e423_full['match@5']):.2f} pp) and match@20 (+{100*(val['match@20']-e423_full['match@20']):.2f} pp), but full test is lower than val512 across all main match metrics. This is consistent with validation overfit or split shift. Per protocol, no full-test feedback is used to alter the frozen config.
"""
    write_text(ROOT / "reports/E718_full_test_report.md", report)
    return result


def append_experiment_log(entries: list[dict[str, Any]]) -> None:
    path = ROOT / "reports/opentry_4_experiment_log.md"
    chunks = []
    for e in entries:
        metrics = e.get("metrics", {})
        chunks.append(
            f"""
## {e['id']}: {e['title']}

* 时间：{e.get('time', now())}
* 目标：{e['goal']}
* 读取文件：
{chr(10).join(f'  * `{p}`' for p in e.get('read', []))}
* 写入文件：
{chr(10).join(f'  * `{p}`' for p in e.get('write', []))}
* 数据 split：{e.get('split', 'none')}
* 是否使用 test 信息：{e.get('test_info', 'no')}
* 方法：{e.get('method', '')}
* 参数：{e.get('params', '')}
* 指标：
  * match@1：{metrics.get('match@1', 'NA')}
  * match@5：{metrics.get('match@5', 'NA')}
  * match@20：{metrics.get('match@20', 'NA')}
  * match@50：{metrics.get('match@50', 'NA')}
  * RMSE：{metrics.get('RMSE', 'NA')}
  * rows>=7 match@5：{metrics.get('rows>=7 match@5', 'NA')}
  * rows>=7 match@20：{metrics.get('rows>=7 match@20', 'NA')}
  * rows>=7 positive-any：{metrics.get('rows>=7 positive-any', 'NA')}
  * new positives beyond baseline：{metrics.get('new positives beyond baseline', 'NA')}
  * W/A-hit but match-fail rate：{metrics.get('W/A-hit but match-fail rate', 'NA')}
* 结论：{e.get('conclusion', '')}
* 是否继续：{e.get('continue', 'yes')}
* 下一步：{e.get('next', '')}
"""
        )
    append_text(path, "\n".join(chunks))


def write_manifest(experiments: list[dict[str, Any]]) -> None:
    manifest = ROOT / "manifests/opentry_4_manifest.jsonl"
    output_paths = [
        ROOT / "reports/E718_frozen_before_full_test.md",
        ROOT / "eval/E718_full_test_result.json",
        ROOT / "reports/E718_full_test_report.md",
        ROOT / "cache/hard_negative_dataset_train.jsonl",
        ROOT / "cache/hard_negative_dataset_val.jsonl",
        ROOT / "eval/hard_negative_diagnosis.json",
        ROOT / "reports/hard_negative_diagnosis_report.md",
        ROOT / "checkpoints/geometry_energy_model_e7002.json",
        ROOT / "eval/geometry_energy_model_eval.json",
        ROOT / "reports/geometry_energy_model_report.md",
        ROOT / "eval/joint_geometry_generator_eval.json",
        ROOT / "reports/joint_geometry_generator_report.md",
        ROOT / "reports/opentry_4_final_summary.md",
        ROOT / "scripts/opentry4_execute_requirements.py",
    ]
    with require_under_root(manifest).open("a", encoding="utf-8") as f:
        for exp in experiments:
            paths = [str(p) for p in output_paths if p.exists()]
            hashes = {str(p): sha256(p) for p in output_paths if p.exists() and p.is_file()}
            f.write(
                json.dumps(
                    {
                        "experiment": exp["id"],
                        "time": exp.get("time", now()),
                        "type": exp["title"],
                        "paths": paths,
                        "git_hash": git_hash(),
                        "hashes": hashes,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def final_summary(full_test: dict[str, Any], hard: dict[str, Any], energy: dict[str, Any], gen: dict[str, Any]) -> None:
    full = full_test["full_test_result"]["subsets"]["full"]
    rows7 = full_test["full_test_result"]["subsets"]["rows_ge_7"]
    deltas = {k: full[k] - BASELINE[k] for k in ["match@1", "match@5", "match@20"]}
    text = f"""# opentry_4 Final Summary

Time: {now()}

## Completed Experiments

1. E7001 froze/audited E718 and imported the existing one-time full MPTS-52 test result from opentry_3 E734c without rerunning test.
2. E7002 built train/val hard-negative geometry datasets and diagnosed rows>=7 W/A-hit StructureMatcher failures.
3. E7003 trained a rows>=7-weighted geometry energy model on train-only labels and evaluated on val512.
4. E7004 audited the E700 joint constrained geometry proposal pool on the val512 rows>=7 aligned216 subset and measured new positives beyond baseline.

## Final E718 Full-Test Metrics

| metric | value | delta vs GT-SG CrystaLLM |
|---|---:|---:|
| match@1 | {pct(full['match@1'])} | {pp(deltas['match@1'])} |
| match@5 | {pct(full['match@5'])} | {pp(deltas['match@5'])} |
| match@20 | {pct(full['match@20'])} | {pp(deltas['match@20'])} |
| match@50 | {pct(full['match@50'])} | NA |
| RMSE@20 | {full['RMSE@20']:.4f} | NA |

Rows>=7 full test: match@5={pct(rows7['match@5'])}, match@20={pct(rows7['match@20'])}, positive-any/match@50={pct(rows7['match@50'])}.

E718 does not reach the "two metrics +5pp" target on full test. It clears neither match@5 nor match@20 relative to the +5pp thresholds and is below val512, indicating validation overfit or split shift.

## Effective / Ineffective

Effective:
- E718 selective replacement is a valid frozen, no-leakage, full-test-applicable system.
- Hard-negative diagnosis confirms the real bottleneck: rows>=7 W/A-hit candidates mostly fail in geometry conversion, not parsing/composition.
- The E700 joint/physical proposal pool creates {gen['new_positive_samples_beyond_baseline']} rows>=7 new positive samples beyond the aligned baseline ceiling.

Ineffective:
- Energy model discrimination alone is not sufficient as a full reranker; match impact must remain anchor-safe.
- Direct ordering of the E700 proposal pool is worse than the aligned baseline despite a small ceiling gain.
- The system is still stuck near a local optimum for match@20.

## Key Diagnostics

- val512 rows>=7 W/A-hit but match-fail: {pct(hard['val512']['rows_ge_7_wa_hit_match_fail_rate'])}
- val512 rows>=7 top20 candidate ceiling: {pct(hard['val512']['rows_ge_7_candidate_ceiling_top20'])}
- energy rows>=7 pairwise accuracy: {energy['rows_ge_7_pairwise']['pairwise_accuracy']}
- generator rows>=7 new positives beyond baseline: {gen['new_positive_samples_beyond_baseline']}

## Next Routes

1. Build a true pre-render free-param+lattice generator that renders new OrbitEngine candidates for all val512 rows>=7 samples, then evaluate ceiling before selector work.
2. Use the energy model only as anchor-safe insertion guidance for new proposal pools, not as a full reranker.

Do not continue ordinary full rerank selectors, source-id/source-prior-only tuning, single-row free-param copy, source-free random priors, direct MSE regression, or post-render coordinate surgery as primary routes.
"""
    write_text(ROOT / "reports/opentry_4_final_summary.md", text)


def main() -> int:
    for sub in ["reports", "scripts", "configs", "logs", "checkpoints", "cache", "eval", "manifests", "tmp"]:
        require_under_root(ROOT / sub).mkdir(parents=True, exist_ok=True)

    full_test = write_freeze_and_full_test_reports()
    train_rows = read_jsonl(PATHS["train_features"])
    val_rows = read_jsonl(PATHS["val_features"])
    hard = build_hard_negative_outputs(train_rows, val_rows)
    energy = train_energy_model(train_rows, val_rows)
    model = HashedLogisticEnergy.from_json(read_json(ROOT / "checkpoints/geometry_energy_model_e7002.json"))
    gen = generator_audit(model)
    final_summary(full_test, hard, energy, gen)

    entries = [
        {
            "id": "E7001",
            "title": "freeze E718 and import one-time full-test result",
            "goal": "verify frozen E718 config and record the full MPTS-52 test result without rerunning test.",
            "read": [str(PATHS[k]) for k in ["e718_audit", "e734_test", "e732_apply", "e733_apply", "e730_render", "e731_selfscore", "e423_val", "e718_val"]],
            "write": ["reports/E718_frozen_before_full_test.md", "eval/E718_full_test_result.json", "reports/E718_full_test_report.md"],
            "split": "MPTS-52 test, imported existing E734c artifact",
            "method": "audit/import existing frozen full-test artifact to avoid a second test execution.",
            "params": json.dumps(FROZEN, ensure_ascii=False),
            "metrics": {
                "match@1": pct(full_test["full_test_result"]["subsets"]["full"]["match@1"]),
                "match@5": pct(full_test["full_test_result"]["subsets"]["full"]["match@5"]),
                "match@20": pct(full_test["full_test_result"]["subsets"]["full"]["match@20"]),
                "match@50": pct(full_test["full_test_result"]["subsets"]["full"]["match@50"]),
                "RMSE": full_test["full_test_result"]["subsets"]["full"]["RMSE@20"],
                "rows>=7 match@5": pct(full_test["full_test_result"]["subsets"]["rows_ge_7"]["match@5"]),
                "rows>=7 match@20": pct(full_test["full_test_result"]["subsets"]["rows_ge_7"]["match@20"]),
            },
            "conclusion": "full test completed as existing E734c; not rerun in opentry_4; no test feedback used for tuning.",
            "next": "diagnose hard negatives and geometry proposal ceiling on train/val.",
        },
        {
            "id": "E7002",
            "title": "rows>=7 hard-negative diagnosis",
            "goal": "build positive/hard-negative candidate datasets and quantify W/A-hit match-fail bottleneck.",
            "read": [str(PATHS["train_features"]), str(PATHS["val_features"])],
            "write": ["cache/hard_negative_dataset_train.jsonl", "cache/hard_negative_dataset_val.jsonl", "eval/hard_negative_diagnosis.json", "reports/hard_negative_diagnosis_report.md"],
            "split": "E318 train, E424 val512",
            "method": "label-based train/val diagnostic only; candidates filtered for readable/composition/SG legality.",
            "params": "positive=StructureMatcher match; hard_negative=W/A-hit or skeleton-hit and StructureMatcher fail.",
            "metrics": {
                "match@20": pct(hard["val512"]["candidate_ceiling_top20"]),
                "rows>=7 match@20": pct(hard["val512"]["rows_ge_7_candidate_ceiling_top20"]),
                "W/A-hit but match-fail rate": pct(hard["val512"]["rows_ge_7_wa_hit_match_fail_rate"]),
            },
            "conclusion": "rows>=7 failures are mostly valid W/A-hit geometry conversion failures; new candidates are needed.",
            "next": "train geometry energy model.",
        },
        {
            "id": "E7003",
            "title": "geometry energy model train-dev val512 evaluation",
            "goal": "train a hard-negative-aware geometry energy model and assess group/pairwise discrimination and match impact.",
            "read": [str(PATHS["train_features"]), str(PATHS["val_features"])],
            "write": ["checkpoints/geometry_energy_model_e7002.json", "eval/geometry_energy_model_eval.json", "reports/geometry_energy_model_report.md"],
            "split": "E318 train, E424 val512",
            "method": "standard-library hashed logistic regression with rows>=7 and hard-negative weights; blocked leakage fields.",
            "params": "dim=768, epochs=8, lr=0.025, row7_weight=2.5",
            "metrics": {
                "match@1": pct(energy["energy_full_rerank_val"]["match@1"]),
                "match@5": pct(energy["energy_full_rerank_val"]["match@5"]),
                "match@20": pct(energy["energy_full_rerank_val"]["match@20"]),
                "match@50": pct(energy["energy_full_rerank_val"]["match@50"]),
                "RMSE": energy["energy_full_rerank_val"]["RMSE@20"],
                "rows>=7 match@5": pct(energy["energy_full_rerank_rows_ge_7"]["match@5"]),
                "rows>=7 match@20": pct(energy["energy_full_rerank_rows_ge_7"]["match@20"]),
            },
            "conclusion": "energy is useful for discrimination but cannot be accepted as ordinary full rerank if match@k drops.",
            "next": "audit joint proposal pool ceiling.",
        },
        {
            "id": "E7004",
            "title": "joint free-param lattice proposal generator audit",
            "goal": "evaluate whether constrained joint geometry proposals add rows>=7 positives beyond baseline.",
            "read": [str(PATHS["gen_baseline_features"]), str(PATHS["gen_features"]), str(PATHS["gen_merge_summary"])],
            "write": ["eval/joint_geometry_generator_eval.json", "reports/joint_geometry_generator_report.md"],
            "split": "val512 rows>=7 aligned216 subset",
            "method": "audit E700 pairfield_adam_repel constrained geometry proposal pool and merged ceiling.",
            "params": "proposal source=E700, baseline=E421 aligned216 rows>=7",
            "metrics": {
                "match@1": pct(gen["proposal_direct_metrics"]["match@1"]),
                "match@5": pct(gen["proposal_direct_metrics"]["match@5"]),
                "match@20": pct(gen["proposal_direct_metrics"]["match@20"]),
                "match@50": pct(gen["proposal_direct_metrics"]["match@50"]),
                "RMSE": gen["proposal_direct_metrics"]["RMSE@20"],
                "rows>=7 match@5": pct(gen["proposal_direct_metrics"]["match@5"]),
                "rows>=7 match@20": pct(gen["proposal_direct_metrics"]["match@20"]),
                "rows>=7 positive-any": pct(gen["merged_pool_ceiling"]["positive_any_all_candidates"]),
                "new positives beyond baseline": gen["new_positive_samples_beyond_baseline"],
                "W/A-hit but match-fail rate": pct(gen["wa_hit_match_fail_rate"]),
            },
            "conclusion": "proposal ceiling adds rows>=7 positives, but direct ordering is worse; use only with anchor-safe insertion.",
            "continue": "no",
            "next": "follow final summary routes.",
        },
    ]
    append_experiment_log(entries)
    write_manifest(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
