from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


# NOTE: banned prefixes are policy-dependent (see _is_banned()).
# In restricted mode we forbid atom-site tables/loops to keep the "prompt editing"
# interface high-level and avoid partial CIF bodies in prompt.txt.
_RESTRICTED_BANNED_PREFIXES = ("loop_", "_atom_site_")

# In restricted mode, we only allow the minimal high-level keys that are both
# stable for CrystaLLM sampling and easy to validate/merge deterministically.
_RESTRICTED_ALLOWED_KEYS = {
    "data_",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_sum",
    "_chemical_formula_structural",
    "_cell_formula_units_Z",
}

# Open mode still forbids atom-site tables, but allows more high-level tags.
_OPEN_ALLOWED_KEY_PREFIXES = (
    "data_",
    "_symmetry_",
    "_chemical_",
    "_cell_formula_units_Z",
)

# Restricted explore profile: allow most high-level CIF tags, but still forbid atom-site tables/loops.
# This is meant for "no given formula / exploration" where you want Qwen to adjust more knobs.
_RESTRICTED_EXPLORE_ALLOWED_KEY_PREFIXES = (
    "data_",
    "_symmetry_",
    "_chemical_",
    "_cell_",
)

# Natural-language comments are allowed only as a single trailing line, and are
# normalized to CIF comment syntax '# ...' so we never keep a ';' line (which
# can start a CIF multi-line text block).
_COMMENT_ALLOWED_PHRASES = (
    "charge neutrality",
    "avoid short bonds",
    "keep space group",
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return default


@dataclass(frozen=True)
class PromptEditPolicy:
    edit_scope: Literal["restricted", "open"]
    task_scenario: Literal["reconstruct", "explore"]
    composition_policy: Literal["locked", "editable"]
    restricted_profile: Literal["minimal", "extended"]
    allow_comments: bool
    max_key_changes: int
    max_prompt_chars: int
    max_prompt_key_lines: int
    restricted_comment_max_lines: int = 1
    open_comment_max_lines: int = 2
    comment_max_len: int = 120
    disable_sg_int_autofix: bool = True

    @staticmethod
    def from_env() -> "PromptEditPolicy":
        scope = (os.environ.get("CRYSTALLM_PROMPT_EDIT_SCOPE") or "restricted").strip().lower()
        if scope not in {"restricted", "open"}:
            scope = "restricted"

        scenario = (os.environ.get("CRYSTALLM_TASK_SCENARIO") or "reconstruct").strip().lower()
        if scenario not in {"reconstruct", "explore"}:
            scenario = "reconstruct"

        comp = (os.environ.get("CRYSTALLM_COMPOSITION_POLICY") or "editable").strip().lower()
        if comp not in {"locked", "editable"}:
            comp = "editable"

        # Restricted profile: minimal (tight) vs extended (exploration). Defaults based on scenario.
        profile = (os.environ.get("CRYSTALLM_RESTRICTED_PROFILE") or "").strip().lower()
        if profile not in {"minimal", "extended"}:
            profile = "extended" if scenario == "explore" else "minimal"

        return PromptEditPolicy(
            edit_scope=scope,  # type: ignore[assignment]
            task_scenario=scenario,  # type: ignore[assignment]
            composition_policy=comp,  # type: ignore[assignment]
            restricted_profile=profile,  # type: ignore[assignment]
            allow_comments=_env_bool("CRYSTALLM_ALLOW_COMMENTS", True),
            max_key_changes=_env_int("CRYSTALLM_LLM_MAX_KEY_CHANGES", 1),
            max_prompt_chars=_env_int("CRYSTALLM_MAX_PROMPT_CHARS", 500),
            max_prompt_key_lines=_env_int("CRYSTALLM_MAX_PROMPT_KEY_LINES", 10),
            # Default: disabled (do NOT auto-fill _symmetry_Int_Tables_number) unless explicitly enabled by env.
            disable_sg_int_autofix=_env_bool("CRYSTALLM_DISABLE_SG_INT_AUTOFIX", True),
        )


def _is_banned(line: str, *, policy: PromptEditPolicy) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if policy.edit_scope == "open":
        return False
    return any(stripped.startswith(p) for p in _RESTRICTED_BANNED_PREFIXES)


def _looks_like_atom_site_row(line: str) -> bool:
    # Typical generated atom-site row looks like:
    #   Na Na0 2 0.0 0.0 0.25 1
    # This must never be allowed into "high-level prompt".
    s = line.strip()
    if not s or s.startswith("_") or s.startswith("data_"):
        return False
    return bool(re.match(r"^[A-Za-z]{1,2}\\s+\\S+\\s+\\d+\\s+[-+0-9.]+\\s+[-+0-9.]+\\s+[-+0-9.]+", s))


def _classify_line(line: str, *, policy: PromptEditPolicy) -> Tuple[str, Optional[str]]:
    """
    Return (kind, key):
      - kind == "comment": key is None
      - kind == "key": key is "data_" or a CIF tag like "_chemical_formula_sum"
      - kind == "other": key is None
      - kind == "banned": key is None
    """
    raw = str(line or "").rstrip("\n")
    stripped = raw.strip()
    if not stripped:
        return "other", None
    if policy.edit_scope != "open" and _looks_like_atom_site_row(raw):
        return "banned", None
    if _is_banned(raw, policy=policy):
        return "banned", None
    if stripped.startswith(";") or stripped.startswith("#"):
        return "comment", None
    if stripped.startswith("data_"):
        return "key", "data_"
    if stripped.startswith("_"):
        return "key", stripped.split()[0]
    return "other", None


def _extract_allowed_comment_phrases(text: str) -> List[str]:
    """
    Safe-clean natural language comments into a strict whitelist.

    We accept that the model may output extra words, but the *final* prompt
    comment must only contain whitelisted phrases (in a fixed order), per the
    agreed restricted-mode scheme in qwen-chatlog.txt.
    """
    raw = str(text or "").strip()
    if raw.startswith(";") or raw.startswith("#"):
        raw = raw[1:].strip()
    raw = raw.lower()
    raw = raw.replace("|", "/")
    raw = re.sub(r"[\t ]+", " ", raw).strip()

    found: List[str] = []
    for phrase in _COMMENT_ALLOWED_PHRASES:
        if phrase in raw:
            found.append(phrase)

    # Support a minimal synonym for "keep space group" (but still normalize to whitelist).
    if ("keep space group" not in found) and ("space group" in raw) and ("maintain" in raw):
        found.append("keep space group")

    # de-dup, stable order
    out: List[str] = []
    for phrase in _COMMENT_ALLOWED_PHRASES:
        if phrase in found and phrase not in out:
            out.append(phrase)
    return out


def _build_canonical_comment(phrases: List[str]) -> Optional[str]:
    if not phrases:
        return None
    normalized = " / ".join(phrases)
    if not normalized:
        return None
    if len(normalized) > 120:
        return None
    return f"# {normalized}"


def _space_group_int_from_symbol(symbol: str) -> Optional[int]:
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return int(SpaceGroup(symbol).int_number)
    except Exception:  # noqa: BLE001
        return None


def _space_group_symbol_from_int(number: int) -> Optional[str]:
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return str(SpaceGroup.from_int_number(int(number)).symbol)
    except Exception:  # noqa: BLE001
        return None


def sanitize_prompt_lines_with_audit(
    candidate_lines: List[str],
    fallback_lines: List[str],
    *,
    policy: Optional[PromptEditPolicy] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    policy = policy or PromptEditPolicy.from_env()

    audit: Dict[str, Any] = {
        "policy": {
            "edit_scope": policy.edit_scope,
            "task_scenario": policy.task_scenario,
            "composition_policy": policy.composition_policy,
            "restricted_profile": policy.restricted_profile,
            "allow_comments": policy.allow_comments,
            "max_key_changes": policy.max_key_changes,
            "max_prompt_chars": policy.max_prompt_chars,
            "max_prompt_key_lines": policy.max_prompt_key_lines,
        },
        "candidate_lines_raw": [str(x).rstrip() for x in (candidate_lines or [])],
        "fallback_lines_raw": [str(x).rstrip() for x in (fallback_lines or [])],
        "discarded_lines": [],
        "key_changes": [],
        "auto_fixes": [],
        "validation_errors": [],
        "fallback_used": False,
        "fallback_reason": None,
        "truncated_changes": [],
    }

    if policy.edit_scope == "open":
        # Open/free mode: keep candidate lines as-is (minus minimal safety cleaning),
        # and do NOT attempt key-wise merging (loops/duplicate tags are allowed).
        kept_lines: List[str] = []
        discarded: List[Dict[str, str]] = []
        for raw in (candidate_lines or []):
            line = str(raw).rstrip()
            if not line.strip():
                continue
            if "<unk>" in line:
                line = line.replace("<unk>", "")
            # Prevent CIF multi-line text block semantics: never keep a leading ';' line.
            if line.lstrip().startswith(";"):
                text = line.lstrip()[1:].strip()
                # Open mode allows arbitrary comment text, but we still cap length.
                text = re.sub(r"[\t ]+", " ", text).strip()
                if len(text) > 120:
                    text = text[:120]
                line = f"# {text}" if text else "#"
            kept_lines.append(line.strip())

        # Ensure required anchors exist; otherwise fallback to last prompt.
        has_data = any(ln.startswith("data_") for ln in kept_lines)
        has_sg = any(ln.startswith("_symmetry_space_group_name_H-M") for ln in kept_lines)
        if not (has_data and has_sg):
            audit["fallback_used"] = True
            audit["fallback_reason"] = "open_mode_missing_required_keys:data_or_space_group"
            kept_lines = [str(x).rstrip() for x in (fallback_lines or []) if str(x).strip()]

        # Apply global size caps (truncate from the end).
        total_chars = sum(len(ln) + 1 for ln in kept_lines)
        while kept_lines and total_chars > policy.max_prompt_chars:
            dropped = kept_lines.pop()
            total_chars -= len(dropped) + 1
            discarded.append({"source": "open", "line": dropped, "reason": "char_truncation"})

        audit["discarded_lines"].extend(discarded)
        audit["candidate_lines_kept"] = kept_lines
        return kept_lines, audit

    def _allowed_key(key: str) -> bool:
        # Restricted mode: choose a profile.
        if policy.restricted_profile == "minimal":
            return key in _RESTRICTED_ALLOWED_KEYS

        # Restricted extended profile (exploration): allow a broader set of high-level tags.
        if key == "data_":
            return True
        return key.startswith(_RESTRICTED_EXPLORE_ALLOWED_KEY_PREFIXES)

    def _parse_lines(lines: List[str], *, source: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        order: List[str] = []
        key_to_line: Dict[str, str] = {}
        comments: List[str] = []
        for raw in lines or []:
            line = str(raw).rstrip()
            kind, key = _classify_line(line, policy=policy)
            if kind == "comment":
                comments.append(line.strip())
                continue
            if kind == "banned":
                audit["discarded_lines"].append({"source": source, "line": line, "reason": "banned_line"})
                continue
            if kind == "key" and key:
                if not _allowed_key(key):
                    audit["discarded_lines"].append({"source": source, "line": line, "reason": "key_not_allowed"})
                    continue
                if key not in order:
                    order.append(key)
                key_to_line[key] = line.strip()
                continue
            if kind == "other":
                audit["discarded_lines"].append({"source": source, "line": line, "reason": "non_prompt_line"})
                continue
        return order, key_to_line, comments

    base_order, base_map, base_comments_raw = _parse_lines(fallback_lines, source="fallback")
    cand_order, cand_map, cand_comments_raw = _parse_lines(candidate_lines, source="candidate")

    def _fallback_output(reason: str) -> Tuple[List[str], Dict[str, Any]]:
        audit["fallback_used"] = True
        audit["fallback_reason"] = reason

        # Rebuild from last prompt (base_map/base_order) deterministically.
        out_key_lines: List[str] = []
        for key in base_order:
            if key in base_map and str(base_map[key]).strip():
                out_key_lines.append(str(base_map[key]).strip())

        if not out_key_lines:
            # Last resort: keep non-empty raw lines (but strip any banned ';' blocks).
            out_key_lines = [str(x).rstrip() for x in (fallback_lines or []) if str(x).strip()]

        # Normalize comments from the last prompt as well (optional).
        comments_out: List[str] = []
        if policy.allow_comments:
            max_lines = policy.restricted_comment_max_lines if policy.edit_scope == "restricted" else policy.open_comment_max_lines
            for raw in base_comments_raw:
                norm = _normalize_comment(raw)
                if norm:
                    comments_out.append(norm)
            if len(comments_out) > max_lines:
                comments_out = comments_out[-max_lines:]

        out_lines = [ln for ln in (out_key_lines + comments_out) if not ln.lstrip().startswith(";")]
        audit["candidate_lines_kept"] = out_lines
        return out_lines, audit

    # Detect whether "locked" can be meaningfully enforced.
    lock_formula_source: Optional[str] = None
    if policy.composition_policy == "locked":
        if "_chemical_formula_sum" in base_map:
            lock_formula_source = "_chemical_formula_sum"
        elif "data_" in base_map:
            # We don't attempt to fully parse the data_ header here; treat as weak lock.
            lock_formula_source = "data_"
        else:
            audit["validation_errors"].append("composition_locked_but_no_formula_in_last_prompt")
            # auto-downgrade to editable to avoid undefined behavior
            policy = PromptEditPolicy(
                edit_scope=policy.edit_scope,
                composition_policy="editable",
                allow_comments=policy.allow_comments,
                max_key_changes=policy.max_key_changes,
                max_prompt_chars=policy.max_prompt_chars,
                max_prompt_key_lines=policy.max_prompt_key_lines,
            )
            audit["policy"]["composition_policy"] = "editable"
            audit["policy"]["composition_policy_downgraded"] = True

    def _is_editable_key(key: str) -> bool:
        if policy.edit_scope == "open":
            return True
        # restricted mode
        if policy.composition_policy == "locked":
            # In locked mode, the chemical formula semantics must not drift.
            # Allow only space-group edits (plus optional extended keys if configured).
            if policy.restricted_profile == "extended":
                # Explore + locked: allow broad non-composition tags.
                if key in {"data_", "_chemical_formula_sum", "_chemical_formula_structural", "_cell_formula_units_Z"}:
                    return False
                return _allowed_key(key)
            return key in {"_symmetry_space_group_name_H-M", "_symmetry_Int_Tables_number"}
        if policy.restricted_profile == "extended":
            return _allowed_key(key)
        return key in _RESTRICTED_ALLOWED_KEYS

    merged_map = dict(base_map)
    changes: List[Tuple[str, str, str]] = []
    for key in cand_order:
        if key not in cand_map:
            continue
        if not _is_editable_key(key):
            audit["discarded_lines"].append(
                {"source": "candidate", "line": cand_map[key], "reason": "key_not_editable_under_policy"}
            )
            continue
        old = merged_map.get(key)
        new = cand_map[key]
        if old is None or old.strip() != new.strip():
            changes.append((key, old or "", new))
        merged_map[key] = new

    # Budget enforcement
    if policy.max_key_changes is not None and policy.max_key_changes >= 0 and len(changes) > policy.max_key_changes:
        if policy.edit_scope == "restricted":
            return _fallback_output(f"max_key_changes_exceeded:{len(changes)}>{policy.max_key_changes}")
        # open: truncate to first N changes (candidate order)
        allowed_keys = {k for (k, _, _) in changes[: policy.max_key_changes]}
        for key, old, new in changes:
            if key not in allowed_keys:
                merged_map[key] = base_map.get(key, merged_map.get(key, ""))
                audit["truncated_changes"].append({"key": key, "old": old, "new": new})
        changes = [c for c in changes if c[0] in allowed_keys]

    for key, old, new in changes:
        audit["key_changes"].append({"key": key, "old": old, "new": new})

    # Space group symbol/number normalization: symbol > number
    # This changes the sampling prefix and can shift the CrystaLLM output distribution.
    # Allow disabling it for A/B tests via CRYSTALLM_DISABLE_SG_INT_AUTOFIX=1.
    if not policy.disable_sg_int_autofix:
        sg_symbol_line = merged_map.get("_symmetry_space_group_name_H-M")
        if sg_symbol_line:
            try:
                sg_symbol = sg_symbol_line.split(None, 1)[1].strip().strip("'\"")
            except Exception:  # noqa: BLE001
                sg_symbol = None
            if sg_symbol:
                sg_int = _space_group_int_from_symbol(sg_symbol)
                if sg_int is not None:
                    desired = f"_symmetry_Int_Tables_number {sg_int}"
                    existing = merged_map.get("_symmetry_Int_Tables_number")
                    if existing is None:
                        merged_map["_symmetry_Int_Tables_number"] = desired
                        if "_symmetry_Int_Tables_number" not in base_order:
                            base_order.append("_symmetry_Int_Tables_number")
                        audit["auto_fixes"].append({"kind": "space_group_number_filled", "value": sg_int})
                    else:
                        try:
                            existing_int = int(existing.split(None, 1)[1])
                        except Exception:  # noqa: BLE001
                            existing_int = None
                        if existing_int is not None and existing_int != sg_int:
                            merged_map["_symmetry_Int_Tables_number"] = desired
                            audit["auto_fixes"].append(
                                {"kind": "space_group_number_corrected", "old": existing_int, "new": sg_int}
                            )

    # Comments: only keep up to max lines, always at the end, and normalize to '# ...'.
    comments_out: List[str] = []
    if policy.allow_comments:
        # Strong restriction: always collapse any number of comment lines into at most ONE canonical line.
        raw_comments: List[str] = []
        raw_comments.extend(base_comments_raw)
        raw_comments.extend(cand_comments_raw)

        collected: List[str] = []
        for raw in raw_comments:
            phrases = _extract_allowed_comment_phrases(raw)
            if not phrases:
                audit["discarded_lines"].append({"source": "comment", "line": raw, "reason": "comment_not_whitelisted"})
                continue
            for p in phrases:
                if p not in collected:
                    collected.append(p)

        canonical = _build_canonical_comment(collected)
        if canonical:
            comments_out = [canonical]

        # Enforce max lines (restricted: 1). If policy was configured looser, still comply.
        max_lines = policy.restricted_comment_max_lines if policy.edit_scope == "restricted" else policy.open_comment_max_lines
        if len(comments_out) > max_lines:
            dropped = comments_out[:-max_lines]
            for d in dropped:
                audit["discarded_lines"].append({"source": "comment", "line": d, "reason": "comment_over_limit"})
            comments_out = comments_out[-max_lines:]

    # Rebuild output in stable base order; append new keys at the end.
    out_key_lines: List[str] = []
    seen_keys: set[str] = set()
    for key in base_order:
        if key in merged_map and merged_map[key].strip():
            out_key_lines.append(merged_map[key].strip())
            seen_keys.add(key)
    for key in cand_order:
        if key in merged_map and key not in seen_keys and merged_map[key].strip():
            out_key_lines.append(merged_map[key].strip())
            seen_keys.add(key)

    # Required anchors
    has_data = any(ln.startswith("data_") for ln in out_key_lines)
    has_sg = any(ln.startswith("_symmetry_space_group_name_H-M") for ln in out_key_lines)
    if not (has_data and has_sg):
        return _fallback_output("missing_required_keys:data_or_space_group")

    # Global size caps
    out_lines = out_key_lines + comments_out
    n_key_lines = len(out_key_lines)
    total_chars = sum(len(ln) + 1 for ln in out_lines)
    if n_key_lines > policy.max_prompt_key_lines or total_chars > policy.max_prompt_chars:
        if policy.edit_scope == "restricted":
            return _fallback_output("prompt_size_limit_exceeded")
        # open: drop comments first, then truncate keys
        while out_lines and out_lines[-1].startswith("#"):
            audit["discarded_lines"].append({"source": "comment", "line": out_lines[-1], "reason": "size_truncation"})
            out_lines.pop()
        while len([ln for ln in out_lines if ln.startswith("_") or ln.startswith("data_")]) > policy.max_prompt_key_lines:
            audit["discarded_lines"].append({"source": "key", "line": out_lines[-1], "reason": "size_truncation"})
            out_lines.pop()
        while sum(len(ln) + 1 for ln in out_lines) > policy.max_prompt_chars and out_lines:
            audit["discarded_lines"].append({"source": "line", "line": out_lines[-1], "reason": "char_truncation"})
            out_lines.pop()

    # Guarantee no ';' lines remain
    out_lines = [ln for ln in out_lines if not ln.lstrip().startswith(";")]

    audit["candidate_lines_kept"] = out_lines
    audit["lock_formula_source"] = lock_formula_source
    return out_lines, audit


def sanitize_prompt_lines(candidate_lines: List[str], fallback_lines: List[str]) -> List[str]:
    """Backward-compatible wrapper (drops audit)."""
    out, _audit = sanitize_prompt_lines_with_audit(candidate_lines, fallback_lines)
    return out
