"""
Qwen(OpenAI-compatible Chat Completions) 客户端：根据评分摘要自动生成新的 CIF prompt。

示例：
```bash
python bin/qwen_client.py \
  --previous-prompt data_Na2Cl2\\n \
  --summary-file experiments/round_1/evaluator_summary.json \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-30B-A3B-Instruct-2507
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

DEFAULT_SYSTEM_PROMPT = (
    # Role & goal
    "你是一名用于编辑晶体生成 CIF 提示行的助手。\n"
    "Goal: adjust the given CIF lines (data_, _chemical_formula_sum, _symmetry_space_group_name_H-M, etc.) "
    "to prioritize (1) lower formation_energy (more negative is better), then (2) higher bandgap, while keeping outputs valid.\n"
    "If validation pass rate is low, prioritize changes that improve validation_ok first, then optimize formation_energy, then bandgap.\n"
    "You MUST ground your decision in EVALUATOR_SUMMARY (failure reasons + representative samples).\n\n"
    # What you must output
    "Reply ONLY in plain text with EXACTLY these sections (no markdown, no code fences, no JSON):\n"
    "ANALYSIS:\n"
    "用中文（简体）写 1-3 句：\n"
    "- 点出 1 个最主要失败原因（带上 count/ratio）或 1 个最主要成功信号；\n"
    "- 引用 1 个代表性 sample_id（来自 TOP_STRUCTURES 或 REPRESENTATIVE_STRUCTURES）；\n"
    "- 说明你将做的“唯一一个关键改动”是什么（或明确说明保持不变）。\n"
    "Do not use placeholders like <...>.\n"
    "NEXT_PROMPT_LINES:\n"
    "Write the next CIF prompt lines as plain text, one line per prompt line.\n"
    "If you are unsure how to improve, copy last_prompt_lines exactly here (line-by-line).\n"
    "Never output an empty list like [] and never leave NEXT_PROMPT_LINES empty.\n"
    "END\n"
    "The word END must appear alone on its own line at the end. Do not add any other text before or after END.\n\n"
    # Hard constraints
    "Rules:\n"
    "- If uncertain, copy last_prompt_lines exactly and make the smallest safe improvement.\n"
    "- If you cannot improve, output last_prompt_lines exactly under NEXT_PROMPT_LINES (line-by-line).\n"
    "- Respect PROMPT_EDIT_POLICY: if composition_policy is locked, you may only change space-group related keys.\n"
    "- Keep edits minimal: propose at most ONE key change (e.g., change _symmetry_space_group_name_H-M).\n"
    "- Output ONLY high-level CIF prompt lines (no atom-site tables).\n"
    "- Do NOT output any of: loop_, _atom_site_*, _cell_length_*, _cell_angle_*, _cell_volume, or raw coordinate rows.\n"
    "- Restricted mode comment line policy (if you output any comment):\n"
    "  - At most ONE line starting with ';' and it MUST be the LAST line.\n"
    "  - Do NOT add any other words (no 'maintain P4/mmm symmetry', no numbers, no extra text).\n"
    "- Do NOT output JSON, YAML, bullet lists, or quote the input.\n"
    "- Do NOT wrap anything in ``` or other fences.\n"
    "- Ensure both sections are present; missing sections are treated as failure.\n\n"
    "Example:\n"
    "ANALYSIS:\n"
    "Space group is fine; keep composition stable and improve validity.\n"
    "NEXT_PROMPT_LINES:\n"
    "data_Na2Cl2\n"
    "_symmetry_space_group_name_H-M   P4/mmm\n"
    "END\n"
)


@dataclass
class QwenConfig:
    api_base: str = os.environ.get("QWEN_API_BASE") or os.environ.get("LLAMA_API_BASE", "http://localhost:8000/v1")
    api_key: str | None = os.environ.get("QWEN_API_KEY") or os.environ.get("LLAMA_API_KEY")
    model: str = os.environ.get("QWEN_MODEL", os.environ.get("LLAMA_MODEL", "Qwen3-30B-A3B-Instruct-2507"))
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.95
    timeout: float = 120.0


class QwenClient:
    def __init__(self, config: QwenConfig):
        self.config = config

    @staticmethod
    def _clean_raw_text(text: str, max_len: int = 2000) -> str:
        if text is None:
            return ""
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.strip()
        if len(text) > max_len:
            return text[:max_len] + "…(truncated)"
        return text

    @staticmethod
    def _parse_template(text: str) -> Dict[str, Any] | None:
        lines = [ln.rstrip() for ln in text.splitlines()]
        state: str | None = None
        analysis_lines: List[str] = []
        prompt_lines: List[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                if state == "analysis":
                    analysis_lines.append("")
                elif state == "prompt":
                    prompt_lines.append("")
                continue
            upper = line.upper()
            if upper == "ANALYSIS:":
                state = "analysis"
                continue
            if upper == "NEXT_PROMPT_LINES:":
                state = "prompt"
                continue
            if upper == "END":
                break
            if state == "analysis":
                analysis_lines.append(line)
            elif state == "prompt":
                prompt_lines.append(line)
        analysis_text = " ".join([ln for ln in analysis_lines if ln.strip()]).strip()
        prompt_lines = [ln for ln in prompt_lines if ln.strip()]
        if not analysis_text or not prompt_lines:
            return None
        return {"analysis": analysis_text, "next_prompt_lines": prompt_lines}

    def _build_messages(self, last_prompt_lines: List[str], evaluator_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        lines: List[str] = []
        lines.append("LAST_PROMPT_LINES:")
        lines.extend(last_prompt_lines or ["(empty)"])
        lines.append("")
        lines.append("EVALUATOR_SUMMARY (compact):")
        round_n = evaluator_summary.get("round")
        if round_n is not None:
            lines.append(f"round: {round_n}")
        score_property = evaluator_summary.get("score_property")
        score_goal = evaluator_summary.get("score_goal")
        if score_property is not None:
            lines.append(f"pipeline_score_property: {score_property}")
        if score_goal is not None:
            lines.append(f"pipeline_score_goal: {score_goal}")
        metrics = evaluator_summary.get("metrics") or {}
        if isinstance(metrics, dict):
            for k in ["n_structures", "n_scored_ok", "n_validation_pass", "validation_pass_ratio"]:
                if k in metrics:
                    lines.append(f"{k}: {metrics.get(k)}")

        optimizer_objective = evaluator_summary.get("optimizer_objective")
        if isinstance(optimizer_objective, dict):
            lines.append("")
            lines.append("OPTIMIZER_OBJECTIVE:")
            primary = optimizer_objective.get("primary") or {}
            secondary = optimizer_objective.get("secondary") or {}
            if isinstance(primary, dict):
                lines.append(f"primary: {primary.get('property')} {primary.get('goal')}")
            if isinstance(secondary, dict):
                lines.append(f"secondary: {secondary.get('property')} {secondary.get('goal')}")
            note = optimizer_objective.get("note")
            if isinstance(note, str) and note.strip():
                lines.append(f"note: {note.strip()}")

        def _fmt_structure(item: Any) -> str:
            if not isinstance(item, dict):
                return str(item)
            props = item.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            sample_id = item.get("sample_id")
            v_ok = item.get("validation_ok")
            fe = props.get("formation_energy")
            bg = props.get("bandgap")
            reasons = item.get("validation_reasons") or []
            tag = item.get("tag")
            parts = []
            if tag:
                parts.append(f"tag={tag}")
            parts.append(f"sample_id={sample_id}")
            parts.append(f"validation_ok={v_ok}")
            parts.append(f"formation_energy={fe}")
            parts.append(f"bandgap={bg}")
            if reasons:
                parts.append(f"reasons={reasons}")
            return ", ".join(parts)

        top_structures = evaluator_summary.get("top_structures") or []
        if isinstance(top_structures, list) and top_structures:
            lines.append("")
            lines.append("TOP_STRUCTURES:")
            for item in top_structures[:5]:
                lines.append(f"- {_fmt_structure(item)}")

        rep = evaluator_summary.get("representative_structures") or []
        if isinstance(rep, list) and rep:
            lines.append("")
            lines.append("REPRESENTATIVE_STRUCTURES:")
            for item in rep[:8]:
                lines.append(f"- {_fmt_structure(item)}")

        failure_top = evaluator_summary.get("validation_failure_reasons_top") or []
        if isinstance(failure_top, list) and failure_top:
            lines.append("")
            lines.append("TOP_VALIDATION_FAILURE_REASONS:")
            for item in failure_top[:8]:
                if isinstance(item, dict):
                    lines.append(f"- reason={item.get('reason')}, count={item.get('count')}, ratio={item.get('ratio')}")
                else:
                    lines.append(f"- {item}")
        try:
            from crystallm.prompt_sanitizer import PromptEditPolicy

            policy = PromptEditPolicy.from_env()
            lines.append("")
            lines.append("PROMPT_EDIT_POLICY:")
            lines.append(f"edit_scope: {policy.edit_scope}")
            lines.append(f"task_scenario: {policy.task_scenario}")
            lines.append(f"composition_policy: {policy.composition_policy}")
            lines.append(f"restricted_profile: {policy.restricted_profile}")
            lines.append(f"max_key_changes: {policy.max_key_changes}")
        except Exception:  # noqa: BLE001
            pass
        lines.append("")
        lines.append("Task: follow the output template exactly.")
        user_text = "\n".join(lines)
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def _request_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = requests.post(
            f"{self.config.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    def optimize_prompt(
        self, last_prompt_lines: List[str], evaluator_summary: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        messages = self._build_messages(last_prompt_lines, evaluator_summary)
        raw_response = self._request_chat(messages)
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"响应格式异常: {raw_response}") from exc
        raw_text_full = content if content is not None else ""
        safe_raw = self._clean_raw_text(raw_text_full)

        parsed = self._parse_template(safe_raw)
        if not parsed:
            parsed = {
                "analysis": "Template parse failed; using fallback.",
                "next_prompt_lines": [line for line in last_prompt_lines if line.strip()],
                "_raw": "<EMPTY or TRUNCATED>" if not safe_raw else safe_raw,
            }
        else:
            parsed["_raw"] = safe_raw or "<EMPTY or TRUNCATED>"

        if not isinstance(parsed.get("next_prompt_lines"), list):
            parsed["next_prompt_lines"] = [str(parsed.get("next_prompt_lines"))]
        return parsed, raw_response, safe_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 提示优化客户端（文本协议 + 本地裁决）")
    parser.add_argument("--previous-prompt", help="上一轮 prompt 字符串；若与 --previous-prompt-file 同时提供，以文件为准")
    parser.add_argument("--previous-prompt-file", help="上一轮 prompt 文件路径")
    parser.add_argument("--summary-file", required=True, help="evaluator_summary JSON 路径（包含打分统计与验证结果）")
    parser.add_argument("--api-base", help="覆盖 QWEN_API_BASE（或兼容 LLAMA_API_BASE）环境变量")
    parser.add_argument("--api-key", help="覆盖 QWEN_API_KEY（或兼容 LLAMA_API_KEY）环境变量")
    parser.add_argument("--model", help="覆盖 QWEN_MODEL（或兼容 LLAMA_MODEL）环境变量")
    parser.add_argument("--temperature", type=float, help="采样温度")
    parser.add_argument("--max-tokens", type=int, help="生成 token 上限")
    parser.add_argument("--top-p", type=float, help="nucleus sampling top_p")
    parser.add_argument("--timeout", type=float, help="HTTP 超时秒数")
    parser.add_argument("--out", help="可选：将新 prompt 写入文件")
    parser.add_argument("--raw-out", help="可选：保存完整 LLM 输出 JSON")
    return parser.parse_args()


def _load_previous_prompt(args: argparse.Namespace) -> str:
    if args.previous_prompt_file:
        return Path(args.previous_prompt_file).read_text(encoding="utf-8")
    if args.previous_prompt:
        return args.previous_prompt
    raise ValueError("必须提供 --previous-prompt 或 --previous-prompt-file")


def main() -> None:
    args = parse_args()
    evaluator_summary = json.loads(Path(args.summary_file).read_text(encoding="utf-8"))
    config = QwenConfig()
    if args.api_base:
        config.api_base = args.api_base
    if args.api_key:
        config.api_key = args.api_key
    if args.model:
        config.model = args.model
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.timeout is not None:
        config.timeout = args.timeout

    client = QwenClient(config)
    previous_prompt = _load_previous_prompt(args)
    last_prompt_lines = [line for line in previous_prompt.splitlines() if line.strip()]
    parsed, raw_response, raw_text = client.optimize_prompt(last_prompt_lines, evaluator_summary)
    next_prompt_lines = parsed.get("next_prompt_lines", [])
    new_prompt = "\n".join(next_prompt_lines)
    if new_prompt and not new_prompt.endswith("\n"):
        new_prompt += "\n"
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(new_prompt, encoding="utf-8")
        print(f"[qwen_client] 写入 prompt -> {args.out}")
    if args.raw_out:
        payload = {
            "raw_text": raw_text,
            "response": raw_response,
            "previous_prompt_lines": last_prompt_lines,
            "evaluator_summary": evaluator_summary,
            "parsed": parsed,
        }
        Path(args.raw_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[qwen_client] 写入原始响应 -> {args.raw_out}")


# Backward-compatible aliases (for older scripts/imports).
LlamaConfig = QwenConfig
LlamaClient = QwenClient


if __name__ == "__main__":
    main()
