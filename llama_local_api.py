"""
Lightweight FastAPI server exposing a local LLaMat-3-Chat model with an
OpenAI-compatible `/v1/chat/completions` endpoint.

Usage:
    LLAMA_MODEL_ID=m3rg-iitd/llamat-3-chat \
    uvicorn llama_local_api:app --host 0.0.0.0 --port 8000

The service expects Hugging Face access to the model (token, cache, etc.)
to be available on this machine.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)


MODEL_ID = os.getenv("LLAMA_MODEL_ID", "m3rg-iitd/llamat-3-chat")
DEFAULT_MAX_NEW_TOKENS = 512
RAW_GEN_LOG = os.getenv(
    "LLAMA_RAW_LOG",
    "/root/autodl-tmp/model/CrystaLLM/llama_raw_generation.log",
)
STRICT_JSON = os.getenv("LLAMA_STRICT_JSON", "false").lower() in ("1", "true", "yes")
# When disabled, do not enforce any output-format constraints (no seeding, no stop-on-END, no bad-words).
CONSTRAINED_OUTPUT = os.getenv("LLAMA_CONSTRAINED_OUTPUT", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Local LLaMat-3-Chat API", version="1.0")


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS


class ChatChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    choices: List[ChatChoice]


_tokenizer = None
_model = None
_text_gen = None


class EndSequenceCriteria(StoppingCriteria):
    """
    Stop generation when the sequence of token ids ends with a target sequence.
    """

    def __init__(self, end_ids: List[int]):
        super().__init__()
        self.end_ids = end_ids

    def __call__(self, input_ids, scores, **kwargs):
        if not self.end_ids:
            return False
        seq = input_ids[0].tolist()
        return seq[-len(self.end_ids) :] == self.end_ids


def _extract_json(text: str) -> Dict[str, object]:
    """
    Best-effort JSON extraction to enforce a clean JSON reply to the client.
    Tries a full parse first; on failure, attempts to parse from the first '{'.
    Raises ValueError if no JSON object can be parsed.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start != -1:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(stripped[start:])
            return obj
    raise ValueError("unable to extract JSON from model output")


def _lazy_load():
    global _tokenizer, _model, _text_gen
    if _text_gen is not None:
        return
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    _text_gen = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        _lazy_load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

    # Try to capture user JSON payload for fallback use.
    user_payload = {}
    for msg in req.messages:
        if msg.get("role") == "user":
            try:
                user_payload = json.loads(msg.get("content", ""))
            except Exception:
                user_payload = {}
            break

    try:
        # Build prompt text for logging; tokenize once for generation.
        system_msg = ""
        roles = []
        for m in req.messages:
            roles.append(m.get("role", ""))
            if not system_msg and m.get("role") == "system":
                system_msg = m.get("content", "") or ""
        prompt_text = _tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # In constrained mode, strongly bias the model into the required template by seeding the first header.
        if CONSTRAINED_OUTPUT:
            prompt_text = prompt_text + "ANALYSIS:\n"
        tokenized = _tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized = {k: v.to(_model.device) for k, v in tokenized.items()}
        input_len = tokenized["input_ids"].shape[-1]

        eot_id = _tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [_tokenizer.eos_token_id]
        if isinstance(eot_id, int) and eot_id >= 0:
            eos_ids.append(eot_id)

        end_ids: List[int] = []
        stopping: StoppingCriteriaList | None = None
        bad_words_ids: List[List[int]] | None = None
        if CONSTRAINED_OUTPUT:
            end_ids = _tokenizer.encode("\nEND", add_special_tokens=False)
            stopping = StoppingCriteriaList([EndSequenceCriteria(end_ids)]) if end_ids else None
            # Prevent common "chat marker" and JSON-like emissions.
            bad_words_ids = [[27], [90], [92]]  # '<', '{', '}'

        gen_kwargs = dict(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            eos_token_id=eos_ids,
            pad_token_id=_tokenizer.eos_token_id,
        )
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping
        if bad_words_ids is not None:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        gen_ids = _model.generate(**tokenized, **gen_kwargs)
        total_len = gen_ids.shape[-1]
        prompt_head = _tokenizer.decode(gen_ids[0][:input_len], skip_special_tokens=False)
        generated_ids = gen_ids[0][input_len:]
        gen_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
        out = gen_text

        # Log debug info and full generation.
        try:
            with open(RAW_GEN_LOG, "a", encoding="utf-8") as log_f:
                log_f.write(f"[DEBUG] roles={roles}\n")
                log_f.write(f"[DEBUG] system_prompt[:400]={system_msg[:400]}\n")
                log_f.write(f"[DEBUG] prompt_text[:200]={prompt_text[:200]}\n")
                log_f.write(f"[DEBUG] input_len={input_len}, total_len={total_len}\n")
                log_f.write(f"[DEBUG] prompt_head[:200]={prompt_head[:200]}\n")
                log_f.write(f"[DEBUG] gen_text[:200]={gen_text[:200]}\n")
                log_f.write(
                    f"[DEBUG] constrained={CONSTRAINED_OUTPUT}, eos_ids={eos_ids}, bad_words_ids={bad_words_ids}, has_end_stop={bool(end_ids)}\n\n"
                )
                log_f.write(out)
                log_f.write("\n\n---\n\n")
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    content_raw = out

    if STRICT_JSON:
        # Enforce JSON output: try to parse; on failure, fall back to a safe stub.
        try:
            parsed = _extract_json(content_raw)
            # Keep only the required fields if present.
            safe_output = {
                "analysis": parsed.get("analysis"),
                "next_prompt_lines": parsed.get("next_prompt_lines"),
            }
            # If missing mandatory fields, fall back to user payload's last_prompt_lines.
            if not safe_output.get("analysis") or not isinstance(safe_output.get("next_prompt_lines"), list):
                raise ValueError("missing required fields")
            content = json.dumps(safe_output, ensure_ascii=False)
        except Exception:
            fallback_lines = []
            if isinstance(user_payload, dict):
                fallback_lines = user_payload.get("last_prompt_lines") or []
            safe_output = {
                "analysis": "LLM output invalid, using fallback.",
                "next_prompt_lines": fallback_lines,
                "_raw": content_raw,
            }
            content = json.dumps(safe_output, ensure_ascii=False)
    else:
        # Return raw model content for local parsing.
        content = content_raw

    choice = ChatChoice(index=0, message={"role": "assistant", "content": content})
    return ChatResponse(choices=[choice])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
