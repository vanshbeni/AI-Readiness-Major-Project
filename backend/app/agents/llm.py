"""
LLM gateway: every agent talks to models through here. OpenRouter (OpenAI-compatible chat completions),
with model fallbacks, retry/backoff, deadlines, usage accounting and robust JSON extraction.
"""
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
FATAL = {401, 402, 403}


class LLMError(RuntimeError):
    """The model could not produce a usable answer (callers fall back to deterministic logic)."""


class LLMUnavailable(LLMError):
    """No key configured, or the key was rejected."""


class LLMQuotaExceeded(LLMUnavailable):
    """The account's daily request quota is used up: retrying or switching free models cannot help until it resets."""


def friendly_error(e: Exception) -> str:
    """One sentence a user can act on."""
    if isinstance(e, LLMQuotaExceeded):
        return ("The OpenRouter account has used its daily free-model quota. It resets daily; adding credits on "
                "openrouter.ai raises the limit (or set a paid model in OPENROUTER_MODEL).")
    if isinstance(e, LLMUnavailable):
        return f"OpenRouter rejected the request: {e}. Check OPENROUTER_API_KEY in backend/.env."
    return f"The AI model could not answer right now ({e}). Please try again in a minute."


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: Dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            if model and model not in self.models:
                self.models.append(model)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "models": list(self.models),
        }


@dataclass
class LLMReply:
    text: str
    model: str


def is_configured() -> bool:
    return bool(settings.OPENROUTER_API_KEY.strip())


def model_chain() -> List[str]:
    chain: List[str] = []
    for m in [settings.OPENROUTER_MODEL, *settings.OPENROUTER_FALLBACK_MODELS]:
        if m and m not in chain:
            chain.append(m)
    return chain


def status() -> Dict[str, Any]:
    return {
        "configured": is_configured(),
        "provider": "openrouter",
        "model": settings.OPENROUTER_MODEL,
        "fallbacks": model_chain()[1:],
        "privacy_mode": settings.AI_PRIVACY_MODE,
    }


def _error_reason(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return str(err.get("message") or err)[:200]
    except ValueError:
        return resp.text[:200]


def chat(
    messages: List[Dict[str, str]],
    *,
    usage: Optional[Usage] = None,
    temperature: float = 0.2,
    max_tokens: int = 1800,
    deadline: Optional[float] = None,
) -> LLMReply:
    """One chat completion with model fallbacks. Raises LLMUnavailable / LLMError."""
    if not is_configured():
        raise LLMUnavailable("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.APP_PUBLIC_URL,
        "X-Title": "Nebulus",
    }
    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    deadline = deadline or (time.monotonic() + settings.AGENT_RUN_BUDGET_SEC)
    last_error = "no model answered"

    for model in model_chain():
        budget = max_tokens
        for attempt in range(settings.LLM_MAX_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                raise LLMError("Agent time budget exhausted while waiting for the model.")
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": budget},
                    timeout=min(settings.LLM_TIMEOUT_SEC, remaining),
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = f"{model}: {type(e).__name__}"
                logger.warning(f"LLM {model} attempt {attempt + 1} failed ({type(e).__name__}).")
                status_code, retry_after = None, None
            else:
                status_code = resp.status_code
                if status_code == 200:
                    try:
                        body = resp.json()
                        if body.get("error"):
                            raise KeyError(str(body["error"])[:200])
                        first = body["choices"][0]
                        text = (first["message"].get("content") or "").strip()
                        finish = first.get("finish_reason")
                        used_model = body.get("model") or model
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        last_error = f"{model}: unexpected response ({e})"
                        logger.warning(f"LLM {model} returned an unexpected payload.")
                        break
                    if usage is not None:
                        usage.add(used_model, body.get("usage") or {})
                    if not text:
                        last_error = f"{model}: empty answer (finish_reason={finish})"
                        logger.warning(f"LLM {used_model} returned an empty answer (finish_reason={finish}).")
                        # Reasoning models can spend the whole token budget thinking; retry with more room.
                        if finish == "length" and attempt < settings.LLM_MAX_RETRIES and budget < 8000:
                            budget = min(budget * 2, 8000)
                            continue
                        break
                    return LLMReply(text=text, model=used_model)
                reason = _error_reason(resp)
                last_error = f"{model}: HTTP {status_code} {reason}"
                logger.warning(f"LLM {model} attempt {attempt + 1} returned {status_code}: {reason}")
                if status_code in FATAL:
                    raise LLMUnavailable(f"OpenRouter rejected the request ({status_code}): {reason}")
                if status_code == 429 and re.search(r"per[- ]day|daily", reason, re.IGNORECASE):
                    raise LLMQuotaExceeded(f"daily quota reached: {reason}")
                if status_code not in RETRYABLE:
                    break  # e.g. 400/404: this model can't serve the request, try the next one
                retry_after = resp.headers.get("Retry-After")
            if attempt < settings.LLM_MAX_RETRIES:
                delay = 2.0 * (2 ** attempt)
                if retry_after and str(retry_after).isdigit():
                    delay = max(delay, min(float(retry_after), 20.0))
                if deadline - time.monotonic() - delay < 5:
                    break
                time.sleep(delay)
    raise LLMError(f"All models failed. Last error: {last_error}")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def extract_json(text: str) -> Any:
    """Parses JSON from a model reply, tolerating code fences, prose around it and <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except ValueError:
                            break
            start = cleaned.find(opener, start + 1)
    raise LLMError("The model did not return valid JSON.")


def chat_json(
    system: str,
    user: str,
    *,
    usage: Optional[Usage] = None,
    deadline: Optional[float] = None,
    max_tokens: int = 1800,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Any, LLMReply]:
    """Chat call that must return JSON. One repair attempt if the first answer isn't valid JSON."""
    messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
    reply = chat(messages, usage=usage, deadline=deadline, max_tokens=max_tokens)
    try:
        return extract_json(reply.text), reply
    except LLMError:
        messages += [
            {"role": "assistant", "content": reply.text[:4000]},
            {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object, no prose, no code fences."},
        ]
        reply = chat(messages, usage=usage, deadline=deadline, max_tokens=max_tokens)
        return extract_json(reply.text), reply
