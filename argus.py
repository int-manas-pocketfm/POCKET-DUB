"""
Thin wrapper around the MadEye/Argus proxy for Claude Sonnet.
Uses httpx directly with the OpenAI-compatible /v1/chat/completions endpoint.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

_API_KEY    = os.getenv("ARGUS_API_KEY", "")
_BASE_URL   = os.getenv("ARGUS_BASE_URL", "https://madeye.internal.pocketfm.org")
_MODEL      = os.getenv("ARGUS_MODEL", "claude-sonnet-4-6")
_USER_EMAIL = os.getenv("ARGUS_USER_EMAIL", "")


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens  = input_tokens
        self.output_tokens = output_tokens


class _Content:
    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_Content(text)]
        self.usage   = _Usage(input_tokens, output_tokens)


class _Messages:
    def __init__(self, api_key: str, base_url: str, user_email: str):
        self._api_key    = api_key
        self._base_url   = base_url.rstrip("/") + "/v1/chat/completions"
        self._user_email = user_email

    def create(self, model: str, max_tokens: int, messages: list, **kwargs) -> _Response:
        # Extract system message from messages list if present
        system = kwargs.pop("system", None)
        filtered = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                filtered.append(m)
        if system:
            filtered = [{"role": "system", "content": system}] + filtered

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._user_email:
            headers["user_email"] = self._user_email

        payload = {"model": model, "max_tokens": max_tokens, "messages": filtered, **kwargs}
        r = httpx.post(self._base_url, headers=headers, json=payload, timeout=180.0)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return _Response(
            text,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


class ArgusClient:
    def __init__(self):
        self.messages = _Messages(_API_KEY, _BASE_URL, _USER_EMAIL)


def make_client() -> ArgusClient:
    return ArgusClient()
