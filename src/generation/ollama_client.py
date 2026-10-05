"""Minimal Ollama client (local LLM, optional). Raises LLMUnavailable on any failure."""
from __future__ import annotations

import httpx


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b", timeout: float = 120.0):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, system: str, prompt: str) -> str:
        try:
            r = httpx.post(f"{self.base_url}/api/generate", timeout=self.timeout, json={
                "model": self.model, "system": system, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096, "seed": 42}})
            r.raise_for_status()
            return r.json()["response"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMUnavailable(f"Ollama call failed: {e}") from e
