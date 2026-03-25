from __future__ import annotations

import os
from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI
import ollama


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class NoLLMClient:
    def generate(self, prompt: str) -> str:
        return (
            "LLM provider is not configured. Retrieval worked, but generation is disabled.\n\n"
            "Set LLM_PROVIDER in .env to one of: anthropic, openai, ollama."
        )


class AnthropicClient:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1800,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


class OpenAICompatibleClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class OllamaClient:
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    def generate(self, prompt: str) -> str:
        resp = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]



def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAICompatibleClient()
    if provider == "ollama":
        return OllamaClient()
    return NoLLMClient()
