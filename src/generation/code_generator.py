from __future__ import annotations

from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_codegen_prompt, build_discovery_prompt


class RagGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def answer_discovery(self, question: str, results: List[Dict]) -> str:
        prompt = build_discovery_prompt(question, results)
        return self.llm_client.generate(prompt)

    def generate_code(self, question: str, results: List[Dict]) -> str:
        prompt = build_codegen_prompt(question, results)
        return self.llm_client.generate(prompt)
