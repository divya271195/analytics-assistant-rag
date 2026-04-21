from __future__ import annotations

from typing import Dict, List


DISCOVERY_PROMPT = """You are a grounded knowledge discovery assistant.
You must answer ONLY from the provided context.
Do not use outside knowledge.
If the answer is not fully supported by the context, say: "Insufficient Information in the provided sources."
Rules:
1. Do not invent facts.
2. Prefer exact business terminology from the sources.
Quote or paraphrase only what is supported by the retrieved context.
3. End with a "sources" section listing the source title, path and chunk ID used.
5. If multiple sources support the answer mention the most relevant ones.

Out
Use only the provided context.
If the answer is not present, say that clearly.
Provide a concise answer followed by source references.
"""

CODEGEN_PROMPT = """You are a senior data engineering assistant.
Use only the provided context.
If information is missing, do not invent it.
Return the following sections:
1. Requirement Summary
2. Extracted Business Rules
3. SQL Code
4. PySpark Code
5. Data Quality Checks
6. Assumptions
7. Missing Information
8. Source References
"""


def build_context(results: List[Dict]) -> str:
    blocks = []
    for idx, item in enumerate(results, start=1):
        meta = item["metadata"]
        blocks.append(
            f"[Source {idx}]\n"
            f"Title: {meta.get('title', '')}\n"
            f"Path: {meta.get('source_path', '')}\n"
            f"Chunk ID: {meta.get('chunk_id', '')}\n"
            f"Content:\n{item['text']}"
        )
    return "\n\n".join(blocks)


def build_discovery_prompt(question: str, results: List[Dict]) -> str:
    context = build_context(results)
    return f"{DISCOVERY_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}"


def build_codegen_prompt(question: str, results: List[Dict]) -> str:
    context = build_context(results)
    return f"{CODEGEN_PROMPT}\n\nContext:\n{context}\n\nUser request:\n{question}"
