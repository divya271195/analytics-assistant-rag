from __future__ import annotations

from typing import Dict, List


def format_retrieval_results(results: List[Dict]) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        meta = item["metadata"]
        lines.append(
            f"[{idx}] score={item['score']:.4f} | title={meta.get('title', '')} | path={meta.get('source_path', '')}\n"
            f"{item['text'][:500]}\n"
        )
    return "\n".join(lines)
