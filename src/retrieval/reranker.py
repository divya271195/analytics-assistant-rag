from __future__ import annotations

from typing import List, Dict


class NoOpReranker:
    def rerank(self, results: List[Dict]) -> List[Dict]:
        return results
