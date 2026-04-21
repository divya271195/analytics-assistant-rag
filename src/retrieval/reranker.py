from __future__ import annotations

from typing import List, Dict

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return results

        pairs = [(question, item["text"]) for item in results]
        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(results, scores):
            row = dict(item)
            row["rerank_score"] = float(score)
            reranked.append(row)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked
#
# class NoOpReranker:
#     def rerank(self, results: List[Dict]) -> List[Dict]:
#         return results
