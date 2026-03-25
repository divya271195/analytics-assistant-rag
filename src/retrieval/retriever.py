from __future__ import annotations

from typing import Dict, List

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import FaissVectorStore


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: FaissVectorStore, top_k: int = 5):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict[str, str | float]]:
        query_embedding = self.embedding_model.encode_query(question)
        rows = self.vector_store.search(query_embedding, self.top_k)
        return [
            {"score": score, "metadata": metadata, "text": text}
            for score, metadata, text in rows
        ]
