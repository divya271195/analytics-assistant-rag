from typing import List, Dict

from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import CrossEncoderReranker

class Retriever:
    def __init__(self, embedding_model, vector_store, top_k=5, initial_k=15):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.initial_k = initial_k
        self.reranker = CrossEncoderReranker()
        self.query_rewriter = QueryRewriter()

    def retrieve(self, question: str) -> List[Dict[str, str | float]]:
        query_embedding = self.embedding_model.encode_query(question)
        rows = self.vector_store.search(query_embedding, self.initial_k)
        results = [
            {"score": score, "metadata": metadata, "text": text}
            for score, metadata, text in rows
        ]
        reranked = self.reranker.rerank(question, results)
        rewritten = self.query_rewriter.rewrite(question)
        query_embedding = self.embedding_model.encode_query(rewritten)

        return reranked[: self.top_k]