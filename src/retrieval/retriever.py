import logging
from typing import List, Dict

from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_entity_matcher import SemanticEntityMatcher

_log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedding_model, vector_store, entity_store=None,
                 top_k=5, initial_k=25, entity_threshold=0.45):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.initial_k = initial_k
        self.reranker = CrossEncoderReranker()
        self.query_rewriter = QueryRewriter()
        self.entity_matcher = (
            SemanticEntityMatcher(embedding_model, entity_store, threshold=entity_threshold)
            if entity_store is not None else None
        )

    def retrieve(self, question: str) -> List[Dict[str, str | float]]:
        # ── 1. Semantic vector search ────────────────────────────────────
        query_embedding = self.embedding_model.encode_query(question)
        rows = self.vector_store.search(query_embedding, self.initial_k)
        results = [
            {"score": score, "metadata": metadata, "text": text}
            for score, metadata, text in rows
        ]
        _log.debug("[Retriever] vector search returned %d candidates", len(results))

        # ── 2. Semantic entity matching ──────────────────────────────────
        if self.entity_matcher is not None:
            matched_entities = self.entity_matcher.match(question)
            if matched_entities:
                entity_chunks = self.entity_matcher.fetch_chunks(
                    matched_entities, self.vector_store
                )
                # Merge into result list, deduplicating by chunk_id.
                existing_ids = {r["metadata"].get("chunk_id") for r in results}
                added = 0
                for chunk in entity_chunks:
                    cid = chunk["metadata"].get("chunk_id")
                    if cid not in existing_ids:
                        results.append(chunk)
                        existing_ids.add(cid)
                        added += 1
                _log.debug("[Retriever] +%d entity chunks → %d total candidates", added, len(results))

        # ── 3. Rerank combined candidate set ────────────────────────────
        reranked = self.reranker.rerank(question, results)
        return reranked[:self.top_k]