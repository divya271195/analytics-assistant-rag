from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

_log = logging.getLogger(__name__)


class SemanticEntityMatcher:
    """
    Bridges the gap between what a user says ("booking revenue table") and
    what the entity catalog stores ("booking_revenue_abc", "booking_swd_revenue").

    At query time:
      1. Embed every entity value (table name etc.) stored in the entity catalog,
         replacing underscores with spaces so the model handles them as words.
      2. Compare with the query embedding via cosine similarity.
      3. Return entity records whose similarity exceeds `threshold`.
      4. Fetch chunks from the vector store that contain those entity names.
    """

    def __init__(self, embedding_model, entity_store, threshold: float = 0.45):
        self.embedding_model = embedding_model
        self.entity_store = entity_store
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, question: str) -> List[Dict]:
        """Return entity records semantically similar to the question."""
        all_entities = self._all_entities()
        if not all_entities:
            return []

        # Embed entity values with underscores replaced by spaces so the
        # language model treats them as real words, not code identifiers.
        entity_texts = [e["value"].replace("_", " ") for e in all_entities]
        entity_embs = self.embedding_model.encode_texts(entity_texts)  # (n, dim) normalized

        query_emb = self.embedding_model.encode_query(question)  # (1, dim) normalized

        # Cosine similarity: both sides are L2-normalised so dot product == cosine.
        scores: np.ndarray = (entity_embs @ query_emb.T).flatten()

        matched = []
        for entity, score in zip(all_entities, scores):
            s = float(score)
            if s >= self.threshold:
                matched.append({**entity, "entity_score": s})
                _log.debug("[SemanticEntity] matched '%s'  score=%.3f", entity["value"], s)

        matched.sort(key=lambda x: x["entity_score"], reverse=True)
        _log.debug("[SemanticEntity] %d / %d entities matched for query: %r",
                   len(matched), len(all_entities), question[:80])
        return matched

    def fetch_chunks(self, matched_entities: List[Dict], vector_store) -> List[Dict]:
        """
        For each matched entity value, retrieve chunks whose text contains
        that value (exact substring, both underscore and space variants).
        Returns dicts in the same shape as the vector-search results so they
        can be merged and reranked without any downstream changes.
        """
        if not matched_entities:
            return []

        values = [e["value"] for e in matched_entities]
        # Build a lookup so each chunk can be tagged with which entity it matched.
        score_by_value = {e["value"]: e["entity_score"] for e in matched_entities}

        raw = vector_store.find_chunks_containing(values)  # List[(score, meta, text)]

        results = []
        for score, meta, text in raw:
            # Find which entity value was responsible for this chunk.
            text_lower = text.lower()
            matched_value = next(
                (v for v in values
                 if v.lower() in text_lower or v.lower().replace("_", " ") in text_lower),
                values[0],
            )
            results.append({
                "score": score_by_value.get(matched_value, score),
                "metadata": meta,
                "text": text,
                "matched_entity": matched_value,
            })

        _log.debug("[SemanticEntity] %d entity-matched chunks fetched", len(results))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_entities(self) -> List[Dict]:
        entities: List[Dict] = []
        for concept in ("table", "schema"):
            entities.extend(self.entity_store.unique_values_by_concept(concept))
        return entities
