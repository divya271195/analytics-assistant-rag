from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.processing.chunker import Chunk
from src.processing.metadata_builder import chunk_to_metadata

_COLLECTION = "rag_chunks"


class FaissVectorStore:
    """Vector store backed by local-persisted Qdrant (drop-in replacement for FAISS)."""

    def __init__(self):
        self._client: QdrantClient | None = None
        self._qdrant_path: str | None = None
        self._ready: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self, qdrant_path: str) -> QdrantClient:
        if self._client is None or self._qdrant_path != qdrant_path:
            self._client = QdrantClient(path=qdrant_path)
            self._qdrant_path = qdrant_path
        return self._client

    # ------------------------------------------------------------------
    # Public interface (same signatures as the old FaissVectorStore)
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.size == 0:
            raise ValueError("Embeddings are empty.")
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]})"
            )

        self._embeddings = embeddings.astype("float32")
        self._chunks = chunks
        self._ready = False  # committed on save()

    def save(self, index_dir: str | Path) -> None:
        if not hasattr(self, "_embeddings"):
            raise ValueError("Index has not been built")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        qdrant_path = str(index_dir / "qdrant_data")

        client = self._ensure_client(qdrant_path)
        dim = self._embeddings.shape[1]

        # Recreate collection so save() is idempotent
        if client.collection_exists(_COLLECTION):
            client.delete_collection(_COLLECTION)
        client.create_collection(
            _COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=self._embeddings[i].tolist(),
                payload={**chunk_to_metadata(self._chunks[i]), "text": self._chunks[i].text},
            )
            for i in range(len(self._chunks))
        ]
        client.upsert(collection_name=_COLLECTION, points=points)
        self._ready = True

    def load(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        qdrant_path = str(index_dir / "qdrant_data")
        self._ensure_client(qdrant_path)
        self._ready = True

    def find_chunks_containing(self, text_values: List[str]) -> List[Tuple[float, Dict[str, str], str]]:
        """Return chunks whose text contains any of the given entity values.

        Checks both the original value (e.g. ``booking_revenue_abc``) and its
        space-separated form (``booking revenue abc``) so it works regardless of
        how the document was written.
        """
        if not self._ready or self._client is None:
            return []

        all_points, _ = self._client.scroll(
            collection_name=_COLLECTION,
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )

        seen: set = set()
        results: List[Tuple[float, Dict[str, str], str]] = []

        for point in all_points:
            payload = dict(point.payload or {})
            raw_text = payload.get("text", "")
            text_lower = raw_text.lower()
            chunk_id = payload.get("chunk_id", "")

            if chunk_id in seen:
                continue

            for value in text_values:
                v_lower = value.lower()
                v_spaced = v_lower.replace("_", " ")
                if v_lower in text_lower or v_spaced in text_lower:
                    seen.add(chunk_id)
                    stored_text = payload.pop("text", "")
                    results.append((1.0, payload, stored_text))
                    break  # one match per chunk is enough

        return results

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, str], str]]:
        if not self._ready or self._client is None:
            raise ValueError("Index is empty. Load or build it first.")
        hits = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_embedding.flatten().astype("float32").tolist(),
            limit=top_k,
        ).points
        results: List[Tuple[float, Dict[str, str], str]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append((hit.score, payload, text))
        return results
