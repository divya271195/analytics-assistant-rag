from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4
import shutil

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.processing.chunker import Chunk
from src.processing.metadata_builder import chunk_to_metadata

_COLLECTION = "rag_chunks"


class VectorStore:
    """Vector store backed by local-persisted Qdrant."""

    def __init__(self):
        self._client: QdrantClient | None = None
        self._qdrant_path: str | None = None
        self._ready: bool = False
        self._embeddings: np.ndarray | None = None
        self._chunks: List[Chunk] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the local Qdrant lock and forget the active client."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._qdrant_path = None
        self._ready = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self, qdrant_path: str) -> QdrantClient:
        """
        Create or reuse a Qdrant client for the requested path.
        If the path changes, close the old client first so Windows releases .lock.
        """
        if self._client is not None and self._qdrant_path != qdrant_path:
            self.close()

        if self._client is None:
            self._client = QdrantClient(path=qdrant_path)
            self._qdrant_path = qdrant_path

        return self._client

    def _get_qdrant_path(self, index_dir: str | Path) -> Path:
        return Path(index_dir) / "qdrant_data"

    def _require_client(self) -> QdrantClient:
        if self._client is None:
            raise ValueError("Qdrant client is not initialized. Load or save the index first.")
        return self._client

    def _require_collection(self) -> QdrantClient:
        client = self._require_client()
        if not client.collection_exists(_COLLECTION):
            location = self._qdrant_path or "<unknown>"
            raise ValueError(
                f"Collection '{_COLLECTION}' not found in {location}. "
                "Build/save the index first."
            )
        return client

    # ------------------------------------------------------------------
    # Build / save / load
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
        self._ready = False

    def save(self, index_dir: str | Path) -> None:
        if self._embeddings is None or self._chunks is None:
            raise ValueError("Index has not been built.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        qdrant_path = str(self._get_qdrant_path(index_dir))
        client = self._ensure_client(qdrant_path)
        dim = int(self._embeddings.shape[1])

        if client.collection_exists(_COLLECTION):
            client.delete_collection(_COLLECTION)

        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=self._embeddings[i].tolist(),
                payload={
                    **chunk_to_metadata(self._chunks[i]),
                    "text": self._chunks[i].text,
                },
            )
            for i in range(len(self._chunks))
        ]

        client.upsert(collection_name=_COLLECTION, points=points)
        self._ready = True

    def load(self, index_dir: str | Path) -> None:
        qdrant_path = str(self._get_qdrant_path(index_dir))
        client = self._ensure_client(qdrant_path)

        if not client.collection_exists(_COLLECTION):
            raise ValueError(
                f"Collection '{_COLLECTION}' not found in {qdrant_path}. "
                "Run indexing first."
            )

        self._ready = True

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def delete_collection(self, index_dir: str | Path) -> None:
        """
        Delete only the RAG collection.
        Prefer this for a normal rebuild.
        """
        qdrant_path = str(self._get_qdrant_path(index_dir))
        client = self._ensure_client(qdrant_path)

        if client.collection_exists(_COLLECTION):
            client.delete_collection(_COLLECTION)

        self._ready = False

    def reset_storage(self, index_dir: str | Path) -> None:
        """
        Wipe the whole local Qdrant folder.
        Always close the client first so .lock is released on Windows.
        """
        qdrant_path = self._get_qdrant_path(index_dir)

        # Release the .lock file before deleting the directory
        self.close()

        if qdrant_path.exists():
            shutil.rmtree(qdrant_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_chunks_containing(
        self, text_values: List[str]
    ) -> List[Tuple[float, Dict[str, str], str]]:
        """
        Return chunks whose text contains any of the given entity values.

        Checks both the original value (e.g. ``booking_revenue_abc``) and its
        space-separated form (``booking revenue abc``) so it works regardless of
        how the document was written.
        """
        if not self._ready:
            return []

        client = self._require_collection()

        all_points, _ = client.scroll(
            collection_name=_COLLECTION,
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )

        seen: set[str] = set()
        results: List[Tuple[float, Dict[str, str], str]] = []

        for point in all_points:
            payload = dict(point.payload or {})
            raw_text = str(payload.get("text", ""))
            text_lower = raw_text.lower()
            chunk_id = str(payload.get("chunk_id", ""))

            if chunk_id and chunk_id in seen:
                continue

            for value in text_values:
                v_lower = value.lower()
                v_spaced = v_lower.replace("_", " ")
                if v_lower in text_lower or v_spaced in text_lower:
                    if chunk_id:
                        seen.add(chunk_id)
                    stored_text = str(payload.pop("text", ""))
                    results.append((1.0, payload, stored_text))
                    break

        return results

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[float, Dict[str, str], str]]:
        if not self._ready:
            raise ValueError("Index is empty. Load or build it first.")

        client = self._require_collection()

        hits = client.query_points(
            collection_name=_COLLECTION,
            query=query_embedding.flatten().astype("float32").tolist(),
            limit=top_k,
        ).points

        results: List[Tuple[float, Dict[str, str], str]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = str(payload.pop("text", ""))
            results.append((float(hit.score), payload, text))

        return results