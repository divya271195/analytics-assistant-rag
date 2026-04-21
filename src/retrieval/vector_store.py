from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from src.processing.chunker import Chunk
from src.processing.metadata_builder import chunk_to_metadata


class FaissVectorStore:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, str]] = []
        self.texts: List[str] = []

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

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))
        self.metadata = [chunk_to_metadata(chunk) for chunk in chunks]
        self.texts = [chunk.text for chunk in chunks]

    def save(self, index_dir: str | Path) -> None:
        if self.index is None:
            raise ValueError("Index has not been built")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": self.metadata, "texts": self.texts}, f, ensure_ascii=False, indent=2)

    def load(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.metadata = payload["metadata"]
        self.texts = payload["texts"]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, str], str]]:
        if self.index is None:
            raise ValueError("Index is empty. Load or build it first.")
        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: List[Tuple[float, Dict[str, str], str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx], self.texts[idx]))
        return results
