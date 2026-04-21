from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("encode_texts received an empty list. No chunks were produced.")
        arr = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def encode_query(self, query: str) -> np.ndarray:
        arr = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr