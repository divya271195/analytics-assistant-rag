from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
