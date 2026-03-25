from __future__ import annotations

from typing import Dict

from src.processing.chunker import Chunk


def chunk_to_metadata(chunk: Chunk) -> Dict[str, str]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "source_path": chunk.source_path,
        **chunk.metadata,
    }
