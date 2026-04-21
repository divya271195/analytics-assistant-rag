from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.ingestion.loaders import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str
    metadata: Dict[str, str]


class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 120):
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[RawDocument]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            for idx, chunk_text in enumerate(self._chunk_text(doc.text)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        source_path=doc.source_path,
                        text=chunk_text,
                        metadata={**doc.metadata,
                                  "chunk_index": str(idx),
                                  "title": doc.title,
                                  "source_path": doc.source_path,
                                  "doc_id": doc.doc_id,
                                  "section_title": "",
                                  "page_number": ""
                                  },
                    )
                )
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = []

        for para in paragraphs:
            candidate = "\n\n".join(current + [para])
            if len(candidate) <= self.chunk_size:
                current.append(para)
            else:
                if current:
                    chunks.append("\n\n".join(current))
                current = [para]

        if current:
            chunks.append("\n\n".join(current))

        return chunks