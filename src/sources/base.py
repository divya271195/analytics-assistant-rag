# from __future__ import annotations
from typing import List, Protocol
from src.ingestion.loaders import RawDocument

class DocumentSource(Protocol):
    def load_documents(self) -> List[RawDocument]: ...