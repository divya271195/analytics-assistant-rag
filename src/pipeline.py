from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.embeddings.embedding_model import EmbeddingModel
from src.generation.code_generator import RagGenerator
from src.generation.llm_client import get_llm_client
from src.ingestion.document_cleaner import DocumentCleaner
from src.ingestion.loaders import DocumentLoader, RawDocument
from src.processing.chunker import TextChunker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FaissVectorStore
from src.utils.config import load_yaml
from src.processing.entity_extractor import EntityExtractor
from src.retrieval.entity_store import EntityStore
from src.retrieval.ontology import infer_requested_concept


class RagPipeline:
    def __init__(self, app_config_path: str = "configs/app_config.yaml"):
            self.config = load_yaml(app_config_path)
            self.embedding_model = EmbeddingModel(self.config["embedding"]["model_name"])
            self.vector_store = FaissVectorStore()
            self.entity_extractor = EntityExtractor()
            self.entity_store = EntityStore()
            self.retriever = Retriever(
                embedding_model=self.embedding_model,
                vector_store=self.vector_store,
                top_k=self.config["retrieval"]["top_k"],
            )
            self.generator = RagGenerator(get_llm_client())

    def ingest_and_index(self, docs_dir: str, index_dir: str) -> Dict[str, int]:
        loader = DocumentLoader()
        cleaner = DocumentCleaner()
        chunker = TextChunker(
            chunk_size=self.config["chunking"]["chunk_size"],
            overlap=self.config["chunking"]["overlap"],
        )

        raw_docs: List[RawDocument] = loader.load_directory(docs_dir)

        cleaned_docs = [
            RawDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                source_path=doc.source_path,
                text=cleaner.clean(doc.text),
                metadata=doc.metadata,
            )
            for doc in raw_docs
        ]

        if not cleaned_docs:
            raise ValueError(f"No documents found in {docs_dir}")

        chunks = chunker.chunk_documents(cleaned_docs)

        if not chunks:
            raise ValueError(f"No chunks were produced from documents in {docs_dir}")

        embeddings = self.embedding_model.encode_texts([chunk.text for chunk in chunks])

        self.vector_store.build(embeddings, chunks)
        self.vector_store.save(index_dir)

        entity_records = self.entity_extractor.extract_from_documents(cleaned_docs)
        self.entity_store.build(entity_records)
        self.entity_store.save(index_dir)

        entity_count = sum(len(r.entities) for r in entity_records)

        return {
            "documents": len(cleaned_docs),
            "chunks": len(chunks),
            "entities": entity_count,
        }

    def load_index(self, index_dir: str) -> None:
        self.vector_store.load(index_dir)
        self.entity_store.load(index_dir)

    def retrieve(self, question: str) -> List[Dict]:
        return self.retriever.retrieve(question)

    def list_entities_for_question(self, question: str) -> List[Dict]:
        concept = infer_requested_concept(question)
        if not concept:
            return []
        return self.entity_store.unique_values_by_concept(concept)

    def discovery(self, question: str) -> str:
        results = self.retrieve(question)
        return self.generator.answer_discovery(question, results)

    def codegen(self, question: str) -> str:
        results = self.retrieve(question)
        return self.generator.generate_code(question, results)
