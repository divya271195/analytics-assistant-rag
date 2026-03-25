from src.processing.chunker import TextChunker
from src.ingestion.loaders import RawDocument


def test_chunker_creates_multiple_chunks():
    text = "a" * 2000
    doc = RawDocument(doc_id="1", title="t", source_path="x", text=text, metadata={})
    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_documents([doc])
    assert len(chunks) >= 4
