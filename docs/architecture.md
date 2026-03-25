# Architecture

This project follows a simple RAG pipeline:

1. Load content documents
2. Clean text and extract metadata
3. Chunk documents with overlap
4. Generate embeddings
5. Build FAISS index
6. Retrieve top-k chunks for a user query
7. Build a grounded prompt
8. Generate structured output with SQL, PySpark, DQ checks, assumptions, and sources
