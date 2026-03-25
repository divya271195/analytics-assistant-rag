# Requirement-to-Code RAG

A Python-based Retrieval Augmented Generation (RAG) project that reads requirement or content documents and generates grounded outputs such as:
- requirement summary
- business rules
- SQL
- PySpark code
- data quality checks
- assumptions and missing information

This repo is designed to be:
- beginner-friendly
- modular
- GitHub-ready
- easy to extend later with Confluence, RBAC, reranking, and schema-aware validation

## What problem this solves

In many teams, product and business requirements are written in documents, but engineers still need to manually translate them into SQL logic, Spark transformations, and validation rules. This project helps by retrieving the relevant requirement context first and then generating technical implementation drafts grounded in that context.

## Features

- document ingestion for `txt`, `md`, `pdf`, `docx`, `html`
- chunking with overlap
- embeddings using `sentence-transformers`
- FAISS vector index
- semantic retrieval
- grounded code generation
- optional support for Anthropic Claude, OpenAI-compatible APIs, or Ollama
- structured output with sources
- starter Confluence loader for future extension

## Architecture

```text
                   +----------------------+
                   |  Requirement Docs    |
                   |  (TXT/PDF/MD/HTML)   |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |   Ingestion Layer     |
                   |  Load + Clean Docs    |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |   Chunking Layer      |
                   | Chunk + Metadata      |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |   Embedding Model     |
                   | Convert Text to Vec   |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |    Vector Store       |
                   |      FAISS Index      |
                   +----------+-----------+
                              |
               User Query     |
        +---------------------+
        |
        v
+----------------------+      +----------------------+
| Query Embedding      | ---> | Retriever            |
| Convert Query to Vec |      | Top-K Relevant Chunks|
+----------------------+      +----------+-----------+
                                          |
                                          v
                               +----------------------+
                               |   Prompt Builder      |
                               | Context + Instruction |
                               +----------+-----------+
                                          |
                                          v
                               +----------------------+
                               |  LLM / Claude / GPT  |
                               | Code + Summary Gen   |
                               +----------+-----------+
                                          |
                                          v
                               +----------------------+
                               | Structured Output     |
                               | SQL / Spark / DQ      |
                               +----------------------+
```

## Repo structure

```text
requirement-to-code-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
└── tests/
```

## Quick start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd requirement-to-code-rag
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Fill `.env` only if you want LLM generation. Retrieval-only usage works without it.

### 3) Add documents

Put your docs in:

```text
data/raw/sample_requirement_docs/
```

Some sample docs are already included.

### 4) Build the index

```bash
python scripts/build_index.py --docs-dir data/raw/sample_requirement_docs --index-dir data/index
```

### 5) Ask discovery questions

```bash
python scripts/query_rag.py --index-dir data/index --question "What business rules are defined for booking revenue?"
```

### 6) Generate SQL / PySpark

```bash
python scripts/generate_code.py --index-dir data/index --question "Generate Spark code for daily booking revenue aggregation"
```

## Supported LLM providers

Set `LLM_PROVIDER` in `.env`.

### Anthropic Claude

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

### OpenAI-compatible API

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Ollama

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434
```

## Example queries

### Discovery
- What source tables are mentioned in the documents?
- What output fields are required?
- What are the business rules for hotel profile ingestion?
- Which fields are used for partitioning?

### Code generation
- Generate SQL for daily booking revenue aggregation
- Generate PySpark code for hotel profile change detection
- List assumptions before generating SQL
- Generate data quality checks from the requirement

## Confluence support

This repo includes a starter `confluence_loader.py` that can pull pages from Confluence Cloud using email + API token. It is optional and not needed for local docs.

## Suggested next improvements

- Confluence space crawling
- hybrid search
- reranking
- RBAC / permission-aware retrieval
- schema-aware validation
- feedback loop
- evaluation harness
- FastAPI or Streamlit UI

## Interview positioning

Use this project as:

> A knowledge-grounded engineering assistant that reads requirement documents and generates SQL and Spark implementation drafts from them.

## License

MIT
