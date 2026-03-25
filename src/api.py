from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import RagPipeline

app = FastAPI(title="Requirement-to-Code RAG")
pipeline = RagPipeline()


class LoadIndexRequest(BaseModel):
    index_dir: str


class QueryRequest(BaseModel):
    question: str


@app.post("/load-index")
def load_index(request: LoadIndexRequest):
    pipeline.load_index(request.index_dir)
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    return {"results": pipeline.retrieve(request.question)}


@app.post("/discover")
def discover(request: QueryRequest):
    return {"answer": pipeline.discovery(request.question)}


@app.post("/generate")
def generate(request: QueryRequest):
    return {"answer": pipeline.codegen(request.question)}
