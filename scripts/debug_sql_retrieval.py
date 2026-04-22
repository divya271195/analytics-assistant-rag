"""
Debug why SQL answer generation returns "Insufficient Information".

Shows every stage of the retrieval → rerank → context assembly pipeline
for a given question, so you can see exactly what the LLM receives.

Usage:
    python scripts/debug_sql_retrieval.py --index-dir data/index \
        --question "Generate SQL for booking_payment_link"

Close Streamlit / any other process holding the Qdrant lock before running.
"""
from __future__ import annotations

import argparse
import textwrap

from src.utils.config import load_environment


def separator(label: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--question", default="Generate SQL for booking_payment_link")
    parser.add_argument("--initial-k", type=int, default=15)
    args = parser.parse_args()

    load_environment()

    from src.utils.config import load_yaml
    from src.embeddings.embedding_model import EmbeddingModel
    from src.retrieval.vector_store import FaissVectorStore
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.entity_store import EntityStore
    from src.generation.prompt_builder import build_context, build_codegen_prompt

    config = load_yaml("configs/app_config.yaml")
    embedding_model = EmbeddingModel(config["embedding"]["model_name"])
    vector_store = FaissVectorStore()
    vector_store.load(args.index_dir)
    entity_store = EntityStore()
    entity_store.load(args.index_dir)

    # ------------------------------------------------------------------ #
    # 1. Collection stats
    # ------------------------------------------------------------------ #
    separator("1. COLLECTION STATS")
    from qdrant_client import QdrantClient
    client = QdrantClient(path=f"{args.index_dir}/qdrant_data")
    info = client.get_collection("rag_chunks")
    print(f"  Total vectors : {info.points_count}")
    print(f"  Distance      : {info.config.params.vectors.distance}")
    print(f"  Vector size   : {info.config.params.vectors.size}")

    # ------------------------------------------------------------------ #
    # 2. Raw vector search (before rerank)
    # ------------------------------------------------------------------ #
    separator(f"2. RAW VECTOR SEARCH  top-{args.initial_k}  (before rerank)")
    query_embedding = embedding_model.encode_query(args.question)
    raw_rows = vector_store.search(query_embedding, args.initial_k)

    for rank, (score, meta, text) in enumerate(raw_rows, 1):
        print(f"\n  [{rank:02d}] score={score:.4f}")
        print(f"       source : {meta.get('source_path', '?')}")
        print(f"       chunk  : {meta.get('chunk_id', '?')}")
        print(f"       title  : {meta.get('title', '?')}")
        print(f"       text   : {textwrap.shorten(text, width=120, placeholder='...')}")

    # ------------------------------------------------------------------ #
    # 3. After cross-encoder rerank
    # ------------------------------------------------------------------ #
    separator("3. AFTER CROSS-ENCODER RERANK  (top-5 cutoff)")
    reranker = CrossEncoderReranker()
    results = [{"score": s, "metadata": m, "text": t} for s, m, t in raw_rows]
    reranked = reranker.rerank(args.question, results)

    print(f"\n  Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"  (trained on web search — may demote schema/DDL chunks)\n")
    for rank, item in enumerate(reranked, 1):
        marker = "  >>>  CUTOFF (top_k=5)" if rank == 6 else ""
        print(f"  [{rank:02d}] rerank={item['rerank_score']:+.4f}  "
              f"cosine={item['score']:.4f}  "
              f"chunk={item['metadata'].get('chunk_id','?')}{marker}")
        print(f"       text: {textwrap.shorten(item['text'], width=100, placeholder='...')}")

    top5 = reranked[:5]

    # ------------------------------------------------------------------ #
    # 4. booking_payment_link presence check
    # ------------------------------------------------------------------ #
    separator("4. KEYWORD PRESENCE CHECK  ('booking_payment_link')")
    keyword = "booking_payment_link"
    all_points, _ = client.scroll(
        collection_name="rag_chunks", limit=2000,
        with_payload=True, with_vectors=False,
    )
    matching = [p for p in all_points if keyword.lower() in (p.payload.get("text") or "").lower()]
    print(f"\n  Chunks containing '{keyword}' in index : {len(matching)}")
    for p in matching:
        pl = p.payload
        print(f"    chunk_id={pl.get('chunk_id')}  source={pl.get('source_path')}")
        print(f"    text: {textwrap.shorten(pl.get('text',''), width=120, placeholder='...')}")

    in_top5 = [r for r in top5 if keyword.lower() in r["text"].lower()]
    print(f"\n  Of those, in top-5 after rerank : {len(in_top5)}")
    if not in_top5 and matching:
        demoted = [r for r in reranked if keyword.lower() in r["text"].lower()]
        if demoted:
            positions = [reranked.index(r)+1 for r in demoted]
            print(f"  *** RERANKER DEMOTION CONFIRMED: '{keyword}' chunks reranked to positions {positions}")
            print(f"      They were fetched but cut off at top_k=5.")
        else:
            print(f"  *** COSINE MISS: '{keyword}' exists in index but not in top-{args.initial_k} cosine results.")
            print(f"      Increase --initial-k or add entity-store lookup.")

    # ------------------------------------------------------------------ #
    # 5. Entity store lookup
    # ------------------------------------------------------------------ #
    separator("5. ENTITY STORE  (structured catalog, currently NOT wired to retrieval)")
    from src.retrieval.ontology import infer_requested_concept
    concept = infer_requested_concept(args.question)
    print(f"\n  Inferred concept from question: {concept!r}")
    if concept:
        entities = entity_store.unique_values_by_concept(concept)
        print(f"  Entities found: {len(entities)}")
        for e in entities[:10]:
            print(f"    {e['entity_subtype']:20s} {e['value']:40s}  (from {e['doc_id']})")
    else:
        print("  No concept inferred — entity store not consulted.")

    # ------------------------------------------------------------------ #
    # 6. Final context sent to LLM
    # ------------------------------------------------------------------ #
    separator("6. FINAL CONTEXT SENT TO LLM  (top-5 after rerank)")
    context = build_context(top5)
    print(context)

    # ------------------------------------------------------------------ #
    # 7. Full prompt
    # ------------------------------------------------------------------ #
    separator("7. FULL CODEGEN PROMPT  (what the LLM actually sees)")
    full_prompt = build_codegen_prompt(args.question, top5)
    print(full_prompt)

    # ------------------------------------------------------------------ #
    # 8. Diagnosis summary
    # ------------------------------------------------------------------ #
    separator("8. DIAGNOSIS SUMMARY")
    if not raw_rows:
        print("  CRITICAL: Zero results from vector search. Index is empty or not built.")
    elif not matching:
        print(f"  CRITICAL: '{keyword}' is NOT in the index at all. Rebuild the index with the right docs.")
    elif not in_top5 and matching:
        demoted = [r for r in reranked if keyword.lower() in r["text"].lower()]
        if demoted:
            print(f"  ROOT CAUSE: Reranker demotion.")
            print(f"  The schema chunk exists and was retrieved (cosine rank {[reranked.index(r)+1 for r in demoted]})")
            print(f"  but cross-encoder pushed it past position 5.")
            print(f"  FIX: increase initial_k in retriever.py (try 25-30) so reranker has more headroom.")
        else:
            print(f"  ROOT CAUSE: Cosine retrieval miss.")
            print(f"  '{keyword}' chunk exists but cosine search doesn't surface it in top-{args.initial_k}.")
            print(f"  FIX: add entity-store pre-retrieval or increase initial_k.")
    else:
        print(f"  Retrieval looks correct — '{keyword}' IS in top-5 context.")
        print(f"  Issue may be prompt wording or LLM grounding strictness.")
        print(f"  Check prompt_builder.py DISCOVERY_PROMPT for 'Out' fragment (line ~17).")


if __name__ == "__main__":
    main()
