"""
Inspect what text was extracted and indexed from .jpg/.jpeg files.
Run ONLY when no other process (Streamlit, build_index, etc.) is using the index.

Usage:
    python scripts/inspect_image_chunks.py --index-dir data/index
"""
from __future__ import annotations

import argparse
from qdrant_client import QdrantClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    client = QdrantClient(path=f"{args.index_dir}/qdrant_data")
    collection = "rag_chunks"

    all_points, _ = client.scroll(
        collection_name=collection,
        limit=2000,
        with_payload=True,
        with_vectors=False,
    )

    img_points = [
        p for p in all_points
        if p.payload.get("extension") in {".jpg", ".jpeg"}
    ]

    print(f"Total chunks in index    : {len(all_points)}")
    print(f"Image chunks (.jpg/.jpeg): {len(img_points)}")
    print()

    if not img_points:
        print("NO image chunks found.")
        print("Possible reasons:")
        print("  1. No .jpg/.jpeg files were in the docs directory during build-index")
        print("  2. Tesseract is not configured — OCR returned empty text")
        print("  3. Index was built before OCR support was added — rebuild with:")
        print("       python scripts/build_index.py --docs-dir <your-docs-dir> --index-dir data/index")
        return

    for i, p in enumerate(img_points, 1):
        pl = p.payload
        text = pl.get("text", "")
        print(f"{'='*60}")
        print(f"Chunk {i} of {len(img_points)}")
        print(f"  file       : {pl.get('source_path')}")
        print(f"  doc_id     : {pl.get('doc_id')}")
        print(f"  chunk_id   : {pl.get('chunk_id')}")
        print(f"  text length: {len(text)} chars")
        print(f"  --- extracted text ---")
        print(text if text.strip() else "(empty — OCR returned nothing)")
        print()


if __name__ == "__main__":
    main()
