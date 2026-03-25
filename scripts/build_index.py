
from __future__ import annotations

import argparse

from src.pipeline import RagPipeline
from src.utils.config import load_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    load_environment()
    pipeline = RagPipeline()
    stats = pipeline.ingest_and_index(args.docs_dir, args.index_dir)
    print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks.")


if __name__ == "__main__":
    main()
