from __future__ import annotations

import argparse

from src.ingestion.loaders import DocumentLoader
from src.ingestion.document_cleaner import DocumentCleaner
from src.utils.helpers import truncate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    loader = DocumentLoader()
    cleaner = DocumentCleaner()
    docs = loader.load_directory(args.input_dir)
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        cleaned = cleaner.clean(doc.text)
        print(f"- {doc.title} | {doc.source_path}")
        print(truncate(cleaned, 300))
        print()


if __name__ == "__main__":
    main()
