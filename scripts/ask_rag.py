from __future__ import annotations

import argparse

from src.pipeline import RagPipeline
from src.generation.output_formatter import format_retrieval_results
from src.utils.config import load_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    load_environment()
    pipeline = RagPipeline()
    pipeline.load_index(args.index_dir)
    answer = pipeline.discovery(args.question)
    print(answer)


if __name__ == "__main__":
    main()
