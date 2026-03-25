from __future__ import annotations

import argparse

from src.pipeline import RagPipeline
from src.utils.config import load_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["codegen", "discover"], default="codegen")
    args = parser.parse_args()

    load_environment()
    pipeline = RagPipeline()
    pipeline.load_index(args.index_dir)
    if args.mode == "discover":
        print(pipeline.discovery(args.question))
    else:
        print(pipeline.codegen(args.question))


if __name__ == "__main__":
    main()
