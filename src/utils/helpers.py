from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


def iter_supported_files(root_dir: str | Path) -> List[Path]:
    root = Path(root_dir)
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def truncate(text: str, size: int = 800) -> str:
    return text if len(text) <= size else text[: size - 3] + "..."
