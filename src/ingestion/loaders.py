from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from src.utils.helpers import iter_supported_files


@dataclass
class RawDocument:
    doc_id: str
    title: str
    source_path: str
    text: str
    metadata: Dict[str, str]


class DocumentLoader:
    def load_directory(self, docs_dir: str) -> List[RawDocument]:
        docs: List[RawDocument] = []
        for file_path in iter_supported_files(docs_dir):
            text = self.load_file(file_path)
            docs.append(
                RawDocument(
                    doc_id=file_path.stem,
                    title=file_path.stem.replace("_", " ").title(),
                    source_path=str(file_path),
                    text=text,
                    metadata={"extension": file_path.suffix.lower()},
                )
            )
        return docs

    def load_file(self, path: str | Path) -> str:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        if suffix in {".html", ".htm"}:
            return self._read_html(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _read_html(self, path: Path) -> str:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text("\n").splitlines()]
        return "\n".join([line for line in lines if line])

    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)

    def _read_docx(self, path: Path) -> str:
        doc = DocxDocument(str(path))
        return "\n".join([para.text for para in doc.paragraphs])
