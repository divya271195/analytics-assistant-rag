from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from src.processing.entity_extractor import DocumentEntityRecord
from src.retrieval.ontology import expand_concept


class EntityStore:
    def __init__(self) -> None:
        self.records: List[Dict] = []

    def build(self, records: List[DocumentEntityRecord]) -> None:
        self.records = [self._to_dict(r) for r in records]

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(index_dir / "entity_catalog.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def load(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        path = index_dir / "entity_catalog.json"

        if not path.exists():
            self.records = []
            return

        with open(path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def find_by_concept(self, concept: str) -> List[Dict]:
        """
        Example:
        concept='table' should return source_table + target_table + fact_table + ...
        """
        allowed_subtypes = set(expand_concept(concept))
        rows: List[Dict] = []

        for record in self.records:
            title = record["title"]
            source_path = record["source_path"]
            doc_id = record["doc_id"]

            for entity in record.get("entities", []):
                if entity.get("subtype") in allowed_subtypes or entity.get("type") == concept:
                    rows.append(
                        {
                            "doc_id": doc_id,
                            "title": title,
                            "source_path": source_path,
                            "entity_type": entity.get("type"),
                            "entity_subtype": entity.get("subtype"),
                            "value": entity.get("value"),
                            "evidence": entity.get("evidence"),
                        }
                    )

        return rows

    def unique_values_by_concept(self, concept: str) -> List[Dict]:
        rows = self.find_by_concept(concept)

        seen = set()
        unique_rows: List[Dict] = []

        for row in rows:
            key = row["value"].lower()
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

        unique_rows.sort(key=lambda x: x["value"].lower())
        return unique_rows

    def _to_dict(self, record: DocumentEntityRecord) -> Dict:
        return {
            "doc_id": record.doc_id,
            "title": record.title,
            "source_path": record.source_path,
            "metadata": record.metadata,
            "entities": [
                {
                    "type": e.type,
                    "subtype": e.subtype,
                    "value": e.value,
                    "evidence": e.evidence,
                }
                for e in record.entities
            ],
        }