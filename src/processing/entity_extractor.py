from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from src.ingestion.loaders import RawDocument


@dataclass
class Entity:
    type: str
    subtype: str
    value: str
    evidence: str


@dataclass
class DocumentEntityRecord:
    doc_id: str
    title: str
    source_path: str
    metadata: Dict[str, str]
    entities: List[Entity]


class EntityExtractor:
    """
    Extracts structured entities from BRD-style documents.

    This first version focuses on:
    - source tables
    - target tables
    - source schema
    - target schema
    - classification
    - version
    - status

    Extend this later for SLA, owners, domains, source systems, etc.
    """

    SOURCE_TABLE_PATTERNS = [
        r"primary source table name\(s\)\s*:\s*(.+)",
        r"source table name\(s\)\s*:\s*(.+)",
        r"source table\(s\)\s*:\s*(.+)",
        r"source tables?\s*:\s*(.+)",
        r"source table\s*:\s*(.+)",
    ]

    TARGET_TABLE_PATTERNS = [
        r"primary target table name\(s\)\s*:\s*(.+)",
        r"target table name\(s\)\s*:\s*(.+)",
        r"target table\(s\)\s*:\s*(.+)",
        r"target tables?\s*:\s*(.+)",
        r"target table\s*:\s*(.+)",
    ]

    CLASSIFICATION_PATTERNS = [
        r"classification\s*:\s*(.+)",
    ]

    VERSION_PATTERNS = [
        r"version\s*:\s*([A-Za-z0-9.\-_ ]+)",
    ]

    STATUS_PATTERNS = [
        r"status\s*:\s*([A-Za-z0-9.\-_ ]+)",
    ]

    def extract_from_documents(self, docs: List[RawDocument]) -> List[DocumentEntityRecord]:
        return [self.extract_from_document(doc) for doc in docs]

    def extract_from_document(self, doc: RawDocument) -> DocumentEntityRecord:
        text = doc.text or ""
        entities: List[Entity] = []

        entities.extend(self._extract_table_entities(text, subtype="source_table", patterns=self.SOURCE_TABLE_PATTERNS))
        entities.extend(self._extract_table_entities(text, subtype="target_table", patterns=self.TARGET_TABLE_PATTERNS))
        entities.extend(self._extract_scalar_entities(text, entity_type="classification", subtype="classification", patterns=self.CLASSIFICATION_PATTERNS))
        entities.extend(self._extract_scalar_entities(text, entity_type="version", subtype="version", patterns=self.VERSION_PATTERNS))
        entities.extend(self._extract_scalar_entities(text, entity_type="status", subtype="status", patterns=self.STATUS_PATTERNS))

        entities = self._dedupe_entities(entities)

        return DocumentEntityRecord(
            doc_id=doc.doc_id,
            title=doc.title,
            source_path=doc.source_path,
            metadata=doc.metadata,
            entities=entities,
        )

    def _extract_table_entities(self, text: str, subtype: str, patterns: List[str]) -> List[Entity]:
        matches: List[Entity] = []

        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw_value = m.group(1).strip()
                evidence = m.group(0).strip()

                for table_name in self._split_values(raw_value):
                    if self._looks_like_identifier(table_name):
                        matches.append(
                            Entity(
                                type="table",
                                subtype=subtype,
                                value=table_name,
                                evidence=evidence,
                            )
                        )

        return matches

    def _extract_scalar_entities(
        self,
        text: str,
        entity_type: str,
        subtype: str,
        patterns: List[str],
    ) -> List[Entity]:
        matches: List[Entity] = []

        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = self._clean_scalar(m.group(1))
                evidence = m.group(0).strip()

                if value:
                    matches.append(
                        Entity(
                            type=entity_type,
                            subtype=subtype,
                            value=value,
                            evidence=evidence,
                        )
                    )

        return matches

    def _split_values(self, raw_value: str) -> List[str]:
        """
        Split things like:
        reservation_inventory_source, maintenance_block_feed, partner_allotment_feed
        """
        raw_value = raw_value.strip().strip(".")
        if not raw_value:
            return []

        parts = re.split(r"[,\n;/|]+", raw_value)
        cleaned = [self._clean_identifier(p) for p in parts]
        return [x for x in cleaned if x]

    def _clean_identifier(self, value: str) -> str:
        value = value.strip()
        value = re.sub(r"^[\-\u2022•\*]+", "", value).strip()
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _clean_scalar(self, value: str) -> str:
        value = value.strip()
        value = value.split("\n")[0].strip()
        value = re.sub(r"\s+", " ", value)
        return value

    def _looks_like_identifier(self, value: str) -> bool:
        """
        Heuristic for table-like identifiers:
        reservation_inventory_source
        daily_booking_revenue
        hotel_profile_current
        """
        if not value:
            return False

        if len(value) < 3:
            return False

        # Typical table-style identifiers: snake_case, alnum, underscore
        if re.fullmatch(r"[A-Za-z0-9_]+", value):
            return True

        return False

    def _dedupe_entities(self, entities: List[Entity]) -> List[Entity]:
        seen = set()
        deduped: List[Entity] = []

        for entity in entities:
            key = (entity.type.lower(), entity.subtype.lower(), entity.value.lower())
            if key not in seen:
                seen.add(key)
                deduped.append(entity)

        return deduped