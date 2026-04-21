from __future__ import annotations

from typing import Dict, List, Optional, Set


ONTOLOGY: Dict[str, List[str]] = {
    "table": [
        "source_table",
        "target_table",
        "staging_table",
        "fact_table",
        "dimension_table",
        "history_table",
        "current_table",
    ],
    "schema": [
        "source_schema",
        "target_schema",
    ],
    "classification": [
        "classification",
    ],
    "version": [
        "version",
    ],
    "status": [
        "status",
    ],
}

CONCEPT_ALIASES: Dict[str, Set[str]] = {
    "table": {
        "table", "tables", "table name", "table names",
        "source table", "source tables",
        "target table", "target tables",
        "fact table", "fact tables",
        "dimension table", "dimension tables",
        "staging table", "staging tables",
    },
    "schema": {
        "schema", "schemas",
        "source schema", "source schemas",
        "target schema", "target schemas",
    },
    "classification": {
        "classification", "security classification", "confidentiality",
    },
    "version": {
        "version", "versions",
    },
    "status": {
        "status", "statuses",
    },
}


def expand_concept(concept: str) -> List[str]:
    concept = concept.lower().strip()
    return ONTOLOGY.get(concept, [concept])


def infer_requested_concept(question: str) -> Optional[str]:
    q = question.lower().strip()

    for concept, aliases in CONCEPT_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return concept

    return None