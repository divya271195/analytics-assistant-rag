from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_environment() -> None:
    load_dotenv()
