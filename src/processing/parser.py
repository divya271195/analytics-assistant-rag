from __future__ import annotations

import re
from typing import List


class SimpleRequirementParser:
    def extract_bullets(self, text: str) -> List[str]:
        return [line.strip("- ").strip() for line in text.splitlines() if re.match(r"^\s*[-*]", line)]
