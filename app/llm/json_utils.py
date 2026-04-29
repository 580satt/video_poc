from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_json_loose(text: str) -> Any:
    """Strip markdown fences and parse JSON (shared by Gemini + WaveSpeed scene JSON)."""
    t = text.strip()
    m = _JSON_FENCE.search(t)
    if m:
        t = m.group(1).strip()
    return json.loads(t)
