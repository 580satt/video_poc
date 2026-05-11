from __future__ import annotations

from typing import Any

_VALID = frozenset({"both", "rag", "brief", "none"})


def normalize_context_sources(raw: dict[str, Any] | None) -> str:
    raw = raw or {}
    cs = str(raw.get("context_sources") or "").strip().lower()
    return cs if cs in _VALID else "both"


def context_source_flags(raw: dict[str, Any] | None) -> tuple[bool, bool]:
    """Return (use_narrative_rag, use_brand_psychology_brief) for prompts and retrieval."""
    cs = normalize_context_sources(raw)
    if cs == "rag":
        return True, False
    if cs == "brief":
        return False, True
    if cs == "none":
        return False, False
    return True, True
