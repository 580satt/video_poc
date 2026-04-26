from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str], settings: Optional[Settings] = None) -> list[list[float]]:
    """Return embedding vectors; uses OpenAI (1536-dim for text-embedding-3-small by default)."""
    s = settings or get_settings()
    if not s.openai_api_key or not texts:
        return []
    try:
        from openai import OpenAI
    except ImportError as e:
        logger.warning("openai not installed: %s", e)
        return []

    client = OpenAI(api_key=s.openai_api_key)
    # Batch to respect token limits
    out: list[list[float]] = []
    size = 16
    for i in range(0, len(texts), size):
        batch = [t if len(t) <= 30000 else t[:30000] for t in texts[i : i + size]]
        kwargs: dict = {"model": s.openai_embedding_model, "input": batch}
        if s.openai_embedding_model.startswith("text-embedding-3"):
            kwargs["dimensions"] = s.openai_embedding_dimensions
        r = client.embeddings.create(**kwargs)
        for d in r.data:
            out.append(list(d.embedding))
    return out


def embed_query(text: str, settings: Optional[Settings] = None) -> list[float]:
    vecs = embed_texts([text], settings=settings)
    return vecs[0] if vecs else []
