from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _azure_configured(s: Settings) -> bool:
    return bool(
        s.azure_openai_endpoint.strip()
        and s.azure_openai_api_key.strip()
        and s.azure_openai_embedding_deployment.strip()
    )


def embeddings_configured(settings: Optional[Settings] = None) -> bool:
    """True if either Azure OpenAI (deployment) or direct OpenAI is configured for embeddings."""
    s = settings or get_settings()
    if _azure_configured(s):
        return True
    return bool(s.openai_api_key.strip())


def _embed_azure(s: Settings, texts: list[str]) -> list[list[float]]:
    try:
        from openai import AzureOpenAI
    except ImportError as e:
        logger.warning("openai package required for Azure embeddings: %s", e)
        return []
    client = AzureOpenAI(
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
        azure_endpoint=s.azure_openai_endpoint.rstrip("/"),
    )
    out: list[list[float]] = []
    size = 16
    for i in range(0, len(texts), size):
        batch = [t if len(t) <= 30000 else t[:30000] for t in texts[i : i + size]]
        r = client.embeddings.create(
            input=batch,
            model=s.azure_openai_embedding_deployment,
        )
        for d in r.data:
            out.append(list(d.embedding))
    return out


def _embed_openai_direct(s: Settings, texts: list[str]) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as e:
        logger.warning("openai not installed: %s", e)
        return []
    client = OpenAI(api_key=s.openai_api_key)
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


def embed_texts(texts: list[str], settings: Optional[Settings] = None) -> list[list[float]]:
    """Embedding vectors: Azure OpenAI (ada-002 deployment) if configured, else direct OpenAI."""
    s = settings or get_settings()
    if not texts:
        return []
    if _azure_configured(s):
        return _embed_azure(s, texts)
    if s.openai_api_key.strip():
        return _embed_openai_direct(s, texts)
    logger.warning("No embedding provider: set Azure OpenAI (endpoint+key+deployment) or OPENAI_API_KEY.")
    return []


def embed_query(text: str, settings: Optional[Settings] = None) -> list[float]:
    vecs = embed_texts([text], settings=settings)
    return vecs[0] if vecs else []
