from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import chromadb

from app.config import Settings, get_settings
from app.rag.embeddings import embed_query

logger = logging.getLogger(__name__)

_client: dict[str, Any] = {}


def _get_client(persist_path: str) -> Any:
    if persist_path in _client:
        return _client[persist_path]
    p = Path(persist_path)
    if not p.is_dir() and not p.exists():
        logger.warning("Chroma path does not exist: %s", persist_path)
    c = chromadb.PersistentClient(path=str(p))
    _client[persist_path] = c
    return c


def template_to_query_string(product_template: dict[str, Any], raw_user_input: dict[str, Any] | None) -> str:
    """Deterministic string for embedding / similarity to existing documents."""
    raw = raw_user_input or {}
    order = [
        "brand",
        "product_name",
        "goal",
        "target_audience",
        "tone",
        "visual_anchors",
        "product_image_insights",
    ]
    parts: list[str] = []
    for k in order:
        v = (product_template or {}).get(k) or raw.get(k)
        if v:
            parts.append(f"{k}: {v}" if not isinstance(v, (dict, list)) else f"{k}: {json.dumps(v, ensure_ascii=False)[:2000]}")
    ex = (product_template or {}).get("extra")
    if isinstance(ex, dict) and ex:
        parts.append(f"extra: {json.dumps(ex, ensure_ascii=False)[:1500]}")
    return " | ".join(parts) or json.dumps(product_template, ensure_ascii=False)[:4000]


def query_narrative_rag(
    product_template: dict[str, Any],
    raw_user_input: dict[str, Any] | None,
    settings: Optional[Settings] = None,
) -> str:
    """
    Return a text block to inject into script/scene prompts, or empty string.
    """
    s = settings or get_settings()
    if not s.rag_enabled or not s.chroma_persist_path or not s.chroma_narrative_collection.strip():
        return ""
    if not s.openai_api_key:
        logger.warning("RAG skipped: openai_api_key not set (needed to match vector dimensions).")
        return ""
    try:
        client = _get_client(s.chroma_persist_path)
        coll = client.get_collection(name=s.chroma_narrative_collection.strip())
    except Exception as e:
        logger.warning("Chroma narrative collection not available: %s", e)
        return ""
    q = template_to_query_string(product_template, raw_user_input)
    if not q.strip():
        return ""
    try:
        emb = embed_query(q, settings=s)
    except Exception as e:
        logger.warning("Embedding failed (narrative RAG): %s", e)
        return ""
    if not emb or len(emb) != s.openai_embedding_dimensions:
        logger.warning("Embedding dim mismatch: got %s expected %s", len(emb or []), s.openai_embedding_dimensions)
        return ""
    try:
        res = coll.query(
            query_embeddings=[emb],
            n_results=max(1, min(s.rag_top_k, 50)),
            include=["documents", "distances", "metadatas"],
        )
    except Exception as e:
        logger.warning("Chroma query failed: %s", e)
        return ""
    docs = (res.get("documents") or [[]])[0]
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs):
        t = d if isinstance(d, str) else str(d)
        t = t.strip()
        if len(t) > 6000:
            t = t[:6000] + "…[truncated]"
        parts.append(f"--- reference {i + 1} ---\n{t}")
    return "Reference only — similar library entries (do not copy verbatim; match tone, category, and visual intent):\n\n" + "\n\n".join(
        parts
    )


def get_or_create_review_collection(client: Any, name: str) -> Any:
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name, metadata={"source": "storybook_reviewer"})


def review_doc_id(run_id: str, step: str, draft: int) -> str:
    return f"{run_id}_{step}_d{draft}"


def review_document_body(
    run_id: str,
    step: str,
    draft: int,
    output_text: str,
    product_snippet: str,
    feedback: str,
    rating: int,
) -> str:
    o = output_text if len(output_text) <= 12000 else output_text[:12000] + "…[truncated]"
    return (
        f"step={step} run_id={run_id} draft={draft} rating={rating}\n"
        f"product_context: {product_snippet[:2000]}\n"
        f"output:\n{o}\n"
        f"reviewer_feedback: {feedback[:4000]}"
    )


def query_review_memory(
    template_summary: str,
    current_output: str,
    last_feedback: str,
    step: str,
    settings: Optional[Settings] = None,
) -> str:
    """High-rated similar past cases for reviewer context."""
    s = settings or get_settings()
    if not s.chroma_persist_path or not s.chroma_review_collection.strip() or not s.openai_api_key:
        return ""
    try:
        client = _get_client(s.chroma_persist_path)
        coll = get_or_create_review_collection(client, s.chroma_review_collection.strip())
    except Exception as e:
        logger.warning("Chroma review collection open failed: %s", e)
        return ""
    q = f"{template_summary}\n{step}\n{current_output[:8000]}\n{last_feedback[:2000]}"
    try:
        emb = embed_query(q, settings=s)
    except Exception as e:
        logger.warning("Review memory embed failed: %s", e)
        return ""
    if not emb:
        return ""
    try:
        res = coll.query(
            query_embeddings=[emb],
            n_results=max(1, s.review_memory_top_k),
            include=["documents", "metadatas", "distances"],
            # Prefer cases that ended well
            where={"rating": {"$gte": 3}},
        )
    except Exception as e1:
        try:
            res = coll.query(
                query_embeddings=[emb],
                n_results=max(1, s.review_memory_top_k),
                include=["documents", "metadatas"],
            )
        except Exception as e2:
            logger.warning("Review memory query failed: %s / %s", e1, e2)
            return ""
    docs = (res.get("documents") or [[]])[0]
    if not docs:
        return ""
    parts = [f"--- {i + 1} ---\n{(d if isinstance(d, str) else str(d))[:3000]}" for i, d in enumerate(docs)]
    return "Similar past review cases (for calibration; do not copy blindly):\n" + "\n\n".join(parts)


def upsert_review_memory(
    run_id: str,
    step: str,
    draft: int,
    output_text: str,
    product_template: dict[str, Any],
    feedback: str,
    rating: int,
    settings: Optional[Settings] = None,
) -> None:
    s = settings or get_settings()
    if not s.chroma_persist_path or not s.chroma_review_collection.strip() or not s.openai_api_key:
        return
    try:
        client = _get_client(s.chroma_persist_path)
        coll = get_or_create_review_collection(client, s.chroma_review_collection.strip())
    except Exception as e:
        logger.warning("Chroma review upsert open failed: %s", e)
        return
    body = review_document_body(
        run_id,
        step,
        draft,
        output_text,
        json.dumps(
            {k: product_template.get(k) for k in ("brand", "product_name", "goal", "target_audience", "tone")},
            ensure_ascii=False,
        )[:2000],
        feedback,
        rating,
    )
    try:
        emb = embed_query(body, settings=s)
    except Exception as e:
        logger.warning("Review memory embed (upsert) failed: %s", e)
        return
    if not emb:
        return
    rid = review_doc_id(run_id, step, draft)
    try:
        try:
            coll.delete(ids=[rid])
        except Exception:
            pass
        coll.add(
            ids=[rid],
            embeddings=[emb],
            documents=[body],
            metadatas=[
                {
                    "run_id": run_id,
                    "step": step,
                    "draft": int(draft),
                    "rating": int(rating),
                }
            ],
        )
    except Exception as e:
        logger.warning("Chroma review upsert failed: %s", e)
