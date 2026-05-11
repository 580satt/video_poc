from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import chromadb

from app.config import Settings, get_settings
from app.rag.embeddings import embed_query, embeddings_configured

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
    cs = str(raw.get("context_sources") or "").strip().lower()
    if cs not in ("both", "rag", "brief", "none"):
        cs = "both"
    include_brand_snippet = cs in ("both", "brief")
    bp = str(raw.get("brand_psychology_context") or "").strip()
    if bp and include_brand_snippet:
        parts.append(f"brand_psychology_context: {bp[:2500]}")
    return " | ".join(parts) or json.dumps(product_template, ensure_ascii=False)[:4000]


def _narrative_trace_base(settings: Settings) -> dict[str, Any]:
    coll = (settings.chroma_narrative_collection or "").strip()
    k = max(1, min(settings.rag_top_k, 50))
    persist_set = bool(str(settings.chroma_persist_path or "").strip())
    return {
        "collection": coll or None,
        "top_k": k,
        "rag_enabled": bool(settings.rag_enabled),
        "chroma_persist_path_set": persist_set,
        "skipped_reason": None,
        "skipped_hint": None,
        "query_preview": None,
        "hits": [],
    }


def _json_safe_metadata(meta: Any) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                out[str(k)] = v
            except (TypeError, ValueError):
                out[str(k)] = str(v)[:500]
        else:
            out[str(k)] = str(v)[:500]
    return out


def query_narrative_rag_with_trace(
    product_template: dict[str, Any],
    raw_user_input: dict[str, Any] | None,
    settings: Optional[Settings] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Narrative RAG context string for prompts plus a JSON-serializable trace for UI / logs.
    """
    s = settings or get_settings()
    trace = _narrative_trace_base(s)
    if not s.rag_enabled:
        trace["skipped_reason"] = "rag_disabled"
        trace["skipped_hint"] = "Set RAG_ENABLED=true (default) when you want narrative retrieval."
        return "", trace
    if not str(s.chroma_persist_path or "").strip():
        trace["skipped_reason"] = "chroma_persist_path_missing"
        trace["skipped_hint"] = (
            "CHROMA_PERSIST_PATH is empty. Set it in .env to your Chroma persist directory "
            "(the folder that contains chroma.sqlite3), then restart the API."
        )
        return "", trace
    if not (s.chroma_narrative_collection or "").strip():
        trace["skipped_reason"] = "narrative_collection_missing"
        trace["skipped_hint"] = "Set CHROMA_NARRATIVE_COLLECTION to the collection name on disk."
        return "", trace
    if not embeddings_configured(s):
        logger.warning("RAG skipped: configure Azure OpenAI (endpoint+key+deployment) or OPENAI_API_KEY for embeddings.")
        trace["skipped_reason"] = "embeddings_not_configured"
        trace["skipped_hint"] = (
            "Configure Azure OpenAI embedding deployment or OPENAI_API_KEY so query embeddings match the collection."
        )
        return "", trace
    coll_name = s.chroma_narrative_collection.strip()
    try:
        client = _get_client(s.chroma_persist_path)
        coll = client.get_collection(name=coll_name)
    except Exception as e:
        logger.warning("Chroma narrative collection not available: %s", e)
        trace["skipped_reason"] = f"collection_unavailable: {e}"
        trace["skipped_hint"] = (
            "Could not open Chroma at CHROMA_PERSIST_PATH or get collection "
            f"{coll_name!r}. Check the path and that the collection name matches this DB."
        )
        return "", trace
    q = template_to_query_string(product_template, raw_user_input)
    if not q.strip():
        trace["skipped_reason"] = "empty_query"
        trace["skipped_hint"] = "Product template fields used for the RAG query were empty."
        return "", trace
    trace["query_preview"] = q[:800] + ("…" if len(q) > 800 else "")
    try:
        emb = embed_query(q, settings=s)
    except Exception as e:
        logger.warning("Embedding failed (narrative RAG): %s", e)
        trace["skipped_reason"] = f"embedding_failed: {e}"
        trace["skipped_hint"] = "Embedding request failed; check API keys and embedding deployment."
        return "", trace
    if not emb or len(emb) != s.openai_embedding_dimensions:
        logger.warning("Embedding dim mismatch: got %s expected %s", len(emb or []), s.openai_embedding_dimensions)
        trace["skipped_reason"] = "embedding_dimension_mismatch"
        trace["skipped_hint"] = (
            f"Embedding length was {len(emb or [])} but OPENAI_EMBEDDING_DIMENSIONS expects "
            f"{s.openai_embedding_dimensions}. Match the model that built the Chroma collection."
        )
        return "", trace
    n_results = trace["top_k"]
    try:
        # Chroma query() does not accept "ids" in include (unlike get()); ids are still
        # returned on the result dict when present.
        res = coll.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
    except Exception as e:
        logger.warning("Chroma query failed: %s", e)
        trace["skipped_reason"] = f"chroma_query_failed: {e}"
        trace["skipped_hint"] = (
            "Chroma query() failed: verify include= fields for your chromadb version, "
            "persist path, collection name, and that embedding dimension matches the index."
        )
        return "", trace
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    if not docs:
        trace["skipped_reason"] = "no_results"
        trace["skipped_hint"] = "Query succeeded but Chroma returned no documents for this embedding."
        return "", trace
    hits: list[dict[str, Any]] = []
    parts = []
    for i, d in enumerate(docs):
        t = d if isinstance(d, str) else str(d)
        t = t.strip()
        raw_len = len(t)
        dist = None
        if dists is not None and i < len(dists):
            try:
                dist = float(dists[i])
            except (TypeError, ValueError):
                dist = None
        meta_i = metas[i] if metas is not None and i < len(metas) else None
        id_i = ids[i] if ids is not None and i < len(ids) else None
        logger.info(
            "[rag] narrative collection=%s rank=%s/%s distance=%s id=%s",
            coll_name,
            i + 1,
            len(docs),
            dist,
            id_i,
        )
        hits.append(
            {
                "rank": i + 1,
                "id": id_i,
                "distance": dist,
                "metadata": _json_safe_metadata(meta_i),
                "document_chars": raw_len,
                "document": t[:6000] + ("…[truncated]" if raw_len > 6000 else ""),
            }
        )
        inj = t[:6000] + ("…[truncated]" if raw_len > 6000 else "")
        parts.append(f"--- reference {i + 1} ---\n{inj}")
    trace["hits"] = hits
    text = (
        "Reference only — similar library entries (do not copy verbatim; match tone, category, and visual intent):\n\n"
        + "\n\n".join(parts)
    )
    return text, trace


def narrative_trace_failed(settings: Settings, reason: str, *, hint: str | None = None) -> dict[str, Any]:
    t = _narrative_trace_base(settings)
    t["skipped_reason"] = reason
    if hint:
        t["skipped_hint"] = hint
    return t


def narrative_trace_user_disabled(settings: Settings, mode: str) -> dict[str, Any]:
    """RAG not run because the run's context_sources excludes retrieval."""
    t = _narrative_trace_base(settings)
    t["skipped_reason"] = "user_disabled"
    m = (mode or "").strip().lower()
    if m == "brief":
        t["skipped_hint"] = (
            "You chose **brand brief only** for this run: Chroma narrative RAG was skipped on purpose. "
            "Script/scene prompts use the template and long brief only."
        )
    elif m == "none":
        t["skipped_hint"] = (
            "You chose **neither** extra source: RAG and the long brand brief are omitted from script/scene prompts."
        )
    else:
        t["skipped_hint"] = "Narrative RAG was turned off for this run (context sources selection)."
    return t


def query_narrative_rag(
    product_template: dict[str, Any],
    raw_user_input: dict[str, Any] | None,
    settings: Optional[Settings] = None,
) -> str:
    """
    Return a text block to inject into script/scene prompts, or empty string.
    """
    text, _ = query_narrative_rag_with_trace(product_template, raw_user_input, settings)
    return text


def get_or_create_review_collection(client: Any, name: str) -> Any:
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "source": "storybook_reviewer"},
        )


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
    if not s.chroma_persist_path or not s.chroma_review_collection.strip() or not embeddings_configured(s):
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
    if len(emb) != s.openai_embedding_dimensions:
        logger.warning(
            "Review memory query skipped: embedding length %s != OPENAI_EMBEDDING_DIMENSIONS %s",
            len(emb),
            s.openai_embedding_dimensions,
        )
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
    if not s.chroma_persist_path or not s.chroma_review_collection.strip() or not embeddings_configured(s):
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
    if not emb or len(emb) != s.openai_embedding_dimensions:
        logger.warning(
            "Review memory upsert skipped: embedding length %s != OPENAI_EMBEDDING_DIMENSIONS %s",
            len(emb or []),
            s.openai_embedding_dimensions,
        )
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
