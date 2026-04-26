from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal, Optional

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.db.mongo import RunStore, get_run_store
from app.graph import nodes as N
from app.graph.state import StorybookState
from app.llm.gemini import GeminiClient
from app.rag.chroma_narrative import query_narrative_rag

logger = logging.getLogger(__name__)


def route_entry(state: StorybookState) -> str:
    sf = state.get("start_from") or "full"
    if sf in ("full", "default", "", None):
        return "build_template"
    if sf == "template":
        return "build_template"
    if sf == "script":
        return "write_script"
    if sf == "scenes":
        return "write_scenes"
    if sf == "images":
        return "generate_images"
    return "build_template"


def route_script(state: StorybookState) -> Literal["ok", "retry", "best"]:
    s = get_settings()
    r = state.get("script_last_rating")
    if r is not None and r >= s.reviewer_min_rating:
        return "ok"
    if (state.get("story_draft_count") or 0) >= N.max_script():
        return "best"
    return "retry"


def route_scenes(state: StorybookState) -> Literal["ok", "retry", "best"]:
    s = get_settings()
    r = state.get("scenes_last_rating")
    if r is not None and r >= s.reviewer_min_rating:
        return "ok"
    if (state.get("scenes_draft_count") or 0) >= N.max_scenes():
        return "best"
    return "retry"


def build_app_graph() -> StateGraph:
    g = StateGraph(StorybookState)
    g.add_node("build_template", N.node_build_template)
    g.add_node("write_script", N.node_write_script)
    g.add_node("review_script", N.node_review_script)
    g.add_node("save_story_approved", N.node_save_story_approved)
    g.add_node("write_scenes", N.node_write_scenes)
    g.add_node("review_scenes", N.node_review_scenes)
    g.add_node("save_scenes_approved", N.node_save_scenes_approved)
    g.add_node("generate_images", N.node_generate_images)
    g.add_node("complete", N.node_complete)
    g.add_node("fail_script", N.node_fail_script)
    g.add_node("fail_scenes", N.node_fail_scenes)
    g.add_node("pick_best_script", N.node_pick_best_script)
    g.add_node("pick_best_scenes", N.node_pick_best_scenes)

    g.add_conditional_edges(
        START,
        route_entry,
        {
            "build_template": "build_template",
            "write_script": "write_script",
            "write_scenes": "write_scenes",
            "generate_images": "generate_images",
        },
    )
    g.add_edge("build_template", "write_script")
    g.add_edge("write_script", "review_script")
    g.add_conditional_edges(
        "review_script",
        route_script,
        {
            "ok": "save_story_approved",
            "retry": "write_script",
            "best": "pick_best_script",
        },
    )
    g.add_edge("pick_best_script", "save_story_approved")
    g.add_edge("save_story_approved", "write_scenes")
    g.add_edge("write_scenes", "review_scenes")
    g.add_conditional_edges(
        "review_scenes",
        route_scenes,
        {
            "ok": "save_scenes_approved",
            "retry": "write_scenes",
            "best": "pick_best_scenes",
        },
    )
    g.add_edge("pick_best_scenes", "save_scenes_approved")
    g.add_edge("save_scenes_approved", "generate_images")
    g.add_edge("generate_images", "complete")
    g.add_edge("complete", END)
    g.add_edge("fail_script", END)
    g.add_edge("fail_scenes", END)
    return g.compile()


def _base_state() -> dict[str, Any]:
    return {
        "errors": [],
        "error_detail": None,
        "pending_error": None,
        "image_pass_full": True,
        "rag_context_narrative": "",
        "script_review_history": [],
        "scenes_review_history": [],
        "review_trace": [],
        "script_chosen": "",
        "scenes_chosen": "",
    }


async def run_full_pipeline(
    *,
    run_id: str,
    raw_user_input: dict[str, Any],
    target_runtime_seconds: float,
    target_runtime_max_seconds: float,
    product_image: bytes | None,
    product_image_mime: str | None,
    settings: Optional[Settings] = None,
    store: Optional[RunStore] = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    st = store or get_run_store()
    gem = GeminiClient(s)
    N.bind_gemini(gem, s)
    N.bind_mongo(st)
    graph = build_app_graph()
    init: dict[str, Any] = {
        "run_id": run_id,
        "raw_user_input": raw_user_input,
        "target_runtime_seconds": target_runtime_seconds,
        "target_runtime_max_seconds": target_runtime_max_seconds,
        "product_image": product_image,
        "product_image_mime": product_image_mime,
        "start_from": "full",
        **_base_state(),
    }
    await st.update_run(
        run_id,
        {
            "status": "running",
            "raw_input": raw_user_input,
            "target_runtime_seconds": target_runtime_seconds,
            "target_runtime_max_seconds": target_runtime_max_seconds,
            "error_detail": None,
        },
    )
    out = await graph.ainvoke(init)
    return out


async def run_regenerate(
    *,
    run_id: str,
    from_step: Literal["script", "scenes", "images"],
    settings: Optional[Settings] = None,
    store: Optional[RunStore] = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    st = store or get_run_store()
    doc = await st.get_run(run_id)
    if not doc:
        return {"status": "failed", "error_detail": "Run not found", "run_id": run_id}
    gem = GeminiClient(s)
    N.bind_gemini(gem, s)
    N.bind_mongo(st)
    graph = build_app_graph()
    tr = float(doc.get("target_runtime_seconds", s.target_runtime_seconds))
    tmax = float(doc.get("target_runtime_max_seconds", s.target_runtime_max_seconds))
    raw = doc.get("raw_input") or {}
    pt = doc.get("product_template")
    story = doc.get("story") or ""
    sc = doc.get("scenes")
    init: dict[str, Any] = {
        **_base_state(),
        "run_id": run_id,
        "raw_user_input": raw,
        "target_runtime_seconds": tr,
        "target_runtime_max_seconds": tmax,
        "product_image": None,
        "product_image_mime": None,
        "product_template": pt,
        "story": story,
        "scenes_payload": sc,
    }
    if from_step == "script":
        if not pt:
            return {"status": "failed", "error_detail": "No product template stored for this run", "run_id": run_id}
        init["start_from"] = "script"
        init["script_reviewer_feedback"] = ""
        init["scenes_reviewer_feedback"] = ""
        init["images_reviewer_feedback"] = ""
        init["story_approved"] = False
        init["scenes_approved"] = False
        init["scenes_payload"] = None
        init["story"] = ""
        init["images_approved"] = False
        init["images"] = []
        init["image_pass_full"] = True
        init["story_draft_count"] = 0
        init["scenes_draft_count"] = 0
        init["images_draft_count"] = 0
        init["script_review_history"] = []
        init["scenes_review_history"] = []
        init["review_trace"] = []
        init["script_chosen"] = ""
        init["scenes_chosen"] = ""
    elif from_step == "scenes":
        if not pt or not story:
            return {"status": "failed", "error_detail": "Need approved story in DB", "run_id": run_id}
        init["start_from"] = "scenes"
        init["scenes_approved"] = False
        init["scenes_payload"] = None
        init["images_approved"] = False
        init["images"] = []
        init["image_pass_full"] = True
        init["scenes_draft_count"] = 0
        init["images_draft_count"] = 0
        init["scenes_reviewer_feedback"] = ""
        init["images_reviewer_feedback"] = ""
        init["scenes_review_history"] = []
        init["review_trace"] = doc.get("review_trace") or []  # keep script trace if any
        init["scenes_chosen"] = ""
    elif from_step == "images":
        if not sc or not doc.get("scenes_approved"):
            return {"status": "failed", "error_detail": "Need approved scenes in DB", "run_id": run_id}
        init["start_from"] = "images"
        init["scenes_approved"] = True
        init["scenes_payload"] = sc
        init["images"] = []
        init["image_pass_full"] = True
        init["images_draft_count"] = 0
    if from_step in ("script", "scenes") and isinstance(pt, dict):
        try:
            init["rag_context_narrative"] = await asyncio.to_thread(
                query_narrative_rag, pt, raw, s
            )
        except Exception as e:
            logger.warning("RAG for regenerate failed: %s", e)
            init["rag_context_narrative"] = ""
    await st.update_run(run_id, {"status": "running", "error_detail": None})
    out = await graph.ainvoke(init)
    return out


def new_run_id() -> str:
    return str(uuid.uuid4())
