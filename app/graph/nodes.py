from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.db.mongo import RunStore

from app.config import Settings, get_settings
from app.storage.s3_images import put_scene_image_async
from app.llm.gemini import GeminiClient
from app.models.schemas import (
    ProductTemplate,
    ReviewResult,
    ScenesPayload,
    SceneItem,
    validate_scenes_against_max,
)
from app.graph.progress import step_status_patch
from app.graph.state import StorybookState
from app.rag.chroma_narrative import (
    query_narrative_rag,
    query_review_memory,
    upsert_review_memory,
)

logger = logging.getLogger(__name__)

_GEM: Optional[GeminiClient] = None
_S: Optional[Settings] = None
_MONGO: Optional["RunStore"] = None


def bind_gemini(gem: GeminiClient, st: Settings) -> None:
    global _GEM, _S
    _GEM = gem
    _S = st


def bind_mongo(store: "RunStore") -> None:
    global _MONGO
    _MONGO = store


def _store() -> "RunStore":
    if _MONGO is None:
        raise RuntimeError("Mongo not bound")
    return _MONGO


def _gem() -> GeminiClient:
    if _GEM is None:
        raise RuntimeError("Gemini not bound")
    return _GEM


def _settings() -> Settings:
    if _S is None:
        from app.config import get_settings

        return get_settings()
    return _S


async def _call_text(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)


def _parse_review_result(text: str) -> ReviewResult:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            return ReviewResult(rating=0, feedback="Could not parse reviewer JSON.")
    if not isinstance(data, dict):
        return ReviewResult(rating=0, feedback="Reviewer returned non-object JSON.")
    if "rating" not in data and "approved" in data:
        data = {
            **data,
            "rating": 5 if data.get("approved") else 2,
        }
    if "rating" not in data:
        data["rating"] = 0
    try:
        r = ReviewResult.model_validate(data)
        r = r.model_copy(
            update={"rating": max(0, min(5, r.rating))}
        )  # clamp
        return r
    except Exception:
        return ReviewResult(rating=0, feedback="Invalid reviewer JSON schema.")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_step_output(step: str, draft: int, rating: int | None, feedback: str, body: str) -> None:
    prev = 8000
    t = body if len(body) <= prev else body[:prev] + f"\n…[truncated, total {len(body)} chars]"
    logger.info(
        "[review] step=%s draft=%s rating=%s feedback=%s\n%s",
        step,
        draft,
        rating,
        (feedback or "")[:2000],
        t,
    )


def _template_prompt(raw: dict[str, Any], tr: float, tmax: float) -> str:
    return f"""You are a brand strategist. Return ONLY a single JSON object (no markdown) with keys:
  brand, product_name, goal, target_audience, tone (strings),
  primary_goal: "advertise_product",
  target_runtime_sec: number,
  target_runtime_max_sec: number,
  goals: array of strings including "advertise_product",
  visual_anchors, product_image_insights (strings),
  extra: object.

Time budget: design for ~{tr}s, must not exceed {tmax}s total story/scenes.

User / product context: {json.dumps(raw, ensure_ascii=False)}."""


def _pt_from_merged(data: dict[str, Any], tr: float, tmax: float) -> dict[str, Any]:
    data = dict(data)
    data.setdefault("target_runtime_sec", tr)
    data.setdefault("target_runtime_max_sec", tmax)
    data.setdefault("primary_goal", "advertise_product")
    data.setdefault("goals", ["advertise_product", "time_fit", "clarity"])
    return ProductTemplate.model_validate(
        {**data, "extra": data.get("extra") or {} if isinstance(data.get("extra"), dict) else {}}
    ).as_dict()


async def node_build_template(state: StorybookState) -> dict[str, Any]:
    tr, tmax = state["target_runtime_seconds"], state["target_runtime_max_seconds"]
    raw = dict(state.get("raw_user_input") or {})
    image_desc = ""
    if state.get("product_image") and state.get("product_image_mime"):
        image_desc = await _call_text(
            _gem().describe_image,
            state["product_image"],
            state["product_image_mime"] or "image/png",
            "List colors, pack shape, logo, materials, and how it should look in an ad. Be concise.",
        )
    data = await _call_text(
        _gem().generate_json,
        _template_prompt({**raw, "image_expert": image_desc}, tr, tmax)
        + (f" Image expert notes: {image_desc}" if image_desc else ""),
    )
    if not data:
        data = {k: v for k, v in raw.items() if k in ("brand", "product_name", "goal", "target_audience", "notes")}
    try:
        pt = _pt_from_merged(
            {**data, "product_image_insights": data.get("product_image_insights") or image_desc},
            tr,
            tmax,
        )
    except Exception:
        pt = _pt_from_merged(
            {
                "brand": str(raw.get("brand", "")),
                "product_name": str(raw.get("product_name", "")),
                "goal": str(raw.get("goal", "")),
                "target_audience": str(raw.get("target_audience", "")),
                "product_image_insights": image_desc,
            },
            tr,
            tmax,
        )
    rag = await _call_text(
        query_narrative_rag,
        pt,
        raw,
        _settings(),
    )
    if rag:
        logger.info("[rag] narrative context length=%s", len(rag))

    await _store().update_run(
        state["run_id"],
        {
            "product_template": pt,
            "status": "running",
            "rag_context_narrative": rag,
            **step_status_patch(
                {
                    "template": "complete",
                    "script": "in_progress",
                }
            ),
        },
    )
    return {"product_template": pt, "status": "running", "rag_context_narrative": rag}


async def node_write_script(state: StorybookState) -> dict[str, Any]:
    st = _settings()
    tr = state["target_runtime_seconds"]
    tmax = state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    n = (state.get("story_draft_count") or 0) + 1
    feedback = (state.get("script_reviewer_feedback") or "").strip()
    rag = (state.get("rag_context_narrative") or "").strip()
    rag_block = f"\n\n{rag}\n" if rag else ""
    p = f"""You write a very short, clear ad story for a storybook. This must be readable aloud within about {tmax} seconds (hard cap) and the PRIMARY goal is to advertise: {pt.get("product_name")} by {pt.get("brand", "")}. Tone: {pt.get("tone")}. Audience: {pt.get("target_audience")}. Goals: {pt.get("goal")}.

Time budget: {tr}-{tmax}s. No subplot that needs more time. Strong product and CTA focus.
Return ONLY the story text, no title line, 3-6 short paragraphs max.

Template JSON: {json.dumps(pt, ensure_ascii=False)}
{rag_block}
{f"Revise per reviewer feedback: {feedback}" if feedback else ""}"""
    text = await _call_text(_gem().generate_text, p, st.gemini_text_model, 0.7)
    story = (text or "").strip()
    await _store().update_run(
        state["run_id"],
        {
            "last_story_draft": story,
            "status": "running",
            **step_status_patch({"script": "in_progress"}),
        },
    )
    return {"story": story, "story_draft_count": n}


async def node_review_script(state: StorybookState) -> dict[str, Any]:
    st = _settings()
    tr, tmax = state["target_runtime_seconds"], state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    story = state.get("story") or ""
    draft_n = int(state.get("story_draft_count") or 0)
    prev_fb = (state.get("script_reviewer_feedback") or "").strip()
    summary = json.dumps(
        {k: pt.get(k) for k in ("brand", "product_name", "goal", "target_audience", "tone")},
        ensure_ascii=False,
    )
    mem = await _call_text(
        query_review_memory,
        summary,
        story,
        prev_fb,
        "script",
        st,
    )
    mem_block = f"\n{mem}\n" if mem.strip() else ""
    p = f"""You are a strict ad reviewer. You MUST score the output from 0 (unusable) to 5 (excellent). Return ONLY valid JSON: {{"rating": <0-5 integer>, "feedback": "brief actionable notes"}}
Rubric in order: (1) product advertising is strong, (2) the story can plausibly fit in {tmax} seconds when read at moderate pace, (3) creative, (4) matches template, (5) no contradictions.

Target runtime: {tr}-{tmax}s. Template: {json.dumps(pt, ensure_ascii=False)}
{mem_block}
STORY:
{story}"""
    text = await _call_text(_gem().review_json, p)
    r = _parse_review_result(text)
    _log_step_output("script", draft_n, r.rating, r.feedback, story)
    hist = list(state.get("script_review_history") or [])
    hist.append(
        {
            "draft": draft_n,
            "story": story,
            "rating": r.rating,
            "feedback": r.feedback,
        }
    )
    trace = list(state.get("review_trace") or [])
    trace.append(
        {
            "step": "script",
            "draft_index": draft_n,
            "rating": r.rating,
            "feedback": r.feedback,
            "full_output": story,
            "at": _now_iso(),
        }
    )
    await _call_text(
        upsert_review_memory,
        state["run_id"],
        "script",
        draft_n,
        story,
        pt,
        r.feedback,
        r.rating,
        st,
    )
    min_ok = r.rating >= st.reviewer_min_rating
    await _store().update_run(
        state["run_id"],
        {
            "script_review_history": hist,
            "review_trace": trace,
            "script_last_rating": r.rating,
            "last_script_review": {"rating": r.rating, "feedback": r.feedback, "draft": draft_n},
        },
    )
    return {
        "script_reviewer_feedback": r.feedback if not min_ok else "",
        "script_review_approved": min_ok,
        "script_last_rating": r.rating,
        "script_review_history": hist,
        "review_trace": trace,
        "script_chosen": "rating_pass" if min_ok else (state.get("script_chosen") or ""),
    }


async def node_save_story_approved(state: StorybookState) -> dict[str, Any]:
    rid = state["run_id"]
    chosen = state.get("script_chosen") or "rating_pass"
    await _store().update_run(
        rid,
        {
            "product_template": state.get("product_template"),
            "story": state.get("story"),
            "story_approved": True,
            "script_revisions": state.get("story_draft_count", 0),
            "script_review_history": state.get("script_review_history"),
            "review_trace": state.get("review_trace"),
            "script_chosen": chosen,
            "script_last_rating": state.get("script_last_rating"),
            **step_status_patch(
                {
                    "script": "complete",
                    "scenes": "in_progress",
                }
            ),
        },
    )
    return {
        "story_approved": True,
    }


async def node_pick_best_script(state: StorybookState) -> dict[str, Any]:
    hist: list[dict[str, Any]] = list(state.get("script_review_history") or [])
    if not hist:
        story = state.get("story") or ""
        logger.info("[script] best_of_run: no history, keeping current story")
        return {
            "story": story,
            "script_chosen": "best_of_run",
            "script_last_rating": state.get("script_last_rating", 0),
        }
    best = max(hist, key=lambda h: (h.get("rating", -1), h.get("draft", 0)))
    story = str(best.get("story") or state.get("story") or "")
    r = int(best.get("rating") or 0)
    logger.info(
        "[script] best_of_run: chose draft %s with rating %s (among %s drafts)",
        best.get("draft"),
        r,
        len(hist),
    )
    tr = list(state.get("review_trace") or [])
    tr.append(
        {
            "step": "script",
            "draft_index": int(best.get("draft") or 0),
            "rating": r,
            "feedback": f"Selected best-of-run: draft {best.get('draft')}",
            "full_output": story,
            "at": _now_iso(),
        }
    )
    await _store().update_run(
        state["run_id"],
        {
            "story": story,
            "script_chosen": "best_of_run",
            "script_last_rating": r,
            "review_trace": tr,
        },
    )
    return {
        "story": story,
        "script_chosen": "best_of_run",
        "script_last_rating": r,
        "review_trace": tr,
    }


async def node_write_scenes(state: StorybookState) -> dict[str, Any]:
    tr, tmax = state["target_runtime_seconds"], state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    story = state.get("story") or ""
    n = (state.get("scenes_draft_count") or 0) + 1
    feedback = (state.get("scenes_reviewer_feedback") or "").strip()
    rag = (state.get("rag_context_narrative") or "").strip()
    rag_block = f"\n\n{rag}\n" if rag else ""
    p = f"""Create scene-by-scene JSON for an illustrated ad storybook. Return ONLY a JSON object with:
  "style_bible": string (fixed visual style, palette, line quality for every frame),
  "character_anchors": string (recurring look of people/mascot if any),
  "scenes": array of objects, each with:
     "index" (0-based int),
     "suggested_duration_sec" (float, positive),
     "visual_description", "camera", "lighting", "environment", "characters", "product_placement", "negative_space", "continuity_tags" (strings)

Rules: The sum of suggested_duration_sec must be <= {tmax}. Primary goal: advertise the product. Cover the full story. Keep scenes consistent with each other.
Target runtime max: {tmax} seconds. Nominal: {tr} seconds.
Story: {story}
Product template: {json.dumps(pt, ensure_ascii=False)}
{rag_block}
{f"Address reviewer feedback: {feedback}" if feedback else ""}"""
    data = await _call_text(_gem().generate_json, p)
    if not data or "scenes" not in data:
        data = {
            "style_bible": "Clean commercial illustration, high-key lighting, brand colors from template.",
            "character_anchors": "Consistent proportions and clothing.",
            "scenes": [],
        }
    scenes = []
    for i, s in enumerate(data.get("scenes") or []):
        if not isinstance(s, dict):
            continue
        s = {**s, "index": s.get("index", i)}
        try:
            scenes.append(SceneItem.model_validate(s))
        except Exception:
            continue
    if not scenes:
        scenes = [
            SceneItem(
                index=0,
                suggested_duration_sec=min(3.0, tmax),
                visual_description="Hero product shot with context.",
                camera="35mm eye level",
                lighting="soft studio",
                environment="minimal set",
                characters="if needed",
                product_placement="center, readable logo",
                negative_space="for pack text if any",
                continuity_tags="A",
            )
        ]
    style_bible = str(data.get("style_bible") or "Polished ad illustration.")
    ch = str(data.get("character_anchors") or "Consistent look.")
    payload = ScenesPayload(
        style_bible=style_bible,
        character_anchors=ch,
        scenes=scenes,
        target_runtime_max_sec=tmax,
    )
    ok, _msg = validate_scenes_against_max(payload)
    if not ok:
        for sc in payload.scenes:
            sc.suggested_duration_sec = max(0.5, tmax / max(1, len(payload.scenes)))
    pl = json.loads(payload.model_dump_json())
    await _store().update_run(
        state["run_id"],
        {
            "last_scenes_draft": pl,
            "status": "running",
            **step_status_patch({"scenes": "in_progress"}),
        },
    )
    return {"scenes_payload": pl, "scenes_draft_count": n}


async def node_review_scenes(state: StorybookState) -> dict[str, Any]:
    st = _settings()
    tr, tmax = state["target_runtime_seconds"], state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    sp = state.get("scenes_payload") or {}
    story = state.get("story") or ""
    draft_n = int(state.get("scenes_draft_count") or 0)
    prev_fb = (state.get("scenes_reviewer_feedback") or "").strip()
    summary = json.dumps(
        {k: pt.get(k) for k in ("brand", "product_name", "goal", "target_audience", "tone")},
        ensure_ascii=False,
    )
    sp_text = json.dumps(sp, ensure_ascii=False)
    mem = await _call_text(
        query_review_memory,
        summary,
        f"{story}\n{sp_text}",
        prev_fb,
        "scenes",
        st,
    )
    mem_block = f"\n{mem}\n" if mem.strip() else ""
    p = f"""You are a strict ad reviewer. You MUST score the output from 0 (unusable) to 5 (excellent). Return ONLY valid JSON: {{"rating": <0-5 integer>, "feedback": "brief actionable notes"}}
Rubric: (1) product-first, (2) total duration sum vs cap {tmax}, (3) full story covered within time, (4) product placement, (5) cross-scene consistency, (6) match template.

Template: {json.dumps(pt, ensure_ascii=False)}
{mem_block}
Story: {story}
Scenes JSON: {json.dumps(sp, ensure_ascii=False)}"""
    text = await _call_text(_gem().review_json, p)
    r = _parse_review_result(text)
    _log_step_output("scenes", draft_n, r.rating, r.feedback, sp_text)
    hist = list(state.get("scenes_review_history") or [])
    hist.append(
        {
            "draft": draft_n,
            "scenes": sp,
            "rating": r.rating,
            "feedback": r.feedback,
        }
    )
    trace = list(state.get("review_trace") or [])
    trace.append(
        {
            "step": "scenes",
            "draft_index": draft_n,
            "rating": r.rating,
            "feedback": r.feedback,
            "full_output": sp_text,
            "at": _now_iso(),
        }
    )
    await _call_text(
        upsert_review_memory,
        state["run_id"],
        "scenes",
        draft_n,
        sp_text,
        pt,
        r.feedback,
        r.rating,
        st,
    )
    min_ok = r.rating >= st.reviewer_min_rating
    await _store().update_run(
        state["run_id"],
        {
            "scenes_review_history": hist,
            "review_trace": trace,
            "scenes_last_rating": r.rating,
            "last_scenes_review": {"rating": r.rating, "feedback": r.feedback, "draft": draft_n},
        },
    )
    return {
        "scenes_reviewer_feedback": r.feedback if not min_ok else "",
        "scenes_review_approved": min_ok,
        "scenes_last_rating": r.rating,
        "scenes_review_history": hist,
        "review_trace": trace,
        "scenes_chosen": "rating_pass" if min_ok else (state.get("scenes_chosen") or ""),
    }


async def node_pick_best_scenes(state: StorybookState) -> dict[str, Any]:
    hist: list[dict[str, Any]] = list(state.get("scenes_review_history") or [])
    if not hist:
        pl = state.get("scenes_payload") or {}
        logger.info("[scenes] best_of_run: no history, keeping current payload")
        return {
            "scenes_payload": pl,
            "scenes_chosen": "best_of_run",
            "scenes_last_rating": state.get("scenes_last_rating", 0),
        }
    best = max(hist, key=lambda h: (h.get("rating", -1), h.get("draft", 0)))
    pl = best.get("scenes")
    if not isinstance(pl, dict):
        pl = state.get("scenes_payload") or {}
    r = int(best.get("rating") or 0)
    logger.info(
        "[scenes] best_of_run: chose draft %s with rating %s (among %s drafts)",
        best.get("draft"),
        r,
        len(hist),
    )
    tr = list(state.get("review_trace") or [])
    tr.append(
        {
            "step": "scenes",
            "draft_index": int(best.get("draft") or 0),
            "rating": r,
            "feedback": f"Selected best-of-run: draft {best.get('draft')}",
            "full_output": json.dumps(pl, ensure_ascii=False)[:50000],
            "at": _now_iso(),
        }
    )
    await _store().update_run(
        state["run_id"],
        {
            "scenes": pl,
            "scenes_chosen": "best_of_run",
            "scenes_last_rating": r,
            "review_trace": tr,
        },
    )
    return {
        "scenes_payload": pl,
        "scenes_chosen": "best_of_run",
        "scenes_last_rating": r,
        "review_trace": tr,
    }


async def node_save_scenes_approved(state: StorybookState) -> dict[str, Any]:
    rid = state["run_id"]
    chosen = state.get("scenes_chosen") or "rating_pass"
    await _store().update_run(
        rid,
        {
            "scenes": state.get("scenes_payload"),
            "scenes_approved": True,
            "scenes_revisions": state.get("scenes_draft_count", 0),
            "scenes_review_history": state.get("scenes_review_history"),
            "review_trace": state.get("review_trace"),
            "scenes_chosen": chosen,
            "scenes_last_rating": state.get("scenes_last_rating"),
            **step_status_patch(
                {
                    "scenes": "complete",
                    "images": "in_progress",
                }
            ),
        },
    )
    return {"scenes_approved": True}


# Minimal 1x1 transparent PNG
_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\x2d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def node_generate_images(state: StorybookState) -> dict[str, Any]:
    sp = state.get("scenes_payload") or {}
    style = str(sp.get("style_bible", ""))
    ch = str(sp.get("character_anchors", ""))
    raw_scenes = sp.get("scenes") or []
    scenes: list[dict] = [s if isinstance(s, dict) else {} for s in raw_scenes]
    tmax = state["target_runtime_max_seconds"]
    nround = (state.get("images_draft_count") or 0) + 1
    await _store().update_run(
        state["run_id"],
        {**step_status_patch({"images": "in_progress"}), "status": "running"},
    )
    n = len(scenes)
    if n == 0:
        return {
            "images": [],
            "image_pass_full": False,
            "scene_ids_to_regen": [],
            "images_draft_count": nround,
        }
    by_idx: dict[int, dict] = {}
    for d in state.get("images") or []:
        if isinstance(d, dict) and "scene_index" in d:
            by_idx[int(d["scene_index"])] = d
    full = bool(state.get("image_pass_full", True))
    regen = [int(x) for x in (state.get("scene_ids_to_regen") or [])]
    if full or not by_idx:
        indices = list(range(n))
    else:
        indices = regen if regen else list(range(n))
    out: dict[int, dict] = {k: dict(v) for k, v in by_idx.items()}
    for i in sorted(set(indices)):
        if i < 0 or i >= n:
            continue
        d = scenes[i]
        v = json.dumps(d, ensure_ascii=False)
        ref: Optional[tuple[bytes, str]] = None
        if i > 0:
            prev = out.get(i - 1) or by_idx.get(i - 1)
            if prev and prev.get("bytes"):
                ref = (bytes(prev["bytes"]), str(prev.get("mime_type") or "image/png"))
        prompt = (
            f"Storybook advertising illustration, frame {i+1} of {n}. "
            f"Match previous frames' characters and product. "
            f"STYLE: {style} CHARACTERS: {ch} SCENE: {v} Ad total time cap: {tmax}s. "
            f"CRITICAL: product is clearly visible, commercial polished look, no ugly watermarks."
        )
        b = await _call_text(_gem().generate_image, prompt, ref)
        if not b:
            b = await _call_text(_gem().generate_image, prompt, None)
        if not b:
            b = _PLACEHOLDER_PNG
        out[i] = {
            "scene_index": i,
            "mime_type": "image/png",
            "prompt_used": prompt[:2000],
            "bytes": b,
        }
    merged: list[dict] = [out[i] for i in range(n) if i in out]
    return {
        "images": merged,
        "image_pass_full": False,
        "scene_ids_to_regen": [],
        "images_draft_count": nround,
    }


async def node_complete(state: StorybookState) -> dict[str, Any]:
    rid = state["run_id"]
    store = _store()
    settings = get_settings()
    image_refs: list[dict] = []
    for im in state.get("images") or []:
        if not isinstance(im, dict):
            continue
        b = im.get("bytes")
        if b and isinstance(b, (bytes, bytearray)):
            ref = await put_scene_image_async(
                settings,
                rid,
                int(im.get("scene_index", 0)),
                bytes(b),
                str(im.get("mime_type") or "image/png"),
            )
            image_refs.append(
                {
                    **ref,
                    "prompt_used": im.get("prompt_used", "")[:2000],
                }
            )
        elif im.get("s3_key") and im.get("s3_bucket"):
            image_refs.append({k: v for k, v in im.items() if k != "bytes"})
    await store.update_run(
        rid,
        {
            "status": "complete",
            "images": image_refs,
            "image_revisions": state.get("images_draft_count", 0),
            "error_detail": None,
            "review_trace": state.get("review_trace"),
            **step_status_patch({"images": "complete"}),
        },
    )
    return {
        "status": "complete",
        "images_approved": True,
        "scenes_approved": True,
        "story_approved": True,
        "images": image_refs,
    }


async def node_fail_script(state: StorybookState) -> dict[str, Any]:
    msg = state.get("pending_error") or "Maximum script review iterations reached."
    await _store().update_run(
        state["run_id"],
        {
            "status": "failed",
            "error_detail": msg,
            **step_status_patch({"script": "error"}),
        },
    )
    return {"status": "failed", "error_detail": msg, "errors": [msg]}


async def node_fail_scenes(state: StorybookState) -> dict[str, Any]:
    msg = state.get("pending_error") or "Maximum scenes review iterations reached."
    await _store().update_run(
        state["run_id"],
        {
            "status": "failed",
            "error_detail": msg,
            **step_status_patch({"scenes": "error"}),
        },
    )
    return {"status": "failed", "error_detail": msg, "errors": [msg]}


# routing helpers
def max_script() -> int:
    return _settings().max_script_revisions


def max_scenes() -> int:
    return _settings().max_scenes_revisions
