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
from app.graph.state import StorybookState
from app.llm.gemini import GeminiClient
from app.llm.wavespeed_client import WaveSpeedClient
from app.models.schemas import (
    ProductTemplate,
    ReviewResult,
    ScenesPayload,
    SceneItem,
    validate_scenes_against_max,
)
from app.graph.context_prefs import context_source_flags, normalize_context_sources
from app.graph.progress import step_status_patch
from app.rag.chroma_narrative import (
    narrative_trace_user_disabled,
    query_narrative_rag_with_trace,
    query_review_memory,
    upsert_review_memory,
)

logger = logging.getLogger(__name__)

_GEM: Optional[GeminiClient] = None
_S: Optional[Settings] = None
_MONGO: Optional["RunStore"] = None
_WAVE: Optional[WaveSpeedClient] = None


def bind_gemini(gem: GeminiClient, st: Settings) -> None:
    global _GEM, _S, _WAVE
    _GEM = gem
    _S = st
    _WAVE = None  # rebuild WaveSpeed client if settings / key changed


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


def _wavespeed_client() -> Optional[WaveSpeedClient]:
    """Lazy WaveSpeed client; None if key missing or init fails."""
    global _WAVE
    s = _settings()
    if not s.wavespeed_api_key.strip():
        return None
    if _WAVE is None:
        try:
            _WAVE = WaveSpeedClient(s)
        except Exception as e:
            logger.warning("WaveSpeed client unavailable: %s", e)
            return None
    return _WAVE


def _story_scenes_provider(state: StorybookState) -> str:
    """Per-run override from raw_user_input, else Settings.story_scenes_llm_provider."""
    raw = state.get("raw_user_input") or {}
    for key in ("story_scenes_provider", "story_scenes_llm_provider"):
        v = raw.get(key)
        if v is not None and str(v).strip() != "":
            w = str(v).strip().lower()
            if w in ("wavespeed", "claude", "claude_opus", "opus", "anthropic"):
                return "wavespeed"
            return "gemini"
    p = (_settings().story_scenes_llm_provider or "gemini").strip().lower()
    if p in ("wavespeed", "claude", "claude_opus", "opus", "anthropic"):
        return "wavespeed"
    return "gemini"


_MAX_BRAND_PSYCHOLOGY_CHARS = 60_000


def _normalize_brand_psychology_context(val: Any) -> str:
    s = str(val or "").strip()
    if len(s) > _MAX_BRAND_PSYCHOLOGY_CHARS:
        return (
            s[:_MAX_BRAND_PSYCHOLOGY_CHARS]
            + "\n\n[Truncated: input exceeded safe limit for model context.]"
        )
    return s


def _rag_block_for_prompt(rag: str) -> str:
    rag = (rag or "").strip()
    if not rag:
        return ""
    return (
        "\n\n---\nRETRIEVED NARRATIVE REFERENCE (RAG — optional library context; "
        "use only where it fits the brief and template):\n"
        f"{rag}\n---\n"
    )


def _brand_psychology_block_for_prompt(brief: str) -> str:
    brief = (brief or "").strip()
    if not brief:
        return ""
    return (
        "\n\n---\nUSER BRAND / PSYCHOLOGY / INSIGHTS BRIEF (mandatory when present — align claims, tone, "
        "emotional angle, and brand cues with this text; do not contradict the product template):\n"
        f"{brief}\n---\n"
    )


def _no_extra_context_note(use_rag: bool, use_brief: bool) -> str:
    if use_rag or use_brief:
        return ""
    return (
        "\n\n---\nNo narrative RAG and no user brand/psychology brief for this run (per user selection). "
        "Use only the template JSON and these instructions.\n---\n"
    )


def _reviewer_context_excerpt(state: StorybookState) -> str:
    """Compact excerpts so reviewers know what the writer was given beyond the template."""
    use_rag, use_brief = context_source_flags(state.get("raw_user_input"))
    parts: list[str] = []
    rag = (state.get("rag_context_narrative") or "").strip()
    if use_rag and rag:
        cap = 4500
        tail = "…" if len(rag) > cap else ""
        parts.append(f"RAG narrative reference (excerpt for review): {rag[:cap]}{tail}")
    brief = (state.get("brand_psychology_context") or "").strip()
    if use_brief and brief:
        cap = 8000
        tail = "…" if len(brief) > cap else ""
        parts.append(f"User brand/psychology brief (excerpt for review): {brief[:cap]}{tail}")
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


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


def _log_step_output(
    step: str, draft: int, rating: int | None, feedback: str, body: str, *, kind: str = "review"
) -> None:
    prev = 8000
    t = body if len(body) <= prev else body[:prev] + f"\n…[truncated, total {len(body)} chars]"
    logger.info(
        "[%s] step=%s draft=%s rating=%s feedback=%s\n%s",
        kind,
        step,
        draft,
        rating,
        (feedback or "")[:2000],
        t,
    )


def _story_pacing_instruction(tr: float, tmax: float) -> str:
    """Scale story length to total runtime (speakable ad read ~2–3 words/s; paragraph count grows with tmax)."""
    p_lo = max(2, int(tmax / 4.0))
    p_hi = max(p_lo + 1, min(24, int(tmax / 1.3)))
    w_lo = int(max(30, tmax * 1.8))
    w_hi = int(max(w_lo + 15, tmax * 2.8))
    return (
        f"**Length (must match the {tr}-{tmax}s ad slot):** write enough prose to use that time at a clear, speakable ad pace. "
        f"Use roughly {p_lo}–{p_hi} paragraphs (about {w_lo}–{w_hi} words total; scale up for longer runtimes). "
        "Longer runtimes = more story beats, character moments, and product messaging—not a single short blurb. "
        "Do not under-write for long slots; the script should feel like a real {tmax}-second ad when read aloud."
    )


def _scene_pacing_instruction(tr: float, tmax: float) -> str:
    """Encourage more scenes for longer runtimes; each scene still gets suggested_duration that sums to <= tmax."""
    n_lo = max(3, int(round(tmax / 3.0)))
    n_hi = min(28, max(n_lo + 1, int(round(tmax / 1.5))))
    per_lo = max(0.8, tmax / n_hi * 0.9)
    per_hi = min(6.0, tmax / n_lo * 1.1) if n_lo else 4.0
    return (
        f"**Scene count:** for a **{tmax}s** (nominal {tr}s) storyboard, use **{n_lo} to {n_hi} scenes** so the ad breathes. "
        f"Shorter total times use fewer frames; **longer 15s+ runtimes need more scenes** (often ~1 scene per 1.5–3.5s of the cap). "
        f"Distribute `suggested_duration_sec` so their sum is ≤{tmax}s (typical per-scene range ~{per_lo:.1f}–{per_hi:.1f}s but vary as needed). "
        "Each scene = one key visual beat; do not default to 5 shots regardless of total time."
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

Time budget: design for ~{tr}s, must not exceed {tmax}s total story/scenes. When target_runtime_max_sec is **large (e.g. 15-25s)**, the creative concept must support a **longer, richer** story and more storyboard beats, not a tiny 5-8s feel.

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
    cs = normalize_context_sources(raw)
    raw = {**raw, "context_sources": cs}
    brief = _normalize_brand_psychology_context(raw.get("brand_psychology_context"))
    if brief:
        raw = {**raw, "brand_psychology_context": brief}
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
    use_rag, _ = context_source_flags(raw)
    if use_rag:
        rag, rag_trace = await _call_text(
            query_narrative_rag_with_trace,
            pt,
            raw,
            _settings(),
        )
        if rag:
            logger.info("[rag] narrative context length=%s", len(rag))
    else:
        rag, rag_trace = "", narrative_trace_user_disabled(_settings(), cs)

    await _store().update_run(
        state["run_id"],
        {
            "product_template": pt,
            "brand_psychology_context": brief,
            "raw_input": raw,
            "status": "running",
            "rag_context_narrative": rag,
            "rag_narrative_trace": rag_trace,
            **step_status_patch(
                {
                    "template": "complete",
                    "script": "in_progress",
                }
            ),
        },
    )
    return {
        "product_template": pt,
        "raw_user_input": raw,
        "brand_psychology_context": brief,
        "status": "running",
        "rag_context_narrative": rag,
        "rag_narrative_trace": rag_trace,
    }


async def node_write_script(state: StorybookState) -> dict[str, Any]:
    st = _settings()
    tr = state["target_runtime_seconds"]
    tmax = state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    n = (state.get("story_draft_count") or 0) + 1
    feedback = (state.get("script_reviewer_feedback") or "").strip()
    use_rag, use_brief = context_source_flags(state.get("raw_user_input"))
    rag = (state.get("rag_context_narrative") or "").strip()
    bp = (state.get("brand_psychology_context") or "").strip()
    rag_block = _rag_block_for_prompt(rag) if use_rag else ""
    bp_block = _brand_psychology_block_for_prompt(bp) if use_brief else ""
    pace = _story_pacing_instruction(tr, tmax)
    p = f"""You write a clear, engaging ad story for a storybook (voiceover / read-aloud). The PRIMARY goal is to advertise: {pt.get("product_name")} by {pt.get("brand", "")}. Tone: {pt.get("tone")}. Audience: {pt.get("target_audience")}. Goals: {pt.get("goal")}.

Hard time cap: the full story must be deliverable within **{tmax} seconds** when read at a **moderate, clear** ad pace (target nominal window {tr}-{tmax}s). Strong product and CTA focus; every paragraph should earn its place.

{pace}

Return ONLY the story body text, no title line, no section headers.

Template JSON: {json.dumps(pt, ensure_ascii=False)}
{rag_block}{bp_block}{_no_extra_context_note(use_rag, use_brief)}
{f"Revise per reviewer feedback: {feedback}" if feedback else ""}"""
    prov = _story_scenes_provider(state)
    if prov == "wavespeed":
        ws = _wavespeed_client()
        if ws:
            text = await _call_text(ws.generate_text, p, temperature=0.7)
            logger.info("[story] provider=wavespeed model=%s", st.wavespeed_model)
        else:
            logger.warning(
                "story_scenes_provider=wavespeed but WAVESPEED_API_KEY missing; using Gemini for story"
            )
            text = await _call_text(_gem().generate_text, p, st.gemini_text_model, 0.7)
    else:
        text = await _call_text(_gem().generate_text, p, st.gemini_text_model, 0.7)
        logger.info("[story] provider=gemini model=%s", st.gemini_text_model)
    story = (text or "").strip()
    _log_step_output("script", n, None, "draft output (before review)", story, kind="output")
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
Rubric in order: (1) product advertising is strong, (2) the story is **appropriately developed for the {tmax}s** slot (at moderate read-aloud pace, longer runtimes need enough substance—not a tiny 5-8s blurb for a 15-20s brief), (3) creative, (4) matches template, (5) respects the **user brand / psychology brief** and any **RAG reference** only when excerpts appear below (the writer did not see sources omitted there), (6) no contradictions.

Target runtime: {tr}-{tmax}s. Template: {json.dumps(pt, ensure_ascii=False)}
{_reviewer_context_excerpt(state)}
{mem_block}
STORY:
{story}"""
    text = await _call_text(_gem().review_json, p)
    r = _parse_review_result(text)
    _log_step_output("script", draft_n, r.rating, r.feedback, story, kind="review")
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
    st = _settings()
    tr, tmax = state["target_runtime_seconds"], state["target_runtime_max_seconds"]
    pt = state.get("product_template") or {}
    story = state.get("story") or ""
    n = (state.get("scenes_draft_count") or 0) + 1
    feedback = (state.get("scenes_reviewer_feedback") or "").strip()
    use_rag, use_brief = context_source_flags(state.get("raw_user_input"))
    rag = (state.get("rag_context_narrative") or "").strip()
    bp = (state.get("brand_psychology_context") or "").strip()
    rag_block = _rag_block_for_prompt(rag) if use_rag else ""
    bp_block = _brand_psychology_block_for_prompt(bp) if use_brief else ""
    scene_pace = _scene_pacing_instruction(tr, tmax)
    p = f"""Create scene-by-scene JSON for a live-action / photoreal TV ad (key visuals as **photographs**, not drawings). Return ONLY a JSON object with:
  "style_bible": string (fixed **photographic** look: camera/lens, light, color grade; professional commercial photo shoot, not cartoon or illustration),
  "character_anchors": string (recurring on-camera look of real people, wardrobe, build),
  "scenes": array of objects, each with:
     "index" (0-based int),
     "suggested_duration_sec" (float, positive),
     "visual_description", "camera", "lighting", "environment", "characters", "product_placement", "negative_space", "continuity_tags" (strings; real camera / on-set language)

{scene_pace}

Rules: The sum of suggested_duration_sec must be **<= {tmax}** (and should usually use the budget well for longer spots). Primary goal: advertise the product. Cover the **entire** story with enough distinct beats; do not compress a long script into 4–5 scenes if the time cap is 15-25s. Keep scenes consistent. No cartoon or illustrated look; photoreal on-set/location only.
Target runtime max: {tmax} seconds. Nominal: {tr} seconds.
Story: {story}
Product template: {json.dumps(pt, ensure_ascii=False)}
{rag_block}{bp_block}{_no_extra_context_note(use_rag, use_brief)}
{f"Address reviewer feedback: {feedback}" if feedback else ""}"""
    prov = _story_scenes_provider(state)
    if prov == "wavespeed":
        ws = _wavespeed_client()
        if ws:
            data = await _call_text(ws.generate_json, p)
            logger.info("[scenes] provider=wavespeed model=%s", st.wavespeed_model)
        else:
            logger.warning(
                "story_scenes_provider=wavespeed but WAVESPEED_API_KEY missing; using Gemini for scenes JSON"
            )
            data = await _call_text(_gem().generate_json, p)
    else:
        data = await _call_text(_gem().generate_json, p)
        logger.info("[scenes] provider=gemini model=%s", st.gemini_text_model)
    if not data or "scenes" not in data:
        data = {
            "style_bible": "Photoreal TV commercial: natural/cinematic light, 35mm or 50mm camera feel, true-to-life color, shallow depth of field when appropriate, brand-true product.",
            "character_anchors": "Consistent real people: same hair, skin tone, wardrobe, and build across frames.",
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
                visual_description="Photoreal hero product on seamless backdrop with believable studio lighting and reflections.",
                camera="35mm eye level",
                lighting="soft studio",
                environment="minimal set",
                characters="if needed",
                product_placement="center, readable logo",
                negative_space="for pack text if any",
                continuity_tags="A",
            )
        ]
    style_bible = str(
        data.get("style_bible")
        or "Photoreal advertising photography: high dynamic range, real locations or believable set, professional lighting, premium TV spot look."
    )
    ch = str(data.get("character_anchors") or "Consistent on-camera talent and wardrobe for continuity between shots.")
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
    pl_text = json.dumps(pl, ensure_ascii=False)
    _log_step_output("scenes", n, None, "draft output (before review)", pl_text, kind="output")
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
Rubric: (1) product-first, (2) total duration sum vs cap {tmax}, (3) full story covered **with a scene count and pacing that fit a {tmax}s spot** (penalize dumping a long script into only ~5 generic scenes when the cap is 15-25s), (4) product placement, (5) cross-scene consistency, (6) match template, (7) honor the **user brand / psychology brief** and **RAG reference** only when excerpts appear below.

Template: {json.dumps(pt, ensure_ascii=False)}
{_reviewer_context_excerpt(state)}
{mem_block}
Story: {story}
Scenes JSON: {json.dumps(sp, ensure_ascii=False)}"""
    text = await _call_text(_gem().review_json, p)
    r = _parse_review_result(text)
    _log_step_output("scenes", draft_n, r.rating, r.feedback, sp_text, kind="review")
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


def _image_style_directive(st: Settings) -> str:
    """Text baked into every image prompt; default is photoreal / live-action TVC still."""
    if not st.image_photorealistic:
        return (
            "Style: polished advertising illustration; match prior frames' characters and product. "
            "No ugly watermarks."
        )
    return (
        "MANDATORY LOOK: photorealistic — must resemble a real photograph or frame from a high-end live-action TV / web commercial, "
        "shot on a real camera (e.g. 35mm or 50mm prime, or cinema glass). "
        "Natural materials and light: real skin, fabric, metal, glass, environment. "
        "Cinematic or natural color grading; believable depth of field and lens behavior. "
        "FORBIDDEN: cartoon, comic book, anime, chibi, vector art, flat illustration, storybook/painterly art, "
        "thick black outlines, cel shading, or toy-like plastic look. "
        "The image must be indistinguishable from a professional photo still, not a drawing. "
        "Product and talent clearly visible; premium commercial look; no ugly watermarks; no on-image slogans or logos unless in scene brief."
    )


async def node_generate_images(state: StorybookState) -> dict[str, Any]:
    st = _settings()
    sp = state.get("scenes_payload") or {}
    style = str(sp.get("style_bible", ""))
    ch = str(sp.get("character_anchors", ""))
    realism = _image_style_directive(st)
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
            f"Advertising key visual — frame {i+1} of {n} in the same campaign. "
            f"{realism} "
            f"Continuity: match people, product, and environment with prior frames in this ad. "
            f"LOOK BIBLE (interpret as real-world lighting and camera, not as drawn art): {style} "
            f"TALENT / PRODUCT ANCHORS: {ch} "
            f"SHOT LIST / SCENE (camera sees): {v} "
            f"Context: {tmax}s max spot length."
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
            **step_status_patch(
                {
                    "template": "complete",
                    "script": "complete",
                    "scenes": "complete",
                    "images": "complete",
                }
            ),
        },
    )
    await store.insert_pipeline_output(
        rid,
        "run_complete",
        {
            "product_template": state.get("product_template"),
            "story": state.get("story"),
            "scenes_payload": state.get("scenes_payload"),
            "images": image_refs,
            "target_runtime_seconds": state.get("target_runtime_seconds"),
            "target_runtime_max_seconds": state.get("target_runtime_max_seconds"),
            "script_chosen": state.get("script_chosen"),
            "scenes_chosen": state.get("scenes_chosen"),
            "rag_context_narrative": (state.get("rag_context_narrative") or "")[:20000],
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
