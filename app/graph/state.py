from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class StorybookState(TypedDict, total=False):
    run_id: str
    raw_user_input: dict[str, Any]
    product_image: bytes | None
    product_image_mime: str | None

    target_runtime_seconds: float
    target_runtime_max_seconds: float

    product_template: dict[str, Any] | None
    # Injected after template; used in write_script / write_scenes
    rag_context_narrative: str
    # User-provided long-form brand / insights / psychology (same lifecycle as RAG for script+scenes)
    brand_psychology_context: str
    # Chroma narrative retrieval debug: collection, top_k, hits, skipped_reason
    rag_narrative_trace: dict[str, Any]

    story: str
    story_approved: bool
    story_draft_count: int
    script_reviewer_feedback: str
    script_last_rating: int
    script_review_history: list[dict[str, Any]]  # {draft, story, rating, feedback}
    script_chosen: str  # e.g. rating_pass | best_of_run

    scenes_payload: dict[str, Any] | None
    scenes_approved: bool
    scenes_draft_count: int
    scenes_reviewer_feedback: str
    scenes_last_rating: int
    scenes_review_history: list[dict[str, Any]]
    scenes_chosen: str

    images: list[dict[str, Any]]
    images_approved: bool
    images_draft_count: int
    images_reviewer_feedback: str
    scene_ids_to_regen: list[int]

    status: str  # running | failed | complete
    errors: list[str]
    error_detail: str | None

    # image gen control
    image_pass_full: bool  # if True, generate all scenes; if False, use scene_ids_to_regen

    # regen from API: set by runner
    start_from: NotRequired[str]  # template | script | scenes | images

    # reviewer: latest pass against min rating (convenience for debugging)
    script_review_approved: bool
    scenes_review_approved: bool
    images_review_approved: bool
    # structured history for API / export
    review_trace: list[dict[str, Any]]

    pending_error: str | None
