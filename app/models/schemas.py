from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    running = "running"
    failed = "failed"
    complete = "complete"


class ReviewResult(BaseModel):
    """Reviewer output: 0-5 (all reviewers must return a score). No separate approve flag — routing uses `rating` vs `reviewer_min_rating` in config."""

    rating: int = Field(0, ge=0, le=5)
    feedback: str = ""
    scene_ids_to_redo: list[int] = Field(default_factory=list)


class ProductTemplate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    brand: str = ""
    product_name: str = ""
    goal: str = ""
    target_audience: str = ""
    tone: str = ""
    primary_goal: str = "advertise_product"
    target_runtime_sec: float = 8.0
    target_runtime_max_sec: float = 10.0
    goals: list[str] = Field(default_factory=lambda: ["advertise_product", "time_fit", "clarity"])
    visual_anchors: str = ""
    product_image_insights: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.model_dump()
        return d


class SceneItem(BaseModel):
    index: int
    suggested_duration_sec: float
    visual_description: str
    camera: str = ""
    lighting: str = ""
    environment: str = ""
    characters: str = ""
    product_placement: str = ""
    negative_space: str = ""
    continuity_tags: str = ""


class ScenesPayload(BaseModel):
    style_bible: str = ""
    character_anchors: str = ""
    scenes: list[SceneItem] = Field(default_factory=list)
    target_runtime_max_sec: float = 10.0

    def duration_sum(self) -> float:
        return sum(s.suggested_duration_sec for s in self.scenes)


class ImageRecord(BaseModel):
    scene_index: int
    filename: str
    mime_type: str = "image/png"
    s3_key: Optional[str] = None
    s3_bucket: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    prompt_used: str = ""


class CreateRunRequest(BaseModel):
    product_name: str = ""
    brand: str = ""
    goal: str = ""
    target_audience: str = ""
    notes: str = ""
    target_runtime_seconds: Optional[float] = None
    target_runtime_max_seconds: Optional[float] = None
    # Image uploaded via multipart field `product_image` in the route, not in JSON


class RegenerateRequest(BaseModel):
    from_step: str = Field(..., pattern="^(script|scenes|images)$")


def validate_scenes_against_max(payload: ScenesPayload) -> tuple[bool, str]:
    s = payload.duration_sum()
    if s > payload.target_runtime_max_sec + 0.01:
        return False, f"Scene durations sum to {s:.2f}s, exceeds cap {payload.target_runtime_max_sec:.2f}s"
    if not payload.scenes:
        return False, "No scenes"
    return True, ""
