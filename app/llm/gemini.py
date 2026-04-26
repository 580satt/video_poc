from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Optional

from google import genai
from google.genai import types

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json_loose(text: str) -> Any:
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


class GeminiClient:
    """Thin async-friendly wrapper: call sync `generate_content` in a thread from FastAPI if needed."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._text = self._settings.gemini_text_model
        self._reviewer = self._settings.gemini_reviewer_model
        self._image = self._settings.gemini_image_model
        if not self._settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        m = model or self._text
        cfg = types.GenerateContentConfig(
            temperature=temperature,
        )
        resp = self._client.models.generate_content(
            model=m,
            contents=prompt,
            config=cfg,
        )
        if not resp or not resp.text:
            return ""
        return resp.text

    def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        repair_hint: str = "",
    ) -> dict[str, Any]:
        m = model or self._text
        full = prompt
        for attempt in range(3):
            text = self.generate_text(full, model=m, temperature=0.2 if attempt else 0.3)
            try:
                return _parse_json_loose(text)
            except json.JSONDecodeError:
                full = (
                    prompt
                    + "\n\nReturn ONLY a single valid JSON object, no markdown.\n"
                    + (repair_hint or "")
                )
        return {}

    def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        instruction: str,
    ) -> str:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        contents: list = [types.Part.from_text(text=instruction), part]
        resp = self._client.models.generate_content(
            model=self._text,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return (resp.text or "").strip() if resp else ""

    def review_json(self, system_and_user: str) -> str:
        return self.generate_text(
            system_and_user,
            model=self._reviewer,
            temperature=0.1,
        )

    def review_multimodal(
        self,
        text: str,
        image_parts: list[tuple[bytes, str]],
    ) -> str:
        parts: list = [types.Part.from_text(text=text)]
        for b, mt in image_parts:
            parts.append(types.Part.from_bytes(data=b, mime_type=mt))
        resp = self._client.models.generate_content(
            model=self._reviewer,
            contents=parts,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return (resp.text or "").strip() if resp else ""

    def generate_image(
        self,
        prompt: str,
        extra_image_ref: Optional[tuple[bytes, str]] = None,
    ) -> Optional[bytes]:
        """Return PNG/JPEG bytes from image-capable model, or None on failure."""
        parts: list = [types.Part.from_text(text=prompt)]
        if extra_image_ref:
            b, mt = extra_image_ref
            parts.append(types.Part.from_bytes(data=b, mime_type=mt))
        try:
            cfg = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
        except TypeError:
            cfg = None
        try:
            if cfg is not None:
                resp = self._client.models.generate_content(
                    model=self._image,
                    contents=parts,
                    config=cfg,
                )
            else:
                resp = self._client.models.generate_content(
                    model=self._image,
                    contents=parts,
                )
        except Exception:
            return None
        if not resp:
            return None
        for c in (resp.candidates or []):
            for p in c.content.parts or []:
                if getattr(p, "inline_data", None) and p.inline_data:
                    d = p.inline_data.data
                    if isinstance(d, (bytes, bytearray)) and d:
                        return bytes(d)
                    if isinstance(d, str) and d:
                        try:
                            return base64.b64decode(d)
                        except (ValueError, TypeError):
                            pass
        logger.warning(
            "Image model %s returned no image bytes; use an image generation model (e.g. gemini-2.5-flash-image).",
            self._image,
        )
        return None
