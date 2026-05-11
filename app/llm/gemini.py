from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional

from google import genai
from google.genai import types

from app.config import Settings, get_settings
from app.llm.json_utils import parse_json_loose

logger = logging.getLogger(__name__)


def _iter_response_parts(resp: Any):
    """Yield Part objects from all candidates; tolerates missing/None content or parts."""
    for cand in getattr(resp, "candidates", None) or []:
        if cand is None:
            continue
        if isinstance(cand, dict):
            content = cand.get("content")
        else:
            content = getattr(cand, "content", None)
        if content is None:
            continue
        if isinstance(content, dict):
            plist = content.get("parts")
        else:
            plist = getattr(content, "parts", None)
        for part in plist or []:
            if part is not None:
                yield part


def _inline_data_payload(part: Any) -> Any:
    inl = part.get("inline_data") if isinstance(part, dict) else getattr(part, "inline_data", None)
    if inl is None:
        return None
    if isinstance(inl, dict):
        return inl.get("data")
    return getattr(inl, "data", None)


def first_inline_image_bytes_from_response(resp: Any) -> Optional[bytes]:
    """First image bytes from a generate_content response, or None."""
    for p in _iter_response_parts(resp):
        d = _inline_data_payload(p)
        if isinstance(d, (bytes, bytearray)) and d:
            return bytes(d)
        if isinstance(d, str) and d:
            try:
                return base64.b64decode(d)
            except (ValueError, TypeError):
                continue
    return None


def text_from_generate_content_response(resp: Any) -> str:
    """Concatenate text parts without using ``resp.text`` (SDK can raise on odd candidates)."""
    if not resp:
        return ""
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""
    chunks: list[str] = []
    for cand in cands[:1]:
        if cand is None:
            continue
        content = cand.get("content") if isinstance(cand, dict) else getattr(cand, "content", None)
        if content is None:
            continue
        plist = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
        for part in plist or []:
            if part is None:
                continue
            if isinstance(part, dict):
                t = part.get("text")
                th = part.get("thought")
            else:
                t = getattr(part, "text", None)
                th = getattr(part, "thought", None)
            if not isinstance(t, str):
                continue
            if isinstance(th, bool) and th:
                continue
            chunks.append(t)
    return "".join(chunks)


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
        mo = int(getattr(self._settings, "gemini_max_output_tokens", 8192) or 8192)
        try:
            cfg = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=mo,
            )
        except (TypeError, ValueError):
            cfg = types.GenerateContentConfig(temperature=temperature)
        resp = self._client.models.generate_content(
            model=m,
            contents=prompt,
            config=cfg,
        )
        return text_from_generate_content_response(resp).strip()

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
                return parse_json_loose(text)
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
        return text_from_generate_content_response(resp).strip()

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
        return text_from_generate_content_response(resp).strip()

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
        try:
            found = first_inline_image_bytes_from_response(resp)
        except (AttributeError, TypeError) as exc:
            logger.warning("Could not parse image from model response: %s", exc)
            found = None
        if found:
            return found
        logger.warning(
            "Image model %s returned no image bytes; use an image generation model (e.g. gemini-2.5-flash-image).",
            self._image,
        )
        return None
