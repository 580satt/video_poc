from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.config import Settings, get_settings
from app.llm.json_utils import parse_json_loose

logger = logging.getLogger(__name__)


class WaveSpeedClient:
    """OpenAI-compatible Chat Completions at llm.wavespeed.ai (e.g. anthropic/claude-opus-4.7)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.wavespeed_api_key.strip():
            raise ValueError("WAVESPEED_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ValueError("openai package required for WaveSpeed") from e
        base = self._settings.wavespeed_base_url.strip().rstrip("/")
        self._client = OpenAI(
            api_key=self._settings.wavespeed_api_key,
            base_url=base,
        )
        self._model = self._settings.wavespeed_model

    def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        m = model or self._model
        try:
            cap = int(getattr(self._settings, "wavespeed_max_output_tokens", 16384) or 16384)
            r = self._client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=cap,
            )
            ch = r.choices[0].message
            return (ch.content or "").strip() if ch else ""
        except Exception as e:
            logger.warning("WaveSpeed chat completion failed: %s", e)
            return ""

    def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        repair_hint: str = "",
    ) -> dict[str, Any]:
        m = model or self._model
        full = prompt
        for attempt in range(3):
            text = self.generate_text(full, model=m, temperature=0.2 if attempt else 0.3)
            try:
                out = parse_json_loose(text)
                if isinstance(out, dict):
                    return out
            except json.JSONDecodeError:
                pass
            full = (
                prompt
                + "\n\nReturn ONLY a single valid JSON object, no markdown.\n"
                + (repair_hint or "")
            )
        return {}
