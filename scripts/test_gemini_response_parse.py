#!/usr/bin/env python3
"""Unit checks for parsing Gemini generate_content responses (no API key)."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from app.llm.gemini import (
    _iter_response_parts,
    first_inline_image_bytes_from_response,
    text_from_generate_content_response,
)


def _fail(msg: str) -> None:
    print("FAIL:", msg)
    sys.exit(1)


def main() -> None:
    png = b"\x89PNG\r\n\x1a\nfake"

    # Normal: one candidate with inline image
    resp_ok = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(inline_data=types.Blob(data=png, mime_type="image/png"))]
                )
            )
        ]
    )
    if first_inline_image_bytes_from_response(resp_ok) != png:
        _fail("expected PNG bytes from normal response")
    if list(_iter_response_parts(resp_ok)) != resp_ok.candidates[0].content.parts:
        _fail("iterator mismatch for normal response")

    # Candidate with content=None (blocked / empty)
    resp_no_content = types.GenerateContentResponse(
        candidates=[types.Candidate(content=None)]
    )
    try:
        b = first_inline_image_bytes_from_response(resp_no_content)
    except Exception as e:
        _fail(f"None content raised {e!r}")
    if b is not None:
        _fail("expected None when candidate.content is None")

    # Content exists but parts=None (valid per SDK types)
    resp_parts_none = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=None))]
    )
    try:
        b = first_inline_image_bytes_from_response(resp_parts_none)
    except Exception as e:
        _fail(f"parts=None raised {e!r}")
    if b is not None:
        _fail("expected None when parts is None")

    # Mixed: first useless candidate, second has image
    b64 = base64.b64encode(png).decode("ascii")
    resp_second = types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=None),
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=b64, mime_type="image/png"),
                        )
                    ]
                )
            ),
        ]
    )
    out = first_inline_image_bytes_from_response(resp_second)
    if out != png:
        _fail("expected image from second candidate")

    # Dict-shaped content (defensive)
    class _C:
        pass

    c0 = _C()
    c0.candidates = [
        {"content": {"parts": [{"inline_data": {"data": png, "mime_type": "image/png"}}]}}
    ]
    if first_inline_image_bytes_from_response(c0) != png:
        _fail("dict-shaped candidate failed")

    # Text extraction without touching SDK ``.text`` (broken when content is odd)
    resp_txt = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello "), types.Part(text="world")])
            )
        ]
    )
    if text_from_generate_content_response(resp_txt) != "Hello world":
        _fail("text concat from parts failed")

    resp_bad = types.GenerateContentResponse(candidates=[types.Candidate(content=None)])
    try:
        s = text_from_generate_content_response(resp_bad)
    except Exception as e:
        _fail(f"text extraction raised {e!r}")
    if s != "":
        _fail("expected empty text when content is None")

    print("OK: all response parsing edge cases passed")


if __name__ == "__main__":
    main()
