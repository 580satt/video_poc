#!/usr/bin/env python3
"""Quick check that GEMINI_IMAGE_MODEL returns image bytes. Run from project root: python scripts/test_image_gen.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.llm.gemini import GeminiClient


def main() -> None:
    s = get_settings()
    print("GEMINI_IMAGE_MODEL =", s.gemini_image_model)
    c = GeminiClient(s)
    b = c.generate_image("A simple red circle on white background, flat vector style")
    if not b:
        print("FAIL: no bytes returned. Use an image-native model, e.g. gemini-2.5-flash-image.")
        sys.exit(1)
    print("OK: received", len(b), "bytes, PNG header:", b[:8].hex() if len(b) >= 8 else b)
    if not b.startswith(b"\x89PNG\r\n\x1a\n"):
        print("Note: not PNG signature; may still be valid image/jpeg.")


if __name__ == "__main__":
    main()
