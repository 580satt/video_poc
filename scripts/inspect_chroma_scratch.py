#!/usr/bin/env python3
"""
Temporary scratch script: open a local Chroma persist directory, list collections,
and print a small sample of documents + metadata.

Not imported by the app. Run from repo root, e.g.:
  python3 scripts/inspect_chroma_scratch.py
  python3 scripts/inspect_chroma_scratch.py --path ./logs/chroma_db --collection VIDEO_DNA_FULL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import chromadb


def _try_load_dotenv() -> None:
    """Pick up CHROMA_* from .env if pydantic-settings is available."""
    try:
        from pydantic_settings import BaseSettings, SettingsConfigDict

        class _Dot(BaseSettings):
            model_config = SettingsConfigDict(
                env_file=str(Path(__file__).resolve().parents[1] / ".env"),
                env_file_encoding="utf-8",
                extra="ignore",
            )
            chroma_persist_path: str = ""
            chroma_narrative_collection: str = ""

        d = _Dot()
        if d.chroma_persist_path and not os.environ.get("CHROMA_PERSIST_PATH"):
            os.environ["CHROMA_PERSIST_PATH"] = d.chroma_persist_path
        if d.chroma_narrative_collection and not os.environ.get("CHROMA_NARRATIVE_COLLECTION"):
            os.environ["CHROMA_NARRATIVE_COLLECTION"] = d.chroma_narrative_collection
    except Exception:
        pass


def _short(text: str, max_len: int) -> str:
    t = text.replace("\n", "\\n")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def main() -> int:
    _try_load_dotenv()
    ap = argparse.ArgumentParser(description="Peek into a Chroma persistent store (scratch / debug).")
    ap.add_argument(
        "--path",
        default="",
        help="Chroma persist directory (default: env CHROMA_PERSIST_PATH)",
    )
    ap.add_argument(
        "--collection",
        default="",
        help="Only show this collection (default: env CHROMA_NARRATIVE_COLLECTION or all)",
    )
    ap.add_argument("--limit", type=int, default=5, help="Max documents to fetch per collection")
    ap.add_argument("--chars", type=int, default=400, help="Max characters of each document preview")
    args = ap.parse_args()

    persist = (args.path or os.environ.get("CHROMA_PERSIST_PATH", "")).strip()
    if not persist:
        print("ERROR: set CHROMA_PERSIST_PATH or pass --path", file=sys.stderr)
        return 1
    p = Path(persist)
    if not p.is_dir():
        print(f"ERROR: not a directory: {p}", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(p.resolve()))
    names = [c.name for c in client.list_collections()]
    names.sort()
    print(f"Persist: {p.resolve()}")
    print(f"Collections ({len(names)}): {', '.join(names) if names else '(none)'}\n")

    want = (args.collection or os.environ.get("CHROMA_NARRATIVE_COLLECTION", "")).strip()
    todo = [want] if want else names

    for name in todo:
        if name not in names:
            print(f"--- {name!r} --- (not in this store)\n")
            continue
        coll = client.get_collection(name=name)
        try:
            n = coll.count()
        except Exception as e:
            n = f"? ({e})"
        print(f"=== {name} ===  count={n}")
        try:
            batch = coll.get(include=["documents", "metadatas"], limit=max(1, args.limit))
        except Exception as e:
            print(f"  get() failed: {e}\n")
            continue
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, doc_id in enumerate(ids):
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else None
            preview = _short(doc or "", args.chars)
            meta_s = json.dumps(meta, ensure_ascii=False) if meta else "{}"
            if len(meta_s) > 300:
                meta_s = meta_s[:299] + "…"
            print(f"  [{i + 1}] id={doc_id!r}")
            print(f"       doc: {preview!r}")
            print(f"       meta: {meta_s}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
