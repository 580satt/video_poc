#!/usr/bin/env python3
"""Inspect one full Chroma row from VIDEO_DNA_FULL (or CHROMA_NARRATIVE_COLLECTION).

By default prints the **complete document text** and **full metadata** in a readable layout.
The embedding vector is **not** dumped to the terminal (it hides the text); fetch it only when needed.

Uses project-root `.env` for `CHROMA_PERSIST_PATH` / `CHROMA_NARRATIVE_COLLECTION`.

  pip install -e .
  python scripts/rag_one_video_dna_full.py
  python scripts/rag_one_video_dna_full.py --format json
  python scripts/rag_one_video_dna_full.py --format json --with-embedding   # huge JSON
  python scripts/rag_one_video_dna_full.py --embedding-out /tmp/emb.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import chromadb
except ModuleNotFoundError:
    print("Missing chromadb. From project root: pip install -e .", file=sys.stderr)
    sys.exit(1)

from app.config import get_settings


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Show one full Chroma record: complete document text + metadata (embedding optional).",
    )
    ap.add_argument("--path", default="", help="Override CHROMA_PERSIST_PATH")
    ap.add_argument("--collection", default="", help="Override CHROMA_NARRATIVE_COLLECTION")
    ap.add_argument("--index", type=int, default=0, help="Row offset (0 = first). Loads index+1 rows internally.")
    ap.add_argument("--id", default="", help="Fetch this Chroma id exactly.")
    ap.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="text = human-readable full document (default); json = one JSON object",
    )
    ap.add_argument(
        "--with-embedding",
        action="store_true",
        help="With --format json: include the full embedding array in stdout (very large).",
    )
    ap.add_argument(
        "--embedding-out",
        default="",
        help="Write the full embedding JSON array to this file (implies fetching embeddings).",
    )
    ap.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Deprecated: default behavior already avoids embeddings unless --with-embedding or --embedding-out.",
    )
    args = ap.parse_args()

    s = get_settings()
    persist = (args.path or str(s.chroma_persist_path or "")).strip()
    name = (args.collection or s.chroma_narrative_collection or "VIDEO_DNA_FULL").strip()

    if not persist:
        print("ERROR: CHROMA_PERSIST_PATH is empty. Set it in .env or pass --path", file=sys.stderr)
        return 1

    p = Path(persist)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.is_dir():
        print(f"ERROR: Chroma persist path is not a directory: {p}", file=sys.stderr)
        return 1

    emb_out_set = bool((args.embedding_out or "").strip())
    if args.skip_embedding and emb_out_set:
        print("WARN: --embedding-out needs vectors; ignoring --skip-embedding", file=sys.stderr)
    want_json_emb = bool(args.with_embedding and args.format == "json")
    if args.skip_embedding and not emb_out_set:
        want_emb = False
    else:
        want_emb = want_json_emb or emb_out_set

    include: list[str] = ["documents", "metadatas"]
    if want_emb:
        include.append("embeddings")

    client = chromadb.PersistentClient(path=str(p.resolve()))
    names = {c.name for c in client.list_collections()}
    if name not in names:
        print(f"ERROR: collection {name!r} not found. Available: {sorted(names)}", file=sys.stderr)
        return 1

    coll = client.get_collection(name=name)
    total = coll.count()
    if total == 0:
        print(f"ERROR: collection {name!r} is empty", file=sys.stderr)
        return 1

    id_arg = (args.id or "").strip()
    if id_arg:
        batch = coll.get(ids=[id_arg], include=include)
        ids = batch.get("ids") or []
        if not ids:
            print(f"ERROR: no row with id {id_arg!r} in {name!r}", file=sys.stderr)
            return 1
        rid = ids[0]
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        emb = batch.get("embeddings")
        doc = docs[0] if docs else None
        meta = metas[0] if metas else None
        vec = None
        if emb is not None and len(emb) > 0 and emb[0] is not None:
            vec = list(emb[0])
        idx_note = None
    else:
        if args.index < 0 or args.index >= total:
            print(f"ERROR: --index {args.index} out of range (count={total})", file=sys.stderr)
            return 1
        batch = coll.get(limit=args.index + 1, include=include)
        ids = batch.get("ids") or []
        if len(ids) <= args.index:
            print(
                f"ERROR: expected at least {args.index + 1} rows, got {len(ids)} (count={total})",
                file=sys.stderr,
            )
            return 1
        rid = ids[args.index]
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        emb = batch.get("embeddings")
        doc = docs[args.index] if args.index < len(docs) else None
        meta = metas[args.index] if args.index < len(metas) else None
        vec = None
        if emb is not None and args.index < len(emb) and emb[args.index] is not None:
            vec = list(emb[args.index])
        idx_note = args.index

    out_emb_path = (args.embedding_out or "").strip()
    if out_emb_path and vec is not None:
        outp = Path(out_emb_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(vec), encoding="utf-8")
    elif out_emb_path and vec is None:
        print("WARN: --embedding-out set but no embedding returned; file not written.", file=sys.stderr)

    if args.format == "text":
        print(f"collection: {name}")
        print(f"persist_path: {p.resolve()}")
        print(f"collection_count: {total}")
        print(f"id: {rid}")
        if idx_note is not None:
            print(f"index: {idx_note}")
        print()
        print("DOCUMENT (complete; nothing truncated)")
        print("=" * 72)
        if doc is None:
            sys.stdout.write("(none)\n")
        else:
            text = doc if isinstance(doc, str) else str(doc)
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
        print("=" * 72)
        print()
        print("METADATA (complete JSON)")
        print("=" * 72)
        print(json.dumps(meta if meta is not None else {}, ensure_ascii=False, indent=2))
        print("=" * 72)
        print()
        if vec is not None:
            print(f"embedding: dim={len(vec)}")
        else:
            print("embedding: not fetched (use --embedding-out PATH or --format json --with-embedding)")
        if out_emb_path:
            print(f"embedding: written to {Path(out_emb_path).resolve()}")
        return 0

    payload: dict = {
        "collection": name,
        "persist_path": str(p.resolve()),
        "collection_count": total,
        "id": rid,
        "document": doc,
        "metadata": meta,
    }
    if idx_note is not None:
        payload["index_in_batch"] = idx_note
    if vec is not None:
        payload["embedding_dim"] = len(vec)
        if args.with_embedding:
            payload["embedding"] = vec
        if out_emb_path:
            payload["embedding_file"] = str(Path(out_emb_path).resolve())
    elif want_emb:
        payload["embedding"] = None
        payload["embedding_note"] = "No embedding returned for this row."

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
