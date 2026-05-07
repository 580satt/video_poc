#!/usr/bin/env python3
"""Verify MongoDB writes without running the LLM pipeline.

From project root, with dependencies installed (``pip install -e .``):

  python scripts/test_mongo_persist.py

Uses the same Settings path and MongoRunStore as the API. Removes the probe doc when done.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ModuleNotFoundError:
    print("Missing dependency: motor. From project root run: pip install -e .", file=sys.stderr)
    sys.exit(1)

from app.config import get_settings
from app.db.mongo import MongoRunStore


async def main_async() -> None:
    s = get_settings()
    print("Settings: USE_MONGODB =", s.use_mongodb)
    print("Settings: MONGODB_DB =", s.mongodb_db)
    print("Settings: MONGODB_COLLECTION =", s.mongodb_collection)

    if not s.use_mongodb:
        print("FAIL: use_mongodb is False — set USE_MONGODB=true in .env")
        sys.exit(1)

    store = MongoRunStore()
    await store.connect()
    rid = f"mongo_probe_{uuid.uuid4().hex[:12]}"
    await store.update_run(
        rid,
        {
            "status": "probe",
            "note": "scripts/test_mongo_persist.py (safe to delete)",
        },
    )
    doc = await store.get_run(rid)
    await store.close()

    if not doc or doc.get("status") != "probe" or doc.get("run_id") != rid:
        print("FAIL: read back unexpected document:", doc)
        sys.exit(1)

    client = AsyncIOMotorClient(s.mongodb_uri, serverSelectionTimeoutMS=15_000)
    try:
        await client.admin.command("ping")
        print("OK: ping succeeded")
        db = client[s.mongodb_db]
        cols = sorted(await db.list_collection_names())
        print("Collections in", repr(s.mongodb_db) + ":", cols)
        if s.mongodb_collection not in cols:
            print(
                "WARN: collection",
                repr(s.mongodb_collection),
                "not in list (probe was still readable via store).",
            )
        doc2 = await db[s.mongodb_collection].find_one({"run_id": rid})
        if not doc2:
            print("FAIL: probe document not found in DB after async read succeeded")
            sys.exit(1)
        print("OK: probe document present in", s.mongodb_collection, "run_id =", rid)
        res = await db[s.mongodb_collection].delete_one({"run_id": rid})
        print("Cleanup: deleted", res.deleted_count, "document(s)")
    finally:
        client.close()

    print("All checks passed.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
