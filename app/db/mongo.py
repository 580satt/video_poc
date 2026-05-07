from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunStore(Protocol):
    async def connect(self) -> None: ...
    async def update_run(self, run_id: str, patch: dict[str, Any]) -> None: ...
    async def get_run(self, run_id: str) -> Optional[dict[str, Any]]: ...
    async def insert_pipeline_output(self, run_id: str, kind: str, payload: dict[str, Any]) -> None: ...
    async def close(self) -> None: ...


def _apply_patch_for_in_memory(existing: dict[str, Any], patch: dict[str, Any]) -> None:
    """Merge patch into existing run doc.

    Pipeline code uses Mongo dotted keys from `step_status_patch` (e.g. ``step_status.template``).
    Motor stores those as nested fields; a plain dict.update() would add a *literal* key
    ``'step_status.template'`` and leave ``step_status`` stale — so the UI never left
    "in progress". We merge dotted keys into ``step_status`` here.
    """
    now = _utcnow()
    incoming = {**patch, "updated_at": now}
    dotted: dict[str, Any] = {}
    normal: dict[str, Any] = {}
    for k, v in incoming.items():
        if k.startswith("step_status."):
            dotted[k[len("step_status.") :]] = v
        else:
            normal[k] = v
    for k, v in normal.items():
        if k == "step_status":
            continue
        existing[k] = v
    cur = existing.get("step_status")
    cur = cur if isinstance(cur, dict) else {}
    if "step_status" in normal and isinstance(normal["step_status"], dict):
        existing["step_status"] = {**cur, **normal["step_status"], **dotted}
    elif dotted:
        existing["step_status"] = {**cur, **dotted}


class InMemoryRunStore:
    """Dev / no-DB: keeps run state in process memory (lost on restart)."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        return

    async def update_run(self, run_id: str, patch: dict[str, Any]) -> None:
        if run_id not in self._runs:
            self._runs[run_id] = {"run_id": run_id, "created_at": _utcnow()}
        _apply_patch_for_in_memory(self._runs[run_id], patch)

    async def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        doc = self._runs.get(run_id)
        return None if doc is None else dict(doc)

    async def insert_pipeline_output(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        if run_id not in self._runs:
            self._runs[run_id] = {"run_id": run_id, "created_at": _utcnow()}
        now = _utcnow()
        cur = self._runs[run_id].get("pipeline_outputs")
        cur_list = list(cur) if isinstance(cur, list) else []
        cur_list.append({"kind": kind, "payload": dict(payload), "created_at": now})
        self._runs[run_id]["pipeline_outputs"] = cur_list
        self._runs[run_id]["updated_at"] = now

    async def close(self) -> None:
        return


class MongoRunStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    def _collection_name(self) -> str:
        n = (self._settings.mongodb_collection or "").strip()
        return n or "video_ad_pipeline"

    def _runs(self) -> Any:
        return self.db[self._collection_name()]

    async def connect(self) -> None:
        if self._client is None:
            self._client = AsyncIOMotorClient(self._settings.mongodb_uri)
            self._db = self._client[self._settings.mongodb_db]
            try:
                coll = self._runs()
                await coll.create_index("run_id", unique=True)
                await coll.create_index("created_at")
            except ServerSelectionTimeoutError as e:
                err = str(e).lower()
                if "ssl" in err or "tls" in err or "handshake" in err:
                    raise RuntimeError(
                        "MongoDB TLS handshake failed before the driver could connect. This is almost always a "
                        "network or Atlas allowlist issue (not a wrong password).\n"
                        "  1. In MongoDB Atlas: Network Access → add your current public IP, or 0.0.0.0/0 for "
                        "local dev only (then tighten later).\n"
                        "  2. Ensure the cluster is running (not paused) and MONGODB_URI is the SRV string "
                        "from Atlas (Connect → Drivers).\n"
                        "  3. If you use a VPN or strict firewall, try without VPN or allow outbound to "
                        "port 27017 to *.mongodb.net.\n"
                        f"Original error: {e}"
                    ) from e
                raise
            except OperationFailure as e:
                if e.code == 13:  # Unauthorized
                    raise RuntimeError(
                        "MongoDB requires authentication. Set MONGODB_URI in .env with a user and password, "
                        "for example:\n"
                        "  MONGODB_URI=mongodb://USER:PASSWORD@localhost:27017/?authSource=admin\n"
                        "Use your actual username, password, and authSource (often 'admin' or the DB name). "
                        "MONGODB_DB selects the database (e.g. ads_scraper_db); MONGODB_COLLECTION names the "
                        "collection for run documents.\n"
                        "For MongoDB Atlas, use the SRV string from the Atlas UI (it includes credentials).\n"
                        f"Original error: {e}"
                    ) from e
                raise

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("MongoRunStore not connected")
        return self._db

    async def update_run(self, run_id: str, patch: dict[str, Any]) -> None:
        now = _utcnow()
        p = {**patch, "updated_at": now}
        await self._runs().update_one(
            {"run_id": run_id},
            {
                "$set": p,
                "$setOnInsert": {"run_id": run_id, "created_at": now},
            },
            upsert=True,
        )

    async def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        return await self._runs().find_one({"run_id": run_id})

    async def insert_pipeline_output(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        now = _utcnow()
        entry = {"kind": kind, "payload": payload, "created_at": now}
        try:
            result = await self._runs().update_one(
                {"run_id": run_id},
                {"$push": {"pipeline_outputs": entry}, "$set": {"updated_at": now}},
            )
            if result.matched_count == 0:
                logger.warning(
                    "MongoDB pipeline_outputs: no run document for run_id=%s (push skipped)",
                    run_id,
                )
        except Exception as e:
            logger.warning("MongoDB pipeline_outputs append failed (run_id=%s kind=%s): %s", run_id, kind, e)

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


_store: Optional[RunStore] = None


def get_run_store() -> RunStore:
    global _store
    if _store is None:
        s = get_settings()
        _store = MongoRunStore() if s.use_mongodb else InMemoryRunStore()
    return _store
