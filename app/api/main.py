from __future__ import annotations

import io
import json
import logging
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db.mongo import get_run_store
from app.storage.s3_images import enrich_run_image_records, get_object_bytes
from app.graph.builder import new_run_id, run_full_pipeline, run_regenerate
from app.models.schemas import RegenerateRequest

logger = logging.getLogger(__name__)

_store = get_run_store()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _store.connect()
    s = get_settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    yield
    await _store.close()


app = FastAPI(title="Storybook Ad Pipeline", lifespan=lifespan)


def _run_dict(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    d = {**doc}
    if "_id" in d and d["_id"] is not None:
        d["_id"] = str(d["_id"])
    for k in list(d.keys()):
        if k.endswith("bytes") or k == "bytes":
            d.pop(k, None)
    return d


async def _pipeline_task(
    run_id: str,
    raw: dict,
    tr: float,
    tmax: float,
    image_bytes: bytes | None,
    image_mime: str | None,
) -> None:
    try:
        s = get_settings()
        await run_full_pipeline(
            run_id=run_id,
            raw_user_input=raw,
            target_runtime_seconds=tr,
            target_runtime_max_seconds=tmax,
            product_image=image_bytes,
            product_image_mime=image_mime,
            settings=s,
            store=_store,
        )
    except Exception as e:
        logger.exception("Pipeline failed for %s", run_id)
        await _store.update_run(run_id, {"status": "failed", "error_detail": str(e)})


async def _regen_task(run_id: str, from_step: str) -> None:
    try:
        s = get_settings()
        await run_regenerate(run_id=run_id, from_step=from_step, settings=s, store=_store)  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("Regenerate failed for %s", run_id)
        await _store.update_run(run_id, {"status": "failed", "error_detail": str(e)})


@app.post("/api/runs", status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
    product_name: str = Form(""),
    brand: str = Form(""),
    goal: str = Form(""),
    target_audience: str = Form(""),
    notes: str = Form(""),
    target_runtime_seconds: Optional[str] = Form(None),
    target_runtime_max_seconds: Optional[str] = Form(None),
    product_image: UploadFile | None = File(None),
):
    s = get_settings()
    tr = float(target_runtime_seconds) if target_runtime_seconds not in (None, "") else s.target_runtime_seconds
    tmax = float(target_runtime_max_seconds) if target_runtime_max_seconds not in (None, "") else s.target_runtime_max_seconds
    rid = new_run_id()
    raw: dict = {
        "product_name": product_name,
        "brand": brand,
        "goal": goal,
        "target_audience": target_audience,
        "notes": notes,
    }
    im_bytes: bytes | None = None
    im_mime: str | None = None
    if product_image and product_image.filename:
        im_bytes = await product_image.read()
        im_mime = product_image.content_type or "image/png"
    background_tasks.add_task(
        _pipeline_task,
        rid,
        raw,
        tr,
        tmax,
        im_bytes,
        im_mime,
    )
    await _store.update_run(
        rid,
        {
            "status": "running",
            "raw_input": raw,
            "target_runtime_seconds": tr,
            "target_runtime_max_seconds": tmax,
            "step_status": {
                "template": "in_progress",
                "script": "pending",
                "scenes": "pending",
                "images": "pending",
            },
        },
    )
    return {"run_id": rid, "status": "running"}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    doc = await _store.get_run(run_id)
    if not doc:
        raise HTTPException(404, "Run not found")
    d = _run_dict(doc)
    return enrich_run_image_records(d)


@app.get("/api/runs/{run_id}/story")
async def get_story_approved(run_id: str):
    doc = await _store.get_run(run_id)
    if not doc:
        raise HTTPException(404, "Run not found")
    if not doc.get("story_approved") or not doc.get("story"):
        raise HTTPException(409, "Approved story is not available yet, or the run failed before script approval")
    return {
        "run_id": run_id,
        "story": doc.get("story", ""),
        "script_revisions": doc.get("script_revisions", 0),
    }


@app.get("/api/runs/{run_id}/script", response_class=RedirectResponse)
async def get_script_alias(run_id: str):
    return RedirectResponse(url=f"/api/runs/{run_id}/story", status_code=307)


@app.get("/api/runs/{run_id}/scenes")
async def get_scenes_approved(run_id: str):
    doc = await _store.get_run(run_id)
    if not doc:
        raise HTTPException(404, "Run not found")
    if not doc.get("scenes_approved"):
        if doc.get("status") == "failed":
            raise HTTPException(409, "Scenes not approved; run may have failed earlier")
        raise HTTPException(409, "Scenes not yet approved")
    return {
        "run_id": run_id,
        "scenes": doc.get("scenes"),
        "scenes_revisions": doc.get("scenes_revisions", 0),
    }


@app.get("/api/runs/{run_id}/download")
async def download_run(run_id: str):
    doc = await _store.get_run(run_id)
    if not doc:
        raise HTTPException(404, "Run not found")
    s = get_settings()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        run_for_export = enrich_run_image_records(_run_dict(doc))
        z.writestr("run.json", json.dumps(run_for_export or {}, indent=2, default=str))
        if doc.get("product_template"):
            z.writestr("product_template.json", json.dumps(doc["product_template"], indent=2, default=str))
        if doc.get("story"):
            z.writestr("story.txt", str(doc.get("story", "")))
        if doc.get("scenes"):
            z.writestr("scenes.json", json.dumps(doc["scenes"], indent=2, default=str))
        for im in doc.get("images") or []:
            if not isinstance(im, dict):
                continue
            key = im.get("s3_key")
            bucket = im.get("s3_bucket") or s.s3_bucket
            if not key or not bucket:
                continue
            b = get_object_bytes(s, str(bucket), str(key))
            if b:
                z.writestr(f"image_scene_{im.get('scene_index', 0)}.png", b)
        z.writestr(
            "README.txt",
            "Storybook ad pipeline export. story.txt = approved script, scenes.json = approved scene list, run.json = run metadata.\n",
        )
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.zip"'},
    )


def _step_status_for_regen(from_step: str) -> dict[str, str]:
    if from_step == "script":
        return {
            "template": "complete",
            "script": "in_progress",
            "scenes": "pending",
            "images": "pending",
        }
    if from_step == "scenes":
        return {
            "template": "complete",
            "script": "complete",
            "scenes": "in_progress",
            "images": "pending",
        }
    if from_step == "images":
        return {
            "template": "complete",
            "script": "complete",
            "scenes": "complete",
            "images": "in_progress",
        }
    return {
        "template": "in_progress",
        "script": "pending",
        "scenes": "pending",
        "images": "pending",
    }


@app.post("/api/runs/{run_id}/regenerate", status_code=202)
async def regenerate(run_id: str, body: RegenerateRequest, background_tasks: BackgroundTasks):
    doc = await _store.get_run(run_id)
    if not doc:
        raise HTTPException(404, "Run not found")
    from_step = body.from_step
    await _store.update_run(
        run_id,
        {
            "status": "running",
            "error_detail": None,
            "step_status": _step_status_for_regen(from_step),
        },
    )
    background_tasks.add_task(_regen_task, run_id, from_step)
    return {"run_id": run_id, "status": "running", "from_step": from_step}


_ROOT = Path(__file__).resolve().parent.parent.parent
_static = _ROOT / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def index():
    p = _static / "index.html"
    if p.is_file():
        return FileResponse(p)
    return HTMLResponse(
        "<h1>Storybook ad pipeline</h1><p>Add static/index.html in the project <code>static</code> folder.</p>"
    )


if __name__ == "__main__":
    import uvicorn

    c = get_settings()
    uvicorn.run("app.api.main:app", host=c.api_host, port=c.api_port, reload=False)
