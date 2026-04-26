from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import quote

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _s3_client(settings: Settings) -> BaseClient:
    kwargs: dict[str, Any] = {"region_name": settings.aws_default_region or "us-east-1"}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    return boto3.client("s3", **kwargs)


def _key_to_url_path(key: str) -> str:
    return "/".join(quote(segment, safe="-_.~") for segment in key.split("/"))


def public_http_url_for_key(settings: Settings, s3_key: str) -> str:
    if settings.s3_public_base_url:
        return f"{settings.s3_public_base_url.rstrip('/')}/{_key_to_url_path(s3_key)}"
    bucket = settings.s3_bucket
    region = settings.aws_default_region or "ap-south-1"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{_key_to_url_path(s3_key)}"


def presigned_get_url(settings: Settings, bucket: str, s3_key: str) -> str:
    c = _s3_client(settings)
    return c.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=settings.s3_presign_expires_seconds,
    )


def http_url_for_key(settings: Settings, bucket: str, s3_key: str) -> str:
    if settings.s3_use_presigned_urls:
        return presigned_get_url(settings, bucket, s3_key)
    return public_http_url_for_key(settings, s3_key)


def put_scene_image(
    settings: Settings,
    run_id: str,
    scene_index: int,
    data: bytes,
    content_type: str,
) -> dict[str, Any]:
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET (s3_bucket) is not set; configure AWS/S3 in .env")
    key = f"{settings.s3_key_prefix.strip('/')}/runs/{run_id}/scene_{int(scene_index)}.png"
    c = _s3_client(settings)
    c.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    b = settings.s3_bucket
    url = http_url_for_key(settings, b, key)
    return {
        "scene_index": int(scene_index),
        "s3_key": key,
        "s3_bucket": b,
        "mime_type": content_type,
        "url": url,
    }


async def put_scene_image_async(
    settings: Settings,
    run_id: str,
    scene_index: int,
    data: bytes,
    content_type: str,
) -> dict[str, Any]:
    return await asyncio.to_thread(put_scene_image, settings, run_id, scene_index, data, content_type)


def get_object_bytes(settings: Settings, bucket: str, s3_key: str) -> Optional[bytes]:
    c = _s3_client(settings)
    try:
        o = c.get_object(Bucket=bucket, Key=s3_key)
    except ClientError as e:
        logger.warning("S3 get_object failed: %s", e)
        return None
    return o["Body"].read()


def enrich_run_image_records(doc: dict[str, Any] | None, settings: Optional[Settings] = None) -> dict[str, Any] | None:
    if not doc:
        return None
    s = settings or get_settings()
    d = {**doc}
    imgs = d.get("images")
    if not isinstance(imgs, list):
        return d
    out_imgs: list[dict] = []
    for im in imgs:
        if not isinstance(im, dict):
            out_imgs.append(im)
            continue
        m = {**im}
        key = m.get("s3_key")
        bucket = m.get("s3_bucket") or s.s3_bucket
        if s.s3_refresh_url_on_get and key and bucket and s.s3_use_presigned_urls:
            try:
                m["url"] = presigned_get_url(s, str(bucket), str(key))
            except ClientError as e:
                logger.warning("Presign failed for %s: %s", key, e)
        out_imgs.append(m)
    d["images"] = out_imgs
    return d
