from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    # When False, the API uses in-process storage only (no MongoDB; survives until process exit).
    use_mongodb: bool = False
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "storybook"

    gemini_text_model: str = "gemini-2.0-flash"
    gemini_reviewer_model: str = "gemini-2.0-flash"
    # Use an image-native model; plain "gemini-2.5-flash" is text-only and will not return image bytes.
    gemini_image_model: str = "gemini-2.5-flash-image"

    # S3: scene frame storage (boto3 also reads standard AWS env vars if these are empty)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "ap-south-1"
    s3_bucket: str = "slm-scraping-data"
    s3_key_prefix: str = "images"
    # e.g. https://mybucket.s3.amazonaws.com or CloudFront base (no trailing slash) — optional; see README
    s3_public_base_url: str = ""
    # If True, stored/returned "url" is a presigned GET. If False, use s3_public_base_url or public virtual-hosted style.
    s3_use_presigned_urls: bool = True
    s3_presign_expires_seconds: int = 604800
    s3_endpoint_url: str = ""  # MinIO / LocalStack, optional
    # When using presigned URLs, regenerate on each GET /api/runs/… so the client always gets a fresh link.
    s3_refresh_url_on_get: bool = True

    max_script_revisions: int = 10
    max_scenes_revisions: int = 10
    max_image_revisions: int = 10

    target_runtime_seconds: float = 8.0
    target_runtime_max_seconds: float = 10.0

    artifacts_dir: Path = Path("artifacts")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Chroma RAG (query existing collections) + review memory (app-managed collection)
    chroma_persist_path: str = ""
    # If empty, RAG is skipped. Use the exact collection name on disk, e.g. MAGIC_IMAGE_DATA_ADA_V1
    chroma_narrative_collection: str = ""
    # Created by the app; embeddings must match openai_embedding_model dimensions (1536 for ada-3-small)
    chroma_review_collection: str = "REVIEWER_FEEDBACK"
    rag_top_k: int = 5
    rag_enabled: bool = True
    # OpenAI embeddings (1536-dim for text-embedding-3-small) — must match your indexed Chroma data
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    reviewer_min_rating: int = 4
    # How many high-rated past reviews to inject into the reviewer prompt
    review_memory_top_k: int = 3


def get_settings() -> Settings:
    return Settings()
