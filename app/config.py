from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load `.env` from the project root (parent of `app/`), not from the process CWD — otherwise
# starting uvicorn from another directory leaves USE_MONGODB unset and runs stay in-memory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    # When False, the API uses in-process storage only (no MongoDB; survives until process exit).
    use_mongodb: bool = False
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "ads_scraper_db"
    # Single collection in MONGODB_DB: run document + `pipeline_outputs` array (final snapshot, etc.).
    mongodb_collection: str = Field(
        default="video_ad_pipeline",
        validation_alias=AliasChoices("mongodb_collection", "mongodb_outputs_collection"),
    )

    # Story + scene JSON (when story_scenes_llm_provider=gemini); template + reviewers also use gemini_text_model where applicable
    gemini_text_model: str = "gemini-2.5-flash"
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
    # If empty, RAG is skipped. Exact name on disk, e.g. MAGIC_IMAGE_DATA_ADA_V1, VIDEO_DNA, VIDEO_DNA_FULL
    chroma_narrative_collection: str = "VIDEO_DNA_FULL"
    # Created by the app; vector dim must match embedding output (typically 1536 for ada-002 / text-embedding-3-small@1536)
    chroma_review_collection: str = "REVIEWER_FEEDBACK"
    rag_top_k: int = 5
    rag_enabled: bool = True
    # Embeddings: use Azure OpenAI (below) or direct OpenAI — must match the model used to build each Chroma collection
    # Azure OpenAI embeddings (e.g. ada-002) — if endpoint + key + deployment are set, this is used; no public OpenAI key needed
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""  # e.g. https://your-resource.openai.azure.com/
    azure_openai_api_version: str = "2023-05-15"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    # Optional: direct OpenAI API (public) — only used when Azure block above is not fully configured
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    # Chroma / review upserts: expected embedding length (name is legacy; applies to Azure ada-002 and OpenAI 3-small@1536)
    openai_embedding_dimensions: int = 1536
    reviewer_min_rating: int = 4
    # How many high-rated past reviews to inject into the reviewer prompt
    review_memory_top_k: int = 3

    # Story + scene-by-scene JSON: Gemini or Claude (OpenAI-compatible WaveSpeed API). Images always use gemini_image_model.
    story_scenes_llm_provider: str = "gemini"  # gemini | wavespeed
    wavespeed_api_key: str = ""
    wavespeed_base_url: str = "https://llm.wavespeed.ai/v1"
    wavespeed_model: str = "anthropic/claude-opus-4.7"
    # Longer story/scene JSON: avoid low default caps on WaveSpeed chat
    wavespeed_max_output_tokens: int = 16384
    # Gemini text generation output budget (story + scene JSON); higher = room for long 15–30s scripts
    gemini_max_output_tokens: int = 8192
    # Image generation: require live-action / photographic look (set false only if you want illustrated storybook art)
    image_photorealistic: bool = True


def get_settings() -> Settings:
    return Settings()
