# Storybook ad pipeline

Python service: product brief → LLM template → creative story (review loop) → scene JSON (review loop) → images (review loop) → **AWS S3** for frames + run metadata in **MongoDB** + ZIP export. **LangGraph** orchestration, **Gemini** (text, vision, image gen), **Motor/MongoDB**, **boto3/S3**.

## Setup

1. Python 3.11+

2. Install dependencies (from the project root):

   ```bash
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`, `MONGODB_URI`, and S3: `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` (or use instance/credential provider and leave access keys empty when appropriate).

   If MongoDB returns **"Command createIndexes requires authentication"**, your server requires a user. Set something like:
   `MONGODB_URI=mongodb://USER:PASSWORD@host:27017/?authSource=admin` (adjust `authSource` to what your server uses, often `admin` or the database name). For Atlas, use the full SRV string from the UI.

4. Start MongoDB locally or point `MONGODB_URI` to Atlas.

5. Run the API:

   ```bash
   uvicorn app.api.main:app --host 0.0.0.0 --port 8000
   ```

6. Open `http://localhost:8000` for the static UI, or use `POST /api/runs` (multipart form) to create a run, then `GET /api/runs/{run_id}` to poll.

## Notable environment variables

- `GEMINI_TEXT_MODEL`, `GEMINI_REVIEWER_MODEL`, `GEMINI_IMAGE_MODEL` — model names for your API tier. Use an **image** model for `GEMINI_IMAGE_MODEL` (e.g. `gemini-2.5-flash-image`); plain `gemini-2.5-flash` is text-only and will produce no real frames (only a tiny placeholder).
- **S3:** `S3_BUCKET` (required for saving frames), `S3_KEY_PREFIX` (default `storybook`), `S3_PUBLIC_BASE_URL` (optional, e.g. CloudFront or static website base; if unset and `S3_USE_PRESIGNED_URLS=false`, a virtual-hosted `https://{bucket}.s3.{region}.amazonaws.com/...` URL is stored), `S3_USE_PRESIGNED_URLS` (default `true` — presigned GET URLs, refreshed on `GET /api/runs/…` when `S3_REFRESH_URL_ON_GET=true`), `S3_PRESIGN_EXPIRES_SECONDS`, `S3_ENDPOINT_URL` (for MinIO). Standard `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` are read by the app and passed to boto3.
- `MAX_*_REVISIONS` — max automated review loops per stage (default 10)
- `TARGET_RUNTIME_SECONDS` / `TARGET_RUNTIME_MAX_SECONDS` — default 8s / 10s pacing

## API

- `POST /api/runs` — form: `product_name`, `brand`, `goal`, `target_audience`, `notes`, optional `target_runtime_seconds`, `target_runtime_max_seconds`, optional `product_image` file
- `GET /api/runs/{run_id}` — run document (no raw image bytes in Mongo; each `images[]` item includes `url`, `s3_key`, `s3_bucket`; presigned `url` may be refreshed on each read when using private buckets)
- `GET /api/runs/{run_id}/story` — approved story only (409 until Reviewer1 passes)
- `GET /api/runs/{run_id}/scenes` — approved scenes only (409 until Reviewer2 passes)
- `GET /api/runs/{run_id}/download` — ZIP of JSON + images

`GET /api/runs/{id}` includes **`step_status`**: for each of `template`, `script`, `scenes`, `images` the value is `pending` | `in_progress` | `complete` | `error`, so the UI can show whether that stage is still running or done.
- `POST /api/runs/{run_id}/regenerate` — JSON `{"from_step": "script"|"scenes"|"images"}`

## Development

- Artifacts: `ARTIFACTS_DIR` (default `./artifacts`).

The pipeline runs in `BackgroundTasks` after you create a run; poll `GET` until `status` is `complete` or `failed`.
