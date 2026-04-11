## Scripts

### `retrieve_image_background.py`

- Reusable multimodal retrieval and grounding pipeline for images.
- Accepts an image as a public URL, `data:` URL, base64 string, or local path.
- Runs a staged process:
  - Stage 1: image-local evidence extraction
  - Stage 2: reverse image search, including temporary GCS upload for local images
  - Stage 3: OCR-based web search
  - Stage 4: source verification through embedded-image visual matching
  - Stage 5: grounded JSON output assembly
- Uses deterministic evidence cleaning and summary generation by default.
- Can optionally apply an OpenRouter-based LLM rewrite layer on top of the deterministic summaries for:
  - `candidate.page_summary_text`
  - `likely_context`
  - `concise_overview`
- Uses a strict rule that a source page must contain the same or a near-identical embedded image; text-only similarity is not enough.
- Returns `source_not_found` when no visually verified source is found.
- Preserves raw reverse-image candidates in the JSON output even when Stage 4 cannot verify a page-level embedded image match.
- Exposes structured `debug_info.reverse_image_search` details for skipped reverse-search attempts and GCS signed-URL failures.
- Exposes structured `debug_info.llm_summary` details for optional summary rewrite usage, fallback, and model configuration.
- Optional environment variables:
  - `OPENROUTER_API_KEY` for vision-based evidence extraction through OpenRouter
  - `SERPAPI_API_KEY` for reverse image search through SerpApi Google Lens
  - `BRAVE_SEARCH_API_KEY` for OCR-based web search through Brave Search
  - `GOOGLE_CLOUD_PROJECT`, `TEMP_IMAGE_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` for temporary GCS uploads and signed URLs when reverse-searching local, base64, or `data:` images
  - `ENABLE_LLM_CONTEXT_SUMMARY` to enable optional LLM summary rewriting
  - `OPENROUTER_SUMMARY_MODEL` to override the summary rewrite model
  - `LLM_SUMMARY_TIMEOUT_SECONDS` to override summary request timeout
  - `LLM_PAGE_SUMMARY_ENABLED`, `LLM_LIKELY_CONTEXT_ENABLED`, `LLM_CONCISE_OVERVIEW_ENABLED` to scope the optional LLM rewrite layer
- If you run the pipeline with a local image, base64 image, or `data:` URL, you must configure GCS. The pipeline uploads the image to a private bucket, creates a signed URL, uses that signed URL for SerpApi Google Lens, and then deletes the uploaded object.
- For local-image reverse search, create a service account in the Google Cloud project, create a JSON key for that service account, download it locally, and export it with `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"`.
- ADC is used for Google Cloud access through `google-cloud-storage`; no separate GCS API key is required.
- Strongly recommended OCR setup:
  - install the Python wrapper with `pip install pytesseract`
  - install the Tesseract OCR engine itself
  - on Linux: `sudo apt update` then `sudo apt install tesseract-ocr`
  - on Windows: install Tesseract OCR and ensure the executable is on `PATH`
- Install Python dependencies as needed, including `pip install google-cloud-storage`.
- Run: `python retrieve_image_background.py --image <url|data-url|base64|local-path>`
- Optional CLI overrides:
  - `--enable-llm-summary` / `--disable-llm-summary`
  - `--enable-page-summary` / `--disable-page-summary`
  - `--enable-likely-context` / `--disable-likely-context`
  - `--enable-concise-overview` / `--disable-concise-overview`
  - `--llm-summary-model <model>`
  - `--llm-summary-timeout <seconds>`
- Config precedence for the optional summary rewrite layer:
  - explicit CLI override or function-call config
  - environment variable
  - hardcoded default
- Python callers can pass `llm_summary_config=LlmSummaryConfig(...)` into `retrieve_image_background(...)` for the same overrides without relying on CLI flags.
- Detailed design and limitations: `RETRIEVE_IMAGE_BACKGROUND_PIPELINE.md`

### `reddit_scraper.py`

- Scrapes `r/dataisugly` via Reddit’s public JSON endpoints (no API keys), iterating multiple sorts/time windows: `new`, `hot`, `rising`, `best`, `top` (`day/week/month/year/all`), and `controversial` (`day/week/month/year/all`).
- Downloads `.jpg/.jpeg/.png` images, computes perceptual hashes, and skips duplicates using a 80% similarity threshold; resumes numbering from existing files.
- Skips posts already recorded in `OUTPUT_CSV` when the image file exists.
- Defaults: `POST_LIMIT=2000`, `SUBREDDIT="dataisugly"`, `OUTPUT_DIR="reddit/mis_fig"`, `OUTPUT_CSV="reddit/data_is_ugly_output.csv"`, `SIMILARITY_THRESHOLD=0.9`.
- Adds lightweight retries per page fetch and requests raw JSON to reduce entity decoding problems; shows progress with `tqdm` and interleaves logs through the bar.
- Run: `python reddit_scraper.py` (ensure `requests`, `Pillow`, `imagehash`, `tqdm`).

### `comments_scraper.py`

- Reads posts metadata from `POSTS_CSV` (default `reddit/data_is_ugly_output.csv`) and fetches full comment trees from Reddit’s public JSON (`/comments/{post_id}.json`).
- Handles rate limits and flaky responses with retry + exponential backoff; adds a small delay between posts for politeness.
- Flattens nested replies into one row per comment with ids, parent ids, depth, path, author, timestamp, and body fields.
- Appends to `COMMENTS_CSV` (default `reddit/data_is_ugly_comments.csv`) while deduplicating by `comment_id`; tracks finished posts in `PROGRESS_CSV` (default `reddit/comments_scrape_progress.csv`) so runs are resumable.
- Run: `python comments_scraper.py` (requires `requests`).
