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
- Uses a strict rule that a source page must contain the same or a near-identical embedded image; text-only similarity is not enough.
- Returns `source_not_found` when no visually verified source is found.
- Preserves raw reverse-image candidates in the JSON output even when Stage 4 cannot verify a page-level embedded image match.
- Exposes structured `debug_info.reverse_image_search` details for skipped reverse-search attempts and GCS signed-URL failures.
- Optional environment variables:
  - `OPENROUTER_API_KEY` for vision-based evidence extraction through OpenRouter
  - `SERPAPI_API_KEY` for reverse image search through SerpApi Google Lens
  - `BRAVE_SEARCH_API_KEY` for OCR-based web search through Brave Search
  - `GOOGLE_CLOUD_PROJECT` and `TEMP_IMAGE_BUCKET` for temporary GCS uploads when reverse-searching local, base64, or `data:` images
- Local, base64, and `data:` image inputs now use a private Google Cloud Storage upload plus a short-lived signed URL so they can reuse the same SerpApi Google Lens flow as public URLs.
- ADC is used for Google Cloud access through `google-cloud-storage`; no separate GCS API key is required.
- Install Python dependencies as needed, including `pip install google-cloud-storage`.
- Run: `python retrieve_image_background.py --image <url|data-url|base64|local-path>`
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
