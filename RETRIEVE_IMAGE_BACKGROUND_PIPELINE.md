# `retrieve_image_background` Pipeline

## Overview

`retrieve_image_background.py` is an evidence-first image grounding pipeline.

It:

1. loads an image from a public URL, local path, base64 string, or `data:` URL
2. extracts observable evidence from the image
3. collects candidate pages with reverse image search and OCR-informed web search
4. verifies candidate pages by checking for the same or a near-identical embedded image
5. returns a grounded JSON result with source status, contextual summary fields, and debug metadata

The pipeline is intentionally strict: text similarity alone is not enough to claim a source.

## Public Interface

### Python

```python
from retrieve_image_background import LlmSummaryConfig
from retrieve_image_background import retrieve_image_background

result = retrieve_image_background(
    image="https://example.com/chart.png",
    user_query="Find the original source page for this image.",
    llm_summary_config=LlmSummaryConfig(
        enable_llm_context_summary=True,
        likely_context_enabled=True,
        concise_overview_enabled=True,
        summary_model="google/gemini-3.1-flash-lite-preview",
        timeout_seconds=20,
    ),
)
```

Function signature:

```python
retrieve_image_background(
    image: str,
    user_query: str = "",
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> Dict[str, Any]
```

### CLI

Basic usage:

```bash
python retrieve_image_background.py --image "https://example.com/chart.png"
```

With optional LLM summary overrides:

```bash
python retrieve_image_background.py \
  --image "https://example.com/chart.png" \
  --user-query "Find the original source page for this image." \
  --enable-llm-summary \
  --enable-likely-context \
  --enable-concise-overview \
  --llm-summary-model "google/gemini-3.1-flash-lite-preview" \
  --llm-summary-timeout 20
```

Current CLI flags:

- `--image` required
- `--user-query`
- `--enable-llm-summary` / `--disable-llm-summary`
- `--enable-page-summary` / `--disable-page-summary`
- `--enable-likely-context` / `--disable-likely-context`
- `--enable-concise-overview` / `--disable-concise-overview`
- `--llm-summary-model`
- `--llm-summary-timeout`

Conflicting enable/disable pairs are rejected with a CLI error.

## Pipeline Stages

### Stage 1: Image Evidence Extraction

Source of truth: the image itself.

Extracted fields include:

- `visible_text`
- `likely_title`
- `chart_figure_type`
- `axes_labels`
- `legend_items`
- `units`
- `key_entities`
- `numbers_percentages_dates`
- `logos_watermarks_branding`
- `visual_description`

Implementation:

- primary path uses OpenRouter vision with structured JSON output
- default vision model: `google/gemini-3.1-flash-lite-preview`
- optional local OCR via `pytesseract`

Fallback behavior:

- if `OPENROUTER_API_KEY` is missing, the stage falls back to OCR-only behavior and conservative placeholders
- OCR failure does not stop the pipeline

### Stage 2: Reverse Image Search

Implementation:

- uses SerpApi Google Lens when `SERPAPI_API_KEY` is available
- for public image URLs, uses the original URL directly
- for local paths, base64, and `data:` URLs, uploads the image to GCS, generates a signed URL, submits that URL to SerpApi, then deletes the object

Candidate fields collected here are later normalized into `CandidateSource`.

If reverse image search is unavailable or fails, the rest of the pipeline still runs.

### Stage 3: OCR-Informed Web Search

Implementation:

- builds deterministic search queries from extracted title, visible text, axes, entities, numbers, and branding
- runs Brave web search when `BRAVE_SEARCH_API_KEY` is available

If Brave search is unavailable, this stage returns no candidates and the pipeline continues.

### Stage 4: Candidate Verification and Source Decision

This is the main anti-hallucination stage.

For each candidate page, the pipeline:

- fetches page HTML
- parses page title, visible text, embedded image URLs, and common date metadata
- downloads candidate page images
- computes perceptual hashes
- compares candidate images against the input image
- computes evidence-token overlap against page text

Source decision rules:

- embedded-image visual match is the primary gate
- text overlap helps rank candidates but cannot claim a source by itself
- a page without a verified matching embedded image cannot become the source

Current decision outputs:

- `exact_source_found`
- `possible_source`
- `reverse_match_found_but_not_page_verified`
- `source_not_found`

### Stage 5: Deterministic Summary Layer

The script builds three grounded summary-related fields from cleaned evidence:

- `candidate.page_summary_text`
- `likely_context`
- `concise_overview`

Current behavior:

- page text is cleaned and stripped of obvious boilerplate
- sentence selection is evidence-aware and deterministic
- `likely_context` is the fuller contextual background
- `concise_overview` is the clearest direct explanation of the image itself

### Optional LLM Summary Rewrite Layer

The script can optionally rewrite the three summary fields above through OpenRouter:

- `candidate.page_summary_text`
- `likely_context`
- `concise_overview`

Important constraints reflected in code:

- deterministic output remains the baseline
- the LLM only sees compact cleaned grounded inputs
- the LLM does not control source verification, source status, source link, match type, or confidence
- any failure falls back to deterministic output

## Dependencies

### Required Python packages

- `requests`
- `Pillow`
- `imagehash`

### Optional Python packages

- `pytesseract`
- `google-cloud-storage`

### External services

- OpenRouter for Stage 1 vision extraction and optional summary rewriting
- SerpApi Google Lens for reverse image search
- Brave Search for OCR-informed web search

## Environment Variables

### Vision extraction

- `OPENROUTER_API_KEY`
- `OPENROUTER_VISION_MODEL` optional, default `google/gemini-3.1-flash-lite-preview`
- `OPENROUTER_BASE_URL` optional
- `OPENROUTER_HTTP_REFERER` optional
- `OPENROUTER_APP_TITLE` optional

### Reverse image search

- `SERPAPI_API_KEY`

### Brave web search

- `BRAVE_SEARCH_API_KEY`
- `BRAVE_SEARCH_COUNTRY` optional
- `BRAVE_SEARCH_LANG` optional

### GCS for local-image reverse search

- `GOOGLE_CLOUD_PROJECT`
- `TEMP_IMAGE_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS`

### Optional LLM summary rewrite

- `ENABLE_LLM_CONTEXT_SUMMARY`
- `LLM_PAGE_SUMMARY_ENABLED`
- `LLM_LIKELY_CONTEXT_ENABLED`
- `LLM_CONCISE_OVERVIEW_ENABLED`
- `OPENROUTER_SUMMARY_MODEL` optional, default `google/gemini-3.1-flash-lite-preview`
- `LLM_SUMMARY_TIMEOUT_SECONDS` optional, default `20`

## LLM Summary Configuration

The optional summary rewrite layer is disabled by default.

### Baseline behavior

- environment variables are the default baseline
- explicit overrides can be passed by CLI or `LlmSummaryConfig`

### Precedence

The script resolves summary settings with this precedence:

1. explicit override
2. environment variable
3. hardcoded default

### Gating rules

- `ENABLE_LLM_CONTEXT_SUMMARY` is the global master switch
- scoped flags only matter if the global switch is effectively enabled
- missing `OPENROUTER_API_KEY` disables optional summary rewriting even if the flags request it

### Override object

```python
LlmSummaryConfig(
    enable_llm_context_summary: Optional[bool] = None,
    page_summary_enabled: Optional[bool] = None,
    likely_context_enabled: Optional[bool] = None,
    concise_overview_enabled: Optional[bool] = None,
    summary_model: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
)
```

`None` means no explicit override.

## GCS Flow for Local-Image Reverse Search

Local paths, base64 strings, and `data:` URLs do not have a public URL that SerpApi can fetch.

Current runtime flow:

1. load the image into memory
2. upload it to `gs://$TEMP_IMAGE_BUCKET/reverse-search/<sha256>.<ext>`
3. generate a short-lived signed URL through `google-cloud-storage`
4. submit that signed URL to SerpApi Google Lens
5. delete the uploaded object in cleanup

Practical setup:

1. Go to Google Cloud Console and create or select a GCP project.
2. Enable billing for that project if required by your account setup.
3. Enable the Cloud Storage API for the project.
4. Create a private Cloud Storage bucket to hold temporary reverse-search uploads.
   Recommended settings:
   - private bucket
   - uniform bucket-level access enabled
   - public access prevention enabled
5. Create a service account in the same project for this pipeline.
6. Grant that service account storage permissions sufficient to:
   - upload objects
   - delete objects
   - sign URLs
7. Create and download a JSON key for that service account.
8. Install the Python dependency:

```bash
pip install google-cloud-storage
```

9. Export the required environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export TEMP_IMAGE_BUCKET="your-private-temp-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

`GOOGLE_APPLICATION_CREDENTIALS` must point to the downloaded service-account JSON key. The pipeline uses that credential to upload temporary images, delete them afterward, and generate signed URLs for SerpApi.

Notes:

- Public image URLs do not need this GCS path.
- For local paths, base64 strings, and `data:` URLs, reverse image search depends on this setup.
- If the service account cannot sign URLs or lacks storage access, reverse image search for non-public images will be skipped.

If GCS upload or signed URL generation fails, reverse image search for non-public images is skipped gracefully.

## OCR Setup

If you want local OCR fallback:

```bash
pip install pytesseract
```

Linux:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Windows:

- install Tesseract OCR
- ensure `tesseract.exe` is on `PATH`, or configure `pytesseract.pytesseract.tesseract_cmd`

## Output Contract

The tool returns one JSON object.

Core top-level fields:

```json
{
  "image_description": "",
  "visible_text": [],
  "key_entities": [],
  "image_type": "",
  "candidate_sources": [],
  "raw_reverse_image_candidates": [],
  "source_status": "",
  "most_likely_source": null,
  "source_link": null,
  "source_evidence": "",
  "source_match_type": "",
  "likely_context": "",
  "concise_overview": "",
  "confidence": 0.0,
  "debug_info": {}
}
```

`debug_info` currently includes:

- `generated_queries`
- `ocr`
- `reverse_image_search`
- `llm_summary`
- `reasoning_notes`

Important output rules:

- `source_status` is one of:
  - `exact_source_found`
  - `possible_source`
  - `reverse_match_found_but_not_page_verified`
  - `source_not_found`
- `source_link` is only populated when code allows a source candidate
- `raw_reverse_image_candidates` preserves raw reverse-search hits even when verification fails
- `likely_context` is contextual background
- `concise_overview` is the clearest explanation of the image itself

### `debug_info.llm_summary`

When present, this contains safe metadata about the optional summary rewrite layer, including:

- whether LLM summary rewriting was requested and enabled
- whether an API key was present
- configured model name
- per-scope usage, fallback, and last error

It does not include API keys or hidden reasoning.

## Failure Behavior

The pipeline is designed to degrade gracefully.

Examples:

- no `OPENROUTER_API_KEY`: Stage 1 falls back to OCR-only placeholders; optional summary rewriting is disabled
- no `SERPAPI_API_KEY`: reverse image search is skipped
- no `BRAVE_SEARCH_API_KEY`: OCR-informed web search is skipped
- missing GCS config for non-public images: reverse image search is skipped for those inputs
- page fetch blocked: candidate cannot be fully verified
- invalid candidate image payload: that candidate image is skipped safely
- optional LLM summary rewrite failure: deterministic summary output is kept

## Practical Examples

Basic public-image run:

```bash
python retrieve_image_background.py --image "https://example.com/chart.png"
```

Local-image run after GCS setup:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export TEMP_IMAGE_BUCKET="your-private-temp-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export SERPAPI_API_KEY="..."
python retrieve_image_background.py --image "reddit/mis_fig/example.png"
```

Env-driven optional LLM summary rewrite:

```bash
export OPENROUTER_API_KEY="..."
export ENABLE_LLM_CONTEXT_SUMMARY="1"
export LLM_PAGE_SUMMARY_ENABLED="1"
export LLM_LIKELY_CONTEXT_ENABLED="1"
export LLM_CONCISE_OVERVIEW_ENABLED="1"
export OPENROUTER_SUMMARY_MODEL="google/gemini-3.1-flash-lite-preview"
export LLM_SUMMARY_TIMEOUT_SECONDS="20"
python retrieve_image_background.py --image "https://example.com/chart.png"
```
