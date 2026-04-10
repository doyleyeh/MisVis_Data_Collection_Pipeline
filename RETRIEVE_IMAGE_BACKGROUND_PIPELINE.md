# `retrieve_image_background` Pipeline

## Purpose

`retrieve_image_background` is a reusable multimodal retrieval and grounding tool for image-centric agent pipelines. Its job is to inspect an input image, extract observable evidence, search for likely source pages, verify whether candidate pages actually embed the same or a near-identical image, and return a strict JSON result that prefers `source_not_found` over unsupported claims.

The design goal is not generic image captioning. It is evidence-first retrieval with explicit guardrails against hallucinated sources.

## End-to-End Workflow

1. Load the image from a public URL, `data:` URL, raw base64 string, or local file path.
2. Extract image-local evidence only:
   - visible text
   - likely title
   - chart or figure type
   - labels, legend items, units
   - entities, numbers, dates, branding
   - short neutral visual description
3. Run reverse image search when `SERPAPI_API_KEY` is available and the pipeline can obtain a reachable image URL. Public URLs are used directly; local, base64, and `data:` inputs are uploaded to a private GCS bucket and exposed through a short-lived signed URL.
4. Generate OCR-informed web queries, prioritizing extracted visible text from the image.
5. Run Brave web search for those queries and collect candidate pages.
6. Fetch candidate pages, parse embedded image URLs, and compare candidate page images to the input image using perceptual hashing.
7. Decide whether an exact, near-exact, possible, or no verified source exists.
8. Return a grounded JSON payload for downstream AI agents.

## Pipeline Stages

### Stage 1: Image Evidence Extraction

This stage extracts evidence directly from the image and avoids unsupported inference.

Primary path:

- Uses the OpenRouter Chat Completions API with structured outputs.
- Default vision model: `google/gemini-3.1-flash-lite-preview`
- Sends the image as a base64 `data:` URL.
- Instructs the model to avoid guessing missing text and to preserve uncertainty.

Fallback path:

- Uses optional local Tesseract OCR if `pytesseract` is installed.
- If no vision model is configured, the tool returns conservative placeholders instead of inventing content.

Why this stage is needed:

- It gives later search stages concrete evidence to work from.
- It separates observation from inference at the start of the pipeline.
- It reduces bad search queries caused by invented labels or titles.

### Stage 2: Reverse Image Search

This stage gathers candidate pages from reverse image search.

Current implementation:

- Uses SerpApi Google Lens when:
  - `SERPAPI_API_KEY` is set
  - the input image is already public, or the pipeline can create a temporary signed URL through Google Cloud Storage

For non-public image inputs:

- the image is uploaded to the private bucket named by `TEMP_IMAGE_BUCKET`
- the object name is deterministic: `reverse-search/<sha256>.<ext>`
- a short-lived signed URL is generated with ADC through `google-cloud-storage`
- that signed URL is sent to SerpApi Google Lens
- the uploaded object is deleted afterward with `try/finally`

Collected fields per candidate:

- URL
- title
- snippet/source text when available
- whether it is likely original, repost, or commentary

Why this stage is needed:

- Reverse image search is the fastest path to visually similar hosting pages.
- It is especially valuable when visible OCR text is sparse or absent.

Failure behavior:

- If GCS upload or signed URL generation fails, reverse image search is skipped gracefully.
- If SerpApi fails, reverse image search returns no candidates.
- The rest of the pipeline still runs rather than fabricating a reverse-search result.

### Stage 3: OCR-Based Web Search

This stage creates 3 to 5 deterministic search queries from Stage 1 evidence.

Query construction sources:

- quoted title fragments
- quoted visible text lines
- entities plus numbers or dates
- publisher or branding hints
- chart type plus key entities

Current implementation:

- Uses Brave Search Web Search.
- Sends the extracted visible text and related OCR evidence as search queries.

Why this stage is needed:

- It recovers pages missed by reverse image search.
- It helps when an image is discussed alongside distinctive text on the page.

### Stage 4: Source Verification and Exact-Match Decision

This is the main anti-hallucination stage.

For each candidate page, the tool:

1. Fetches the page HTML.
2. Parses:
   - page title
   - visible text
   - embedded image URLs
   - common publication-date metadata
3. Downloads embedded images found in:
   - `og:image`
   - `twitter:image`
   - `<img src=...>`
   - common lazy-load image attributes
4. Computes perceptual hashes for candidate page images.
5. Compares them to the input image hash.
6. Scores text overlap between page text and extracted image evidence.

Invalid downloaded candidate images are skipped safely. A bad `Content-Type` header or undecodable image payload cannot crash the pipeline.

Decision priorities:

- Visual similarity is the primary gate.
- Text overlap cannot produce a source claim by itself.
- A page without a verified embedded visual match cannot become the source.

Thresholds used:

- `exact`: perceptual similarity `>= 0.85`
- `near_exact`: perceptual similarity `>= 0.80`
- `possible`: perceptual similarity `>= 0.70` with supporting evidence
- otherwise: no verified match

Decision outputs:

- `exact_source_found`
- `possible_source`
- `reverse_match_found_but_not_page_verified`
- `source_not_found`

The tool prefers:

- original publisher pages over reposts
- pages with embedded image matches over text-only similarity
- earlier plausible publication pages when available

### Stage 5: Grounded Summary

The final stage builds the required JSON response.

It includes:

- neutral image description
- visible text list
- key entities
- candidate source list
- source status
- most likely source and link when allowed
- source evidence
- likely context
- concise grounded overview
- confidence
- debug notes

Why this stage is needed:

- Downstream AI agents need structured outputs, not free-form prose.
- The JSON keeps retrieval evidence and confidence visible for later decision-making.

## APIs and Tools Used

### Required Python Libraries

- `requests`
- `Pillow`
- `imagehash`

### Optional Python Libraries

- `pytesseract`
- `google-cloud-storage`

### External APIs

#### OpenRouter API

Used for Stage 1 image evidence extraction.

Environment variables:

- `OPENROUTER_API_KEY`
- optional `OPENROUTER_BASE_URL`
- optional `OPENROUTER_VISION_MODEL`
- optional `OPENROUTER_HTTP_REFERER`
- optional `OPENROUTER_APP_TITLE`

#### SerpApi

Used for:

- Stage 2 reverse image search via Google Lens

Environment variable:

- `SERPAPI_API_KEY`

#### Google Cloud Storage

Used for:

- Stage 2 temporary uploads for local file paths, raw base64 inputs, and `data:` URLs

Environment variables:

- `GOOGLE_CLOUD_PROJECT`
- `TEMP_IMAGE_BUCKET`
- `GOOGLE_APPLICATION_CREDENTIALS`

Authentication:

- ADC is used through `google-cloud-storage`
- no separate GCS API key is needed
- for local-image signed URLs, use a service-account JSON key via `GOOGLE_APPLICATION_CREDENTIALS`

#### Local OCR

Used for:

- Stage 1 visible-text extraction when you want a local OCR fallback

Required setup:

- install the Python wrapper with `pip install pytesseract`
- install the Tesseract OCR engine itself

Recommended installation:

- Linux:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

- Windows:
  install Tesseract OCR and ensure the executable is on `PATH`, or configure `pytesseract.pytesseract.tesseract_cmd` explicitly if needed

## Google Cloud Setup for Local Reverse Search

This feature exists because SerpApi Google Lens needs a reachable image URL. Local files, base64 strings, and `data:` URLs do not have one, so the pipeline temporarily uploads those inputs to a private GCS bucket, signs a short-lived URL, uses that URL for reverse image search, and then deletes the object.

Important requirement:

- If you run the pipeline with a local image path, raw base64, or a `data:` URL, you must configure GCS for Stage 2 reverse image search.
- Public image URLs do not need the temporary GCS upload path.

### 1. Create or Use a Google Cloud Account

- Sign in with an existing Google account or create one at Google Cloud.
- If the account is new to Google Cloud, complete the initial account setup in the Cloud Console.

### 2. Create a Project and Enable Billing

- Create a new Google Cloud project in the Cloud Console, or pick an existing one dedicated to this pipeline.
- Enable billing for that project because Cloud Storage and signed-URL access depend on an active billable project.

### 3. Enable the Cloud Storage API

- In the Cloud Console for the selected project, enable the Cloud Storage API before using the bucket from local development.

### 4. Create a Service Account for Signed URLs

- In the same Google Cloud project, create a service account dedicated to this pipeline.
- Grant it the minimum storage permissions needed to upload and delete temporary objects in the bucket.
- Create a JSON key for that service account and download it to your local machine.
- Keep the JSON key private and do not commit it into source control.

Why this is needed:

- Local-image reverse search depends on generating signed URLs.
- Token-only user ADC can authenticate API calls but may fail to sign URLs.
- A service-account JSON key provides the private key material required for signed URL generation.

### 5. Install and Initialize the Google Cloud CLI Locally

Install the Google Cloud CLI for your operating system, then run:

```bash
gcloud init
gcloud auth application-default login
```

Notes:

- `gcloud init` selects the active account and project for local development.
- `gcloud auth application-default login` creates the ADC credentials used by `google-cloud-storage`.
- No custom GCS API key is required for this pipeline.
- Even with ADC configured, local-image signed URL generation should use the service-account JSON key exported through `GOOGLE_APPLICATION_CREDENTIALS`.

### 6. Create a Bucket for Temporary Reverse-Search Uploads

Recommended bucket settings:

- private bucket
- Region location type
- Standard storage class
- Uniform access control enabled
- Public access prevention set to ON
- Google-managed encryption
- object versioning disabled
- no retention lock

Soft delete considerations:

- If soft delete is enabled in your org, deleted temporary objects may remain recoverable for a period and still count toward storage usage.
- For a pure temporary-upload bucket, review soft delete retention carefully so cleanup behavior matches your cost and recovery requirements.

### 7. Install Python Dependencies

For GCS-backed local reverse search:

```bash
pip install google-cloud-storage
```

For local OCR:

```bash
pip install pytesseract
```

### 8. Configure Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export TEMP_IMAGE_BUCKET="your-private-temp-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export SERPAPI_API_KEY="..."
```

`GOOGLE_CLOUD_PROJECT` selects the project used by ADC-backed `google-cloud-storage`.

`TEMP_IMAGE_BUCKET` is the private bucket that stores temporary reverse-search uploads.

`GOOGLE_APPLICATION_CREDENTIALS` must point to the downloaded service-account JSON key used to sign GCS URLs for local-image reverse search.

If you also use Stage 1 local OCR, ensure both the Python wrapper and the Tesseract engine are installed:

```bash
pip install pytesseract
sudo apt update
sudo apt install tesseract-ocr
```

On Windows, install Tesseract OCR and ensure `tesseract.exe` is available on `PATH`.

## Minimum Local Environment Setup

Recommended local setup for the full pipeline:

```bash
pip install requests Pillow imagehash google-cloud-storage pytesseract
```

Then configure:

- OpenRouter credentials if you want vision-model extraction
- SerpApi credentials if you want reverse image search
- Brave Search credentials if you want OCR-based web search
- GCS bucket plus service-account JSON key if you want local-image reverse search
- Tesseract OCR if you want local OCR fallback

## Runtime Flow for Local Images

1. The pipeline loads the local file, base64 input, or `data:` URL into memory.
2. If the image already has a public URL, that URL is reused and no GCS upload occurs.
3. Otherwise, the image is uploaded to `gs://$TEMP_IMAGE_BUCKET/reverse-search/<sha256>.<ext>`.
4. A short-lived signed URL is generated using the service-account credentials pointed to by `GOOGLE_APPLICATION_CREDENTIALS`.
5. The signed URL is sent to SerpApi Google Lens.
6. The uploaded object is deleted afterward, even if reverse image search fails.

## Failure Behavior

- If `google-cloud-storage` is not installed, reverse image search for local images is skipped gracefully.
- If `GOOGLE_CLOUD_PROJECT` or `TEMP_IMAGE_BUCKET` is missing, reverse image search for local images is skipped gracefully.
- If `GOOGLE_APPLICATION_CREDENTIALS` is missing or points to credentials without a private key, signed URL generation for local images may fail.
- If GCS upload or signed URL generation fails, reverse image search is skipped gracefully.
- If `pytesseract` is installed but the `tesseract` executable is missing from the machine or `PATH`, local OCR fails and the pipeline records the OCR failure in `debug_info.ocr`.
- If a candidate page returns invalid image bytes, that candidate image is skipped safely during verification.
- None of these failures should crash the full retrieval pipeline.

#### Brave Search API

Used for:

- Stage 3 OCR-based web search via Brave Web Search

Environment variables:

- `BRAVE_SEARCH_API_KEY`
- optional `BRAVE_SEARCH_COUNTRY`
- optional `BRAVE_SEARCH_LANG`

## Source Verification Logic

The source verification rules are intentionally strict.

A page can only be treated as a source if:

- the page contains the same or a near-identical embedded image
- and the page is a plausible original host or at least a strong source candidate

Textual similarity alone is never enough.

Role classification heuristics:

- `repost`: social and repost-heavy domains such as Reddit, X/Twitter, Pinterest, Imgur
- `commentary`: titles or snippets that look analytical, reactive, or discussion-oriented
- `original source`: default for pages that are not obvious reposts or commentary

These role heuristics influence ranking, but they do not override the visual verification gate.

## JSON Output Schema

The tool returns:

```json
{
  "image_description": "",
  "visible_text": [],
  "key_entities": [],
  "image_type": "",
  "candidate_sources": [
    {
      "url": "",
      "title": "",
      "evidence_type": "reverse_image_search | web_search",
      "match_strength": "high | medium | low",
      "is_probable_original": true,
      "notes": ""
    }
  ],
  "raw_reverse_image_candidates": [
    {
      "url": "",
      "title": "",
      "snippet": "",
      "result_role": "original source | commentary | repost",
      "notes": ""
    }
  ],
  "source_status": "exact_source_found | possible_source | reverse_match_found_but_not_page_verified | source_not_found",
  "most_likely_source": null,
  "source_link": null,
  "source_evidence": "",
  "source_match_type": "exact | near_exact | possible | none",
  "likely_context": "",
  "concise_overview": "",
  "confidence": 0.0,
  "debug_info": {
    "generated_queries": [],
    "reverse_image_search": {},
    "reasoning_notes": ""
  }
}
```

Field notes:

- `most_likely_source` is the candidate title when a source is found, otherwise `null`.
- `source_link` is only populated for `exact_source_found` and `possible_source`.
- `raw_reverse_image_candidates` preserves unverified Google Lens hits even if Stage 4 cannot confirm a page-level embedded image match.
- `reverse_match_found_but_not_page_verified` means reverse image search surfaced candidate pages, but none passed page-level verification.
- `debug_info.reverse_image_search` exposes why Stage 2 was skipped or failed, including GCS signed-URL setup failures.
- `source_evidence` begins with the required source assessment sentence, then adds the verification rationale.

## Failure Cases, Uncertainty Handling, and Limitations

### Failure Cases

- OpenRouter unavailable:
  - image description and higher-level visual extraction may be unavailable
- SerpApi unavailable:
  - reverse image search returns no candidates
- GCS upload or signed URL generation unavailable:
  - reverse image search for non-public image inputs returns no candidates
- Brave Search unavailable:
  - OCR web search returns no candidates
- page fetch blocked:
  - candidate remains listed, but cannot be promoted to verified source
- client-rendered pages:
  - if the image is injected only after JavaScript runs, static HTML verification may miss it
- invalid downloaded candidate image:
  - that image is skipped rather than crashing Stage 4 verification

### Uncertainty Handling

- OCR uncertainty is preserved rather than resolved by guesswork.
- When evidence is weak, the tool lowers confidence and keeps the source status conservative.
- If no matching image is verified on candidate pages, the result is `source_not_found`.

### Known Limitations

- The original-source vs commentary classification is heuristic.
- Perceptual hashing is strong for exact and near-exact copies, but it can miss aggressive crops or edits.
- Without a browser renderer, some dynamic sites cannot be fully verified.
- The implementation currently uses one provider per search role rather than provider fallbacks.
- Temporary local-image reverse search depends on working ADC, Cloud Storage access, and a bucket configuration that allows upload plus signed URL generation.

## Suggested Future Improvements

- Add browser-based rendering for pages that lazy-load or client-render images.
- Support multiple search providers with a clean backend interface.
- Add crop-aware or local-feature visual matching for harder near-duplicate cases.
- Persist candidate-page fetch artifacts for auditability and offline review.
- Add unit tests for:
  - input loading
  - query generation
  - candidate deduplication
  - source decision thresholds
  - output-schema compliance

## Tool Usage

### Tool Name

`retrieve_image_background`

### Input Parameters

```json
{
  "image": "string",
  "user_query": "string"
}
```

Parameter notes:

- `image` is required.
- `image` may be:
  - a public image URL
  - a local file path
  - a base64 string
  - a `data:` URL
- `user_query` is optional and can be used to bias Stage 1 extraction toward the caller's question without relaxing the evidence-only rules.

### Example Tool Call

```json
{
  "image": "https://example.com/chart.png",
  "user_query": "Find the original source page for this image."
}
```

### Example Python Usage

```python
from retrieve_image_background import retrieve_image_background

result = retrieve_image_background(
    image="https://example.com/chart.png",
    user_query="Find the original source page for this image.",
)
```

### Environment Variables

- `OPENROUTER_API_KEY`: required for vision-based evidence extraction
- `SERPAPI_API_KEY`: required for reverse image search
- `GOOGLE_CLOUD_PROJECT`: required for temporary GCS uploads when the input image is not already public
- `TEMP_IMAGE_BUCKET`: required for temporary GCS uploads when the input image is not already public
- `GOOGLE_APPLICATION_CREDENTIALS`: strongly recommended for local-image reverse search so signed URLs can be generated with a service-account private key
- `BRAVE_SEARCH_API_KEY`: required for OCR-based web search
- `OPENROUTER_VISION_MODEL`: optional override for the default OpenRouter model
- `OPENROUTER_BASE_URL`: optional OpenRouter endpoint override
- `OPENROUTER_HTTP_REFERER`: optional OpenRouter attribution header
- `OPENROUTER_APP_TITLE`: optional OpenRouter attribution header
- `BRAVE_SEARCH_COUNTRY`: optional Brave search country, default `us`
- `BRAVE_SEARCH_LANG`: optional Brave search language, default `en`

### Output Contract

The tool returns one JSON object matching the schema described above.

Important output rules:

- `source_status` is one of `exact_source_found`, `possible_source`, `reverse_match_found_but_not_page_verified`, or `source_not_found`
- `source_link` is only populated for `exact_source_found` and `possible_source`
- `source_not_found` is returned when no candidate page contains a verified matching embedded image
- `reverse_match_found_but_not_page_verified` is returned when reverse image search finds raw candidates but Stage 4 cannot verify any page as embedding the same or a near-identical image
- `candidate_sources` may include search hits that were investigated but not verified as the final source
- `raw_reverse_image_candidates` always preserves the raw reverse-search hits collected from SerpApi Google Lens

## Local Usage

```bash
python retrieve_image_background.py --image "https://example.com/chart.png"
```

```bash
python retrieve_image_background.py --image "reddit/mis_fig/000001_ffa08055552a7eaa.png"
```

Optional environment setup:

```bash
export OPENROUTER_API_KEY="..."
export SERPAPI_API_KEY="..."
export BRAVE_SEARCH_API_KEY="..."
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export TEMP_IMAGE_BUCKET="your-private-temp-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
python retrieve_image_background.py --image "https://example.com/chart.png"
```

Example local-image run after GCS setup:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export TEMP_IMAGE_BUCKET="your-private-temp-bucket"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export SERPAPI_API_KEY="..."
python retrieve_image_background.py --image "reddit/mis_fig/001100_bf839715c06be01d.jpeg"
```

Example local OCR setup on Linux:

```bash
pip install pytesseract
sudo apt update
sudo apt install tesseract-ocr
python retrieve_image_background.py --image "reddit/mis_fig/001100_bf839715c06be01d.jpeg"
```
