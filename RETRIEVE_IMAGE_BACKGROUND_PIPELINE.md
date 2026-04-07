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
3. Run reverse image search when a public image URL and `SERPAPI_API_KEY` are available.
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
  - the input image is already a public URL

Collected fields per candidate:
- URL
- title
- snippet/source text when available
- whether it is likely original, repost, or commentary

Why this stage is needed:
- Reverse image search is the fastest path to visually similar hosting pages.
- It is especially valuable when visible OCR text is sparse or absent.

Important limitation:
- Reverse image search is skipped for local files, base64 strings, and `data:` URLs unless the caller first makes the image publicly reachable.
- The pipeline continues rather than fabricating a reverse-search result.

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

Decision priorities:
- Visual similarity is the primary gate.
- Text overlap cannot produce a source claim by itself.
- A page without a verified embedded visual match cannot become the source.

Thresholds used:
- `exact`: perceptual similarity `>= 0.98`
- `near_exact`: perceptual similarity `>= 0.93`
- `possible`: perceptual similarity `>= 0.88` with supporting evidence
- otherwise: no verified match

Decision outputs:
- `exact_source_found`
- `possible_source`
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
  "source_status": "exact_source_found | possible_source | source_not_found",
  "most_likely_source": null,
  "source_link": null,
  "source_evidence": "",
  "source_match_type": "exact | near_exact | possible | none",
  "likely_context": "",
  "concise_overview": "",
  "confidence": 0.0,
  "debug_info": {
    "generated_queries": [],
    "reasoning_notes": ""
  }
}
```

Field notes:
- `most_likely_source` is the candidate title when a source is found, otherwise `null`.
- `source_link` is only populated for `exact_source_found` and `possible_source`.
- `source_evidence` begins with the required source assessment sentence, then adds the verification rationale.

## Failure Cases, Uncertainty Handling, and Limitations

### Failure Cases

- OpenRouter unavailable:
  - image description and higher-level visual extraction may be unavailable
- SerpApi unavailable:
  - reverse image search returns no candidates
- Brave Search unavailable:
  - OCR web search returns no candidates
- page fetch blocked:
  - candidate remains listed, but cannot be promoted to verified source
- client-rendered pages:
  - if the image is injected only after JavaScript runs, static HTML verification may miss it
- non-public image input:
  - reverse image search cannot run without a publicly reachable image URL

### Uncertainty Handling

- OCR uncertainty is preserved rather than resolved by guesswork.
- When evidence is weak, the tool lowers confidence and keeps the source status conservative.
- If no matching image is verified on candidate pages, the result is `source_not_found`.

### Known Limitations

- The original-source vs commentary classification is heuristic.
- Perceptual hashing is strong for exact and near-exact copies, but it can miss aggressive crops or edits.
- Without a browser renderer, some dynamic sites cannot be fully verified.
- The implementation currently uses one provider per search role rather than provider fallbacks.

## Suggested Future Improvements

- Add an image uploader adapter so local/base64 inputs can also run reverse image search.
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
- `source_status` is one of `exact_source_found`, `possible_source`, or `source_not_found`
- `source_link` is only populated for `exact_source_found` and `possible_source`
- `source_not_found` is returned when no candidate page contains a verified matching embedded image
- `candidate_sources` may include search hits that were investigated but not verified as the final source

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
python retrieve_image_background.py --image "https://example.com/chart.png"
```
