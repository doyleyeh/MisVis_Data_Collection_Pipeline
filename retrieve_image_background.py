import argparse
import base64
import binascii
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse

import imagehash
import requests
from PIL import Image, UnidentifiedImageError


USER_AGENT = "retrieve-image-background/0.1"
REQUEST_TIMEOUT_SECONDS = 20
MAX_REVERSE_CANDIDATES = 8
MAX_WEB_CANDIDATES = 10
MAX_PAGE_IMAGES = 12
MAX_VISIBLE_TEXT_LINES = 20
MAX_QUERY_COUNT = 5
DEFAULT_OPENROUTER_VISION_MODEL = "google/gemini-3.1-flash-lite-preview"
SERPAPI_ENDPOINT = "https://serpapi.com/search"
OPENROUTER_CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
EXACT_MATCH_THRESHOLD = 0.98
NEAR_EXACT_MATCH_THRESHOLD = 0.93
POSSIBLE_MATCH_THRESHOLD = 0.88
TEXT_OVERLAP_FOR_POSSIBLE = 0.18
REPOST_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "linkedin.com",
    "www.linkedin.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "pinterest.com",
    "www.pinterest.com",
    "imgur.com",
    "www.imgur.com",
    "tumblr.com",
    "www.tumblr.com",
    "9gag.com",
    "www.9gag.com",
}
COMMENTARY_HINTS = {
    "analysis",
    "opinion",
    "review",
    "commentary",
    "explained",
    "reaction",
    "thread",
    "discussion",
    "reddit",
    "forum",
    "blog",
}
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass
class LoadedImage:
    """Normalized image payload used by all pipeline stages."""

    original_input: str
    source_kind: str
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    sha256: str
    public_url: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass
class ImageEvidence:
    """Stage 1 evidence extracted directly from the image."""

    visible_text: List[str] = field(default_factory=list)
    likely_title: str = ""
    chart_figure_type: str = ""
    axes_labels: List[str] = field(default_factory=list)
    legend_items: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    numbers_percentages_dates: List[str] = field(default_factory=list)
    logos_watermarks_branding: List[str] = field(default_factory=list)
    visual_description: str = ""
    extraction_notes: List[str] = field(default_factory=list)


@dataclass
class CandidateSource:
    """Candidate page collected from reverse image search or OCR-based web search."""

    url: str
    title: str
    evidence_type: str
    snippet: str = ""
    result_role: str = "commentary"
    image_embedded: bool = False
    match_strength: str = "low"
    is_probable_original: bool = False
    notes: str = ""
    published_date: Optional[str] = None
    best_image_match_url: Optional[str] = None
    best_image_similarity: float = 0.0
    text_overlap: float = 0.0
    page_title: str = ""
    page_summary_text: str = ""


class PageParser(HTMLParser):
    """Minimal HTML parser for titles, text, image URLs, and metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.image_urls: List[str] = []
        self.meta_image_urls: List[str] = []
        self.meta_dates: List[str] = []
        self.title_chunks: List[str] = []
        self.text_chunks: List[str] = []
        self._in_title = False
        self._skip_text = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs if key}
        tag = tag.lower()

        if tag == "title":
            self._in_title = True
            return

        if tag in {"script", "style", "noscript"}:
            self._skip_text = True
            return

        if tag == "img":
            for key in ("src", "data-src", "data-lazy-src", "data-original", "data-image"):
                candidate = attr_map.get(key)
                if candidate:
                    self.image_urls.append(candidate.strip())
                    break
            alt_text = attr_map.get("alt")
            if alt_text:
                self.text_chunks.append(alt_text.strip())
            return

        if tag == "meta":
            meta_key = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if not content:
                return
            if meta_key in {"og:image", "twitter:image", "twitter:image:src"}:
                self.meta_image_urls.append(content)
            if meta_key in {
                "article:published_time",
                "og:updated_time",
                "publish-date",
                "date",
                "pubdate",
                "dc.date",
            }:
                self.meta_dates.append(content)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript"}:
            self._skip_text = False

    def handle_data(self, data: str) -> None:
        if self._skip_text:
            return
        cleaned = collapse_whitespace(data)
        if not cleaned:
            return
        if self._in_title:
            self.title_chunks.append(cleaned)
            return
        self.text_chunks.append(cleaned)


def retrieve_image_background(image: str, user_query: str = "") -> Dict[str, Any]:
    """
    Run the full multimodal retrieval pipeline and return the strict JSON payload.

    Stage 1 extracts evidence from the image itself.
    Stage 2 performs reverse image search to collect likely hosting pages.
    Stage 3 generates OCR-informed web queries and collects additional pages.
    Stage 4 verifies candidates by checking whether the same or a near-identical image is
    actually embedded on the candidate page.
    Stage 5 assembles the final grounded JSON and refuses to name a source without a verified
    visual match.
    """

    loaded_image = load_image_input(image)
    evidence = extract_image_evidence(loaded_image, user_query)

    reverse_candidates = reverse_image_search(loaded_image)
    search_queries = generate_ocr_search_queries(evidence)
    web_candidates = collect_web_search_candidates(search_queries)

    combined_candidates = deduplicate_candidates(reverse_candidates + web_candidates)
    verified_candidates = verify_candidate_pages(loaded_image, evidence, combined_candidates)
    source_decision = determine_source_decision(verified_candidates)

    return build_output_json(
        evidence=evidence,
        candidates=verified_candidates,
        source_decision=source_decision,
        search_queries=search_queries,
    )


def load_image_input(image_input: str) -> LoadedImage:
    """
    Normalize the image input into bytes plus metadata.

    This loader accepts:
    - public URLs
    - `data:` URLs
    - raw base64 strings
    - local filesystem paths
    """

    stripped = image_input.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        response = requests.get(
            stripped,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        image_bytes = response.content
        mime_type = response.headers.get("Content-Type", "image/png").split(";")[0].strip() or "image/png"
        width, height = image_dimensions(image_bytes)
        return LoadedImage(
            original_input=image_input,
            source_kind="url",
            image_bytes=image_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            sha256=sha256_hex(image_bytes),
            public_url=stripped,
        )

    if stripped.startswith("data:image/"):
        header, payload = stripped.split(",", 1)
        mime_type = header.split(";")[0].split(":", 1)[1]
        image_bytes = base64.b64decode(payload)
        width, height = image_dimensions(image_bytes)
        return LoadedImage(
            original_input=image_input,
            source_kind="data_url",
            image_bytes=image_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            sha256=sha256_hex(image_bytes),
        )

    maybe_path = Path(stripped)
    if maybe_path.exists():
        image_bytes = maybe_path.read_bytes()
        mime_type = guess_mime_type(maybe_path.suffix)
        width, height = image_dimensions(image_bytes)
        return LoadedImage(
            original_input=image_input,
            source_kind="file_path",
            image_bytes=image_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            sha256=sha256_hex(image_bytes),
            local_path=str(maybe_path.resolve()),
        )

    try:
        image_bytes = base64.b64decode(stripped, validate=True)
    except binascii.Error as exc:
        raise ValueError("Unsupported image input. Expected URL, data URL, base64 string, or local path.") from exc

    width, height = image_dimensions(image_bytes)
    return LoadedImage(
        original_input=image_input,
        source_kind="base64",
        image_bytes=image_bytes,
        mime_type="image/png",
        width=width,
        height=height,
        sha256=sha256_hex(image_bytes),
    )


def extract_image_evidence(loaded_image: LoadedImage, user_query: str) -> ImageEvidence:
    """
    Stage 1: extract only evidence directly observable in the image.

    The primary path uses OpenRouter chat completions with the
    `google/gemini-3.1-flash-lite-preview` model and a strict JSON schema. If no OpenRouter
    credentials are available, the function falls back to optional Tesseract OCR and returns
    conservative placeholders instead of inventing content.
    """

    evidence = ImageEvidence()
    ocr_lines = run_optional_tesseract_ocr(loaded_image.image_bytes)
    if ocr_lines:
        evidence.visible_text = dedupe_preserve_order(ocr_lines)[:MAX_VISIBLE_TEXT_LINES]
        evidence.extraction_notes.append("Visible text includes optional local OCR output.")
    else:
        evidence.extraction_notes.append("Local OCR unavailable or no readable text was extracted.")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        evidence.visual_description = (
            "Visual description unavailable because no vision model is configured in the environment."
        )
        evidence.chart_figure_type = "unknown"
        evidence.extraction_notes.append("OPENROUTER_API_KEY not set; using OCR-only fallback.")
        return evidence

    prompt = build_stage_one_prompt(user_query=user_query)
    payload = {
        "model": os.getenv("OPENROUTER_VISION_MODEL", DEFAULT_OPENROUTER_VISION_MODEL),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": loaded_image.data_url}},
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_evidence",
                "strict": True,
                "schema": evidence_schema(),
            }
        },
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "retrieve_image_background"),
    }
    endpoint = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_CHAT_COMPLETIONS_ENDPOINT).rstrip("/")

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_json = response.json()
        parsed = parse_openrouter_output_json(response_json)
        evidence = merge_evidence(evidence, parsed)
        evidence.extraction_notes.append("Vision extraction completed with a strict JSON schema.")
    except Exception as exc:  # noqa: BLE001
        if not evidence.visual_description:
            evidence.visual_description = (
                "Visual description unavailable because the vision extraction request failed."
            )
        if not evidence.chart_figure_type:
            evidence.chart_figure_type = "unknown"
        evidence.extraction_notes.append(f"Vision extraction failed: {exc}")

    return evidence


def build_stage_one_prompt(user_query: str) -> str:
    """
    Build the strict Stage 1 instruction used by the vision model.

    The prompt emphasizes observation over inference and requires uncertainty to stay explicit.
    """

    user_context = f"User question: {user_query}\n" if user_query else ""
    return (
        "Return JSON only.\n"
        "Extract evidence from the image without guessing missing text.\n"
        f"{user_context}"
        "Rules:\n"
        "- Preserve uncertainty for blurry or partial text by marking the string with '(unclear)'.\n"
        "- Do not infer facts not visible in the image.\n"
        "- Use empty strings or empty arrays when evidence is absent.\n"
        "- visual_description must be a short neutral description of visible content only.\n"
        "- likely_title may be empty if no clear title exists.\n"
        "- chart_figure_type should be specific when visible (for example: bar chart, line chart, infographic, screenshot, map, poster).\n"
    )


def evidence_schema() -> Dict[str, Any]:
    """JSON schema for Stage 1 structured extraction."""

    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "visible_text": string_array,
            "likely_title": {"type": "string"},
            "chart_figure_type": {"type": "string"},
            "axes_labels": string_array,
            "legend_items": string_array,
            "units": string_array,
            "key_entities": string_array,
            "numbers_percentages_dates": string_array,
            "logos_watermarks_branding": string_array,
            "visual_description": {"type": "string"},
        },
        "required": [
            "visible_text",
            "likely_title",
            "chart_figure_type",
            "axes_labels",
            "legend_items",
            "units",
            "key_entities",
            "numbers_percentages_dates",
            "logos_watermarks_branding",
            "visual_description",
        ],
        "additionalProperties": False,
    }


def merge_evidence(existing: ImageEvidence, parsed: Dict[str, Any]) -> ImageEvidence:
    """Merge OCR-first evidence with model output while preserving non-empty fields."""

    model_visible_text = parsed.get("visible_text") or []
    merged_visible_text = dedupe_preserve_order(existing.visible_text + clean_string_list(model_visible_text))
    return ImageEvidence(
        visible_text=merged_visible_text[:MAX_VISIBLE_TEXT_LINES],
        likely_title=clean_string(parsed.get("likely_title", "")),
        chart_figure_type=clean_string(parsed.get("chart_figure_type", "")) or existing.chart_figure_type,
        axes_labels=clean_string_list(parsed.get("axes_labels")),
        legend_items=clean_string_list(parsed.get("legend_items")),
        units=clean_string_list(parsed.get("units")),
        key_entities=clean_string_list(parsed.get("key_entities")),
        numbers_percentages_dates=clean_string_list(parsed.get("numbers_percentages_dates")),
        logos_watermarks_branding=clean_string_list(parsed.get("logos_watermarks_branding")),
        visual_description=clean_string(parsed.get("visual_description", "")) or existing.visual_description,
        extraction_notes=list(existing.extraction_notes),
    )


def run_optional_tesseract_ocr(image_bytes: bytes) -> List[str]:
    """
    Optional OCR fallback used inside Stage 1.

    The tool stays functional without Tesseract. When Tesseract is unavailable, the pipeline
    returns an empty OCR result instead of failing.
    """

    try:
        import pytesseract  # type: ignore
    except ImportError:
        return []

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            raw_text = pytesseract.image_to_string(image)
    except Exception:  # noqa: BLE001
        return []

    lines = [collapse_whitespace(line) for line in raw_text.splitlines()]
    return [line for line in lines if line]


def reverse_image_search(loaded_image: LoadedImage) -> List[CandidateSource]:
    """
    Stage 2: collect candidate pages from reverse image search.

    SerpApi Google Lens is used when both `SERPAPI_API_KEY` and a public image URL are available.
    Reverse image search is intentionally skipped for local/base64-only inputs because Google Lens
    requires a reachable URL. The pipeline records the limitation and continues.
    """

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key or not loaded_image.public_url:
        return []

    params = {
        "engine": "google_lens",
        "url": loaded_image.public_url,
        "api_key": api_key,
    }

    try:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:  # noqa: BLE001
        return []

    results: List[CandidateSource] = []
    for field_name, result_role, note_prefix in (
        ("exact_matches", "original source", "Google Lens exact match"),
        ("visual_matches", "commentary", "Google Lens visual match"),
    ):
        items = data.get(field_name)
        if isinstance(items, list):
            iterable = items
        elif isinstance(items, dict):
            iterable = items.get("items", [])
        else:
            iterable = []

        for item in iterable[:MAX_REVERSE_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            url = clean_string(item.get("link", ""))
            title = clean_string(item.get("title", "")) or clean_string(item.get("source", ""))
            if not url:
                continue
            results.append(
                CandidateSource(
                    url=url,
                    title=title or url,
                    evidence_type="reverse_image_search",
                    snippet=clean_string(item.get("source", "")),
                    result_role=result_role,
                    match_strength="medium",
                    is_probable_original=result_role == "original source",
                    notes=note_prefix,
                )
            )

    return results


def generate_ocr_search_queries(evidence: ImageEvidence) -> List[str]:
    """
    Stage 3: build 3-5 OCR-informed web queries.

    Queries are assembled primarily from extracted visible text, then from key entities,
    numbers, dates, and branding. The output is deterministic so downstream pipelines can
    debug and compare runs.
    """

    queries: List[str] = []

    quoted_fragments = []
    for line in evidence.visible_text:
        word_count = len(line.split())
        if 3 <= word_count <= 12:
            quoted_fragments.append(f"\"{line}\"")
        if len(quoted_fragments) == 2:
            break

    queries.extend(quoted_fragments)
    if evidence.likely_title:
        queries.append(f"\"{evidence.likely_title}\"")

    entities = " ".join(evidence.key_entities[:3]).strip()
    numbers = " ".join(evidence.numbers_percentages_dates[:3]).strip()
    branding = " ".join(evidence.logos_watermarks_branding[:2]).strip()

    if entities and numbers:
        queries.append(collapse_whitespace(f"{entities} {numbers}"))
    if entities and branding:
        queries.append(collapse_whitespace(f"{entities} {branding}"))
    if evidence.chart_figure_type and entities:
        queries.append(collapse_whitespace(f"{evidence.chart_figure_type} {entities}"))
    if branding and evidence.likely_title:
        queries.append(collapse_whitespace(f"{branding} \"{evidence.likely_title}\""))

    if not queries and evidence.visible_text:
        queries.append(f"\"{evidence.visible_text[0]}\"")

    deduped = dedupe_preserve_order([query for query in queries if query])
    return deduped[:MAX_QUERY_COUNT]


def collect_web_search_candidates(queries: Sequence[str]) -> List[CandidateSource]:
    """
    Stage 3: run OCR-based web searches and collect candidate pages.

    The implementation uses Brave Search Web Search. Queries are driven from extracted visible
    text and related OCR evidence from the image.
    """

    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return []

    candidates: List[CandidateSource] = []
    for query in queries:
        try:
            response = requests.get(
                BRAVE_WEB_SEARCH_ENDPOINT,
                params={
                    "q": query,
                    "count": MAX_WEB_CANDIDATES,
                    "country": os.getenv("BRAVE_SEARCH_COUNTRY", "us"),
                    "search_lang": os.getenv("BRAVE_SEARCH_LANG", "en"),
                    "extra_snippets": "true",
                },
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:  # noqa: BLE001
            continue

        web_section = data.get("web", {})
        organic_results = web_section.get("results", []) if isinstance(web_section, dict) else []
        for item in organic_results[:MAX_WEB_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            url = clean_string(item.get("url", ""))
            title = clean_string(item.get("title", ""))
            snippet = clean_string(item.get("description", ""))
            extra_snippets = item.get("extra_snippets", [])
            if isinstance(extra_snippets, list):
                snippet = dedupe_join([snippet] + clean_string_list(extra_snippets))
            if not url:
                continue

            result_role = classify_candidate_role(url, title, snippet)
            candidates.append(
                CandidateSource(
                    url=url,
                    title=title or url,
                    evidence_type="web_search",
                    snippet=snippet,
                    result_role=result_role,
                    match_strength="low",
                    is_probable_original=result_role == "original source",
                    notes=f"Brave web search query: {query}",
                )
            )

    return candidates


def deduplicate_candidates(candidates: Sequence[CandidateSource]) -> List[CandidateSource]:
    """Deduplicate candidates by normalized URL while preserving the strongest metadata seen."""

    merged: Dict[str, CandidateSource] = {}
    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if not normalized:
            continue
        existing = merged.get(normalized)
        if existing is None:
            merged[normalized] = candidate
            continue
        if candidate.evidence_type == "reverse_image_search":
            existing.evidence_type = "reverse_image_search"
        if candidate.is_probable_original:
            existing.is_probable_original = True
        if len(candidate.title) > len(existing.title):
            existing.title = candidate.title
        if len(candidate.snippet) > len(existing.snippet):
            existing.snippet = candidate.snippet
        existing.notes = dedupe_join([existing.notes, candidate.notes])
        if existing.result_role != "original source" and candidate.result_role == "original source":
            existing.result_role = candidate.result_role
    return list(merged.values())


def verify_candidate_pages(
    loaded_image: LoadedImage,
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
) -> List[CandidateSource]:
    """
    Stage 4: verify whether a candidate page actually embeds the same or a near-identical image.

    This stage is the core hallucination guardrail. The final source decision only considers
    candidates whose fetched page contains a strong visual match to the input image.
    """

    target_hash = compute_image_hash(loaded_image.image_bytes)
    evidence_tokens = build_evidence_tokens(evidence)

    verified: List[CandidateSource] = []
    for candidate in candidates:
        page_data = fetch_page_images_and_text(candidate.url)
        if page_data is None:
            candidate.notes = dedupe_join([candidate.notes, "Page fetch failed or was blocked."])
            verified.append(candidate)
            continue

        page_title, page_text, image_urls, published_date = page_data
        candidate.page_title = page_title
        candidate.page_summary_text = page_text[:1000]
        candidate.published_date = published_date

        best_similarity = 0.0
        best_image_url: Optional[str] = None
        for image_url in image_urls[:MAX_PAGE_IMAGES]:
            image_bytes = fetch_candidate_image(image_url)
            if not image_bytes:
                continue
            try:
                candidate_hash = compute_image_hash(image_bytes)
            except ValueError:
                continue
            similarity = phash_similarity(target_hash, candidate_hash)
            if similarity > best_similarity:
                best_similarity = similarity
                best_image_url = image_url

        candidate.image_embedded = best_similarity >= POSSIBLE_MATCH_THRESHOLD
        candidate.best_image_similarity = round(best_similarity, 4)
        candidate.best_image_match_url = best_image_url
        candidate.text_overlap = round(text_overlap_score(evidence_tokens, page_title + " " + page_text), 4)

        if candidate.image_embedded and not candidate.is_probable_original:
            candidate.is_probable_original = classify_candidate_role(
                candidate.url,
                candidate.title or page_title,
                candidate.snippet or page_text[:200],
            ) == "original source"

        candidate.match_strength = classify_match_strength(best_similarity, candidate.text_overlap)
        candidate.notes = build_candidate_notes(candidate, best_similarity)
        verified.append(candidate)

    return sort_candidates(verified)


def fetch_page_images_and_text(url: str) -> Optional[Tuple[str, str, List[str], Optional[str]]]:
    """
    Fetch candidate HTML and parse its title, text, image URLs, and publication date.

    Only lightweight HTML signals are used. If the page requires client-side rendering and the
    image never appears in static HTML, the pipeline treats it as unverified rather than guessing.
    """

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return None

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and not response.text:
        return None

    parser = PageParser()
    parser.feed(response.text)
    page_title = collapse_whitespace(" ".join(parser.title_chunks))
    page_text = collapse_whitespace(" ".join(parser.text_chunks))

    image_urls = [urljoin(response.url, raw_url) for raw_url in parser.meta_image_urls + parser.image_urls]
    image_urls = dedupe_preserve_order(image_urls)
    published_date = normalize_date_candidates(parser.meta_dates)
    return page_title, page_text, image_urls, published_date


def fetch_candidate_image(url: str) -> Optional[bytes]:
    """Fetch an embedded candidate image while avoiding non-image responses."""

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "image" not in content_type:
        return None
    return response.content


def determine_source_decision(candidates: Sequence[CandidateSource]) -> Dict[str, Any]:
    """
    Stage 4 decision logic.

    The visual match threshold is the primary gate. Text overlap only helps distinguish between
    multiple already-visual candidates and cannot produce a source decision by itself.
    """

    if not candidates:
        return {
            "status": "source_not_found",
            "candidate": None,
            "match_type": "none",
            "source_evidence": "No candidate pages were collected.",
            "confidence": 0.05,
        }

    best_visual = max(candidates, key=lambda candidate: candidate.best_image_similarity)
    similarity = best_visual.best_image_similarity

    if similarity < POSSIBLE_MATCH_THRESHOLD or not best_visual.image_embedded:
        return {
            "status": "source_not_found",
            "candidate": None,
            "match_type": "none",
            "source_evidence": (
                "Candidate pages were collected, but none of them contained a verified matching "
                "or near-matching embedded image."
            ),
            "confidence": 0.15,
        }

    if similarity >= EXACT_MATCH_THRESHOLD and best_visual.is_probable_original:
        return {
            "status": "exact_source_found",
            "candidate": best_visual,
            "match_type": "exact",
            "source_evidence": (
                f"The page contains a visually identical embedded image "
                f"(phash similarity {best_visual.best_image_similarity:.2f}) and appears to be an original host."
            ),
            "confidence": 0.96,
        }

    if similarity >= NEAR_EXACT_MATCH_THRESHOLD and best_visual.is_probable_original:
        return {
            "status": "exact_source_found",
            "candidate": best_visual,
            "match_type": "near_exact",
            "source_evidence": (
                f"The page contains a near-identical embedded image "
                f"(phash similarity {best_visual.best_image_similarity:.2f}) and appears to be an original host."
            ),
            "confidence": 0.84,
        }

    if similarity >= NEAR_EXACT_MATCH_THRESHOLD or (
        similarity >= POSSIBLE_MATCH_THRESHOLD and best_visual.text_overlap >= TEXT_OVERLAP_FOR_POSSIBLE
    ):
        return {
            "status": "possible_source",
            "candidate": best_visual,
            "match_type": "possible" if similarity < NEAR_EXACT_MATCH_THRESHOLD else "near_exact",
            "source_evidence": (
                f"The page contains a strong embedded visual match "
                f"(phash similarity {best_visual.best_image_similarity:.2f}), "
                "but originality or exact publication status is not fully verified."
            ),
            "confidence": 0.68 if similarity >= NEAR_EXACT_MATCH_THRESHOLD else 0.6,
        }

    return {
        "status": "source_not_found",
        "candidate": None,
        "match_type": "none",
        "source_evidence": "No candidate page met the visual verification threshold for a source decision.",
        "confidence": 0.18,
    }


def build_output_json(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    search_queries: Sequence[str],
) -> Dict[str, Any]:
    """
    Stage 5: assemble the final grounded JSON output.

    Only verified evidence and explicitly marked inference are surfaced here. When no source is
    verified, the function returns `source_not_found` with `source_link: null`.
    """

    winning_candidate = source_decision.get("candidate")
    likely_context = build_likely_context(evidence)
    concise_overview = build_concise_overview(evidence, source_decision, likely_context)

    source_status = source_decision["status"]
    if source_status == "exact_source_found":
        source_assessment = "An exact or near-exact source page was found."
    elif source_status == "possible_source":
        source_assessment = "A possible source page was found, but it is not fully verified."
    else:
        source_assessment = "No exact matching source page was found."

    candidate_sources = [
        {
            "url": candidate.url,
            "title": candidate.title,
            "evidence_type": candidate.evidence_type,
            "match_strength": candidate.match_strength,
            "is_probable_original": candidate.is_probable_original,
            "notes": candidate.notes,
        }
        for candidate in list(candidates)[:10]
    ]

    combined_source_evidence = source_assessment
    if source_decision["source_evidence"]:
        combined_source_evidence = f"{source_assessment} {source_decision['source_evidence']}"

    return {
        "image_description": evidence.visual_description,
        "visible_text": evidence.visible_text,
        "key_entities": dedupe_preserve_order(
            evidence.key_entities + evidence.logos_watermarks_branding
        ),
        "image_type": evidence.chart_figure_type or "unknown",
        "candidate_sources": candidate_sources,
        "source_status": source_status,
        "most_likely_source": winning_candidate.title if winning_candidate else None,
        "source_link": winning_candidate.url if winning_candidate else None,
        "source_evidence": combined_source_evidence,
        "source_match_type": source_decision["match_type"],
        "likely_context": likely_context,
        "concise_overview": concise_overview,
        "confidence": round(float(source_decision["confidence"]), 2),
        "debug_info": {
            "generated_queries": list(search_queries),
            "reasoning_notes": dedupe_join(
                evidence.extraction_notes
                + [source_assessment]
                + build_reasoning_notes(candidates, source_decision)
            ),
        },
    }


def build_likely_context(evidence: ImageEvidence) -> str:
    """
    Create a cautious context statement grounded in observed evidence.

    This is phrased as inference, not fact. The sentence deliberately references the signals used.
    """

    observed_signals: List[str] = []
    if evidence.likely_title:
        observed_signals.append(f'the apparent title "{evidence.likely_title}"')
    if evidence.key_entities:
        observed_signals.append(f"visible entities such as {', '.join(evidence.key_entities[:3])}")
    if evidence.numbers_percentages_dates:
        observed_signals.append(
            f"numbers or dates including {', '.join(evidence.numbers_percentages_dates[:3])}"
        )

    signal_text = ", ".join(observed_signals)
    if signal_text:
        return (
            f"This likely relates to {summarize_subject(evidence)}, based on {signal_text}."
        )
    return f"This likely relates to {summarize_subject(evidence)}, but the available visual evidence is limited."


def build_concise_overview(
    evidence: ImageEvidence,
    source_decision: Dict[str, Any],
    likely_context: str,
) -> str:
    """Build a 3-5 sentence overview grounded strictly in extracted evidence and verification output."""

    sentences = [truncate_sentence(evidence.visual_description or "A clear visual description could not be extracted.")]

    if evidence.visible_text:
        text_preview = "; ".join(evidence.visible_text[:3])
        sentences.append(f'Visible text includes: "{text_preview}".')
    else:
        sentences.append("No reliable visible text was extracted from the image.")

    sentences.append(likely_context)

    source_status = source_decision["status"]
    candidate = source_decision.get("candidate")
    if source_status == "source_not_found":
        sentences.append("Reverse image search and OCR-based web search did not produce a page with a verified matching embedded image.")
    elif candidate is not None:
        sentences.append(
            f'The strongest verified candidate is "{candidate.title}" ({candidate.url}), '
            f'where the embedded image match was scored as {candidate.best_image_similarity:.2f}.'
        )

    return " ".join(sentences[:5])


def build_reasoning_notes(
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
) -> List[str]:
    """Summarize search and verification behavior for debugging without exposing chain-of-thought."""

    notes = [f"Collected {len(candidates)} deduplicated candidate pages."]
    if candidates:
        best = max(candidates, key=lambda candidate: candidate.best_image_similarity)
        notes.append(
            f"Best visual candidate similarity: {best.best_image_similarity:.2f} at {best.url}."
        )
    notes.append(f"Final source status: {source_decision['status']}.")
    return notes


def sort_candidates(candidates: Sequence[CandidateSource]) -> List[CandidateSource]:
    """Rank candidates with visual similarity first, then probable originality, then text overlap."""

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.best_image_similarity,
            1 if candidate.is_probable_original else 0,
            candidate.text_overlap,
            1 if candidate.evidence_type == "reverse_image_search" else 0,
        ),
        reverse=True,
    )


def classify_match_strength(similarity: float, text_overlap: float) -> str:
    """Map combined verification evidence to the required high/medium/low label."""

    if similarity >= NEAR_EXACT_MATCH_THRESHOLD:
        return "high"
    if similarity >= POSSIBLE_MATCH_THRESHOLD or text_overlap >= 0.2:
        return "medium"
    return "low"


def build_candidate_notes(candidate: CandidateSource, similarity: float) -> str:
    """Create short candidate notes, including likely role and visual verification outcome."""

    role = candidate.result_role
    parts = [candidate.notes, f"Classified as {role}."]
    if candidate.best_image_match_url:
        parts.append(
            f"Embedded image verified at {candidate.best_image_match_url} with similarity {similarity:.2f}."
        )
    else:
        parts.append("No embedded image on the page reached the visual match threshold.")
    if candidate.published_date:
        parts.append(f"Earliest parsed page date: {candidate.published_date}.")
    return dedupe_join(parts)


def classify_candidate_role(url: str, title: str, snippet: str) -> str:
    """Heuristic source-role classifier for original source vs repost vs commentary."""

    domain = urlparse(url).netloc.lower()
    if domain in REPOST_DOMAINS:
        return "repost"

    combined = f"{title} {snippet}".lower()
    if any(token in combined for token in COMMENTARY_HINTS):
        return "commentary"

    return "original source"


def build_evidence_tokens(evidence: ImageEvidence) -> Set[str]:
    """Tokenize image evidence for text-overlap checks during candidate verification."""

    raw_parts = (
        evidence.visible_text
        + [evidence.likely_title]
        + evidence.axes_labels
        + evidence.legend_items
        + evidence.key_entities
        + evidence.numbers_percentages_dates
        + evidence.logos_watermarks_branding
    )
    tokens: Set[str] = set()
    for part in raw_parts:
        for token in tokenize(part):
            if token not in COMMON_STOPWORDS:
                tokens.add(token)
    return tokens


def text_overlap_score(evidence_tokens: Set[str], page_text: str) -> float:
    """Compute simple evidence-token recall against page text."""

    if not evidence_tokens:
        return 0.0
    page_tokens = set(tokenize(page_text))
    if not page_tokens:
        return 0.0
    overlap = len(evidence_tokens & page_tokens)
    return overlap / max(len(evidence_tokens), 1)


def compute_image_hash(image_bytes: bytes) -> imagehash.ImageHash:
    """Compute perceptual hash for visual matching."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return imagehash.phash(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Image bytes are not a decodable raster image.") from exc


def phash_similarity(hash_a: imagehash.ImageHash, hash_b: imagehash.ImageHash) -> float:
    """Convert imagehash hamming distance into a 0-1 similarity score."""

    hash_length = hash_a.hash.size
    distance = hash_a - hash_b
    return max(0.0, 1.0 - (distance / hash_length))


def parse_openrouter_output_json(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the structured JSON object from an OpenRouter chat completions response."""

    choices = response_json.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError("OpenRouter response does not contain choices.")

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = extract_text_from_message_content(content)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue

    raise ValueError("Structured JSON output not found in OpenRouter response.")


def extract_text_from_message_content(content: Any) -> str:
    """Extract plain text from OpenRouter chat completion message content."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()
    return ""


def normalize_date_candidates(values: Sequence[str]) -> Optional[str]:
    """Normalize parsed meta date values to ISO date strings when possible."""

    for value in values:
        cleaned = clean_string(value)
        if not cleaned:
            continue
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            return parsed.date().isoformat()
        except ValueError:
            match = re.search(r"\d{4}-\d{2}-\d{2}", cleaned)
            if match:
                return match.group(0)
    return None


def summarize_subject(evidence: ImageEvidence) -> str:
    """Build a short subject phrase from the strongest observed signals."""

    if evidence.likely_title:
        return evidence.likely_title
    if evidence.key_entities:
        joined = ", ".join(evidence.key_entities[:3])
        if evidence.chart_figure_type:
            return f"{evidence.chart_figure_type} involving {joined}"
        return joined
    if evidence.chart_figure_type and evidence.chart_figure_type.lower() != "unknown":
        return evidence.chart_figure_type
    return "the visible image content"


def image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Read image dimensions from bytes."""

    with Image.open(BytesIO(image_bytes)) as image:
        return image.size


def sha256_hex(data: bytes) -> str:
    """Return SHA-256 digest for the loaded image."""

    return hashlib.sha256(data).hexdigest()


def guess_mime_type(extension: str) -> str:
    """Map common filename extensions to MIME types."""

    extension = extension.lower()
    if extension in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if extension == ".png":
        return "image/png"
    if extension == ".webp":
        return "image/webp"
    if extension == ".gif":
        return "image/gif"
    return "image/png"


def normalize_url(url: str) -> str:
    """Normalize URLs for candidate deduplication."""

    cleaned = clean_string(url)
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def tokenize(text: str) -> List[str]:
    """Tokenize strings into lowercase alphanumeric terms."""

    return re.findall(r"[a-z0-9%./:-]+", text.lower())


def clean_string(value: Any) -> str:
    """Normalize possibly-null strings."""

    if not isinstance(value, str):
        return ""
    return collapse_whitespace(value)


def clean_string_list(value: Any) -> List[str]:
    """Normalize string arrays and drop blank items."""

    if not isinstance(value, list):
        return []
    cleaned = [clean_string(item) for item in value if isinstance(item, str)]
    return [item for item in cleaned if item]


def collapse_whitespace(value: str) -> str:
    """Collapse repeated whitespace into single spaces."""

    return re.sub(r"\s+", " ", value).strip()


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Deduplicate strings while preserving the first seen order."""

    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def dedupe_join(parts: Sequence[str]) -> str:
    """Join note fragments while removing blanks and duplicates."""

    cleaned = dedupe_preserve_order([clean_string(part) for part in parts if clean_string(part)])
    return " ".join(cleaned)


def truncate_sentence(text: str, limit: int = 220) -> str:
    """Trim overly long generated sentences without changing their meaning."""

    cleaned = clean_string(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def main() -> None:
    """CLI entrypoint for local runs or agent pipelines."""

    parser = argparse.ArgumentParser(description="Retrieve grounded background and source evidence for an image.")
    parser.add_argument("--image", required=True, help="Image input as URL, data URL, base64, or local path.")
    parser.add_argument("--user-query", default="", help="Optional user question about the image.")
    args = parser.parse_args()

    result = retrieve_image_background(args.image, args.user_query)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
