import argparse
import base64
import binascii
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse

import imagehash
import requests
from PIL import Image, UnidentifiedImageError


USER_AGENT = "retrieve-image-background/0.1"
LOGGER = logging.getLogger("retrieve_image_background")
REQUEST_TIMEOUT_SECONDS = 20
MAX_REVERSE_CANDIDATES = 8
MAX_WEB_CANDIDATES = 10
MAX_PAGE_IMAGES = 12
MAX_VISIBLE_TEXT_LINES = 20
MAX_QUERY_COUNT = 5
DEFAULT_OPENROUTER_VISION_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_OPENROUTER_SUMMARY_MODEL = "google/gemini-3.1-flash-lite-preview"
SERPAPI_ENDPOINT = "https://serpapi.com/search"
OPENROUTER_CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TEMP_IMAGE_SIGNED_URL_TTL_MINUTES = 15
DEFAULT_LLM_SUMMARY_TIMEOUT_SECONDS = 20
EXACT_MATCH_THRESHOLD = 0.85
NEAR_EXACT_MATCH_THRESHOLD = 0.80
POSSIBLE_MATCH_THRESHOLD = 0.70
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
INLINE_BOILERPLATE_PATTERNS = (
    r"\bskip to (?:main )?content\b",
    r"\bcookie(?: policy| settings| consent)?\b",
    r"\bprivacy policy\b",
    r"\bterms(?: of use| and conditions| of service)?\b",
    r"\bsign in\b",
    r"\blog in\b",
    r"\badvanced search\b",
    r"\ball rights reserved\b",
)
NAVIGATION_TERMS = {
    "home",
    "about",
    "submit",
    "alerts",
    "rss",
    "menu",
    "navigation",
    "search",
    "subscribe",
    "signin",
    "login",
    "privacy",
    "terms",
    "cookie",
}
PROSE_HINT_TERMS = {
    "analyzes",
    "analysis",
    "compares",
    "compare",
    "describes",
    "discusses",
    "examines",
    "explains",
    "finds",
    "found",
    "includes",
    "indicates",
    "measures",
    "plots",
    "ranks",
    "reports",
    "shows",
    "suggests",
    "tracks",
}
INCOMPLETE_TRAILING_TERMS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "because",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "without",
}
GENERIC_CONTEXT_TOKENS = {
    "article",
    "chart",
    "data",
    "figure",
    "graph",
    "image",
    "page",
    "plot",
    "report",
    "study",
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
    ocr_debug: Dict[str, Any] = field(default_factory=dict)


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
    deterministic_page_summary_text: str = ""
    llm_page_summary_text: str = ""


@dataclass
class ReverseImageSearchResult:
    """Stage 2 reverse-search candidates plus structured execution debug details."""

    candidates: List[CandidateSource] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OcrResult:
    """Optional OCR output plus structured execution debug details."""

    lines: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmSummaryConfig:
    """Optional explicit overrides for the env-based LLM summary settings."""

    enable_llm_context_summary: Optional[bool] = None
    page_summary_enabled: Optional[bool] = None
    likely_context_enabled: Optional[bool] = None
    concise_overview_enabled: Optional[bool] = None
    summary_model: Optional[str] = None
    timeout_seconds: Optional[int] = None


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


def retrieve_image_background(
    image: str,
    user_query: str = "",
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> Dict[str, Any]:
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
    llm_summary_debug = initialize_llm_summary_debug(llm_summary_config=llm_summary_config)

    reverse_search_result = reverse_image_search(loaded_image)
    reverse_candidates = reverse_search_result.candidates
    search_queries = generate_ocr_search_queries(evidence)
    web_candidates = collect_web_search_candidates(search_queries)

    combined_candidates = deduplicate_candidates(reverse_candidates + web_candidates)
    verified_candidates = verify_candidate_pages(
        loaded_image,
        evidence,
        combined_candidates,
    )
    source_decision = determine_source_decision(verified_candidates, reverse_candidates)
    apply_post_decision_page_summary_rewrites(
        evidence,
        verified_candidates,
        source_decision,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
    )

    return build_output_json(
        evidence=evidence,
        candidates=verified_candidates,
        raw_reverse_candidates=reverse_candidates,
        source_decision=source_decision,
        search_queries=search_queries,
        reverse_search_debug=reverse_search_result.debug,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
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
    ocr_result = run_optional_tesseract_ocr(loaded_image.image_bytes)
    evidence.ocr_debug = ocr_result.debug
    if ocr_result.lines:
        evidence.visible_text = dedupe_preserve_order(ocr_result.lines)[:MAX_VISIBLE_TEXT_LINES]
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
        error_message = format_exception(exc)
        if not evidence.visual_description:
            evidence.visual_description = (
                "Visual description unavailable because the vision extraction request failed."
            )
        if not evidence.chart_figure_type:
            evidence.chart_figure_type = "unknown"
        evidence.extraction_notes.append(f"Vision extraction failed: {error_message}")
        LOGGER.warning("Vision extraction failed: %s", error_message)

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


def format_exception(exc: BaseException) -> str:
    """Format exceptions consistently for logs and debug payloads."""

    return f"{type(exc).__name__}: {exc}"


def env_flag_enabled(name: str, default: bool = False) -> bool:
    """Parse common boolean environment-variable strings."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_bool_override(
    explicit_override: Optional[bool],
    env_name: str,
    default: bool,
) -> bool:
    """Resolve booleans with explicit override first, then env, then default."""

    if explicit_override is not None:
        return explicit_override
    return env_flag_enabled(env_name, default=default)


def resolve_string_override(
    explicit_override: Optional[str],
    env_name: str,
    default: str,
) -> str:
    """Resolve strings with explicit override first, then env, then default."""

    if explicit_override is not None and explicit_override.strip():
        return explicit_override.strip()
    return os.getenv(env_name, default).strip() or default


def resolve_int_override(
    explicit_override: Optional[int],
    env_name: str,
    default: int,
    minimum: int = 1,
) -> int:
    """Resolve integers with explicit override first, then env, then default."""

    if explicit_override is not None:
        return max(minimum, int(explicit_override))
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default
    try:
        return max(minimum, int(float(raw_value)))
    except ValueError:
        return default


def llm_summary_timeout_seconds(llm_summary_config: Optional[LlmSummaryConfig] = None) -> int:
    """Return the configured timeout for optional summary rewrites."""

    return resolve_int_override(
        llm_summary_config.timeout_seconds if llm_summary_config else None,
        "LLM_SUMMARY_TIMEOUT_SECONDS",
        DEFAULT_LLM_SUMMARY_TIMEOUT_SECONDS,
        minimum=5,
    )


def llm_summary_model_name(llm_summary_config: Optional[LlmSummaryConfig] = None) -> str:
    """Return the configured OpenRouter model for optional summary rewrites."""

    return resolve_string_override(
        llm_summary_config.summary_model if llm_summary_config else None,
        "OPENROUTER_SUMMARY_MODEL",
        DEFAULT_OPENROUTER_SUMMARY_MODEL,
    )


def llm_context_summary_enabled(
    scope: Optional[str] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> bool:
    """Return whether optional LLM-assisted summary rewriting is enabled."""

    # Env is the baseline config; explicit overrides take precedence.
    if not resolve_bool_override(
        llm_summary_config.enable_llm_context_summary if llm_summary_config else None,
        "ENABLE_LLM_CONTEXT_SUMMARY",
        False,
    ):
        return False
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        return False

    if not scope:
        return True

    # The global flag still gates all scoped flags.
    scoped_settings = {
        "page_summary": (
            llm_summary_config.page_summary_enabled if llm_summary_config else None,
            "LLM_PAGE_SUMMARY_ENABLED",
        ),
        "likely_context": (
            llm_summary_config.likely_context_enabled if llm_summary_config else None,
            "LLM_LIKELY_CONTEXT_ENABLED",
        ),
        "concise_overview": (
            llm_summary_config.concise_overview_enabled if llm_summary_config else None,
            "LLM_CONCISE_OVERVIEW_ENABLED",
        ),
    }
    explicit_override, env_name = scoped_settings.get(scope, (None, ""))
    if not env_name:
        return True
    return resolve_bool_override(explicit_override, env_name, True)


def initialize_llm_summary_debug(llm_summary_config: Optional[LlmSummaryConfig] = None) -> Dict[str, Any]:
    """Create safe debug metadata for optional summary rewriting."""

    requested = resolve_bool_override(
        llm_summary_config.enable_llm_context_summary if llm_summary_config else None,
        "ENABLE_LLM_CONTEXT_SUMMARY",
        False,
    )
    api_key_present = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    enabled = requested and api_key_present
    return {
        "requested": requested,
        "enabled": enabled,
        "api_key_present": api_key_present,
        "model": llm_summary_model_name(llm_summary_config),
        "page_summary": {
            "enabled": llm_context_summary_enabled("page_summary", llm_summary_config),
            "attempted": 0,
            "used": 0,
            "failed": 0,
            "fallback_used": False,
            "last_error": None,
            "post_decision_rewrites": {
                "selected_count": 0,
                "candidates": [],
            },
        },
        "likely_context": {
            "enabled": llm_context_summary_enabled("likely_context", llm_summary_config),
            "attempted": False,
            "used": False,
            "fallback_used": False,
            "error": None,
        },
        "concise_overview": {
            "enabled": llm_context_summary_enabled("concise_overview", llm_summary_config),
            "attempted": False,
            "used": False,
            "fallback_used": False,
            "error": None,
        },
    }


def update_llm_summary_debug(
    llm_summary_debug: Optional[Dict[str, Any]],
    scope: str,
    *,
    attempted: bool = False,
    used: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record safe LLM summary status without exposing hidden model reasoning."""

    if not isinstance(llm_summary_debug, dict):
        return
    scope_debug = llm_summary_debug.get(scope)
    if not isinstance(scope_debug, dict):
        return

    if scope == "page_summary":
        if attempted:
            scope_debug["attempted"] = int(scope_debug.get("attempted", 0)) + 1
        if used:
            scope_debug["used"] = int(scope_debug.get("used", 0)) + 1
        if error:
            scope_debug["failed"] = int(scope_debug.get("failed", 0)) + 1
            scope_debug["fallback_used"] = True
            scope_debug["last_error"] = error
        return

    if attempted:
        scope_debug["attempted"] = True
    if used:
        scope_debug["used"] = True
    if error:
        scope_debug["error"] = error
        scope_debug["fallback_used"] = True


def page_summary_debug_scope(llm_summary_debug: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the page-summary debug object when available."""

    if not isinstance(llm_summary_debug, dict):
        return None
    scope_debug = llm_summary_debug.get("page_summary")
    if not isinstance(scope_debug, dict):
        return None
    return scope_debug


def page_summary_rewrite_counts(scope_debug: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Capture current page-summary rewrite counters for delta-based debug records."""

    if not isinstance(scope_debug, dict):
        return {"attempted": 0, "used": 0, "failed": 0}
    return {
        "attempted": int(scope_debug.get("attempted", 0)),
        "used": int(scope_debug.get("used", 0)),
        "failed": int(scope_debug.get("failed", 0)),
    }


def record_post_decision_page_summary_rewrite(
    llm_summary_debug: Optional[Dict[str, Any]],
    candidate: CandidateSource,
    *,
    deterministic_summary_present: bool,
    attempted: bool = False,
    used: bool = False,
    changed: bool = False,
    fallback_used: bool = False,
    skipped_reason: Optional[str] = None,
) -> None:
    """Record which candidate page summaries were rewritten after source decision."""

    scope_debug = page_summary_debug_scope(llm_summary_debug)
    if scope_debug is None:
        return
    rewrite_debug = scope_debug.setdefault(
        "post_decision_rewrites",
        {
            "selected_count": 0,
            "candidates": [],
        },
    )
    if not isinstance(rewrite_debug, dict):
        return
    records = rewrite_debug.setdefault("candidates", [])
    if not isinstance(records, list):
        return
    records.append(
        {
            "url": candidate.url,
            "title": candidate_headline(candidate) or candidate.title,
            "source_candidate": candidate.evidence_type,
            "image_embedded": bool(candidate.image_embedded),
            "best_image_similarity": float(candidate.best_image_similarity),
            "text_overlap": float(candidate.text_overlap),
            "deterministic_summary_present": deterministic_summary_present,
            "attempted": attempted,
            "used": used,
            "changed_text": changed,
            "fallback_used": fallback_used,
            "skipped_reason": skipped_reason,
        }
    )


def summary_response_text(response_json: Dict[str, Any]) -> str:
    """Extract plain text from an OpenRouter summary response."""

    choices = response_json.get("choices", [])
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        text = extract_text_from_message_content(message.get("content"))
        if text:
            return text.strip()
    return ""


def call_openrouter_summary_model(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> Optional[Any]:
    """Call the optional OpenRouter summary model with plain-text or schema output."""

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    payload: Dict[str, Any] = {
        "model": llm_summary_model_name(llm_summary_config),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary_output",
                "strict": True,
                "schema": schema,
            },
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "retrieve_image_background"),
    }
    endpoint = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_CHAT_COMPLETIONS_ENDPOINT).rstrip("/")
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=llm_summary_timeout_seconds(llm_summary_config),
    )
    response.raise_for_status()
    response_json = response.json()

    if schema is not None:
        return parse_openrouter_output_json(response_json)
    text = summary_response_text(response_json)
    if not text:
        raise ValueError("Summary model returned no text.")
    return text


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
        ocr_debug=dict(existing.ocr_debug),
    )


def run_optional_tesseract_ocr(image_bytes: bytes) -> OcrResult:
    """
    Optional OCR fallback used inside Stage 1.

    The tool stays functional without Tesseract. When Tesseract is unavailable, the pipeline
    returns an empty OCR result instead of failing.
    """

    debug: Dict[str, Any] = {
        "attempted": False,
        "dependency_available": False,
        "image_opened": False,
        "succeeded": False,
        "engine": "pytesseract",
        "config": "--psm 6",
        "preprocess": "grayscale",
        "early_return_reason": None,
        "error": None,
        "line_count": 0,
    }

    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        debug["early_return_reason"] = "pytesseract_not_installed"
        debug["error"] = format_exception(exc)
        LOGGER.info("OCR skipped because pytesseract is not installed: %s", debug["error"])
        return OcrResult(debug=debug)

    try:
        debug["dependency_available"] = True
        debug["attempted"] = True
        with Image.open(BytesIO(image_bytes)) as image:
            debug["image_opened"] = True
            image = image.convert("L")
            raw_text = pytesseract.image_to_string(image, config="--psm 6")
    except Exception as exc:  # noqa: BLE001
        debug["early_return_reason"] = "ocr_execution_failed"
        debug["error"] = format_exception(exc)
        LOGGER.warning("OCR execution failed: %s", debug["error"])
        return OcrResult(debug=debug)

    lines = [collapse_whitespace(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    debug["succeeded"] = True
    debug["line_count"] = len(lines)

    if not lines:
        debug["early_return_reason"] = "no_readable_text_extracted"
        LOGGER.info("OCR completed but did not extract readable text.")

    return OcrResult(lines=lines, debug=debug)


def reverse_image_search(loaded_image: LoadedImage) -> ReverseImageSearchResult:
    """
    Stage 2: collect candidate pages from reverse image search.

    SerpApi Google Lens is used when `SERPAPI_API_KEY` is available and the pipeline can obtain
    a reachable image URL. Public image inputs keep their existing URL. Local/base64/data inputs
    are uploaded to a temporary private GCS object, accessed through a short-lived signed URL,
    and deleted immediately after the reverse search request completes.
    """

    debug: Dict[str, Any] = {
        "attempted": False,
        "early_return_reason": None,
        "url_resolution": {},
        "serpapi_request": {
            "attempted": False,
            "succeeded": False,
            "error": None,
        },
        "cleanup": {
            "attempted": False,
            "deleted": False,
            "reason": None,
            "error": None,
        },
        "raw_candidate_count": 0,
    }

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        debug["early_return_reason"] = "missing_serpapi_api_key"
        return ReverseImageSearchResult(debug=debug)

    reverse_search_url, uploaded_object_name, url_resolution_debug = get_reverse_search_url(loaded_image)
    debug["url_resolution"] = url_resolution_debug
    if not reverse_search_url:
        debug["early_return_reason"] = url_resolution_debug.get("early_return_reason") or "reverse_search_url_unavailable"
        return ReverseImageSearchResult(debug=debug)

    params = {
        "engine": "google_lens",
        "url": reverse_search_url,
        "api_key": api_key,
    }

    debug["attempted"] = True
    debug["serpapi_request"]["attempted"] = True
    try:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        debug["serpapi_request"]["succeeded"] = True
    except Exception as exc:  # noqa: BLE001
        debug["early_return_reason"] = "serpapi_request_failed"
        debug["serpapi_request"]["error"] = format_exception(exc)
        LOGGER.warning("Reverse image search request failed: %s", debug["serpapi_request"]["error"])
        return ReverseImageSearchResult(debug=debug)
    finally:
        # Temporary uploads are only needed long enough for SerpApi to fetch the image URL.
        if uploaded_object_name:
            debug["cleanup"] = delete_uploaded_image(uploaded_object_name)

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

    debug["raw_candidate_count"] = len(results)
    return ReverseImageSearchResult(candidates=results, debug=debug)


def mime_type_to_extension(mime_type: str) -> str:
    """Map common image MIME types to stable filename extensions."""

    normalized = clean_string(mime_type).split(";", 1)[0].lower()
    if normalized == "image/jpeg":
        return ".jpg"
    if normalized == "image/png":
        return ".png"
    if normalized == "image/webp":
        return ".webp"
    if normalized == "image/gif":
        return ".gif"
    if normalized == "image/bmp":
        return ".bmp"
    if normalized in {"image/tiff", "image/x-tiff"}:
        return ".tiff"
    return ".png"


def upload_image_and_get_signed_url(
    loaded_image: LoadedImage,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Upload a non-public input image to GCS and return a short-lived signed URL plus object name.

    ADC is used through `google-cloud-storage`. Any failure returns `(None, None)` so reverse
    image search can be skipped without aborting the full pipeline.
    """

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    bucket_name = os.getenv("TEMP_IMAGE_BUCKET", "").strip()
    object_name = f"reverse-search/{loaded_image.sha256}{mime_type_to_extension(loaded_image.mime_type)}"
    debug = {
        "used_public_url": False,
        "upload_attempted": False,
        "upload_succeeded": False,
        "signed_url_generated": False,
        "credential_source": None,
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip() or None,
        "service_account_email": None,
        "project": project or None,
        "bucket": bucket_name or None,
        "object_name": object_name,
        "early_return_reason": None,
        "error": None,
    }

    if not project or not bucket_name:
        debug["early_return_reason"] = "missing_google_cloud_project_or_temp_image_bucket"
        return None, None, debug

    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError as exc:
        debug["early_return_reason"] = "google_cloud_storage_dependency_missing"
        debug["error"] = format_exception(exc)
        LOGGER.warning("GCS dependency import failed during temporary upload: %s", debug["error"])
        return None, None, debug

    try:
        credentials = None
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            debug["credential_source"] = "google_application_credentials_service_account"
            debug["service_account_email"] = getattr(credentials, "service_account_email", None)
        else:
            debug["credential_source"] = "adc_default_credentials"

        debug["upload_attempted"] = True
        client = storage.Client(project=project, credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_string(loaded_image.image_bytes, content_type=loaded_image.mime_type)
        debug["upload_succeeded"] = True
        signed_url_kwargs: Dict[str, Any] = {
            "version": "v4",
            "expiration": timedelta(minutes=TEMP_IMAGE_SIGNED_URL_TTL_MINUTES),
            "method": "GET",
        }
        if credentials is not None:
            signed_url_kwargs["credentials"] = credentials
            signed_url_kwargs["service_account_email"] = getattr(credentials, "service_account_email", None)
        signed_url = blob.generate_signed_url(**signed_url_kwargs)
        debug["signed_url_generated"] = True
        return signed_url, object_name, debug
    except Exception as exc:  # noqa: BLE001
        debug["early_return_reason"] = (
            "gcs_signed_url_generation_failed" if debug["upload_succeeded"] else "gcs_upload_failed"
        )
        debug["error"] = format_exception(exc)
        LOGGER.warning("Temporary GCS upload or signed URL generation failed: %s", debug["error"])
        delete_uploaded_image(object_name)
        return None, None, debug


def delete_uploaded_image(object_name: str) -> Dict[str, Any]:
    """Best-effort cleanup for temporary reverse-search uploads."""

    debug = {
        "attempted": False,
        "deleted": False,
        "reason": None,
        "error": None,
    }
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    bucket_name = os.getenv("TEMP_IMAGE_BUCKET", "").strip()
    if not object_name or not project or not bucket_name:
        debug["reason"] = "missing_cleanup_context"
        return debug

    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError as exc:
        debug["reason"] = "google_cloud_storage_dependency_missing"
        debug["error"] = format_exception(exc)
        LOGGER.warning("GCS dependency import failed during cleanup: %s", debug["error"])
        return debug

    try:
        credentials = None
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        debug["attempted"] = True
        client = storage.Client(project=project, credentials=credentials)
        bucket = client.bucket(bucket_name)
        bucket.blob(object_name).delete()
        debug["deleted"] = True
        debug["reason"] = "deleted"
        return debug
    except Exception as exc:  # noqa: BLE001
        debug["reason"] = "delete_failed"
        debug["error"] = format_exception(exc)
        LOGGER.warning("Temporary GCS cleanup failed for %s: %s", object_name, debug["error"])
        return debug


def get_reverse_search_url(
    loaded_image: LoadedImage,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Return a URL SerpApi Google Lens can fetch.

    Public inputs preserve existing behavior. Non-public inputs use a temporary signed GCS URL.
    The returned object name is only set when cleanup is required.
    """

    debug: Dict[str, Any] = {
        "resolved": False,
        "url_source": "public_url" if loaded_image.public_url else "gcs_signed_url",
        "early_return_reason": None,
        "gcs": None,
    }

    if loaded_image.public_url:
        debug["resolved"] = True
        return loaded_image.public_url, None, debug

    signed_url, object_name, gcs_debug = upload_image_and_get_signed_url(loaded_image)
    debug["gcs"] = gcs_debug
    if not signed_url:
        debug["early_return_reason"] = gcs_debug.get("early_return_reason") or "gcs_signed_url_unavailable"
        return None, None, debug

    debug["resolved"] = True
    return signed_url, object_name, debug


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

    axis_labels = dedupe_preserve_order([label for label in clean_string_list(evidence.axes_labels) if label])
    axis_phrase = " ".join(axis_labels[:2]).strip()
    entities = " ".join(evidence.key_entities[:3]).strip()
    numbers = " ".join(evidence.numbers_percentages_dates[:3]).strip()
    branding = " ".join(evidence.logos_watermarks_branding[:2]).strip()

    if axis_phrase:
        queries.append(collapse_whitespace(axis_phrase))
    if evidence.likely_title and axis_phrase:
        queries.append(collapse_whitespace(f"\"{evidence.likely_title}\" {axis_phrase}"))
    if entities and numbers:
        queries.append(collapse_whitespace(f"{entities} {numbers}"))
    if entities and branding:
        queries.append(collapse_whitespace(f"{entities} {branding}"))
    if axis_phrase and entities:
        queries.append(collapse_whitespace(f"{axis_phrase} {entities}"))
    if evidence.chart_figure_type and entities:
        queries.append(collapse_whitespace(f"{evidence.chart_figure_type} {entities}"))
    if evidence.chart_figure_type and axis_phrase:
        queries.append(collapse_whitespace(f"{evidence.chart_figure_type} {axis_phrase}"))
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
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                'Brave web search request failed for query "%s": %s',
                query,
                format_exception(exc),
            )
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
        candidate.page_summary_text = extract_best_article_excerpt(page_text, evidence)
        candidate.deterministic_page_summary_text = candidate.page_summary_text
        candidate.published_date = published_date

        best_similarity = 0.0
        best_image_url: Optional[str] = None
        for image_url in image_urls[:MAX_PAGE_IMAGES]:
            image_bytes = fetch_candidate_image(image_url)
            if not image_bytes:
                continue
            try:
                candidate_hash = compute_image_hash(image_bytes)
            except ValueError as exc:
                LOGGER.info(
                    "Skipping undecodable candidate image for %s from %s: %s",
                    candidate.url,
                    image_url,
                    format_exception(exc),
                )
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Candidate page fetch failed for %s: %s", url, format_exception(exc))
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Candidate image fetch failed for %s: %s", url, format_exception(exc))
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "image" not in content_type:
        return None
    return response.content


def determine_source_decision(
    candidates: Sequence[CandidateSource],
    raw_reverse_candidates: Sequence[CandidateSource],
) -> Dict[str, Any]:
    """
    Stage 4 decision logic.

    The visual match threshold is the primary gate. Text overlap only helps distinguish between
    multiple already-visual candidates and cannot produce a source decision by itself.
    """

    had_reverse_matches = bool(raw_reverse_candidates)

    if not candidates:
        if had_reverse_matches:
            return {
                "status": "reverse_match_found_but_not_page_verified",
                "candidate": None,
                "match_type": "none",
                "source_evidence": (
                    "Reverse image search returned candidate pages, but none could be carried through "
                    "page-level verification."
                ),
                "confidence": 0.12,
            }
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
        if had_reverse_matches:
            return {
                "status": "reverse_match_found_but_not_page_verified",
                "candidate": None,
                "match_type": "none",
                "source_evidence": (
                    "Reverse image search found candidate pages, but Stage 4 did not verify any candidate "
                    "page as embedding the same or a near-identical image."
                ),
                "confidence": 0.2,
            }
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


def apply_post_decision_page_summary_rewrites(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> None:
    """Rewrite page summaries only after source verification has already been decided."""

    if not llm_context_summary_enabled("page_summary", llm_summary_config):
        return

    context_candidates = select_grounded_context_candidates(candidates, source_decision)
    scope_debug = page_summary_debug_scope(llm_summary_debug)
    if scope_debug is not None:
        rewrite_debug = scope_debug.setdefault(
            "post_decision_rewrites",
            {
                "selected_count": 0,
                "candidates": [],
            },
        )
        if isinstance(rewrite_debug, dict):
            rewrite_debug["selected_count"] = len(context_candidates)

    seen_urls: Set[str] = set()
    for candidate in context_candidates:
        normalized_url = normalize_url(candidate.url)
        if normalized_url in seen_urls:
            record_post_decision_page_summary_rewrite(
                llm_summary_debug,
                candidate,
                deterministic_summary_present=bool(
                    candidate.deterministic_page_summary_text
                    or candidate.page_summary_text
                ),
                skipped_reason="duplicate_candidate_url",
            )
            continue
        seen_urls.add(normalized_url)

        deterministic_summary = (
            candidate.deterministic_page_summary_text
            or candidate.page_summary_text
        )
        if not deterministic_summary:
            record_post_decision_page_summary_rewrite(
                llm_summary_debug,
                candidate,
                deterministic_summary_present=False,
                skipped_reason="missing_deterministic_summary",
            )
            continue

        before_counts = page_summary_rewrite_counts(scope_debug)
        rewritten = maybe_llm_rewrite_page_summary(
            candidate,
            evidence,
            deterministic_summary,
            llm_summary_debug=llm_summary_debug,
            llm_summary_config=llm_summary_config,
        )
        after_counts = page_summary_rewrite_counts(scope_debug)
        attempted = after_counts["attempted"] > before_counts["attempted"]
        used = after_counts["used"] > before_counts["used"]
        failed = after_counts["failed"] > before_counts["failed"]
        changed = bool(rewritten and rewritten != deterministic_summary)
        candidate.page_summary_text = rewritten
        if changed:
            candidate.llm_page_summary_text = rewritten
        record_post_decision_page_summary_rewrite(
            llm_summary_debug,
            candidate,
            deterministic_summary_present=True,
            attempted=attempted,
            used=used,
            changed=changed,
            fallback_used=failed or not used,
            skipped_reason=None,
        )


def build_output_json(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    raw_reverse_candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    search_queries: Sequence[str],
    reverse_search_debug: Dict[str, Any],
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> Dict[str, Any]:
    """
    Stage 5: assemble the final grounded JSON output.

    Only verified evidence and explicitly marked inference are surfaced here. When no source is
    verified, the function returns `source_not_found` with `source_link: null`.
    """

    winning_candidate = source_decision.get("candidate")
    likely_context = build_likely_context(
        evidence,
        candidates,
        source_decision,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
    )
    concise_overview = build_concise_overview(
        evidence,
        candidates,
        source_decision,
        likely_context,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
    )

    source_status = source_decision["status"]
    if source_status == "exact_source_found":
        source_assessment = "An exact or near-exact source page was found."
    elif source_status == "possible_source":
        source_assessment = "A possible source page was found, but it is not fully verified."
    elif source_status == "reverse_match_found_but_not_page_verified":
        source_assessment = "Reverse image search found matches, but no candidate page was visually verified."
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
    raw_reverse_image_candidates = [
        {
            "url": candidate.url,
            "title": candidate.title,
            "snippet": candidate.snippet,
            "result_role": candidate.result_role,
            "notes": candidate.notes,
        }
        for candidate in list(raw_reverse_candidates)[:10]
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
        "raw_reverse_image_candidates": raw_reverse_image_candidates,
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
            "ocr": evidence.ocr_debug,
            "reverse_image_search": reverse_search_debug,
            "llm_summary": llm_summary_debug or initialize_llm_summary_debug(llm_summary_config),
            "reasoning_notes": dedupe_join(
                evidence.extraction_notes
                + [source_assessment]
                + build_reasoning_notes(candidates, raw_reverse_candidates, source_decision)
            ),
        },
    }


def compact_evidence_for_summary(evidence: ImageEvidence) -> Dict[str, Any]:
    """Return a compact grounded evidence payload for optional summary prompts."""

    return {
        "visual_description": trim_to_complete_sentences(evidence.visual_description, max_chars=220, max_sentences=2),
        "likely_title": clean_string(evidence.likely_title),
        "chart_figure_type": clean_string(evidence.chart_figure_type),
        "axes_labels": dedupe_preserve_order(clean_string_list(evidence.axes_labels))[:3],
        "key_entities": dedupe_preserve_order(clean_string_list(evidence.key_entities))[:5],
        "visible_text": dedupe_preserve_order(clean_string_list(evidence.visible_text))[:4],
        "legend_items": dedupe_preserve_order(clean_string_list(evidence.legend_items))[:4],
        "units": dedupe_preserve_order(clean_string_list(evidence.units))[:3],
        "numbers_percentages_dates": dedupe_preserve_order(clean_string_list(evidence.numbers_percentages_dates))[:4],
        "branding": dedupe_preserve_order(clean_string_list(evidence.logos_watermarks_branding))[:3],
    }


def compact_candidate_for_summary(candidate: CandidateSource) -> Dict[str, Any]:
    """Return compact grounded candidate fields for optional summary prompts."""

    deterministic_summary = (
        candidate.deterministic_page_summary_text
        or candidate.page_summary_text
    )
    return {
        "title": trim_to_complete_sentences(candidate.title, max_chars=180, max_sentences=1),
        "snippet": trim_to_complete_sentences(strip_boilerplate_text(candidate.snippet), max_chars=220, max_sentences=2),
        "page_title": trim_to_complete_sentences(candidate.page_title, max_chars=180, max_sentences=1),
        "page_summary_text": trim_to_complete_sentences(
            strip_boilerplate_text(candidate.page_summary_text),
            max_chars=260,
            max_sentences=2,
        ),
        "deterministic_page_summary_text": trim_to_complete_sentences(
            strip_boilerplate_text(deterministic_summary),
            max_chars=320,
            max_sentences=2,
        ),
        "llm_page_summary_text": trim_to_complete_sentences(
            strip_boilerplate_text(candidate.llm_page_summary_text),
            max_chars=260,
            max_sentences=2,
        ),
        "result_role": candidate.result_role,
        "image_embedded": bool(candidate.image_embedded),
        "text_overlap": float(candidate.text_overlap),
    }


def compact_candidates_for_summary(
    candidates: Sequence[CandidateSource],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    """Return compact grounded candidate summaries for prompt inputs."""

    compact: List[Dict[str, Any]] = []
    for candidate in candidates[:limit]:
        item = compact_candidate_for_summary(candidate)
        if any(
            item.get(key)
            for key in (
                "title",
                "snippet",
                "page_title",
                "page_summary_text",
                "deterministic_page_summary_text",
                "llm_page_summary_text",
            )
        ):
            compact.append(item)
    return compact


def make_json_safe(value: Any) -> Any:
    """Convert prompt payloads into stdlib-JSON-safe primitive types."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return make_json_safe(item_method())
        except Exception:  # noqa: BLE001
            pass
    return clean_string(str(value))


def sanitize_llm_single_text(text: str, max_chars: int, max_sentences: int) -> str:
    """Normalize plain-text model output into a compact grounded summary block."""

    cleaned = clean_string(text.replace("\n", " "))
    cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)
    return trim_to_complete_sentences(cleaned, max_chars=max_chars, max_sentences=max_sentences)


def sanitize_llm_multiline_text(text: str, max_lines: int = 6) -> str:
    """Normalize multiline model output into short explanatory lines."""

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = clean_string(re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line))
        if not line:
            continue
        cleaned = trim_to_complete_sentences(line, max_chars=320, max_sentences=2)
        if cleaned and cleaned not in lines:
            lines.append(cleaned)
        if len(lines) == max_lines:
            break
    if lines:
        return "\n".join(lines)
    return trim_to_complete_sentences(clean_string(text), max_chars=420, max_sentences=4)


def build_page_summary_llm_prompt(
    candidate: CandidateSource,
    evidence: ImageEvidence,
    cleaned_excerpt: str,
) -> str:
    """Build the grounded rewrite prompt for a short page-summary excerpt."""

    prompt_payload = make_json_safe({
        "image_evidence": compact_evidence_for_summary(evidence),
        "candidate": {
            "title": candidate.title,
            "snippet": candidate.snippet,
            "page_title": candidate.page_title,
            "cleaned_excerpt": cleaned_excerpt,
        },
    })
    return (
        "Rewrite the cleaned candidate-page excerpt into a short grounded article-style summary.\n"
        "Use only the provided evidence.\n"
        "Rules:\n"
        "- Do not invent source names, communities, trends, or details.\n"
        "- Do not add source-verification claims.\n"
        "- Do not quote or paste long excerpts.\n"
        "- Avoid boilerplate, ellipses, and unfinished fragments.\n"
        "- Return plain text only.\n"
        "- Write 1 to 2 complete sentences.\n"
        f"Evidence:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def build_likely_context_llm_prompt(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    deterministic_context: str,
) -> str:
    """Build the grounded prompt for optional likely-context synthesis."""

    prompt_payload = make_json_safe({
        "image_evidence": compact_evidence_for_summary(evidence),
        "grounded_candidates": compact_candidates_for_summary(candidates, limit=2),
        "source_status": source_decision["status"],
        "source_match_type": source_decision["match_type"],
        "deterministic_context_reference": deterministic_context,
    })
    return (
        "Write one grounded contextual paragraph about what the image likely represents.\n"
        "Use only the provided evidence.\n"
        "Rules:\n"
        "- Explain what the image appears to show and what it is trying to communicate when supported.\n"
        "- Mention axes, categories, or discussion setting only when supported by the provided evidence.\n"
        "- Treat deterministic_page_summary_text as the detail-preserving page evidence when present.\n"
        "- Treat llm_page_summary_text only as a smoother rewrite of that page evidence.\n"
        "- Do not invent chart names, communities, rankings, or source claims.\n"
        "- Do not change or embellish source verification.\n"
        "- If evidence is weak, stay general.\n"
        "- Use complete sentences only. No ellipses.\n"
        "- Return plain text only.\n"
        f"Evidence:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def build_concise_overview_llm_prompt(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    deterministic_overview: str,
) -> str:
    """Build the grounded prompt for the explanatory concise-overview block."""

    prompt_payload = make_json_safe({
        "image_evidence": compact_evidence_for_summary(evidence),
        "grounded_candidates": compact_candidates_for_summary(candidates, limit=2),
        "source_status": source_decision["status"],
        "source_match_type": source_decision["match_type"],
        "deterministic_overview_reference": deterministic_overview,
    })
    return (
        "Write a grounded explanatory summary of the image itself.\n"
        "Use only the provided evidence.\n"
        "Rules:\n"
        "- Start by explaining clearly what the image shows.\n"
        "- When supported, include chart focus, axes, rankings, trends, categories, data types, or main takeaway.\n"
        "- Treat deterministic_page_summary_text as the detail-preserving page evidence when present.\n"
        "- Treat llm_page_summary_text only as a smoother rewrite of that page evidence.\n"
        "- Add a source-status caveat only when relevant to the provided source status.\n"
        "- Avoid repeating raw excerpt text and avoid overclaiming source or originality.\n"
        "- Do not invent chart names, categories, or trends that are not supported.\n"
        "- Use complete sentences only. No ellipses.\n"
        "- Return either one compact paragraph or 3 to 6 short lines separated by newline characters.\n"
        f"Evidence:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def maybe_llm_rewrite_page_summary(
    candidate: CandidateSource,
    evidence: ImageEvidence,
    deterministic_summary: str,
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> str:
    """Optionally rewrite a cleaned page summary while preserving deterministic fallback."""

    if not deterministic_summary or not llm_context_summary_enabled("page_summary", llm_summary_config):
        return deterministic_summary

    try:
        update_llm_summary_debug(llm_summary_debug, "page_summary", attempted=True)
        prompt = build_page_summary_llm_prompt(candidate, evidence, deterministic_summary)
        rewritten = call_openrouter_summary_model(prompt, llm_summary_config=llm_summary_config)
        if not isinstance(rewritten, str):
            update_llm_summary_debug(llm_summary_debug, "page_summary", error="Unusable summary-model response.")
            return deterministic_summary
        cleaned = sanitize_llm_single_text(rewritten, max_chars=320, max_sentences=2)
        if cleaned:
            update_llm_summary_debug(llm_summary_debug, "page_summary", used=True)
            return cleaned
        update_llm_summary_debug(llm_summary_debug, "page_summary", error="Empty summary-model rewrite after sanitization.")
    except Exception as exc:  # noqa: BLE001
        error_message = format_exception(exc)
        update_llm_summary_debug(llm_summary_debug, "page_summary", error=error_message)
        LOGGER.warning("Optional LLM page-summary rewrite failed: %s", error_message)
    return deterministic_summary


def maybe_llm_generate_likely_context(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    deterministic_context: str,
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> str:
    """Optionally generate a smoother likely-context paragraph."""

    if not llm_context_summary_enabled("likely_context", llm_summary_config):
        return deterministic_context

    try:
        update_llm_summary_debug(llm_summary_debug, "likely_context", attempted=True)
        prompt = build_likely_context_llm_prompt(evidence, candidates, source_decision, deterministic_context)
        rewritten = call_openrouter_summary_model(prompt, llm_summary_config=llm_summary_config)
        if isinstance(rewritten, str):
            cleaned = sanitize_llm_single_text(rewritten, max_chars=760, max_sentences=4)
            if cleaned:
                update_llm_summary_debug(llm_summary_debug, "likely_context", used=True)
                return cleaned
        update_llm_summary_debug(llm_summary_debug, "likely_context", error="Empty summary-model likely-context rewrite.")
    except Exception as exc:  # noqa: BLE001
        error_message = format_exception(exc)
        update_llm_summary_debug(llm_summary_debug, "likely_context", error=error_message)
        LOGGER.warning("Optional LLM likely-context rewrite failed: %s", error_message)
    return deterministic_context


def maybe_llm_generate_concise_overview(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    deterministic_overview: str,
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> str:
    """Optionally generate a smoother explanatory concise-overview block."""

    if not llm_context_summary_enabled("concise_overview", llm_summary_config):
        return deterministic_overview

    try:
        update_llm_summary_debug(llm_summary_debug, "concise_overview", attempted=True)
        prompt = build_concise_overview_llm_prompt(evidence, candidates, source_decision, deterministic_overview)
        rewritten = call_openrouter_summary_model(prompt, llm_summary_config=llm_summary_config)
        if isinstance(rewritten, str):
            cleaned = sanitize_llm_multiline_text(rewritten, max_lines=6)
            if cleaned:
                update_llm_summary_debug(llm_summary_debug, "concise_overview", used=True)
                return cleaned
        update_llm_summary_debug(llm_summary_debug, "concise_overview", error="Empty summary-model concise-overview rewrite.")
    except Exception as exc:  # noqa: BLE001
        error_message = format_exception(exc)
        update_llm_summary_debug(llm_summary_debug, "concise_overview", error=error_message)
        LOGGER.warning("Optional LLM concise-overview rewrite failed: %s", error_message)
    return deterministic_overview


def evidence_context_tokens(values: Sequence[str]) -> Set[str]:
    """Normalize evidence strings into informative lowercase tokens."""

    tokens: Set[str] = set()
    for value in values:
        for token in tokenize(value):
            if token not in COMMON_STOPWORDS and len(token) > 2:
                tokens.add(token)
    return tokens


def split_into_sentences(text: str) -> List[str]:
    """Split plain text into sentence-like chunks while preserving order."""

    cleaned = clean_string(text)
    if not cleaned:
        return []

    normalized = cleaned.replace("\xa0", " ")
    normalized = re.sub(r"\s*([|•·])\s*", ". ", normalized)
    normalized = re.sub(r"([.!?])([A-Z0-9])", r"\1 \2", normalized)

    fragments = re.split(r"(?<=[.!?])\s+", normalized)
    sentences: List[str] = []
    for fragment in fragments:
        sentence = clean_string(fragment.strip(" \t\r\n|"))
        if not sentence:
            continue
        word_count = len(re.findall(r"[A-Za-z0-9]+", sentence))
        if word_count < 2 and not re.search(r"[.!?]$", sentence):
            continue
        sentences.append(sentence)
    return sentences


def looks_like_complete_clause(text: str) -> bool:
    """Check whether a clause looks readable enough to stand alone."""

    cleaned = clean_string(text).rstrip(",;:")
    words = re.findall(r"[A-Za-z0-9]+", cleaned.lower())
    if len(words) < 4:
        return False
    if words[0] in {"and", "or", "but", "because", "that", "which", "who"}:
        return False
    if words[-1] in INCOMPLETE_TRAILING_TERMS:
        return False
    return True


def trim_to_complete_clause(text: str, max_chars: int) -> str:
    """Trim long text to a readable clause ending at punctuation before a hard cut."""

    cleaned = clean_string(text)
    if not cleaned:
        return ""

    candidate = cleaned[:max_chars].rstrip()
    clause_breaks = [match.start() for match in re.finditer(r"[,;:]", candidate)]
    for position in reversed(clause_breaks):
        clause = candidate[:position].strip()
        if not looks_like_complete_clause(clause):
            continue
        if clause[-1] not in ".!?":
            clause += "."
        return clause
    return ""


def is_boilerplate_sentence(sentence: str) -> bool:
    """Detect conservative navigation, cookie, and footer-like fragments."""

    cleaned = clean_string(sentence)
    if not cleaned:
        return True

    lowered = cleaned.lower()
    words = re.findall(r"[A-Za-z0-9]+", lowered)
    word_count = len(words)
    compact_words = {word for word in words if word}

    if any(re.search(pattern, lowered) for pattern in INLINE_BOILERPLATE_PATTERNS):
        return True
    if "copyright" in lowered or "newsletter" in lowered:
        return True

    navigation_hits = sum(1 for term in NAVIGATION_TERMS if term in compact_words)
    if navigation_hits >= 2 and word_count <= 14:
        return True
    if word_count <= 3 and compact_words and compact_words.issubset(NAVIGATION_TERMS):
        return True
    if word_count <= 8 and not re.search(r"[.!?]$", cleaned) and navigation_hits >= 1:
        return True
    return False


def strip_boilerplate_text(text: str) -> str:
    """Remove obvious non-article boilerplate while keeping readable prose."""

    cleaned = clean_string(text)
    if not cleaned:
        return ""

    stripped = cleaned
    for pattern in INLINE_BOILERPLATE_PATTERNS:
        stripped = re.sub(pattern, " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(
        r"\b(?:home|about|submit|alerts?|rss|menu|navigation|search|subscribe|privacy|terms|cookie|sign in|log in)\b"
        r"(?:\s+\b(?:home|about|submit|alerts?|rss|menu|navigation|search|subscribe|privacy|terms|cookie|sign in|log in)\b){2,}",
        " ",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = collapse_whitespace(stripped)

    sentences = split_into_sentences(stripped)
    filtered = [sentence for sentence in sentences if not is_boilerplate_sentence(sentence)]
    if filtered:
        return " ".join(filtered)
    return stripped


def score_sentence_for_context(sentence: str, evidence: ImageEvidence) -> float:
    """Rank candidate article sentences against visible image evidence."""

    cleaned = clean_string(sentence)
    if not cleaned:
        return -10.0
    if is_boilerplate_sentence(cleaned):
        return -10.0

    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    word_count = len(words)
    sentence_tokens = {token for token in tokenize(cleaned) if token not in COMMON_STOPWORDS}
    if not sentence_tokens:
        return -5.0

    title_tokens = evidence_context_tokens([evidence.likely_title])
    axis_tokens = evidence_context_tokens(evidence.axes_labels)
    entity_tokens = evidence_context_tokens(evidence.key_entities)
    visible_tokens = evidence_context_tokens(evidence.visible_text)
    branding_tokens = evidence_context_tokens(evidence.logos_watermarks_branding)

    score = 0.0
    score += 3.5 * len(sentence_tokens & title_tokens)
    score += 2.5 * len(sentence_tokens & axis_tokens)
    score += 2.5 * len(sentence_tokens & entity_tokens)
    score += 1.5 * len(sentence_tokens & visible_tokens)
    score += 1.0 * len(sentence_tokens & branding_tokens)

    if 8 <= word_count <= 36:
        score += 1.5
    elif 5 <= word_count <= 50:
        score += 0.8
    elif word_count < 5:
        score -= 1.5
    else:
        score -= 0.4

    if re.search(r"[.!?]$", cleaned):
        score += 0.4
    if re.search(r"\b(?:%s)\b" % "|".join(sorted(PROSE_HINT_TERMS)), cleaned.lower()):
        score += 0.6
    if any(marker in cleaned for marker in ("|", "»", "•")):
        score -= 0.8

    return score


def extract_best_article_excerpt(
    page_text: str,
    evidence: ImageEvidence,
    max_sentences: int = 3,
) -> str:
    """Select a short article-style excerpt that best matches the image evidence."""

    cleaned = strip_boilerplate_text(page_text)
    sentences = split_into_sentences(cleaned)
    if not sentences:
        return trim_to_complete_sentences(cleaned, max_chars=420, max_sentences=max_sentences)

    scored: List[Tuple[float, int, str]] = []
    fallback_indices: List[int] = []
    for index, sentence in enumerate(sentences):
        word_count = len(re.findall(r"[A-Za-z0-9]+", sentence))
        if word_count >= 6 and not is_boilerplate_sentence(sentence):
            fallback_indices.append(index)
        scored.append((score_sentence_for_context(sentence, evidence), index, sentence))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected_indices = [index for score, index, _ in ranked if score > 0][:max_sentences]
    if not selected_indices:
        selected_indices = fallback_indices[:max_sentences]
    if not selected_indices and scored:
        selected_indices = [scored[0][1]]

    excerpt = " ".join(sentences[index] for index in sorted(set(selected_indices)))
    return trim_to_complete_sentences(excerpt, max_chars=420, max_sentences=max_sentences)


def trim_to_complete_sentences(text: str, max_chars: int = 320, max_sentences: int = 2) -> str:
    """Prefer complete sentences over clipped fragments, with hard-trim fallback only when needed."""

    cleaned = clean_string(text)
    if not cleaned:
        return ""

    sentences = split_into_sentences(cleaned)
    if sentences:
        selected: List[str] = []
        for sentence in sentences:
            if len(selected) >= max_sentences:
                break
            candidate = " ".join(selected + [sentence]).strip()
            if selected and len(candidate) > max_chars:
                break
            if not selected and len(sentence) > max_chars:
                break
            selected.append(sentence)
        if selected:
            return " ".join(selected)

        for start_index, sentence in enumerate(sentences[1:], start=1):
            if len(sentence) > max_chars:
                continue
            if len(re.findall(r"[A-Za-z0-9]+", sentence)) < 5:
                continue
            selected = [sentence]
            for follow_sentence in sentences[start_index + 1 :]:
                if len(selected) >= max_sentences:
                    break
                candidate = " ".join(selected + [follow_sentence]).strip()
                if len(candidate) > max_chars:
                    break
                selected.append(follow_sentence)
            if selected:
                return " ".join(selected)

        clause = trim_to_complete_clause(sentences[0], max_chars)
        if clause:
            return clause

    if len(cleaned) <= max_chars:
        return cleaned
    clause = trim_to_complete_clause(cleaned, max_chars)
    if clause:
        return clause
    return cleaned[: max_chars - 3].rsplit(" ", 1)[0].rstrip(",;:-") + "..."


def build_likely_context(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> str:
    """Build a richer article-style context paragraph grounded in image and retrieval evidence."""

    context_candidates = select_grounded_context_candidates(candidates, source_decision)
    subject_sentence = build_subject_context_sentence(evidence, context_candidates)
    framing_sentence = build_visual_framing_sentence(evidence)
    retrieved_sentence = build_retrieved_context_sentence(context_candidates, evidence)
    discussion_sentence = build_discussion_context_sentence(context_candidates)

    sentences = [sentence for sentence in (
        subject_sentence,
        framing_sentence,
        retrieved_sentence,
        discussion_sentence,
    ) if sentence]

    if sentences:
        deterministic_context = trim_to_complete_sentences(" ".join(sentences[:4]), max_chars=720, max_sentences=4)
        return maybe_llm_generate_likely_context(
            evidence,
            context_candidates,
            source_decision,
            deterministic_context,
            llm_summary_debug=llm_summary_debug,
            llm_summary_config=llm_summary_config,
        )
    deterministic_context = trim_to_complete_sentences(
        f"This likely relates to {summarize_subject(evidence)}, but the available grounded context is limited."
    )
    return maybe_llm_generate_likely_context(
        evidence,
        context_candidates,
        source_decision,
        deterministic_context,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
    )


def build_concise_overview(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    likely_context: str,
    llm_summary_debug: Optional[Dict[str, Any]] = None,
    llm_summary_config: Optional[LlmSummaryConfig] = None,
) -> str:
    """Build a grounded human-readable overview after evidence extraction and candidate verification."""

    deterministic_overview = build_grounded_summary(evidence, candidates, source_decision, likely_context)
    context_candidates = select_grounded_context_candidates(candidates, source_decision)
    return maybe_llm_generate_concise_overview(
        evidence,
        context_candidates,
        source_decision,
        deterministic_overview,
        llm_summary_debug=llm_summary_debug,
        llm_summary_config=llm_summary_config,
    )


def build_grounded_summary(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    likely_context: str,
) -> str:
    """Build a compact explanatory summary of the image itself."""

    context_candidates = select_grounded_context_candidates(candidates, source_decision)
    return format_concise_explanatory_summary(
        evidence,
        context_candidates,
        source_decision,
        likely_context,
    )


def format_summary_line(label: str, text: str, max_chars: int = 260, max_sentences: int = 2) -> str:
    """Format a labeled explanatory line while preserving readable sentence endings."""

    cleaned = trim_to_complete_sentences(text, max_chars=max_chars, max_sentences=max_sentences)
    if not cleaned:
        return ""
    return f"{label}: {cleaned}"


def build_concise_focus_line(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
) -> str:
    """Summarize the chart or graphic focus in one grounded line."""

    subject_label = subject_label_from_evidence(evidence, candidates)
    candidate_topics = ""
    if candidates:
        candidate_topics = join_readable_list(extract_candidate_context_terms(candidates[0], evidence))

    if evidence.likely_title:
        return format_summary_line("Chart focus", f'The clearest visible title is "{evidence.likely_title}".')
    if candidate_topics:
        return format_summary_line("Chart focus", f"The retrieved context points to themes such as {candidate_topics}.")
    if subject_label:
        return format_summary_line("Chart focus", f"The image appears to center on {subject_label}.")
    return ""


def build_axis_structure_line(evidence: ImageEvidence) -> str:
    """Explain visible axes, ordering, or other chart structure."""

    axis_labels = dedupe_preserve_order(clean_string_list(evidence.axes_labels))
    if len(axis_labels) >= 2:
        return format_summary_line("Axes", f"The chart compares {axis_labels[0]} against {axis_labels[1]}.")
    if len(axis_labels) == 1:
        return format_summary_line("Structure", f'One visible organizing label is "{axis_labels[0]}".')
    if evidence.numbers_percentages_dates:
        values = join_readable_list(evidence.numbers_percentages_dates[:3])
        return format_summary_line("Structure", f"The image includes numeric or date-like values such as {values}.")
    return ""


def build_data_types_line(evidence: ImageEvidence) -> str:
    """Summarize visible categories, legend items, or entities shown in the image."""

    labels = dedupe_preserve_order(
        clean_string_list(evidence.key_entities)
        + clean_string_list(evidence.legend_items)
        + clean_string_list(evidence.units)
    )
    if labels:
        return format_summary_line("Data shown", f"It includes categories or labels such as {join_readable_list(labels[:5])}.")
    if evidence.visible_text:
        preview = join_readable_list(evidence.visible_text[:3])
        return format_summary_line("Visible text", f"The graphic includes labels such as {preview}.")
    return ""


def build_main_takeaway_line(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
) -> str:
    """Summarize the clearest visible takeaway from the image."""

    framing_sentence = build_visual_framing_sentence(evidence)
    if framing_sentence:
        return format_summary_line("Main takeaway", framing_sentence, max_chars=300, max_sentences=2)

    candidate_takeaway = build_candidate_takeaway_sentence(candidates, evidence)
    if candidate_takeaway:
        return format_summary_line("Main takeaway", candidate_takeaway, max_chars=260, max_sentences=1)

    if evidence.visual_description:
        return format_summary_line("Main takeaway", evidence.visual_description, max_chars=300, max_sentences=2)
    return ""


def build_source_caveat_line(source_decision: Dict[str, Any]) -> str:
    """Add a source-verification caveat when the source is not fully verified."""

    if source_decision["status"] == "exact_source_found":
        return ""
    source_sentence = build_source_status_sentence(source_decision)
    if not source_sentence:
        return ""
    return format_summary_line("Source note", source_sentence, max_chars=300, max_sentences=2)


def format_concise_explanatory_summary(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    likely_context: str,
) -> str:
    """Build a structured explanatory summary distinct from the contextual background paragraph."""

    summary_lines = [
        trim_to_complete_sentences(
            build_subject_context_sentence(evidence, candidates),
            max_chars=320,
            max_sentences=2,
        ),
        build_concise_focus_line(evidence, candidates),
        build_axis_structure_line(evidence),
        build_data_types_line(evidence),
        build_main_takeaway_line(evidence, candidates),
        build_source_caveat_line(source_decision),
    ]

    lines = dedupe_preserve_order([line for line in summary_lines if line])
    if len(lines) < 3:
        fallback_lines = [
            format_summary_line("Visible signal", build_text_signal_sentence(evidence), max_chars=280, max_sentences=2),
            format_summary_line("Context", likely_context, max_chars=320, max_sentences=2),
        ]
        lines.extend([line for line in fallback_lines if line and line not in lines])

    if lines:
        return "\n".join(lines[:6])
    return trim_to_complete_sentences(
        evidence.visual_description or "A clear grounded overview could not be assembled from the available evidence.",
        max_chars=320,
        max_sentences=2,
    )


def select_grounded_context_candidates(
    candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
    limit: int = 3,
) -> List[CandidateSource]:
    """Choose the strongest grounded candidates for context and summary synthesis."""

    selected: List[CandidateSource] = []
    winning_candidate = source_decision.get("candidate")
    if isinstance(winning_candidate, CandidateSource):
        selected.append(winning_candidate)

    ranked_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if candidate not in selected
        ),
        key=lambda candidate: (-score_candidate_for_context(candidate), -candidate.best_image_similarity, candidate.url),
    )

    for candidate in ranked_candidates:
        context_score = score_candidate_for_context(candidate)
        if context_score < 2.0:
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break

    return selected[:limit]


def score_candidate_for_context(candidate: CandidateSource) -> float:
    """Rank candidates for summary context without affecting source verification."""

    summary_text = strip_boilerplate_text(
        dedupe_join([
            candidate.page_summary_text,
            candidate.deterministic_page_summary_text,
        ])
    )
    snippet = strip_boilerplate_text(candidate.snippet)
    headline = candidate_headline(candidate)

    score = 0.0
    if candidate.image_embedded:
        score += 3.0
    score += min(candidate.text_overlap, 0.35) * 10.0
    if summary_text:
        score += 2.5
        word_count = len(re.findall(r"[A-Za-z0-9]+", summary_text))
        if 8 <= word_count <= 80:
            score += 0.8
    if headline:
        score += 1.0
    if snippet:
        score += 0.8
    if not candidate.image_embedded and candidate.text_overlap < 0.08:
        score -= 1.5
    if not summary_text and not snippet:
        score -= 1.0
    return score


def candidate_headline(candidate: CandidateSource) -> str:
    """Return the best available human-readable headline for a grounded candidate."""

    for value in (candidate.page_title, candidate.title):
        cleaned = clean_string(value)
        if cleaned:
            return cleaned
    return ""


def candidate_context_excerpt(candidate: CandidateSource, limit: int = 240) -> str:
    """Return the strongest contextual text available from a grounded candidate page."""

    for value in (
        candidate.page_summary_text,
        candidate.deterministic_page_summary_text,
        candidate.snippet,
        candidate.page_title,
        candidate.title,
    ):
        cleaned = strip_boilerplate_text(clean_string(value))
        if cleaned:
            return trim_to_complete_sentences(cleaned, max_chars=limit, max_sentences=2)
    return ""


def join_readable_list(values: Sequence[str]) -> str:
    """Join short topic labels into readable prose."""

    items = [clean_string(value) for value in values if clean_string(value)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def extract_candidate_context_terms(
    candidate: CandidateSource,
    evidence: ImageEvidence,
    max_terms: int = 3,
) -> List[str]:
    """Extract short grounded topic terms from candidate text fields."""

    fields = [
        candidate.page_title,
        candidate.title,
        candidate.snippet,
        candidate.page_summary_text,
        candidate.deterministic_page_summary_text,
        candidate.llm_page_summary_text,
    ]
    evidence_tokens = build_evidence_tokens(evidence)
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0

    for field in fields:
        cleaned = strip_boilerplate_text(field)
        if not cleaned:
            continue
        for token in tokenize(cleaned):
            token = token.lower()
            if (
                token in COMMON_STOPWORDS
                or token in NAVIGATION_TERMS
                or token in COMMENTARY_HINTS
                or token in GENERIC_CONTEXT_TOKENS
                or len(token) < 4
                or not re.search(r"[a-z]", token)
                or not re.fullmatch(r"[a-z][a-z0-9'-]*", token)
            ):
                continue
            if token not in first_seen:
                first_seen[token] = order
                order += 1
            scores[token] = scores.get(token, 0.0) + 1.0
            if token in evidence_tokens:
                scores[token] += 1.0

    ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return [token for token, _ in ranked[:max_terms]]


def summarize_candidate_context(candidate: CandidateSource, evidence: ImageEvidence) -> str:
    """Synthesize grounded candidate context without pasting excerpt text verbatim."""

    headline = candidate_headline(candidate)
    topics = join_readable_list(extract_candidate_context_terms(candidate, evidence))
    if topics and headline:
        return trim_to_complete_sentences(
            f'A grounded candidate page titled "{headline}" frames the image as part of a discussion about {topics}.',
            max_chars=320,
            max_sentences=1,
        )
    if topics:
        return trim_to_complete_sentences(
            f"Retrieved candidate evidence suggests the image is discussed in a context related to {topics}.",
            max_chars=300,
            max_sentences=1,
        )
    if headline:
        return trim_to_complete_sentences(
            f'A grounded candidate page titled "{headline}" appears to discuss the same subject as the image.',
            max_chars=280,
            max_sentences=1,
        )
    excerpt = candidate_context_excerpt(candidate, limit=220)
    if excerpt:
        return trim_to_complete_sentences(
            "Retrieved candidate evidence supports the same general subject area as the image.",
            max_chars=220,
            max_sentences=1,
        )
    return ""


def subject_label_from_evidence(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
) -> str:
    """Choose the best grounded subject label for the image."""

    if evidence.likely_title:
        return clean_string(evidence.likely_title)
    if evidence.key_entities:
        return ", ".join(evidence.key_entities[:3])
    for line in evidence.visible_text:
        cleaned = clean_string(line)
        token_count = len(re.findall(r"[A-Za-z0-9]+", cleaned))
        if 2 <= token_count <= 8:
            return cleaned
    for candidate in candidates:
        headline = candidate_headline(candidate)
        if headline:
            return headline
    return summarize_subject(evidence)


def build_subject_context_sentence(
    evidence: ImageEvidence,
    candidates: Sequence[CandidateSource],
) -> str:
    """Describe what the image appears to be about using the strongest available subject signals."""

    subject_label = subject_label_from_evidence(evidence, candidates)
    chart_type = clean_string(evidence.chart_figure_type)
    if evidence.likely_title and chart_type and chart_type.lower() != "unknown":
        return trim_to_complete_sentences(f'This image appears to be a {chart_type} titled "{subject_label}".')
    if evidence.likely_title:
        return trim_to_complete_sentences(f'This image appears to center on "{subject_label}".')
    if chart_type and chart_type.lower() != "unknown":
        return trim_to_complete_sentences(f"This image appears to be a {chart_type} about {subject_label}.")
    if subject_label:
        return trim_to_complete_sentences(f"This image appears to relate to {subject_label}.")
    return ""


def build_visual_framing_sentence(evidence: ImageEvidence) -> str:
    """Describe what the chart or image appears to be communicating from visible labels and entities."""

    axis_labels = dedupe_preserve_order(clean_string_list(evidence.axes_labels))
    if len(axis_labels) >= 2:
        sentence = f"The visible framing suggests it is comparing {axis_labels[0]} against {axis_labels[1]}."
        if evidence.key_entities:
            sentence += f" The labeled items point to topics such as {', '.join(evidence.key_entities[:3])}."
        return trim_to_complete_sentences(sentence, max_chars=300, max_sentences=2)
    if len(axis_labels) == 1:
        return trim_to_complete_sentences(
            f"The visible labeling suggests the image is organized around {axis_labels[0]}."
        )
    if evidence.key_entities:
        return trim_to_complete_sentences(
            f"The visible labels and text suggest it is explaining or comparing topics such as {', '.join(evidence.key_entities[:3])}."
        )
    if evidence.visible_text:
        preview = "; ".join(evidence.visible_text[:2])
        return trim_to_complete_sentences(
            f'Visible text suggests a labeled comparison or discussion that includes "{preview}".',
            max_chars=300,
        )
    return ""


def build_retrieved_context_sentence(
    candidates: Sequence[CandidateSource],
    evidence: ImageEvidence,
) -> str:
    """Add grounded page context from the strongest verified candidate pages."""

    if not candidates:
        return ""

    anchor = candidates[0]
    return summarize_candidate_context(anchor, evidence)


def build_discussion_context_sentence(candidates: Sequence[CandidateSource]) -> str:
    """Describe where the image appears to be shared or discussed when grounded candidates support it."""

    discussion_domains = dedupe_preserve_order(
        [
            urlparse(candidate.url).netloc.lower().removeprefix("www.")
            for candidate in candidates
            if candidate.result_role in {"commentary", "repost"}
        ]
    )
    if discussion_domains:
        preview = ", ".join(discussion_domains[:3])
        return trim_to_complete_sentences(
            f"Additional grounded candidates suggest the image is also being shared or discussed in commentary-oriented settings such as {preview}."
        )
    return ""


def build_text_signal_sentence(evidence: ImageEvidence) -> str:
    """Surface the strongest visible-text or title signal without collapsing into a visual caption."""

    if evidence.likely_title:
        return trim_to_complete_sentences(f'The strongest visible title signal is "{evidence.likely_title}".')
    if evidence.visible_text:
        text_preview = "; ".join(evidence.visible_text[:3])
        return trim_to_complete_sentences(
            f'Visible labels and text include "{text_preview}".',
            max_chars=300,
        )
    return ""


def build_candidate_takeaway_sentence(
    candidates: Sequence[CandidateSource],
    evidence: ImageEvidence,
) -> str:
    """Summarize what grounded candidate pages seem to say the image is communicating."""

    if not candidates:
        return ""

    anchor = candidates[0]
    topics = join_readable_list(extract_candidate_context_terms(anchor, evidence))
    if topics:
        return trim_to_complete_sentences(
            f"Grounded candidate evidence points to a discussion about {topics}.",
            max_chars=240,
            max_sentences=1,
        )
    if candidate_headline(anchor):
        return trim_to_complete_sentences(
            "Grounded candidate evidence supports the same interpretation of the image.",
            max_chars=240,
            max_sentences=1,
        )
    return ""


def build_source_status_sentence(source_decision: Dict[str, Any]) -> str:
    """Describe the current source-verification state in summary-friendly prose."""

    source_status = source_decision["status"]
    candidate = source_decision.get("candidate")
    if source_status == "source_not_found":
        return "No candidate page was verified as the source through an embedded image match."
    if source_status == "reverse_match_found_but_not_page_verified":
        return "Reverse image search surfaced candidate pages, but none was verified at the page level."
    if isinstance(candidate, CandidateSource):
        display_title = candidate_headline(candidate) or candidate.url
        return trim_to_complete_sentences(
            f'The strongest grounded candidate is "{display_title}", with an embedded-image similarity score of {candidate.best_image_similarity:.2f}.',
            max_chars=260,
        )
    return ""


def build_reasoning_notes(
    candidates: Sequence[CandidateSource],
    raw_reverse_candidates: Sequence[CandidateSource],
    source_decision: Dict[str, Any],
) -> List[str]:
    """Summarize search and verification behavior for debugging without exposing chain-of-thought."""

    notes = [f"Collected {len(candidates)} deduplicated candidate pages."]
    notes.append(f"Collected {len(raw_reverse_candidates)} raw reverse-image candidates.")
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
        except ValueError as exc:
            LOGGER.debug("Could not parse meta date '%s' as ISO datetime: %s", cleaned, format_exception(exc))
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

    return trim_to_complete_sentences(text, max_chars=limit, max_sentences=1)


def resolve_cli_bool_override(
    enabled_flag: bool,
    disabled_flag: bool,
    setting_name: str,
) -> Optional[bool]:
    """Resolve paired CLI enable/disable flags, or raise on conflicts."""

    if enabled_flag and disabled_flag:
        raise ValueError(f"Conflicting CLI flags for {setting_name}: both enable and disable were provided.")
    if enabled_flag:
        return True
    if disabled_flag:
        return False
    return None


def build_llm_summary_config_from_args(args: argparse.Namespace) -> Optional[LlmSummaryConfig]:
    """Build explicit CLI overrides while leaving env as the baseline when omitted."""

    config = LlmSummaryConfig(
        enable_llm_context_summary=resolve_cli_bool_override(
            args.enable_llm_summary,
            args.disable_llm_summary,
            "LLM summary",
        ),
        page_summary_enabled=resolve_cli_bool_override(
            args.enable_page_summary,
            args.disable_page_summary,
            "page summary",
        ),
        likely_context_enabled=resolve_cli_bool_override(
            args.enable_likely_context,
            args.disable_likely_context,
            "likely context",
        ),
        concise_overview_enabled=resolve_cli_bool_override(
            args.enable_concise_overview,
            args.disable_concise_overview,
            "concise overview",
        ),
        summary_model=args.llm_summary_model.strip() if args.llm_summary_model else None,
        timeout_seconds=args.llm_summary_timeout,
    )

    if all(
        value is None
        for value in (
            config.enable_llm_context_summary,
            config.page_summary_enabled,
            config.likely_context_enabled,
            config.concise_overview_enabled,
            config.summary_model,
            config.timeout_seconds,
        )
    ):
        return None
    return config


def main() -> None:
    """CLI entrypoint for local runs or agent pipelines."""

    log_level_name = os.getenv("RETRIEVE_IMAGE_BACKGROUND_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Retrieve grounded background and source evidence for an image.")
    parser.add_argument("--image", required=True, help="Image input as URL, data URL, base64, or local path.")
    parser.add_argument("--user-query", default="", help="Optional user question about the image.")
    parser.add_argument("--enable-llm-summary", action="store_true", help="Enable optional LLM-assisted summary rewriting.")
    parser.add_argument("--disable-llm-summary", action="store_true", help="Disable optional LLM-assisted summary rewriting.")
    parser.add_argument("--enable-page-summary", action="store_true", help="Enable LLM rewriting for candidate page summaries.")
    parser.add_argument("--disable-page-summary", action="store_true", help="Disable LLM rewriting for candidate page summaries.")
    parser.add_argument("--enable-likely-context", action="store_true", help="Enable LLM rewriting for likely_context.")
    parser.add_argument("--disable-likely-context", action="store_true", help="Disable LLM rewriting for likely_context.")
    parser.add_argument("--enable-concise-overview", action="store_true", help="Enable LLM rewriting for concise_overview.")
    parser.add_argument("--disable-concise-overview", action="store_true", help="Disable LLM rewriting for concise_overview.")
    parser.add_argument("--llm-summary-model", help="Override the OpenRouter model used for optional summary rewrites.")
    parser.add_argument("--llm-summary-timeout", type=int, help="Override the timeout in seconds for optional summary rewrites.")
    args = parser.parse_args()

    try:
        llm_summary_config = build_llm_summary_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    result = retrieve_image_background(args.image, args.user_query, llm_summary_config=llm_summary_config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
