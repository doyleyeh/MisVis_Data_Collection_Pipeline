from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from retrieve_image_background import LlmSummaryConfig, retrieve_image_background


LOGGER = logging.getLogger("retrieve_pipeline_test")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
REVIEW_FIELDS = (
    "likely_context",
    "concise_overview",
    "source_status",
    "most_likely_source",
    "source_link",
    "confidence",
)
VARIANT_ORDER = ("det", "llm-page-summary", "llm-all", "llm-final-no-page")
VARIANT_OUTPUT_FILENAMES = {
    "det": "{image_index}_det.json",
    "llm-page-summary": "{image_index}_llm-page-summary.json",
    "llm-all": "{image_index}_llm-all.json",
    "llm-final-no-page": "{image_index}_llm-final-no-page.json",
}
VARIANT_DISPLAY_NAMES = {
    "det": "Deterministic",
    "llm-page-summary": "LLM Page Summary Only",
    "llm-all": "LLM All",
    "llm-final-no-page": "LLM Context + Overview (No Page Summary)",
}


@dataclass(frozen=True)
class DiscoveredImage:
    """Local image selected for pipeline testing."""

    path: Path
    image_index: str
    image_index_value: int


@dataclass
class VariantRunResult:
    """In-memory record of a single variant run."""

    variant_name: str
    output_path: Path
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    """Build the CLI for local pipeline testing."""

    parser = argparse.ArgumentParser(
        description="Batch-run retrieve_image_background.py on local images and save comparison artifacts."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Single local image file or a folder containing local test images.",
    )
    parser.add_argument(
        "--user-query",
        default="",
        help="Optional shared user query passed into all retrieve_image_background runs.",
    )
    parser.add_argument(
        "--image-index",
        help="Optional numeric image index filter. Accepts values like 14 or 000014.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of images to process after discovery and sorting.",
    )
    parser.add_argument(
        "--json-output-dir",
        default="test_json_output",
        help="Directory for full JSON outputs.",
    )
    parser.add_argument(
        "--md-output-dir",
        default="test_md_output",
        help="Directory for markdown comparison files.",
    )
    return parser.parse_args()


def extract_image_index(path: Path) -> tuple[str, int]:
    """Extract the leading numeric filename prefix used as the image index."""

    match = re.match(r"^(\d+)", path.name)
    if not match:
        raise ValueError(f"Filename does not start with a numeric image index: {path.name}")

    image_index = match.group(1)
    return image_index, int(image_index)


def is_supported_image_file(path: Path) -> bool:
    """Return whether the path looks like a supported local image file."""

    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def normalize_requested_image_index(value: Optional[str]) -> Optional[int]:
    """Normalize a CLI image-index filter into an integer for matching."""

    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned or not cleaned.isdigit():
        raise ValueError("--image-index must contain only digits.")
    return int(cleaned)


def discover_images(
    input_path: Path,
    image_index_filter: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[DiscoveredImage]:
    """Discover local images, sort them by numeric index, then apply filters."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    discovered: List[DiscoveredImage] = []
    seen_output_indices: set[str] = set()

    if input_path.is_file():
        if not is_supported_image_file(input_path):
            raise ValueError(
                f"Unsupported image file: {input_path}. Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        image_index, image_index_value = extract_image_index(input_path)
        discovered.append(
            DiscoveredImage(path=input_path.resolve(), image_index=image_index, image_index_value=image_index_value)
        )
    else:
        for child in input_path.iterdir():
            if not is_supported_image_file(child):
                continue
            try:
                image_index, image_index_value = extract_image_index(child)
            except ValueError:
                LOGGER.warning("Skipping supported image without numeric prefix: %s", child)
                continue

            if image_index in seen_output_indices:
                LOGGER.warning(
                    "Skipping %s because image index %s would overwrite another output file.",
                    child,
                    image_index,
                )
                continue

            discovered.append(
                DiscoveredImage(path=child.resolve(), image_index=image_index, image_index_value=image_index_value)
            )
            seen_output_indices.add(image_index)

    discovered.sort(key=lambda item: (item.image_index_value, item.path.name.lower()))

    if image_index_filter is not None:
        discovered = [item for item in discovered if item.image_index_value == image_index_filter]

    if limit is not None:
        discovered = discovered[:limit]

    return discovered


def ensure_output_dirs(*paths: Path) -> None:
    """Create output directories when they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_variant_config(variant_name: str) -> LlmSummaryConfig:
    """Build an explicit LLM summary config for a named test variant."""

    configs = {
        "det": LlmSummaryConfig(
            enable_llm_context_summary=False,
            page_summary_enabled=False,
            likely_context_enabled=False,
            concise_overview_enabled=False,
        ),
        "llm-page-summary": LlmSummaryConfig(
            enable_llm_context_summary=True,
            page_summary_enabled=True,
            likely_context_enabled=False,
            concise_overview_enabled=False,
        ),
        "llm-all": LlmSummaryConfig(
            enable_llm_context_summary=True,
            page_summary_enabled=True,
            likely_context_enabled=True,
            concise_overview_enabled=True,
        ),
        "llm-final-no-page": LlmSummaryConfig(
            enable_llm_context_summary=True,
            page_summary_enabled=False,
            likely_context_enabled=True,
            concise_overview_enabled=True,
        ),
    }
    try:
        return configs[variant_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported variant: {variant_name}") from exc


def build_error_payload(
    image: DiscoveredImage,
    variant_name: str,
    error_message: str,
) -> Dict[str, Any]:
    """Persist structured failure information alongside successful JSON outputs."""

    return {
        "image_index": image.image_index,
        "image_path": str(image.path),
        "variant": variant_name,
        "error": error_message,
    }


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Write a JSON payload with stable UTF-8 formatting."""

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_variant_output_path(
    image: DiscoveredImage,
    variant_name: str,
    json_output_dir: Path,
) -> Path:
    """Resolve the exact JSON output path for a variant."""

    filename_template = VARIANT_OUTPUT_FILENAMES[variant_name]
    return json_output_dir / filename_template.format(image_index=image.image_index)


def build_enabled_variants(args: argparse.Namespace) -> List[str]:
    """Return the ordered list of variants requested for this run."""

    return list(VARIANT_ORDER)


def run_one_variant(
    image: DiscoveredImage,
    variant_name: str,
    user_query: str,
    json_output_dir: Path,
) -> VariantRunResult:
    """Run one explicit variant, save JSON output, and capture failures without aborting the batch."""

    output_path = format_variant_output_path(image, variant_name, json_output_dir)
    config = build_variant_config(variant_name)

    LOGGER.info(
        "Running variant %s for image %s (%s)",
        variant_name,
        image.image_index,
        image.path.name,
    )
    try:
        result = retrieve_image_background(
            str(image.path),
            user_query=user_query,
            llm_summary_config=config,
        )
        save_json(result, output_path)
        LOGGER.info("Saved %s", output_path)
        return VariantRunResult(variant_name=variant_name, output_path=output_path, data=result)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        LOGGER.exception(
            "Variant %s failed for image %s (%s)",
            variant_name,
            image.image_index,
            image.path,
        )
        error_payload = build_error_payload(image, variant_name, error_message)
        save_json(error_payload, output_path)
        LOGGER.info("Saved failure report %s", output_path)
        return VariantRunResult(
            variant_name=variant_name,
            output_path=output_path,
            data=error_payload,
            error=error_message,
        )


def render_review_value(value: Any) -> str:
    """Render a markdown-friendly string for a comparison field."""

    if value is None:
        return "_Not available._"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "_Not available._"
    return str(value)


def build_skipped_variant_result(image: DiscoveredImage, variant_name: str, json_output_dir: Path) -> VariantRunResult:
    """Create an in-memory placeholder for a variant that was intentionally not run."""

    return VariantRunResult(
        variant_name=variant_name,
        output_path=format_variant_output_path(image, variant_name, json_output_dir),
        data={"variant": variant_name, "status": "skipped", "note": "Variant not requested for this run."},
        skipped=True,
    )


def build_review_section(title: str, result: Optional[VariantRunResult]) -> str:
    """Build one markdown section for human review."""

    lines = [f"## {title}", ""]
    if result and result.skipped:
        for field_name in REVIEW_FIELDS:
            lines.append(f"### {field_name}")
            lines.append("_Not run._")
            lines.append("")
        return "\n".join(lines).rstrip()

    payload = result.data if result else None
    error_message = ""
    if isinstance(payload, Mapping):
        error_value = payload.get("error")
        if isinstance(error_value, str) and error_value.strip():
            error_message = error_value.strip()

    for field_name in REVIEW_FIELDS:
        lines.append(f"### {field_name}")
        if error_message:
            lines.append(f"ERROR: {error_message}")
        else:
            field_value = payload.get(field_name) if isinstance(payload, Mapping) else None
            lines.append(render_review_value(field_value))
        lines.append("")

    return "\n".join(lines).rstrip()


def build_markdown_comparison(
    image: DiscoveredImage,
    variant_results: Mapping[str, VariantRunResult],
) -> str:
    """Build the markdown comparison document for all configured variants."""

    parts = [f"# Image {image.image_index}", ""]

    for variant_name in VARIANT_ORDER:
        parts.append(build_review_section(VARIANT_DISPLAY_NAMES[variant_name], variant_results.get(variant_name)))
        parts.append("")

    parts.extend(
        [
            "## Review Notes",
            "",
        ]
    )
    return "\n".join(parts)


def save_markdown(markdown_text: str, output_path: Path) -> None:
    """Write the comparison markdown file."""

    output_path.write_text(markdown_text, encoding="utf-8")


def process_image(
    image: DiscoveredImage,
    user_query: str,
    json_output_dir: Path,
    md_output_dir: Path,
    enabled_variants: List[str],
) -> Dict[str, VariantRunResult]:
    """Run all variants for one image and create the comparison markdown file."""

    variant_results: Dict[str, VariantRunResult] = {}

    for variant_name in enabled_variants:
        variant_results[variant_name] = run_one_variant(
            image=image,
            variant_name=variant_name,
            user_query=user_query,
            json_output_dir=json_output_dir,
        )

    for variant_name in VARIANT_ORDER:
        if variant_name not in variant_results:
            variant_results[variant_name] = build_skipped_variant_result(image, variant_name, json_output_dir)

    markdown_output_path = md_output_dir / f"{image.image_index}_compare.md"
    markdown_text = build_markdown_comparison(image=image, variant_results=variant_results)
    save_markdown(markdown_text, markdown_output_path)
    LOGGER.info("Saved %s", markdown_output_path)

    return variant_results


def iter_selected_images(images: Iterable[DiscoveredImage]) -> Iterable[tuple[int, DiscoveredImage]]:
    """Enumerate selected images in a dedicated helper for future extension."""

    return enumerate(images, start=1)


def main() -> None:
    """CLI entrypoint for local retrieval-pipeline tests."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    input_path = Path(args.input_path).expanduser()
    json_output_dir = Path(args.json_output_dir).expanduser()
    md_output_dir = Path(args.md_output_dir).expanduser()

    try:
        selected_image_index = normalize_requested_image_index(args.image_index)
        images = discover_images(
            input_path=input_path,
            image_index_filter=selected_image_index,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not images:
        LOGGER.warning("No images matched the provided input and filters.")
        return

    ensure_output_dirs(json_output_dir, md_output_dir)
    enabled_variants = build_enabled_variants(args)
    LOGGER.info("Discovered %s image(s) to process.", len(images))
    LOGGER.info("JSON output directory: %s", json_output_dir.resolve())
    LOGGER.info("Markdown output directory: %s", md_output_dir.resolve())
    LOGGER.info("Enabled variants: %s", ", ".join(enabled_variants))

    processed_count = 0
    failure_count = 0

    for position, image in iter_selected_images(images):
        LOGGER.info(
            "[%s/%s] Processing image %s from %s",
            position,
            len(images),
            image.image_index,
            image.path,
        )
        try:
            results = process_image(
                image=image,
                user_query=args.user_query,
                json_output_dir=json_output_dir,
                md_output_dir=md_output_dir,
                enabled_variants=enabled_variants,
            )
        except Exception:
            failure_count += 1
            LOGGER.exception("Image-level failure for %s (%s)", image.image_index, image.path)
            continue

        processed_count += 1
        failure_count += sum(1 for result in results.values() if result.error)

    LOGGER.info(
        "Finished processing. Images completed: %s/%s. Variant failures recorded: %s.",
        processed_count,
        len(images),
        failure_count,
    )


if __name__ == "__main__":
    main()
