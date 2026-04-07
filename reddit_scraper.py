import csv
import html
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, TextIO

import requests
from PIL import Image
import imagehash
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


# Configurable parameters
POST_LIMIT = 2000
SUBREDDIT = "dataisugly"
OUTPUT_DIR = "reddit/mis_fig"
OUTPUT_CSV = "reddit/data_is_ugly_output.csv"
SIMILARITY_THRESHOLD = 0.8  # phash similarity threshold (80%) for considering duplicates

# Request settings
USER_AGENT = "dataisugly-scraper/0.1 (contact: local script)"
HEADERS = {"User-Agent": USER_AGENT}

# Allowed image extensions
VALID_EXTS = {".jpg", ".jpeg", ".png"}

# Sort order priority (sort, time_filter)
SORTS: List[Tuple[str, Optional[str]]] = [
    ("new", None),
    ("hot", None),
    ("rising", None),
    ("best", None),
    ("top", "day"),
    ("top", "week"),
    ("top", "month"),
    ("top", "year"),
    ("top", "all"),
    ("controversial", "day"),
    ("controversial", "week"),
    ("controversial", "month"),
    ("controversial", "year"),
    ("controversial", "all"),
]

# Network resilience
FETCH_RETRIES = 3
FETCH_RETRY_BACKOFF = 1.5

current_progress = None
POST_ID_REGEX = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)


class NullTqdm:
    """Minimal tqdm-like fallback when tqdm is unavailable."""

    def __init__(self, total: int, initial: int = 0, desc: str = "", unit: str = "") -> None:
        self.total = total
        self.n = initial
        self.desc = desc
        self.unit = unit

    def update(self, n: int = 1) -> None:
        self.n += n

    def write(self, msg: str) -> None:
        print(msg)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def load_existing(csv_path: Path, output_dir: Path) -> Tuple[Set[str], Set[str], List[imagehash.ImageHash], Set[str], int]:
    """Load already-processed links, post IDs, and phashes from CSV; find the highest index."""
    existing_links: Set[str] = set()
    existing_post_ids: Set[str] = set()
    existing_phash_strings: Set[str] = set()
    existing_hashes: List[imagehash.ImageHash] = []
    max_index = 0

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                link = row.get("post_link")
                phash = row.get("phash")
                file_name = row.get("file_name")
                post_id = row.get("post_id")

                if phash:
                    existing_phash_strings.add(phash)
                    try:
                        existing_hashes.append(imagehash.hex_to_hash(phash))
                    except Exception:
                        # If a stored hash cannot be parsed, skip it silently
                        pass

                if post_id:
                    existing_post_ids.add(post_id)

                # Skip only when CSV entry has a corresponding file
                if link and file_name and (output_dir / file_name).exists():
                    existing_links.add(link)

    for file in output_dir.iterdir():
        match = re.match(r"(\d{6})_", file.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return existing_links, existing_phash_strings, existing_hashes, existing_post_ids, max_index


def ensure_csv(csv_path: Path) -> Tuple[csv.writer, TextIO]:
    """Create CSV with header if missing and return writer and handle in append mode."""
    new_file = not csv_path.exists()
    ensure_parent(csv_path)
    file_handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    if new_file:
        writer.writerow(["post_id", "file_name", "post_link", "post_title", "post_context", "phash"])
    return writer, file_handle


def ensure_parent(path: Path) -> None:
    """Ensure parent directories exist."""
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_post_id_from_url(url: str) -> Optional[str]:
    """Extract base36 post id from a reddit comments URL."""
    match = POST_ID_REGEX.search(url)
    if match:
        return match.group(1)
    return None


def extract_post_id(data: Dict, post_link: Optional[str]) -> Optional[str]:
    """Derive post id from JSON payload or URL pattern."""
    post_id = data.get("id") if isinstance(data, dict) else None
    if post_id:
        return str(post_id)
    if post_link:
        from_link = parse_post_id_from_url(post_link)
        if from_link:
            return from_link
    return None


def upgrade_csv_schema(csv_path: Path) -> None:
    """If an existing CSV lacks post_id, rewrite it to add the column safely."""
    if not csv_path.exists():
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "post_id" in fieldnames:
            return
        rows = list(reader)

    upgraded_rows = []
    for row in rows:
        post_link = row.get("post_link", "")
        derived_post_id = parse_post_id_from_url(post_link) or ""
        upgraded_rows.append({
            "post_id": derived_post_id,
            "file_name": row.get("file_name", ""),
            "post_link": post_link,
            "post_title": row.get("post_title", ""),
            "post_context": row.get("post_context", ""),
            "phash": row.get("phash", ""),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["post_id", "file_name", "post_link", "post_title", "post_context", "phash"],
        )
        writer.writeheader()
        writer.writerows(upgraded_rows)

    log(f"[info] Upgraded CSV schema to include post_id ({len(upgraded_rows)} rows).")


def fetch_posts(sort: str, after: Optional[str], time_filter: Optional[str]) -> Tuple[Optional[Iterable[Dict]], Optional[str]]:
    """Fetch a page of posts for a given sort/time filter and after token."""
    url = f"https://www.reddit.com/r/{SUBREDDIT}/{sort}.json"
    params: Dict[str, str] = {"limit": "100", "raw_json": "1"}

    if time_filter:
        params["t"] = time_filter

    if after:
        params["after"] = after

    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            log(f"[error] Failed to fetch {sort} (t={time_filter or 'none'}, after={after}) on attempt {attempt}/{FETCH_RETRIES}: {exc}")
        except ValueError as exc:
            log(f"[error] Invalid JSON for {sort} (t={time_filter or 'none'}, after={after}) on attempt {attempt}/{FETCH_RETRIES}: {exc}")

        if attempt == FETCH_RETRIES:
            return None, None

        time.sleep(FETCH_RETRY_BACKOFF * attempt)

    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    children = data.get("children", [])
    next_after = data.get("after")
    return children, next_after


def extract_image_url(data: Dict) -> Tuple[Optional[str], List[str]]:
    """Extract a usable image URL with an allowed extension; return debug info when missing."""
    debug: List[str] = []
    candidates = [
        data.get("url_overridden_by_dest"),
        data.get("url"),
    ]

    preview = data.get("preview", {}).get("images") if isinstance(data.get("preview"), dict) else None
    if isinstance(preview, list) and preview:
        source_url = preview[0].get("source", {}).get("url")
        if source_url:
            candidates.append(source_url)
        else:
            debug.append("preview: missing source url")
    elif data.get("preview") is not None:
        debug.append("preview: unexpected structure")

    for raw_url in candidates:
        if raw_url is None:
            debug.append("candidate: None")
            continue
        if not isinstance(raw_url, str):
            debug.append(f"candidate: non-string ({type(raw_url).__name__})")
            continue
        cleaned = html.unescape(raw_url)
        ext = Path(cleaned.split("?")[0]).suffix.lower()
        if ext in VALID_EXTS:
            return cleaned, debug
        debug.append(f"candidate rejected: {cleaned} (ext '{ext or 'none'}')")

    return None, debug


def download_image(url: str) -> Optional[bytes]:
    """Download image content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        log(f"[error] Failed to download image {url}: {exc}")
        return None


def compute_phash(image_bytes: bytes) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for image content."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return imagehash.phash(img)
    except Exception as exc:
        log(f"[error] Could not compute phash: {exc}")
        return None


def format_link(permalink: Optional[str], fallback_url: Optional[str]) -> Optional[str]:
    """Build a full Reddit link for the CSV."""
    if permalink:
        if permalink.startswith("/"):
            return f"https://www.reddit.com{permalink}"
        return permalink
    return fallback_url


def close_writer(file_handle: TextIO) -> None:
    """Close the CSV file handle."""
    file_handle.close()


def log(message: str) -> None:
    """Print a log message, integrating with tqdm when available."""
    global current_progress
    if current_progress is not None:
        try:
            current_progress.write(message)
            return
        except Exception:
            pass
    print(message)


def is_duplicate_hash(phash: imagehash.ImageHash, existing_hashes: Iterable[imagehash.ImageHash]) -> bool:
    """Check whether phash is within the similarity threshold of any prior hash."""
    bits = getattr(phash, "hash", None)
    bit_count = bits.size if bits is not None else 64  # default phash is 8x8 -> 64 bits
    max_distance = int((1 - SIMILARITY_THRESHOLD) * bit_count)
    for prior in existing_hashes:
        try:
            if phash - prior <= max_distance:
                return True
        except Exception:
            continue
    return False


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(OUTPUT_CSV)
    upgrade_csv_schema(csv_path)
    existing_links, existing_phash_strings, existing_hashes, existing_post_ids, max_index = load_existing(csv_path, output_dir)
    next_index = max_index + 1

    # Open CSV once so we can append new rows while preserving any existing dataset
    writer, file_handle = ensure_csv(csv_path)
    saved_count = len(existing_phash_strings)
    new_saved = 0
    seen_links_this_run: Set[str] = set()

    try:
        # Multi-sort scraping in priority order; stop entirely if we hit the post cap
        if saved_count >= POST_LIMIT:
            log(f"[info] Existing dataset already meets limit ({POST_LIMIT}); nothing to do.")
            return

        progress_total = POST_LIMIT
        initial_progress = min(saved_count, progress_total)
        progress_factory = tqdm if tqdm is not None else NullTqdm

        global current_progress
        with progress_factory(total=progress_total, initial=initial_progress, desc="Downloading", unit="img") as pbar:
            current_progress = pbar

            for sort, time_filter in SORTS:
                after = None
                sort_label = f"{sort} (t={time_filter})" if time_filter else sort
                log(f"[info] Starting sort '{sort_label}'")

                # Pagination for this sort continues until post limit or pages are exhausted
                while saved_count < POST_LIMIT:
                    posts, after = fetch_posts(sort, after, time_filter)
                    if posts is None:
                        break  # Network/JSON issue; move to next sort
                    if not posts:
                        break  # No posts returned; move to next sort

                    for child in posts:
                        if saved_count >= POST_LIMIT:
                            break

                        data = child.get("data", {}) if isinstance(child, dict) else {}
                        title = data.get("title", "") or ""
                        permalink = data.get("permalink")
                        fallback_url = data.get("url")
                        post_link = format_link(permalink, fallback_url)

                        if not post_link:
                            log("[warn] Missing post link; skipping.")
                            continue

                        # Skip items already processed with existing files
                        if post_link in existing_links:
                            continue

                        if post_link in seen_links_this_run:
                            continue

                        post_id = extract_post_id(data, post_link)
                        if not post_id:
                            log("[warn] Could not determine post_id; skipping.")
                            continue

                        if post_id in existing_post_ids:
                            # If already recorded but missing file, still skip duplicate entry
                            continue

                        image_url, image_debug = extract_image_url(data)
                        if not image_url:
                            detail = "; ".join(image_debug) if image_debug else "no candidates"
                            log(f"[warn] No valid image URL; skipping post. Details: {detail}")
                            continue

                        image_ext = Path(image_url.split("?")[0]).suffix.lower()
                        if image_ext not in VALID_EXTS:
                            continue

                        # Download image content for hashing/deduplication
                        image_bytes = download_image(image_url)
                        if not image_bytes:
                            continue

                        # Compute phash for deduplication across runs and within this session
                        phash_obj = compute_phash(image_bytes)
                        if not phash_obj:
                            continue

                        # Deduplicate using similarity threshold against prior runs and this run
                        if is_duplicate_hash(phash_obj, existing_hashes):
                            log(f"[info] Duplicate image detected by phash similarity (>= {int(SIMILARITY_THRESHOLD * 100)}%); skipping.")
                            continue

                        phash_str = str(phash_obj)
                        file_name = f"{next_index:06d}_{phash_str}{image_ext}"
                        file_path = output_dir / file_name

                        try:
                            with file_path.open("wb") as img_file:
                                img_file.write(image_bytes)
                        except OSError as exc:
                            log(f"[error] Could not save image {file_name}: {exc}")
                            continue

                        post_context = data.get("selftext", "") or ""
                        # Append to CSV to continue building the dataset incrementally
                        writer.writerow([post_id, file_name, post_link, title, post_context, phash_str])

                        existing_phash_strings.add(phash_str)
                        existing_hashes.append(phash_obj)
                        existing_links.add(post_link)
                        existing_post_ids.add(post_id)
                        seen_links_this_run.add(post_link)

                        saved_count += 1
                        new_saved += 1
                        next_index += 1
                        pbar.update(1)

                    # Pagination cursor; stop if none for this sort
                    if not after:
                        break

                if saved_count >= POST_LIMIT:
                    log(f"[info] Reached post limit ({POST_LIMIT}); stopping further sorts.")
                    break
    finally:
        current_progress = None
        close_writer(file_handle)

    log(f"[done] Added {new_saved} new images; total now {saved_count}/{POST_LIMIT}.")


if __name__ == "__main__":
    main()
