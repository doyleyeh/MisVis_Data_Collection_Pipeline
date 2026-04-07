import csv
import re
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TextIO

import requests

# Paths
POSTS_CSV = Path("reddit/data_is_ugly_output.csv")
COMMENTS_CSV = Path("reddit/data_is_ugly_comments.csv")
PROGRESS_CSV = Path("reddit/comments_scrape_progress.csv")

# Request settings
USER_AGENT = "dataisugly-scraper/0.1 (contact: local script)"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY = 1.0  # polite delay between posts
FETCH_RETRIES = 6
FETCH_RETRY_BACKOFF = 1.5
# On 429, sleep generously: first wait ~30s, then back off steeply up to 3 minutes.
RATE_LIMIT_BASE_SLEEP = 30.0
RATE_LIMIT_GROWTH = 2.0
RATE_LIMIT_MAX_SLEEP = 180.0

POST_ID_REGEX = re.compile(r"/comments/([a-z0-9]+)/", re.IGNORECASE)

COMMENT_FIELDS = [
    "comment_id",
    "comment_fullname",
    "post_id",
    "parent_fullname",
    "parent_comment_id",
    "depth",
    "path",
    "author",
    "created_utc",
    "body",
]


def log(message: str) -> None:
    print(message)


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def ensure_comments_csv(csv_path: Path) -> Tuple[csv.DictWriter, TextIO]:
    new_file = not csv_path.exists()
    ensure_parent(csv_path)
    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=COMMENT_FIELDS)
    if new_file:
        writer.writeheader()
    return writer, handle


def ensure_progress_csv(progress_path: Path) -> Tuple[csv.writer, TextIO]:
    new_file = not progress_path.exists()
    ensure_parent(progress_path)
    handle = progress_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if new_file:
        writer.writerow(["post_id", "completed_utc"])
    return writer, handle


def load_posts(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        log(f"[error] Posts CSV not found at {csv_path}")
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_existing_comment_ids(csv_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not csv_path.exists():
        return ids
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("comment_id")
            if cid:
                ids.add(cid)
    return ids


def load_completed(progress_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not progress_path.exists():
        return done
    with progress_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "post_id" not in reader.fieldnames:
            # Malformed progress file; ignore to avoid accidental skips
            return done
        for row in reader:
            pid = row.get("post_id")
            if pid:
                done.add(pid)
    return done


def parse_post_id_from_url(url: str) -> Optional[str]:
    match = POST_ID_REGEX.search(url)
    if match:
        return match.group(1)
    return None


def fetch_comments_json(post_id: str) -> Optional[Any]:
    url = f"https://www.reddit.com/comments/{post_id}.json"
    params = {"raw_json": "1", "limit": "500"}
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                retry_after_hdr = resp.headers.get("Retry-After")
                if retry_after_hdr:
                    try:
                        wait = float(retry_after_hdr)
                    except ValueError:
                        wait = RATE_LIMIT_BASE_SLEEP
                else:
                    wait = RATE_LIMIT_BASE_SLEEP * (RATE_LIMIT_GROWTH ** (attempt - 1))
                wait = min(wait * (1 + random.uniform(0, 0.35)), RATE_LIMIT_MAX_SLEEP)
                log(f"[warn] Rate limited (429) for {post_id}; sleeping {wait:.1f}s before retry {attempt}/{FETCH_RETRIES}")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log(f"[error] Failed to fetch comments for {post_id} (attempt {attempt}/{FETCH_RETRIES}): {exc}")
        except ValueError as exc:
            log(f"[error] Invalid JSON for {post_id} (attempt {attempt}/{FETCH_RETRIES}): {exc}")

        if attempt < FETCH_RETRIES:
            wait = FETCH_RETRY_BACKOFF * attempt
            time.sleep(wait)
    return None


def flatten_comments(
    children: List[Dict[str, Any]],
    post_id: str,
    parent_fullname: str,
    parent_depth: int,
    path_prefix: str,
    rows: List[Dict[str, Any]],
) -> None:
    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("kind") != "t1":
            # Skip non-comment items (including "more" for now)
            continue
        data = child.get("data", {}) or {}
        comment_id = data.get("id")
        if not comment_id:
            continue

        comment_fullname = f"t1_{comment_id}"
        parent_id = data.get("parent_id") or parent_fullname
        parent_comment_id = parent_id[3:] if parent_id.startswith("t1_") else ""
        depth = parent_depth + 1
        path = f"{path_prefix}/t1_{comment_id}" if path_prefix else f"t1_{comment_id}"

        rows.append({
            "comment_id": comment_id,
            "comment_fullname": comment_fullname,
            "post_id": post_id,
            "parent_fullname": parent_id,
            "parent_comment_id": parent_comment_id,
            "depth": depth,
            "path": path,
            "author": data.get("author", "[deleted]") or "[deleted]",
            "created_utc": data.get("created_utc", ""),
            "body": data.get("body", ""),
        })

        # Recurse into replies if present
        replies = data.get("replies")
        if isinstance(replies, dict):
            sub_children = replies.get("data", {}).get("children", [])
            flatten_comments(sub_children, post_id, comment_fullname, depth, path, rows)


def extract_comment_children(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    comments_listing = payload[1]
    if not isinstance(comments_listing, dict):
        return []
    data = comments_listing.get("data", {})
    children = data.get("children")
    return children if isinstance(children, list) else []


def process_post(
    row: Dict[str, str],
    existing_comment_ids: Set[str],
    writer: csv.DictWriter,
    progress_writer: csv.writer,
    completed: Set[str],
) -> None:
    post_link = row.get("post_link", "")
    post_id = row.get("post_id") or parse_post_id_from_url(post_link or "")
    if not post_id:
        log("[warn] Skipping row without post_id and unparsable post_link")
        return

    if post_id in completed:
        return

    payload = fetch_comments_json(post_id)
    if payload is None:
        log(f"[warn] No comments fetched for {post_id}; will retry on next run")
        return

    children = extract_comment_children(payload)
    rows: List[Dict[str, Any]] = []
    flatten_comments(children, post_id, f"t3_{post_id}", -1, f"t3_{post_id}", rows)

    new_rows = [r for r in rows if r["comment_id"] not in existing_comment_ids]
    if new_rows:
        writer.writerows(new_rows)
        for r in new_rows:
            existing_comment_ids.add(r["comment_id"])
    progress_writer.writerow([post_id, int(time.time())])
    completed.add(post_id)
    log(f"[info] Processed post {post_id}: {len(new_rows)} new comments")
    time.sleep(REQUEST_DELAY)


def main() -> None:
    posts = load_posts(POSTS_CSV)
    if not posts:
        return

    existing_comment_ids = load_existing_comment_ids(COMMENTS_CSV)
    completed = load_completed(PROGRESS_CSV)

    writer, comments_handle = ensure_comments_csv(COMMENTS_CSV)
    progress_writer, progress_handle = ensure_progress_csv(PROGRESS_CSV)

    try:
        for row in posts:
            try:
                process_post(row, existing_comment_ids, writer, progress_writer, completed)
            except Exception as exc:  # noqa: BLE001
                log(f"[error] Unexpected error for row: {exc}")
                time.sleep(REQUEST_DELAY)
    finally:
        comments_handle.close()
        progress_handle.close()


if __name__ == "__main__":
    main()
