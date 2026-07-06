"""
Chunker — AskHusky
Loads raw OGS pages, filters junk, splits into
overlapping chunks ready for Chroma embedding.
"""

import json
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = Path("data/raw/ogs_pages.json")
OUTPUT_FILE = Path("data/processed/ogs_chunks.json")
CHUNK_SIZE  = 500   # characters per chunk
OVERLAP     = 100   # overlap between chunks to preserve context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Junk Filter ───────────────────────────────────────────────────────────────

JUNK_KEYWORDS = [
    "wp-login",
    "wp-content",
    "event/",
    "venue/",
    "monthly-treats",
    "calendar",
    "raytheon",
    "amphitheater",
    "cookie",
    "privacy-policy",
    "sitemap",
]

def is_junk(page: dict) -> bool:
    """Return True if the page is not visa-relevant content."""
    url = page["url"].lower()
    text = page["text"].lower()

    if any(keyword in url for keyword in JUNK_KEYWORDS):
        return True

    # Skip pages with very little content
    if len(text.strip()) < 100:
        return True

    return False


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character-level chunks.
    Each chunk starts OVERLAP chars before the previous one ended.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    """
    Filter junk pages, chunk remaining content,
    and attach metadata to each chunk.
    """
    all_chunks = []
    skipped = 0

    for page in pages:
        if is_junk(page):
            skipped += 1
            continue

        chunks = chunk_text(page["text"], CHUNK_SIZE, OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id":   f"{page['url']}__chunk_{i}",
                "url":        page["url"],
                "title":      page["title"],
                "text":       chunk,
                "chunk_index": i,
                "scraped_at": page["scraped_at"],
            })

    logger.info(f"Pages kept:    {len(pages) - skipped}")
    logger.info(f"Pages skipped: {skipped}")
    logger.info(f"Total chunks:  {len(all_chunks)}")
    return all_chunks


# ── Save ──────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Loading pages from {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    logger.info(f"Loaded {len(pages)} pages")
    chunks = build_chunks(pages)
    save_chunks(chunks)