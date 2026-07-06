"""
OGS Scraper — AskHusky
Scrapes all public pages from international.northeastern.edu/ogs
and saves raw HTML content as structured JSON.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://international.northeastern.edu/ogs/"
ALLOWED_DOMAIN = "international.northeastern.edu"
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "ogs_pages.json"
HEADERS = {"User-Agent": "AskHusky-Bot/1.0 (student research project)"}
CRAWL_DELAY = 1.5  # seconds between requests — be polite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid_url(url: str) -> bool:
    """Only follow links within the OGS subdirectory."""
    parsed = urlparse(url)
    return (
        parsed.netloc == ALLOWED_DOMAIN
        and parsed.path.startswith("/ogs")
        and not url.endswith((".pdf", ".jpg", ".png", ".docx", ".zip"))
    )


def clean_text(soup: BeautifulSoup) -> str:
    """Extract readable text — strip nav, header, footer, scripts."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "content"})
        or soup.body
    )

    if not main:
        return ""

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def check_robots_txt() -> bool:
    """Verify we're allowed to scrape OGS pages."""
    try:
        resp = requests.get(
            "https://international.northeastern.edu/robots.txt",
            headers=HEADERS,
            timeout=10
        )
        if "Disallow: /ogs" in resp.text:
            logger.warning("robots.txt disallows /ogs — stopping.")
            return False
        logger.info("robots.txt check passed.")
        return True
    except Exception as e:
        logger.warning(f"Could not fetch robots.txt: {e} — proceeding anyway.")
        return True


# ── Crawler ───────────────────────────────────────────────────────────────────

def scrape_ogs() -> list[dict]:
    """
    BFS crawl of all OGS pages.
    Returns a list of dicts with url, title, text, and scraped_at.
    """
    if not check_robots_txt():
        return []

    visited = set()
    queue = [BASE_URL]
    pages = []

    logger.info(f"Starting crawl from {BASE_URL}")

    while queue:
        url = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            logger.info(f"Scraping: {url}")
            response = requests.get(url, headers=HEADERS, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Skipping {url} — status {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = clean_text(soup)

            if not text:
                logger.warning(f"No text extracted from {url} — skipping.")
                continue

            pages.append({
                "url": url,
                "title": title,
                "text": text,
                "scraped_at": datetime.utcnow().isoformat()
            })

            # Find new links to follow
            for a_tag in soup.find_all("a", href=True):
                full_url = urljoin(url, a_tag["href"]).split("#")[0]  # strip anchors
                if full_url not in visited and is_valid_url(full_url):
                    queue.append(full_url)

            time.sleep(CRAWL_DELAY)

        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            continue

    logger.info(f"Crawl complete — {len(pages)} pages scraped.")
    return pages


# ── Save ──────────────────────────────────────────────────────────────────────

def save_pages(pages: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(pages)} pages to {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pages = scrape_ogs()
    if pages:
        save_pages(pages)
    else:
        logger.error("No pages scraped — check logs above.")