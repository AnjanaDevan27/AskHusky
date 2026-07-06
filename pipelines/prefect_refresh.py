"""
Prefect Weekly Refresh — AskHusky
Orchestrates the full RAG pipeline on a weekly schedule.
Scrape → Chunk → Embed → Log
"""

import time
import logging
from prefect import flow, task

from data.scraper.ogs_scraper import scrape_ogs, save_pages
from data.scraper.chunker import build_chunks, save_chunks
from data.embeddings.embed_and_store import load_chunks, embed_and_store
from evaluation.mlflow_logger import (
    log_scraper_run,
    log_chunker_run,
    log_embedding_run
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="scrape-ogs", retries=2, retry_delay_seconds=60)
def scrape_task() -> list[dict]:
    """Scrape all OGS pages."""
    logger.info("Starting OGS scrape...")
    start = time.time()
    pages = scrape_ogs()
    duration = time.time() - start

    save_pages(pages)
    log_scraper_run(
        pages_scraped=len(pages),
        pages_skipped=0,
        run_duration_seconds=round(duration, 2)
    )
    logger.info(f"Scrape complete — {len(pages)} pages in {duration:.1f}s")
    return pages


@task(name="chunk-pages", retries=1)
def chunk_task(pages: list[dict]) -> list[dict]:
    """Filter and chunk scraped pages."""
    logger.info("Starting chunking...")
    start = time.time()

    chunks = build_chunks(pages)
    save_chunks(chunks)
    duration = time.time() - start

    pages_kept    = sum(1 for p in pages if chunks)
    pages_filtered = len(pages) - pages_kept

    log_chunker_run(
        pages_kept=len(pages),
        pages_filtered=pages_filtered,
        total_chunks=len(chunks),
        chunk_size=500,
        overlap=100
    )
    logger.info(f"Chunking complete — {len(chunks)} chunks in {duration:.1f}s")
    return chunks


@task(name="embed-and-upsert", retries=2, retry_delay_seconds=30)
def embed_task(chunks: list[dict]) -> None:
    """Embed chunks and upsert to Pinecone."""
    logger.info("Starting embedding and upsert...")
    start = time.time()

    embed_and_store(chunks)
    duration = time.time() - start

    log_embedding_run(
        total_chunks=len(chunks),
        embed_model="all-MiniLM-L6-v2",
        index_name="askhusky",
        run_duration_seconds=round(duration, 2)
    )
    logger.info(f"Embedding complete — {len(chunks)} chunks in {duration:.1f}s")


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(
    name="askhusky-weekly-refresh",
    description="Weekly OGS scrape → chunk → embed → Pinecone upsert",
)
def weekly_refresh_flow() -> None:
    """Full RAG pipeline refresh."""
    logger.info("AskHusky weekly refresh starting...")

    pages  = scrape_task()
    chunks = chunk_task(pages)
    embed_task(chunks)

    logger.info(f"Weekly refresh complete — {len(chunks)} chunks in Pinecone")


# ── Schedule ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run once immediately for testing
    weekly_refresh_flow()