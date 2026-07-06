"""
Embed and Store — AskHusky
Loads chunked OGS content, generates embeddings using
sentence-transformers, and stores in Pinecone vector DB.
"""

import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_FILE  = Path("data/processed/ogs_chunks.json")
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 100
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "askhusky")
DIMENSIONS   = 384

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Load Chunks ───────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ── Embed and Store ───────────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict]) -> None:
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logger.info(f"Index '{INDEX_NAME}' already exists — upserting.")

    index = pc.Index(INDEX_NAME)

    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        texts      = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        vectors = [
            {
                "id":     c["chunk_id"],
                "values": embedding,
                "metadata": {
                    "url":         c["url"],
                    "title":       c["title"],
                    "text":        c["text"],
                    "chunk_index": str(c["chunk_index"]),
                    "scraped_at":  c["scraped_at"],
                }
            }
            for c, embedding in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)

        logger.info(f"Upserted batch {i // BATCH_SIZE + 1} "
                    f"({min(i + BATCH_SIZE, total)}/{total} chunks)")

    logger.info(f"Done — {total} chunks stored in Pinecone index '{INDEX_NAME}'")


# ── Sanity Check ──────────────────────────────────────────────────────────────

def sanity_check() -> None:
    model = SentenceTransformer(EMBED_MODEL)
    pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    query = "How many months of OPT can an F-1 student get?"
    logger.info(f"Sanity check query: '{query}'")

    embedding = model.encode([query]).tolist()[0]
    results   = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )

    print("\n── Top 3 Retrieved Chunks ──\n")
    for i, match in enumerate(results["matches"]):
        meta = match["metadata"]
        print(f"  [{i+1}] {meta['title']}")
        print(f"       {meta['url']}")
        print(f"       {meta['text'][:200]}...")
        print(f"       score: {round(match['score'], 4)}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_FILE)
    embed_and_store(chunks)
    sanity_check()