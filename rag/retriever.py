"""
RAG Retriever — AskHusky
Queries Pinecone to retrieve the most relevant OGS chunks
for a given student question.
"""

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "askhusky")
TOP_K       = 8  # number of chunks to retrieve per query

# ── Lazy singletons — load once, reuse across calls ───────────────────────────

_model  = None
_index  = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_index():
    global _index
    if _index is None:
        logger.info("Connecting to Pinecone...")
        pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(INDEX_NAME)
    return _index


# ── Retriever ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if not query or not query.strip():
        return []

    model = _get_model()
    index = _get_index()

    # Expand query for better retrieval
    expanded_query = f"F-1 student visa Northeastern OGS: {query}"

    embedding = model.encode([expanded_query]).tolist()[0]

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    chunks = []
    for match in results["matches"]:
        chunks.append({
            "text":  match["metadata"].get("text", ""),
            "url":   match["metadata"].get("url", ""),
            "title": match["metadata"].get("title", ""),
            "score": round(match["score"], 4)
        })

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string
    for the LLM prompt.

    Example output:
        [Source: OPT page]
        F-1 students may apply for 12 months of OPT...

        [Source: CPT page]
        CPT is available after one academic year...
    """
    if not chunks:
        return "No relevant OGS content found."

    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk['title']}]\n{chunk['text']}")

    return "\n\n".join(parts)


# ── Main (sanity check) ───────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "How many months of OPT can an F-1 student get?",
        "What is the 364 day CPT limit?",
        "Do I need a travel signature to re-enter the US?",
        "What happens if I work more than 20 hours during the semester?",
    ]

    print("\n── Retriever Sanity Check ──\n")
    for query in test_queries:
        print(f"Q: {query}")
        chunks = retrieve(query)
        print(f"   Top result: {chunks[0]['title']} (score: {chunks[0]['score']})")
        print(f"   URL: {chunks[0]['url']}")
        print()