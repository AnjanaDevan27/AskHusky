"""
MLflow Logger — AskHusky
Tracks pipeline runs, chunk stats, and RAG quality metrics.
"""

import os
import mlflow
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "askhusky-rag-pipeline"
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_URI}")
    logger.info(f"MLflow experiment:   {EXPERIMENT_NAME}")


# ── Log Scraper Run ───────────────────────────────────────────────────────────

def log_scraper_run(
    pages_scraped: int,
    pages_skipped: int,
    run_duration_seconds: float
) -> None:
    setup_mlflow()
    with mlflow.start_run(run_name=f"scraper_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tag("pipeline_stage", "scraper")
        mlflow.log_param("base_url", "https://international.northeastern.edu/ogs/")
        mlflow.log_metric("pages_scraped",         pages_scraped)
        mlflow.log_metric("pages_skipped",         pages_skipped)
        mlflow.log_metric("run_duration_seconds",  run_duration_seconds)
        logger.info(f"Logged scraper run — {pages_scraped} pages scraped")


# ── Log Chunker Run ───────────────────────────────────────────────────────────

def log_chunker_run(
    pages_kept: int,
    pages_filtered: int,
    total_chunks: int,
    chunk_size: int,
    overlap: int
) -> None:
    setup_mlflow()
    with mlflow.start_run(run_name=f"chunker_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tag("pipeline_stage", "chunker")
        mlflow.log_param("chunk_size",      chunk_size)
        mlflow.log_param("overlap",         overlap)
        mlflow.log_metric("pages_kept",     pages_kept)
        mlflow.log_metric("pages_filtered", pages_filtered)
        mlflow.log_metric("total_chunks",   total_chunks)
        logger.info(f"Logged chunker run — {total_chunks} chunks")


# ── Log Embedding Run ─────────────────────────────────────────────────────────

def log_embedding_run(
    total_chunks: int,
    embed_model: str,
    index_name: str,
    run_duration_seconds: float
) -> None:
    setup_mlflow()
    with mlflow.start_run(run_name=f"embeddings_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tag("pipeline_stage", "embeddings")
        mlflow.log_param("embed_model",           embed_model)
        mlflow.log_param("vector_db",             "pinecone")
        mlflow.log_param("index_name",            index_name)
        mlflow.log_metric("total_chunks_upserted", total_chunks)
        mlflow.log_metric("run_duration_seconds",  run_duration_seconds)
        logger.info(f"Logged embedding run — {total_chunks} chunks upserted")


# ── Log RAG Quality ───────────────────────────────────────────────────────────

def log_rag_quality(
    faithfulness: float,
    answer_relevance: float,
    context_recall: float,
    num_questions: int
) -> None:
    """
    Log RAGAS evaluation scores.
    Called after running ragas_eval.py in Week 2.
    """
    setup_mlflow()
    with mlflow.start_run(run_name=f"ragas_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tag("pipeline_stage", "evaluation")
        mlflow.log_param("num_questions",      num_questions)
        mlflow.log_metric("faithfulness",      faithfulness)
        mlflow.log_metric("answer_relevance",  answer_relevance)
        mlflow.log_metric("context_recall",    context_recall)
        logger.info(
            f"Logged RAGAS scores — faithfulness: {faithfulness:.3f}, "
            f"relevance: {answer_relevance:.3f}, recall: {context_recall:.3f}"
        )


# ── Main (test run) ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate logging today's pipeline run
    log_scraper_run(
        pages_scraped=268,
        pages_skipped=0,
        run_duration_seconds=127.4
    )
    log_chunker_run(
        pages_kept=249,
        pages_filtered=19,
        total_chunks=3106,
        chunk_size=500,
        overlap=100
    )
    log_embedding_run(
        total_chunks=3106,
        embed_model="all-MiniLM-L6-v2",
        index_name="askhusky",
        run_duration_seconds=184.2
    )
    print("\nMLflow runs logged. Run 'mlflow ui' to view them.")