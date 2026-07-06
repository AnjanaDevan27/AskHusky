"""
RAGAS Evaluation — AskHusky
Evaluates RAG pipeline quality using the golden test set.
Scores faithfulness, answer relevance, and context recall.
Logs results to MLflow.
"""
import os
import re
import json
import logging
from pathlib import Path
from anthropic import Anthropic
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from agents.cpt import CPTAgent
from agents.opt import OPTAgent
from agents.travel import TravelAgent
from agents.visa_status import VisaStatusAgent
from agents.coop import CoopAgent
from agents.urgency import UrgencyAgent
from agents.appointment import AppointmentAgent
from rag.retriever import retrieve, format_context
from evaluation.mlflow_logger import log_rag_quality
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GOLDEN_TEST_SET = Path("evaluation/golden_test_set.json")

AGENT_MAP = {
    "cpt":          CPTAgent(),
    "opt":          OPTAgent(),
    "travel":       TravelAgent(),
    "visa_status":  VisaStatusAgent(),
    "coop":         CoopAgent(),
    "urgency":      UrgencyAgent(),
    "appointment":  AppointmentAgent(),
}

anthropic_client = Anthropic()


# ── Disclaimer Stripping ──────────────────────────────────────────────────────

def strip_disclaimer(answer: str) -> str:
    """
    Strip OGS disclaimer from answer before RAGAS evaluation.
    RAGAS should score answer content only, not the disclaimer.
    """
    # Split on the disclaimer separator or the warning emoji
    cleaned = re.split(r'\n---\n|⚠️ \*\*Important:\*\*', answer)[0].strip()
    return cleaned


# ── Raw Answer (bypasses safety layer for evaluation) ─────────────────────────

def get_raw_answer(query: str, agent_name: str) -> str:
    """
    Get agent answer bypassing the safety layer.
    Safety layer is correct for production but interferes with RAGAS scoring —
    urgency responses return GSOC routing text instead of factual content.
    """
    chunks  = retrieve(query)
    context = format_context(chunks)

    user_message = f"""Use the following OGS content to answer the student's question.
Only use information from the provided context.
If the context does not contain enough information, say so clearly.

--- OGS CONTEXT ---
{context}
--- END CONTEXT ---

Student question: {query}"""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=AGENT_MAP[agent_name].system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text.strip()


# ── Build Evaluation Dataset ──────────────────────────────────────────────────

def build_eval_dataset(test_set: list[dict]) -> dict:
    """
    Run each question through its agent and collect:
    - question
    - generated answer (raw, no safety layer)
    - retrieved contexts
    - ground truth
    """
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    total = len(test_set)
    for i, item in enumerate(test_set):
        query        = item["question"]
        ground_truth = item["ground_truth"]
        agent_name   = item["agent"]

        logger.info(f"[{i+1}/{total}] Evaluating: '{query[:60]}'")

        # Get raw answer bypassing safety layer
        answer = get_raw_answer(query, agent_name)

        # Get retrieved contexts
        chunks  = retrieve(query)
        context = [c["text"] for c in chunks]

        questions.append(query)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    return {
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    }


# ── Judge Setup ───────────────────────────────────────────────────────────────

def setup_judges():
    claude = ChatAnthropic(
        model="claude-haiku-4-5",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    judge_llm = LangchainLLMWrapper(claude)

    hf_embeddings    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    faithfulness.llm            = judge_llm
    answer_relevancy.llm        = judge_llm
    answer_relevancy.embeddings = judge_embeddings
    context_recall.llm          = judge_llm


# ── Run RAGAS ─────────────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    logger.info(f"Loading golden test set from {GOLDEN_TEST_SET}")
    with open(GOLDEN_TEST_SET, "r") as f:
        test_set = json.load(f)

    logger.info(f"Running evaluation on {len(test_set)} questions...")
    data    = build_eval_dataset(test_set)
    dataset = Dataset.from_dict(data)

    setup_judges()

    logger.info("Running RAGAS scoring with Claude as judge...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    scores = {
        "faithfulness":     round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall":   round(float(results["context_recall"]), 4),
    }

    logger.info(f"RAGAS scores: {scores}")

    log_rag_quality(
        faithfulness=scores["faithfulness"],
        answer_relevance=scores["answer_relevancy"],
        context_recall=scores["context_recall"],
        num_questions=len(test_set)
    )

    return scores


# ── Detailed Evaluation ───────────────────────────────────────────────────────

def run_detailed_evaluation() -> None:
    """
    Run evaluation question by question and print
    which ones are scoring low so we can fix them.
    """
    logger.info("Loading golden test set...")
    with open(GOLDEN_TEST_SET, "r") as f:
        test_set = json.load(f)

    setup_judges()

    print("\n── Per-Question RAGAS Breakdown ──\n")
    print(f"{'Question':<55} {'Faith':>6} {'Relev':>6} {'Recall':>6}")
    print("-" * 77)

    for item in test_set:
        query        = item["question"]
        ground_truth = item["ground_truth"]
        agent_name   = item["agent"]

        # Raw answer bypassing safety layer
        answer  = get_raw_answer(query, agent_name)
        chunks  = retrieve(query)
        context = [c["text"] for c in chunks]

        single_dataset = Dataset.from_dict({
            "question":    [query],
            "answer":      [answer],
            "contexts":    [context],
            "ground_truth": [ground_truth],
        })

        result = evaluate(
            single_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )

        f = round(float(result["faithfulness"]), 3)
        r = round(float(result["answer_relevancy"]), 3)
        c = round(float(result["context_recall"]), 3)

        flag = " <--" if f < 0.6 or r < 0.6 or c < 0.5 else ""
        print(f"{query[:54]:<55} {f:>6} {r:>6} {c:>6}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--detailed" in sys.argv:
        run_detailed_evaluation()
    else:
        scores = run_evaluation()
        print("\n── RAGAS Evaluation Results ──\n")
        print(f"  Faithfulness:     {scores['faithfulness']}")
        print(f"  Answer Relevancy: {scores['answer_relevancy']}")
        print(f"  Context Recall:   {scores['context_recall']}")
        print()
        print("Results logged to MLflow.")