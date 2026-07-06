"""
Orchestrator Agent — AskHusky
Classifies student intent and routes to the correct
specialized agent. Does not answer questions directly.
"""

import os
import logging
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = Anthropic()

# ── Agent Registry ────────────────────────────────────────────────────────────

AGENTS = {
    "cpt":          "CPT Agent — Curricular Practical Training eligibility and rules",
    "opt":          "OPT Agent — Optional Practical Training and STEM OPT",
    "travel":       "Travel Agent — travel signatures, I-94, port of entry",
    "visa_status":  "Visa Status Agent — F-1 maintenance, SEVIS, enrollment rules",
    "coop":         "Co-op Agent — international co-op rules and GPA requirements",
    "urgency":      "Urgency Agent — SEVIS termination, deportation, emergency situations",
    "appointment":  "Appointment Agent — booking OGS advisor appointments",
}

# ── System Prompt ─────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator for AskHusky, an AI help desk for Northeastern University
F-1 international students. Your only job is to classify the student's question
and route it to the correct agent.

Available agents:
- cpt: Questions about Curricular Practical Training, co-op authorization, 364-day limit
- opt: Questions about Optional Practical Training, STEM OPT, post-completion work
- travel: Questions about travel signatures, re-entry, I-94, visa stamps, port of entry
- visa_status: Questions about maintaining F-1 status, SEVIS, enrollment requirements
- coop: Questions about Northeastern co-op rules, GPA requirements, international eligibility
- urgency: SEVIS termination, deportation risk, out of status, detained, visa revoked
- appointment: Booking or scheduling an OGS advisor appointment

Respond with ONLY the agent name in lowercase. Nothing else.
No explanation. No punctuation. Just the agent name.

Examples:
Student: "Can I work off campus during my first year?"
Response: cpt

Student: "My SEVIS record was terminated"
Response: urgency

Student: "I want to schedule a meeting with my advisor"
Response: appointment

Student: "Do I need a travel signature before I go home for winter break?"
Response: travel
""".strip()

# ── Orchestrator ──────────────────────────────────────────────────────────────

def classify_intent(query: str) -> str:
    """
    Classify the student's query and return the agent name to route to.

    Args:
        query: The student's question

    Returns:
        Agent name string (e.g. 'cpt', 'opt', 'urgency')
    """
    if not query or not query.strip():
        logger.warning("Empty query received — defaulting to visa_status")
        return "visa_status"

    logger.info(f"Classifying intent for: '{query[:80]}'")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": query}
        ]
    )

    agent = response.content[0].text.strip().lower()

    # Validate — fall back to visa_status if unexpected response
    if agent not in AGENTS:
        logger.warning(f"Unknown agent '{agent}' — falling back to visa_status")
        agent = "visa_status"

    logger.info(f"Routed to: {agent}")
    return agent


def get_agent_description(agent: str) -> str:
    return AGENTS.get(agent, "Unknown agent")


# ── Main (test) ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "Can I do CPT while taking one class?",
        "How many months of OPT can I get after graduation?",
        "Do I need a travel signature to go home for winter break?",
        "My SEVIS was terminated yesterday, what do I do?",
        "What are the GPA requirements for international co-op?",
        "I want to book an appointment with my OGS advisor",
        "Can I maintain F-1 status while taking online classes?",
        "What happens if I work more than 20 hours during the semester?",
    ]

    print("\n── Orchestrator Routing Test ──\n")
    for query in test_queries:
        agent = classify_intent(query)
        print(f"  Q: {query}")
        print(f"  -> {agent} ({get_agent_description(agent)})")
        print()