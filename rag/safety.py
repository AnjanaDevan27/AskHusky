"""
Safety Layer — AskHusky
Appends OGS disclaimer to every immigration response.
Detects urgent situations and routes to GSOC emergency line.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Disclaimer ────────────────────────────────────────────────────────────────

OGS_DISCLAIMER = """
---
⚠️ **Important:** This information is for general guidance only and is not legal advice. Immigration regulations change frequently. Always verify with your OGS advisor before taking action.

📞 **Office of Global Services (OGS):**
- Hours: Monday–Friday, 8:30am–4:30pm ET
- Website: international.northeastern.edu/ogs
- Appointments: Available via the OGS portal

🚨 **24/7 Emergency:** GSOC Emergency Line — available for urgent immigration situations outside OGS hours.
""".strip()

# ── Urgency Keywords ──────────────────────────────────────────────────────────

URGENCY_KEYWORDS = [
    # SEVIS issues
    "sevis termination",
    "sevis terminated",
    "sevis violation",
    "sevis record",
    "status terminated",
    "sevis was terminated",
    "sevis has been terminated",

    # Deportation / removal
    "deportation",
    "deported",
    "removal order",
    "ice",
    "detained",
    "detention",

    # Visa denial / overstay
    "visa denied",
    "visa revoked",
    "overstayed",
    "out of status",
    "unlawful presence",

    # Emergency travel
    "missed flight",
    "stranded",
    "port of entry denied",
    "denied entry",
    "cbp",

    # Health / family emergency
    "medical emergency",
    "family emergency",
    "death in family",
]

GSOC_EMERGENCY_RESPONSE = """
🚨 **This sounds like an urgent immigration situation.**

Please contact the **GSOC 24/7 Emergency Line** immediately. Do not wait until OGS opens — time-sensitive immigration issues require immediate attention.

**GSOC Emergency Line:**
Available 24 hours a day, 7 days a week for Northeastern students.
Contact information is available at: international.northeastern.edu/ogs/emergency

An OGS advisor can assess your situation and advise on next steps right away.

---
⚠️ AskHusky is not a substitute for professional immigration advising. Always contact OGS or GSOC directly for urgent matters.
""".strip()


# ── Safety Functions ──────────────────────────────────────────────────────────

def is_urgent(query: str) -> bool:
    """
    Detect if a student query describes an urgent immigration situation.
    Returns True if any urgency keyword is found.
    """
    query_lower = query.lower()
    for keyword in URGENCY_KEYWORDS:
        if keyword in query_lower:
            logger.warning(f"Urgency detected — keyword: '{keyword}'")
            return True
    return False


def add_disclaimer(response: str) -> str:
    """
    Append the OGS disclaimer to every immigration response.
    This is non-negotiable — called on every agent output.
    """
    return f"{response.strip()}\n\n{OGS_DISCLAIMER}"


def handle_urgent(query: str) -> str:
    """
    Return the GSOC emergency response for urgent situations.
    Bypasses normal agent routing entirely.
    """
    logger.warning(f"Routing to GSOC emergency — query: '{query[:100]}'")
    return GSOC_EMERGENCY_RESPONSE


def safe_response(query: str, response: str) -> str:
    """
    Main safety wrapper — call this on every agent response.

    Flow:
        1. Check if query is urgent → route to GSOC
        2. Otherwise append disclaimer to response

    Args:
        query:    The original student question
        response: The agent's generated answer

    Returns:
        Safe response with disclaimer or GSOC emergency routing
    """
    if is_urgent(query):
        return handle_urgent(query)
    return add_disclaimer(response)


# ── Main (test) ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Safety Layer Test ──\n")

    # Normal query — should get disclaimer
    normal_query    = "How many months of OPT can I get?"
    normal_response = "F-1 students are eligible for 12 months of OPT."
    print("NORMAL QUERY:")
    print(safe_response(normal_query, normal_response))
    print()

    # Urgent query — should route to GSOC
    urgent_query    = "My SEVIS was terminated yesterday, what do I do?"
    urgent_response = "This should never be shown."
    print("URGENT QUERY:")
    print(safe_response(urgent_query, urgent_response))