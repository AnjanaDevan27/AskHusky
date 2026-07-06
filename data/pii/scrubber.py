"""
PII Scrubber — AskHusky
Scrubs personally identifiable information from student
messages before logging or storing them anywhere.
Uses Microsoft Presidio for detection and anonymization.
"""

import logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Engine Setup ──────────────────────────────────────────────────────────────

analyzer  = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ── PII Entities to Detect ────────────────────────────────────────────────────

ENTITIES = [
    "PERSON",           # student names
    "EMAIL_ADDRESS",    # northeastern emails
    "PHONE_NUMBER",     # phone numbers
    "US_PASSPORT",      # passport numbers
    "US_SSN",           # social security numbers
    "US_ITIN",          # SEVIS numbers often match this pattern
    "DATE_TIME",        # birth dates (not travel dates — tradeoff)
    "LOCATION",         # home addresses
    "IP_ADDRESS",       # IP addresses
]

# ── Scrubber ──────────────────────────────────────────────────────────────────

def scrub(text: str, language: str = "en") -> str:
    """
    Detect and anonymize PII in a student message.
    Returns the cleaned text with PII replaced by entity type labels.
    
    Example:
        Input:  "My name is Anjana, email is a@neu.edu"
        Output: "My name is [PERSON], email is [EMAIL_ADDRESS]"
    """
    if not text or not text.strip():
        return text

    try:
        results = analyzer.analyze(
            text=text,
            entities=ENTITIES,
            language=language
        )

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        return anonymized.text

    except Exception as e:
        logger.error(f"PII scrubbing failed: {e}")
        # Fail safe — return empty string rather than expose PII
        return ""


def scrub_metadata(metadata: dict) -> dict:
    """
    Scrub PII from a metadata dict (e.g. LangSmith trace payload).
    Scrubs all string values recursively.
    """
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            cleaned[key] = scrub(value)
        elif isinstance(value, dict):
            cleaned[key] = scrub_metadata(value)
        else:
            cleaned[key] = value
    return cleaned


# ── Main (manual test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_inputs = [
        "My name is Anjana and my email is anjana@northeastern.edu",
        "My SEVIS ID is N1234567890 and my passport is A12345678",
        "Can I do CPT while taking one class?",  # no PII — should be unchanged
        "Call me at 617-555-1234, I need help with OPT",
    ]

    print("\n── PII Scrubber Test ──\n")
    for text in test_inputs:
        scrubbed = scrub(text)
        print(f"  Input:   {text}")
        print(f"  Output:  {scrubbed}")
        print()