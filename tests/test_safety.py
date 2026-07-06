"""
Safety Layer Tests — AskHusky
Tests OGS disclaimer appending and urgency detection.
No API calls — pure logic tests.
Good and bad paths covered for production readiness.
"""

import pytest
from rag.safety import (
    is_urgent,
    add_disclaimer,
    safe_response,
    OGS_DISCLAIMER,
    GSOC_EMERGENCY_RESPONSE
)


# -- Disclaimer Tests ----------------------------------------------------------

class TestAddDisclaimer:

    # Good paths
    def test_disclaimer_appended_to_normal_answer(self):
        answer = "F-1 students can work 20 hours per week."
        result = add_disclaimer(answer)
        assert OGS_DISCLAIMER in result

    def test_answer_content_preserved(self):
        answer = "OPT is available for 12 months."
        result = add_disclaimer(answer)
        assert answer in result

    def test_disclaimer_appears_after_answer(self):
        answer = "CPT requires one academic year."
        result = add_disclaimer(answer)
        assert result.startswith(answer)

    def test_long_answer_gets_disclaimer(self):
        answer = "Some visa info. " * 100
        result = add_disclaimer(answer)
        assert OGS_DISCLAIMER in result

    def test_answer_with_special_characters(self):
        answer = "F-1 status requires 12+ credits/semester."
        result = add_disclaimer(answer)
        assert OGS_DISCLAIMER in result
        assert answer in result

    def test_answer_with_numbers(self):
        answer = "Students can work 20 hours during semester and 40 hours during breaks."
        result = add_disclaimer(answer)
        assert OGS_DISCLAIMER in result

    # Bad paths
    def test_empty_answer_still_gets_disclaimer(self):
        result = add_disclaimer("")
        assert OGS_DISCLAIMER in result

    def test_whitespace_only_answer_gets_disclaimer(self):
        result = add_disclaimer("   ")
        assert OGS_DISCLAIMER in result

    def test_disclaimer_not_doubled(self):
        answer = "Some visa info."
        result = add_disclaimer(answer)
        assert result.count(OGS_DISCLAIMER) == 1

    def test_disclaimer_contains_ogs_hours(self):
        result = add_disclaimer("Some answer.")
        assert "8:30" in result or "4:30" in result

    def test_disclaimer_contains_emergency_contact(self):
        result = add_disclaimer("Some answer.")
        assert "GSOC" in result or "emergency" in result.lower()


# -- Urgency Detection Tests ---------------------------------------------------

class TestIsUrgent:

    # Good paths — non-urgent queries that should pass through
    def test_normal_cpt_query(self):
        assert is_urgent("Can I do CPT while taking one class?") is False

    def test_normal_opt_query(self):
        assert is_urgent("How many months of OPT can I get?") is False

    def test_normal_travel_query(self):
        assert is_urgent("Do I need a travel signature?") is False

    def test_normal_enrollment_query(self):
        assert is_urgent("What is the minimum course load for F-1?") is False

    def test_normal_appointment_query(self):
        assert is_urgent("How do I book an OGS appointment?") is False

    def test_normal_grace_period_query(self):
        assert is_urgent("What is the 60-day grace period?") is False

    def test_empty_query_not_urgent(self):
        assert is_urgent("") is False

    def test_whitespace_only_not_urgent(self):
        assert is_urgent("   ") is False

    # Bad paths — urgent queries that must be detected
    def test_sevis_termination(self):
        assert is_urgent("My SEVIS was terminated yesterday") is True

    def test_sevis_terminated_variant(self):
        assert is_urgent("I got a notice that my SEVIS has been terminated") is True

    def test_sevis_violation(self):
        assert is_urgent("I committed a SEVIS violation") is True

    def test_deportation(self):
        assert is_urgent("I am facing deportation") is True

    def test_deported(self):
        assert is_urgent("I am afraid I will be deported") is True

    def test_removal_order(self):
        assert is_urgent("I received a removal order") is True

    def test_out_of_status(self):
        assert is_urgent("I think I am out of status") is True

    def test_unlawful_presence(self):
        assert is_urgent("I may have accumulated unlawful presence") is True

    def test_visa_revoked(self):
        assert is_urgent("My visa was revoked at the airport") is True

    def test_visa_denied(self):
        assert is_urgent("My visa was denied at the embassy") is True

    def test_detained(self):
        assert is_urgent("I was detained by ICE") is True

    def test_port_of_entry_denied(self):
        assert is_urgent("I was denied entry at port of entry") is True

    def test_cbp_encounter(self):
        assert is_urgent("I had an issue with CBP at the border") is True

    def test_medical_emergency(self):
        assert is_urgent("I have a medical emergency") is True

    def test_case_insensitive_upper(self):
        assert is_urgent("MY SEVIS WAS TERMINATED") is True

    def test_case_insensitive_mixed(self):
        assert is_urgent("My Sevis Was Terminated") is True

    def test_urgency_mid_sentence(self):
        assert is_urgent("I just found out my status terminated last week") is True


# -- Safe Response Tests -------------------------------------------------------

class TestSafeResponse:

    # Good paths — normal responses get disclaimer
    def test_normal_response_gets_disclaimer(self):
        result = safe_response(
            "How many OPT months do I get?",
            "You get 12 months of OPT."
        )
        assert OGS_DISCLAIMER in result

    def test_normal_response_answer_preserved(self):
        answer = "CPT requires one academic year of enrollment."
        result = safe_response("Can I do CPT?", answer)
        assert answer in result

    def test_normal_response_not_routed_to_gsoc(self):
        result = safe_response(
            "How many credits do I need?",
            "F-1 students need 12 credits."
        )
        assert GSOC_EMERGENCY_RESPONSE not in result

    def test_different_normal_queries_all_get_disclaimer(self):
        queries = [
            ("What is CPT?", "CPT is curricular practical training."),
            ("What is OPT?", "OPT is optional practical training."),
            ("What is the grace period?", "The grace period is 60 days."),
        ]
        for query, answer in queries:
            result = safe_response(query, answer)
            assert OGS_DISCLAIMER in result

    # Bad paths — urgent responses route to GSOC
    def test_urgent_routes_to_gsoc(self):
        result = safe_response(
            "My SEVIS was terminated",
            "This should never appear"
        )
        assert GSOC_EMERGENCY_RESPONSE in result

    def test_urgent_hides_original_answer(self):
        result = safe_response(
            "My SEVIS was terminated",
            "This should never appear in output"
        )
        assert "This should never appear in output" not in result

    def test_detained_routes_to_gsoc(self):
        result = safe_response(
            "I was detained by ICE",
            "Some answer"
        )
        assert GSOC_EMERGENCY_RESPONSE in result

    def test_deportation_routes_to_gsoc(self):
        result = safe_response(
            "I am facing deportation",
            "Some answer"
        )
        assert GSOC_EMERGENCY_RESPONSE in result

    def test_gsoc_response_contains_emergency_info(self):
        result = safe_response(
            "My visa was revoked",
            "Some answer"
        )
        assert "GSOC" in result

    def test_gsoc_response_contains_urgency_acknowledgment(self):
        result = safe_response(
            "My SEVIS was terminated",
            "Some answer"
        )
        assert "urgent" in result.lower() or "immediately" in result.lower()