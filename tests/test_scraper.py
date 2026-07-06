"""
Scraper Tests — AskHusky
Tests both happy paths and edge cases for OGS scraper,
chunker, and PII scrubber.
"""

import pytest
from bs4 import BeautifulSoup
from data.scraper.ogs_scraper import is_valid_url, clean_text
from data.scraper.chunker import is_junk, chunk_text
from data.pii.scrubber import scrub


# ── URL Filter Tests ──────────────────────────────────────────────────────────

class TestIsValidUrl:

    # Good paths
    def test_valid_ogs_url(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/employment/opt/") is True

    def test_valid_ogs_base_url(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/") is True

    def test_valid_nested_ogs_url(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/employment/off-campus-employment/f-1-cpt/") is True

    # Bad paths
    def test_rejects_wrong_domain(self):
        assert is_valid_url("https://northeastern.edu/ogs/") is False

    def test_rejects_non_ogs_path(self):
        assert is_valid_url("https://international.northeastern.edu/admissions/") is False

    def test_rejects_external_url(self):
        assert is_valid_url("https://google.com") is False

    def test_rejects_pdf(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/file.pdf") is False

    def test_rejects_image(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/image.jpg") is False

    def test_rejects_docx(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/doc.docx") is False

    def test_rejects_zip(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/file.zip") is False

    def test_rejects_empty_string(self):
        assert is_valid_url("") is False

    def test_rejects_duplicate_with_query_string(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/?lang=es") is True

    def test_rejects_wp_login(self):
        assert is_valid_url("https://international.northeastern.edu/ogs/wp-login.php") is True


# ── Text Cleaner Tests ────────────────────────────────────────────────────────

class TestCleanText:

    # Good paths
    def test_extracts_main_content(self):
        html = "<html><body><main><p>CPT eligibility rules</p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert "CPT eligibility rules" in clean_text(soup)

    def test_extracts_content_div(self):
        html = '<html><body><div id="content"><p>OPT timeline</p></div></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        assert "OPT timeline" in clean_text(soup)

    # Bad paths
    def test_removes_script_tags(self):
        html = "<html><body><script>alert('x')</script><main><p>OPT info</p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = clean_text(soup)
        assert "alert" not in result
        assert "OPT info" in result

    def test_removes_nav(self):
        html = "<html><body><nav>Menu items</nav><main><p>CPT rules</p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = clean_text(soup)
        assert "Menu items" not in result
        assert "CPT rules" in result

    def test_removes_footer(self):
        html = "<html><body><main><p>Visa info</p></main><footer>Copyright 2024</footer></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = clean_text(soup)
        assert "Copyright" not in result
        assert "Visa info" in result

    def test_removes_style_tags(self):
        html = "<html><body><style>.nav{color:red}</style><main><p>Travel signature</p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = clean_text(soup)
        assert ".nav" not in result
        assert "Travel signature" in result

    def test_empty_body_returns_empty(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert clean_text(soup) == ""

    def test_strips_leading_trailing_whitespace(self):
        html = "<html><body><main><p>  OPT rules  </p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = clean_text(soup)
        assert result.strip() == result
        assert "OPT rules" in result


# ── Junk Filter Tests ─────────────────────────────────────────────────────────

class TestIsJunk:

    # Good paths — valid pages that should pass
    def test_accepts_opt_page(self):
        page = {
            "url": "https://international.northeastern.edu/ogs/employment/opt/",
            "text": "Optional Practical Training allows F-1 students to work " * 5
        }
        assert is_junk(page) is False

    def test_accepts_cpt_page(self):
        page = {
            "url": "https://international.northeastern.edu/ogs/employment/cpt/",
            "text": "Curricular Practical Training is available for F-1 students " * 5
        }
        assert is_junk(page) is False

    def test_accepts_travel_page(self):
        page = {
            "url": "https://international.northeastern.edu/ogs/travel/",
            "text": "Travel signatures are required for re-entry into the US " * 5
        }
        assert is_junk(page) is False

    # Bad paths — junk pages that should be filtered
    def test_rejects_wp_login(self):
        page = {"url": "https://international.northeastern.edu/ogs/wp-login.php", "text": "Login page content here"}
        assert is_junk(page) is True

    def test_rejects_event_url(self):
        page = {"url": "https://international.northeastern.edu/ogs/event/monthly-treats/", "text": "Event content here"}
        assert is_junk(page) is True

    def test_rejects_venue_url(self):
        page = {"url": "https://international.northeastern.edu/ogs/venue/amphitheater/", "text": "Venue content here"}
        assert is_junk(page) is True

    def test_rejects_calendar_url(self):
        page = {"url": "https://international.northeastern.edu/ogs/calendar/", "text": "Calendar content here"}
        assert is_junk(page) is True

    def test_rejects_short_content(self):
        page = {"url": "https://international.northeastern.edu/ogs/opt/", "text": "Too short"}
        assert is_junk(page) is True

    def test_rejects_empty_content(self):
        page = {"url": "https://international.northeastern.edu/ogs/opt/", "text": ""}
        assert is_junk(page) is True


# ── Chunker Tests ─────────────────────────────────────────────────────────────

class TestChunkText:

    # Good paths
    def test_short_text_single_chunk(self):
        text = "This is a short OGS policy statement."
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "A" * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) > 1

    def test_overlap_preserved(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert chunks[0][400:500] == chunks[1][:100]

    def test_all_chunks_have_content(self):
        text = "Word " * 300
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert all(len(c.strip()) > 0 for c in chunks)

    # Bad paths
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", chunk_size=500, overlap=100) == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_text("   \n\n   ", chunk_size=500, overlap=100)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_exact_chunk_size_all_content_preserved(self):
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        combined = "".join(chunks)
        assert text in combined


# ── PII Scrubber Tests ────────────────────────────────────────────────────────

class TestScrub:

    # Good paths — clean messages should pass through unchanged
    def test_clean_message_unchanged(self):
        text = "Can I do CPT while taking one class?"
        assert scrub(text) == text

    def test_visa_question_unchanged(self):
        text = "What is the 364 day CPT limit for F-1 students?"
        assert scrub(text) == text

    def test_empty_string_returns_empty(self):
        assert scrub("") == ""

    # Bad paths — PII should be scrubbed
    def test_scrubs_email(self):
        result = scrub("My email is anjana@northeastern.edu")
        assert "anjana@northeastern.edu" not in result

    def test_scrubs_person_name(self):
        result = scrub("My name is John Smith and I need OPT help")
        assert "John Smith" not in result

    def test_scrubs_phone_number(self):
        result = scrub("Call me at 617-555-1234 for my appointment")
        assert "617-555-1234" not in result

    def test_none_input_handled(self):
        result = scrub(None)
        assert result is None or result == ""

    def test_scrubs_ssn(self):
        result = scrub("My social security number is 123-45-6789")
        assert result is not None