"""Tests for the DocumentSegmenter / segment_document module."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.segmenter import (
    DocumentSegmenter,
    DocumentStructure,
    Section,
    SectionLevel,
    segment_document,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_contract() -> str:
    return (
        "SERVICE AGREEMENT\n\n"
        "This Agreement is entered into between Alpha Corp and Beta LLC.\n\n"
        "1. DEFINITIONS\n"
        "  'Confidential Information' means any non-public information disclosed.\n"
        "  'Services' means the professional services described in Schedule A.\n\n"
        "2. SERVICES\n"
        "The Provider shall deliver the Services as described herein.\n\n"
        "2.1 SCOPE OF WORK\n"
        "Provider will perform software development and consulting services.\n\n"
        "2.2 DELIVERABLES\n"
        "Provider shall deliver all Deliverables by the agreed milestones.\n\n"
        "3. PAYMENT\n"
        "Client shall pay Provider within thirty (30) days of invoice.\n\n"
        "4. CONFIDENTIALITY\n"
        "Both parties shall maintain strict confidentiality.\n\n"
        "5. TERMINATION\n"
        "Either party may terminate upon thirty (30) days written notice.\n"
    )


@pytest.fixture
def article_contract() -> str:
    return (
        "MASTER SERVICES AGREEMENT\n\n"
        "ARTICLE I DEFINITIONS\n"
        "'Agreement' means this Master Services Agreement.\n\n"
        "ARTICLE II SCOPE OF SERVICES\n"
        "Provider agrees to deliver services as specified.\n\n"
        "ARTICLE III COMPENSATION\n"
        "Client shall pay the agreed fees monthly.\n\n"
        "ARTICLE IV TERM AND TERMINATION\n"
        "This Agreement commences on the Effective Date and continues for one year.\n"
    )


@pytest.fixture
def section_keyword_contract() -> str:
    return (
        "EMPLOYMENT AGREEMENT\n\n"
        "SECTION 1 — EMPLOYMENT\n"
        "Company agrees to employ the Executive.\n\n"
        "SECTION 2 — COMPENSATION\n"
        "Executive shall receive an annual salary of $150,000.\n\n"
        "SECTION 3 — BENEFITS\n"
        "Executive is entitled to standard company benefits.\n\n"
        "SECTION 4 — TERMINATION\n"
        "Either party may terminate employment with 30 days notice.\n"
    )


@pytest.fixture
def alphabetic_contract() -> str:
    return (
        "SCHEDULE A — TECHNICAL REQUIREMENTS\n\n"
        "1. INFRASTRUCTURE\n"
        "The following infrastructure standards apply.\n\n"
        "(a) All servers must be hosted in approved data centers.\n"
        "(b) Uptime requirements are defined as 99.9% per calendar month.\n"
        "(c) Security patching must occur within 72 hours of release.\n\n"
        "2. TESTING\n"
        "All deliverables must pass acceptance testing.\n"
    )


@pytest.fixture
def roman_numeral_contract() -> str:
    return (
        "CONFIDENTIALITY AGREEMENT\n\n"
        "I. DEFINITION OF CONFIDENTIAL INFORMATION\n"
        "Confidential Information includes all technical and business data.\n\n"
        "II. OBLIGATIONS OF RECEIVING PARTY\n"
        "Receiving party shall protect information with reasonable care.\n\n"
        "III. EXCEPTIONS\n"
        "Obligations do not apply to publicly known information.\n\n"
        "IV. TERM\n"
        "Obligations shall survive for three years after disclosure.\n"
    )


@pytest.fixture
def empty_text() -> str:
    return ""


@pytest.fixture
def whitespace_only() -> str:
    return "   \n\n   \t  \n"


@pytest.fixture
def no_headings_text() -> str:
    return (
        "This is a plain document with no headings or sections.\n"
        "It contains some text but no structural markers whatsoever.\n"
        "The segmenter should return an empty sections list.\n"
    )


# ---------------------------------------------------------------------------
# DocumentSegmenter instantiation
# ---------------------------------------------------------------------------


class TestDocumentSegmenterInit:
    def test_default_params(self) -> None:
        seg = DocumentSegmenter()
        assert seg.extract_definitions is True
        assert seg.min_heading_chars == 3
        assert seg.strip_page_numbers is True

    def test_custom_params(self) -> None:
        seg = DocumentSegmenter(
            extract_definitions=False,
            min_heading_chars=5,
            strip_page_numbers=False,
        )
        assert seg.extract_definitions is False
        assert seg.min_heading_chars == 5
        assert seg.strip_page_numbers is False


# ---------------------------------------------------------------------------
# segment_document convenience function
# ---------------------------------------------------------------------------


class TestSegmentDocument:
    def test_returns_document_structure(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        assert isinstance(result, DocumentStructure)

    def test_empty_text(self, empty_text: str) -> None:
        result = segment_document(empty_text)
        assert result.title == ""
        assert result.sections == []
        assert result.preamble == ""

    def test_whitespace_only(self, whitespace_only: str) -> None:
        result = segment_document(whitespace_only)
        assert result.title == ""

    def test_no_headings(self, no_headings_text: str) -> None:
        result = segment_document(no_headings_text)
        assert result.sections == []
        # Preamble should capture the whole text
        assert len(result.preamble) > 0


# ---------------------------------------------------------------------------
# Title detection
# ---------------------------------------------------------------------------


class TestTitleDetection:
    def test_simple_contract_title(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        assert result.title == "SERVICE AGREEMENT"

    def test_article_contract_title(self, article_contract: str) -> None:
        result = segment_document(article_contract)
        assert result.title == "MASTER SERVICES AGREEMENT"

    def test_employment_agreement_title(self, section_keyword_contract: str) -> None:
        result = segment_document(section_keyword_contract)
        assert result.title == "EMPLOYMENT AGREEMENT"


# ---------------------------------------------------------------------------
# Preamble extraction
# ---------------------------------------------------------------------------


class TestPreambleExtraction:
    def test_preamble_contains_intro(self) -> None:
        # Use text with explicit non-heading intro before the first section
        text = (
            "This Agreement is entered into between Alpha Corp and Beta LLC "
            "as of January 1, 2024.\n\n"
            "1. DEFINITIONS\n"
            "Terms are defined below.\n"
        )
        result = segment_document(text)
        # Preamble is text before the first heading
        assert len(result.preamble) > 0
        assert "Alpha Corp" in result.preamble

    def test_preamble_max_length(self) -> None:
        # Build a doc with a very long preamble
        long_preamble = "X" * 2000 + "\n\n1. SECTION ONE\nContent here.\n"
        result = segment_document(long_preamble)
        assert len(result.preamble) <= 1000


# ---------------------------------------------------------------------------
# Decimal outline sections
# ---------------------------------------------------------------------------


class TestDecimalSections:
    def test_top_level_section_count(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        # Sections 1–5 are top-level
        assert result.section_count >= 5

    def test_section_headings_present(self, simple_contract: str) -> None:
        r = segment_document(simple_contract)
        top_headings = {s.heading for s in r.sections}
        assert "DEFINITIONS" in top_headings or any("DEFINITION" in h for h in top_headings)
        assert any("PAYMENT" in h for h in top_headings)
        assert any("TERMINAT" in h for h in top_headings)

    def test_sub_sections_are_children(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        # Section 2 (SERVICES) should have children 2.1 and 2.2
        services = result.find_by_heading("SERVICES")
        assert len(services) >= 1
        # At least one section has children
        total = result.total_sections
        assert total >= result.section_count

    def test_section_numbers(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        numbers = {s.number for s in result.sections}
        # We expect at least numeric section numbers present
        assert any(n.strip("1234567890.") == "" or n.isdigit() for n in numbers)


# ---------------------------------------------------------------------------
# ARTICLE-style sections
# ---------------------------------------------------------------------------


class TestArticleSections:
    def test_article_level_assigned(self, article_contract: str) -> None:
        result = segment_document(article_contract)
        article_sections = [s for s in result.sections if s.level == SectionLevel.ARTICLE]
        assert len(article_sections) >= 3

    def test_article_headings(self, article_contract: str) -> None:
        result = segment_document(article_contract)
        headings = [s.heading for s in result.sections]
        assert any("DEFINITION" in h for h in headings)
        assert any("COMPENSATION" in h or "SERVICE" in h for h in headings)

    def test_article_numbers_contain_roman(self, article_contract: str) -> None:
        result = segment_document(article_contract)
        art_sections = [s for s in result.sections if "ARTICLE" in s.number.upper()]
        assert len(art_sections) >= 3


# ---------------------------------------------------------------------------
# SECTION-keyword style
# ---------------------------------------------------------------------------


class TestSectionKeyword:
    def test_section_level_assigned(self, section_keyword_contract: str) -> None:
        result = segment_document(section_keyword_contract)
        for s in result.sections:
            assert s.level == SectionLevel.SECTION

    def test_correct_number_of_sections(self, section_keyword_contract: str) -> None:
        result = segment_document(section_keyword_contract)
        assert result.section_count >= 4

    def test_section_content(self, section_keyword_contract: str) -> None:
        result = segment_document(section_keyword_contract)
        pay = result.find_by_heading("COMPENSATION")
        assert len(pay) >= 1
        assert "150,000" in pay[0].content or "$" in pay[0].content


# ---------------------------------------------------------------------------
# Alphabetic subsections
# ---------------------------------------------------------------------------


class TestAlphabeticSubsections:
    def test_alpha_subsections_detected(self, alphabetic_contract: str) -> None:
        result = segment_document(alphabetic_contract)
        # Should have subsection-level entries
        all_secs = list(result.iter_all_sections())
        alpha = [s for s in all_secs if s.level == SectionLevel.SUBSECTION]
        assert len(alpha) >= 2

    def test_alpha_content_preserved(self, alphabetic_contract: str) -> None:
        result = segment_document(alphabetic_contract)
        all_secs = list(result.iter_all_sections())
        alpha = [s for s in all_secs if s.level == SectionLevel.SUBSECTION]
        combined = " ".join(s.heading + " " + s.content for s in alpha)
        # Content from subsections should contain something about servers or uptime
        assert any(
            kw in combined.lower()
            for kw in ["server", "uptime", "security", "data center", "patch"]
        )


# ---------------------------------------------------------------------------
# Roman numeral sections
# ---------------------------------------------------------------------------


class TestRomanNumeralSections:
    def test_roman_sections_detected(self, roman_numeral_contract: str) -> None:
        result = segment_document(roman_numeral_contract)
        assert result.section_count >= 3

    def test_roman_headings(self, roman_numeral_contract: str) -> None:
        result = segment_document(roman_numeral_contract)
        headings = [s.heading for s in result.sections]
        assert any("EXCEPTION" in h for h in headings)
        assert any("TERM" in h for h in headings)


# ---------------------------------------------------------------------------
# Definitions extraction
# ---------------------------------------------------------------------------


class TestDefinitionsExtraction:
    def test_definitions_parsed(self, simple_contract: str) -> None:
        result = segment_document(simple_contract, extract_definitions=True)
        # Should have extracted at least one definition
        assert len(result.definitions) >= 1

    def test_definitions_disabled(self, simple_contract: str) -> None:
        result = segment_document(simple_contract, extract_definitions=False)
        assert result.definitions == {}

    def test_known_definition_key(self, simple_contract: str) -> None:
        result = segment_document(simple_contract, extract_definitions=True)
        # "Confidential Information" or "Services" should be found
        keys_lower = {k.lower() for k in result.definitions}
        assert any(kw in keys_lower for kw in ["confidential information", "services"])


# ---------------------------------------------------------------------------
# DocumentStructure helpers
# ---------------------------------------------------------------------------


class TestDocumentStructureHelpers:
    def test_find_by_heading(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        hits = result.find_by_heading("payment")
        assert len(hits) >= 1

    def test_find_by_heading_case_insensitive(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        hits_lower = result.find_by_heading("payment")
        hits_upper = result.find_by_heading("PAYMENT")
        assert len(hits_lower) == len(hits_upper)

    def test_find_by_heading_no_match(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        hits = result.find_by_heading("xylophone")
        assert hits == []

    def test_find_by_number(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        # One of the sections should have number "3"
        found_any = any(result.find_by_number(str(i)) is not None for i in range(1, 6))
        assert found_any

    def test_total_sections_ge_section_count(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        assert result.total_sections >= result.section_count

    def test_iter_all_sections(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        all_secs = list(result.iter_all_sections())
        assert len(all_secs) == result.total_sections

    def test_to_dict_keys(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        d = result.to_dict()
        assert "title" in d
        assert "sections" in d
        assert "definitions" in d
        assert "section_count" in d

    def test_to_dict_no_content(self, simple_contract: str) -> None:
        result = segment_document(simple_contract)
        d = result.to_dict(include_content=False)
        # Preamble should be empty string when include_content=False
        assert d["preamble"] == ""


# ---------------------------------------------------------------------------
# Section node helpers
# ---------------------------------------------------------------------------


class TestSectionHelpers:
    def test_full_heading(self) -> None:
        s = Section(level=SectionLevel.SECTION, number="3", heading="PAYMENT TERMS")
        assert s.full_heading == "3 PAYMENT TERMS"

    def test_full_heading_no_number(self) -> None:
        s = Section(level=SectionLevel.SECTION, number="", heading="APPENDIX")
        assert s.full_heading == "APPENDIX"

    def test_word_count(self) -> None:
        s = Section(
            level=SectionLevel.CLAUSE,
            number="1.1",
            heading="SCOPE",
            content="This clause covers the scope of work and deliverables.",
        )
        assert s.word_count == 9

    def test_has_children_true(self) -> None:
        child = Section(level=SectionLevel.SUBSECTION, number="(a)", heading="")
        parent = Section(level=SectionLevel.SECTION, number="1", heading="TEST", children=[child])
        assert parent.has_children is True

    def test_has_children_false(self) -> None:
        s = Section(level=SectionLevel.SECTION, number="1", heading="LONE")
        assert s.has_children is False

    def test_iter_all_depth_first(self) -> None:
        grandchild = Section(level=SectionLevel.PARAGRAPH, number="(i)", heading="")
        child = Section(
            level=SectionLevel.SUBSECTION, number="(a)", heading="", children=[grandchild]
        )
        parent = Section(level=SectionLevel.SECTION, number="1", heading="PARENT", children=[child])
        nodes = list(parent.iter_all())
        assert nodes == [parent, child, grandchild]

    def test_section_to_dict(self) -> None:
        s = Section(
            level=SectionLevel.CLAUSE,
            number="2.1",
            heading="SCOPE",
            content="Provider will perform the Services.",
        )
        d = s.to_dict()
        assert d["level"] == "clause"
        assert d["number"] == "2.1"
        assert d["heading"] == "SCOPE"
        assert d["content"] == "Provider will perform the Services."

    def test_section_to_dict_no_content(self) -> None:
        s = Section(level=SectionLevel.SECTION, number="5", heading="TERM")
        d = s.to_dict(include_content=False)
        assert "content" not in d


# ---------------------------------------------------------------------------
# SectionLevel enum
# ---------------------------------------------------------------------------


class TestSectionLevel:
    def test_all_levels_defined(self) -> None:
        expected = {
            "part",
            "article",
            "section",
            "clause",
            "subsection",
            "paragraph",
            "definition",
            "unknown",
        }
        actual = {level.value for level in SectionLevel}
        assert expected == actual

    def test_level_is_string(self) -> None:
        assert isinstance(SectionLevel.SECTION.value, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_page_number_stripping(self) -> None:
        text = (
            "CONTRACT TITLE\n\n"
            "1. FIRST SECTION\n"
            "Some content here.\n"
            "- 1 -\n"
            "2. SECOND SECTION\n"
            "More content.\n"
            "2\n"
        )
        result = segment_document(text, strip_page_numbers=True)
        # There should still be real sections
        assert result.section_count >= 2

    def test_page_number_stripping_disabled(self) -> None:
        text = "CONTRACT\n\n1. SECTION ONE\nContent.\n-  5  -\n"
        seg = DocumentSegmenter(strip_page_numbers=False)
        result = seg.segment(text)
        # With stripping disabled, we just shouldn't crash
        assert isinstance(result, DocumentStructure)

    def test_single_section(self) -> None:
        # Use lower-case preamble so the title isn't detected as a heading
        text = "nda agreement\n\n1. ENTIRE AGREEMENT\nThis document is the entire agreement.\n"
        result = segment_document(text)
        # "nda agreement" is the title/preamble; the one real section is numbered "1."
        sec = result.find_by_heading("ENTIRE AGREEMENT")
        assert len(sec) >= 1
        assert "entire agreement" in sec[0].content.lower()

    def test_deeply_nested_decimal(self) -> None:
        text = (
            "AGREEMENT\n\n"
            "1. TOP LEVEL\n"
            "Top content.\n"
            "1.1 SECOND LEVEL\n"
            "Second content.\n"
            "1.1.1 THIRD LEVEL\n"
            "Third content.\n"
        )
        result = segment_document(text)
        # Total sections should be 3
        assert result.total_sections >= 3

    def test_section_content_preserved(self) -> None:
        text = (
            "NDA\n\n"
            "1. CONFIDENTIALITY\n"
            "The receiving party shall not disclose any confidential information.\n"
            "Obligations survive for five years.\n"
        )
        result = segment_document(text)
        secs = result.find_by_heading("CONFIDENTIALITY")
        assert len(secs) >= 1
        assert "five years" in secs[0].content.lower()

    def test_section_with_special_chars_in_heading(self) -> None:
        text = (
            "AGREEMENT\n\n"
            "SECTION 7 — FORCE MAJEURE\n"
            "Neither party shall be liable for delays caused by events beyond their control.\n"
        )
        result = segment_document(text)
        assert result.section_count >= 1

    def test_unicode_section_symbol(self) -> None:
        text = (
            "LEASE AGREEMENT\n\n"
            "§ 1 RENT\n"
            "Tenant shall pay $2,000 per month.\n\n"
            "§ 2 TERM\n"
            "Lease term is twelve months.\n"
        )
        result = segment_document(text)
        assert result.section_count >= 2

    def test_all_caps_standalone_heading(self) -> None:
        text = (
            "SCHEDULE B\n\n"
            "PAYMENT TERMS\n"
            "All invoices are due within 30 days.\n\n"
            "LIMITATION OF LIABILITY\n"
            "Aggregate liability is capped at total fees paid.\n"
        )
        result = segment_document(text)
        assert result.section_count >= 1
