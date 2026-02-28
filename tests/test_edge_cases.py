"""Edge-case and regression tests for extractors and models."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.extractors import ClauseExtractor, EntityExtractor, RiskDetector
from legal_doc_analyzer.models import (
    AnalysisResult,
    Clause,
    ClauseType,
    Entity,
    EntityType,
    Risk,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# ClauseExtractor edge cases
# ---------------------------------------------------------------------------


class TestClauseExtractorEdgeCases:
    """Edge-case tests for ClauseExtractor."""

    @pytest.fixture
    def extractor(self) -> ClauseExtractor:
        return ClauseExtractor()

    def test_whitespace_only(self, extractor: ClauseExtractor) -> None:
        assert extractor.extract("   \n\n\t  ") == []

    def test_none_like_empty(self, extractor: ClauseExtractor) -> None:
        assert extractor.extract("") == []

    def test_very_short_text(self, extractor: ClauseExtractor) -> None:
        """Text shorter than the 20-char section threshold yields nothing."""
        assert extractor.extract("Short.") == []

    def test_repeated_clause_type_deduplication(self, extractor: ClauseExtractor) -> None:
        """Two identical sections should not produce duplicate clauses at same offset."""
        text = (
            "1. TERMINATION\n"
            "Either party may terminate upon 30 days notice for material breach.\n\n"
            "1. TERMINATION\n"
            "Either party may terminate upon 30 days notice for material breach.\n"
        )
        clauses = extractor.extract(text)
        # Each section sits at a different offset, so we may get two, but never
        # more than two (one per unique start_char).
        assert len(clauses) <= 2

    def test_unicode_text(self, extractor: ClauseExtractor) -> None:
        """Extractor should not crash on non-ASCII legal text."""
        text = (
            "1. CONFIDENTIALITÉ\n"
            "Les parties doivent maintenir la confidentialité des informations "
            "propriétaires et des secrets commerciaux communiqués. "
            "La partie réceptrice ne divulguera pas d'informations confidentielles."
        )
        clauses = extractor.extract(text)
        # May or may not detect (English keywords), but must not crash.
        assert isinstance(clauses, list)

    def test_mixed_numbering_styles(self, extractor: ClauseExtractor) -> None:
        """Handles various heading numbering formats."""
        text = (
            "Section 1: PAYMENT\n"
            "Client shall pay $100,000 upon execution. Invoice due net 30.\n\n"
            "Article 2: TERMINATION\n"
            "Either party may terminate this Agreement upon 30 days written notice "
            "for material breach.\n"
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.PAYMENT in types or ClauseType.TERMINATION in types

    def test_lowercase_headings(self, extractor: ClauseExtractor) -> None:
        """Keywords in body should still match even if heading is lowercase."""
        text = (
            "payment terms\n"
            "The client shall pay the provider $50,000 upon execution. "
            "All invoices are due within net 30 days of receipt. "
            "Late payments shall accrue interest at a rate of 1.5% per month."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.PAYMENT in types

    def test_extract_by_type_empty_filter(self, extractor: ClauseExtractor) -> None:
        text = (
            "1. TERMINATION\nEither party may terminate upon 30 days notice for material breach.\n"
        )
        clauses = extractor.extract_by_type(text, [ClauseType.PAYMENT])
        assert clauses == []

    def test_very_long_section(self, extractor: ClauseExtractor) -> None:
        """A massive section should still be processable."""
        filler = "The parties agree to the terms stated herein. " * 500
        text = f"1. CONFIDENTIALITY\n{filler}confidential information trade secrets.\n"
        clauses = extractor.extract(text)
        assert isinstance(clauses, list)

    def test_clause_confidence_never_exceeds_one(self, extractor: ClauseExtractor) -> None:
        """Confidence should be capped at 1.0 even with strong signals."""
        text = (
            "CONFIDENTIALITY AND NON-DISCLOSURE\n"
            "Confidential information and trade secrets. The confidentiality "
            "obligations require each party to maintain confidentiality. "
            "The receiving party and disclosing party must treat all "
            "confidential information as proprietary information under "
            "this non-disclosure agreement."
        )
        clauses = extractor.extract(text)
        for c in clauses:
            assert c.confidence <= 1.0


# ---------------------------------------------------------------------------
# EntityExtractor edge cases
# ---------------------------------------------------------------------------


class TestEntityExtractorEdgeCases:
    """Edge-case tests for EntityExtractor."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_whitespace_only(self, extractor: EntityExtractor) -> None:
        assert extractor.extract("   ") == []

    def test_no_entities_in_gibberish(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("asdfghjkl qwerty zxcvbnm 12345")
        # Might pick up numbers as durations; verify no crash.
        assert isinstance(entities, list)

    def test_overlapping_date_formats(self, extractor: EntityExtractor) -> None:
        """Multiple date formats in one sentence."""
        text = "Signed on January 1, 2024; effective 2024-06-15; expires 12/31/2025."
        entities = extractor.extract(text)
        dates = [e for e in entities if e.type == EntityType.DATE]
        assert len(dates) >= 2

    def test_money_without_currency_symbol(self, extractor: EntityExtractor) -> None:
        """'500 dollars' format detection."""
        text = "The fee is 500 dollars plus 200 euros in expenses."
        entities = extractor.extract(text)
        money = [e for e in entities if e.type == EntityType.MONEY]
        assert len(money) >= 1

    def test_party_with_special_chars(self, extractor: EntityExtractor) -> None:
        """Party names with apostrophes or ampersands."""
        text = 'between O\'Brien & Associates Ltd. ("Firm") and Smith-Jones Corp. ("Client")'
        entities = extractor.extract(text)
        parties = [e for e in entities if e.type == EntityType.PARTY]
        assert len(parties) >= 1

    def test_no_false_positive_dates(self, extractor: EntityExtractor) -> None:
        """Ensure random numbers aren't flagged as dates."""
        text = "The result was 42 and the code is 9876."
        entities = extractor.extract(text)
        dates = [e for e in entities if e.type == EntityType.DATE]
        assert len(dates) == 0

    def test_obligation_detection(self, extractor: EntityExtractor) -> None:
        text = "The provider shall deliver all documentation within 14 business days."
        entities = extractor.extract(text)
        obligations = [e for e in entities if e.type == EntityType.OBLIGATION]
        assert len(obligations) >= 1

    def test_reference_with_sub_sections(self, extractor: EntityExtractor) -> None:
        text = "As described in Section 4.2(a) and Article 7."
        entities = extractor.extract(text)
        refs = [e for e in entities if e.type == EntityType.REFERENCE]
        assert len(refs) >= 1

    def test_duration_with_words(self, extractor: EntityExtractor) -> None:
        """Duration like 'sixty (60) days'."""
        text = "Notice must be given sixty (60) days prior to termination."
        entities = extractor.extract(text)
        durations = [e for e in entities if e.type == EntityType.DURATION]
        assert len(durations) >= 1


# ---------------------------------------------------------------------------
# RiskDetector edge cases
# ---------------------------------------------------------------------------


class TestRiskDetectorEdgeCases:
    """Edge-case tests for RiskDetector."""

    @pytest.fixture
    def detector(self) -> RiskDetector:
        return RiskDetector()

    def test_empty_clauses_and_empty_text(self, detector: RiskDetector) -> None:
        """Should still flag missing essentials even with empty text."""
        risks = detector.analyze(clauses=[], text="")
        assert len(risks) > 0

    def test_all_essential_plus_recommended(self, detector: RiskDetector) -> None:
        """No missing-clause risks when everything is present."""
        all_types = RiskDetector.ESSENTIAL_CLAUSES | RiskDetector.RECOMMENDED_CLAUSES
        clauses = [Clause(type=ct, text=f"Clause {ct.value}", confidence=0.9) for ct in all_types]
        risks = detector.analyze(clauses=clauses, text="Some contract text.")
        missing = [r for r in risks if r.category == "missing_clause"]
        assert len(missing) == 0

    def test_multiple_liability_clauses(self, detector: RiskDetector) -> None:
        """Two liability clauses: one capped, one not."""
        clauses = [
            Clause(
                type=ClauseType.LIABILITY,
                text="Liability is governed by this Agreement.",
                confidence=0.8,
            ),
            Clause(
                type=ClauseType.LIABILITY,
                text="Aggregate liability shall not exceed the total fees paid.",
                confidence=0.9,
            ),
        ]
        risks = detector.analyze(clauses=clauses, text="")
        unlimited = [r for r in risks if r.category == "unlimited_liability"]
        # First clause should trigger, second should not.
        assert len(unlimited) == 1

    def test_unilateral_termination_detected(self, detector: RiskDetector) -> None:
        text = "The Company may terminate this Agreement at any time without cause."
        risks = detector.analyze(clauses=[], text=text)
        unilateral = [r for r in risks if r.category == "unilateral_termination"]
        assert len(unilateral) >= 1

    def test_broad_ip_without_limitation(self, detector: RiskDetector) -> None:
        text = "The Contractor hereby assigns all rights, title, and interest in the work product."
        risks = detector.analyze(clauses=[], text=text)
        ip_risks = [r for r in risks if r.category == "broad_ip_assignment"]
        assert len(ip_risks) >= 1

    def test_broad_ip_with_limitation(self, detector: RiskDetector) -> None:
        text = (
            "The Contractor assigns all rights, title, and interest "
            "arising from and in connection with this project only."
        )
        risks = detector.analyze(clauses=[], text=text)
        ip_risks = [r for r in risks if r.category == "broad_ip_assignment"]
        assert len(ip_risks) == 0

    def test_ambiguous_language_below_threshold(self, detector: RiskDetector) -> None:
        """Only one ambiguous phrase — below the threshold of 2."""
        text = "The provider shall use reasonable efforts to complete work."
        risks = detector.analyze(clauses=[], text=text)
        ambiguous = [r for r in risks if r.category == "ambiguous_language"]
        assert len(ambiguous) == 0

    def test_ambiguous_language_above_threshold(self, detector: RiskDetector) -> None:
        text = (
            "Provider shall use reasonable efforts and best efforts to deliver "
            "as soon as practicable."
        )
        risks = detector.analyze(clauses=[], text=text)
        ambiguous = [r for r in risks if r.category == "ambiguous_language"]
        assert len(ambiguous) >= 1

    def test_annotate_clauses_sets_risk_level(self, detector: RiskDetector) -> None:
        """Clauses linked to risks should be annotated."""
        clause = Clause(
            type=ClauseType.LIABILITY,
            text="Each party is liable for all damages.",
            confidence=0.8,
        )
        detector.analyze(clauses=[clause], text="")
        assert clause.risk_level == RiskLevel.HIGH

    def test_risk_sorting_comprehensive(self, detector: RiskDetector) -> None:
        """Many risk types together should sort correctly."""
        text = (
            "This contract automatically renews each year. "
            "The Company may terminate at any time without cause. "
            "Provider shall use reasonable efforts and best efforts "
            "as soon as practicable."
        )
        risks = detector.analyze(clauses=[], text=text)
        severity = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2, RiskLevel.INFO: 3}
        for i in range(len(risks) - 1):
            assert severity[risks[i].level] <= severity[risks[i + 1].level]


# ---------------------------------------------------------------------------
# Model edge cases
# ---------------------------------------------------------------------------


class TestModelEdgeCases:
    """Edge-case tests for data models."""

    def test_clause_to_dict_no_optional_fields(self) -> None:
        clause = Clause(type=ClauseType.OTHER, text="misc", confidence=0.5)
        d = clause.to_dict()
        assert d["page"] is None
        assert d["risk_reason"] is None

    def test_entity_default_confidence(self) -> None:
        e = Entity(type=EntityType.PARTY, text="Acme")
        assert e.confidence == 1.0
        assert e.normalized is None

    def test_risk_without_suggestion(self) -> None:
        r = Risk(level=RiskLevel.INFO, category="note", description="FYI")
        d = r.to_dict()
        assert d["suggestion"] is None

    def test_analysis_result_risk_score_all_info(self) -> None:
        """INFO-level risks should contribute 0 to score."""
        risks = [
            Risk(level=RiskLevel.INFO, category=f"i{i}", description=f"info {i}") for i in range(5)
        ]
        result = AnalysisResult(filename="f.pdf", summary="s", risks=risks)
        assert result.risk_score == 0.0

    def test_analysis_result_risk_score_mixed(self) -> None:
        risks = [
            Risk(level=RiskLevel.HIGH, category="a", description="a"),
            Risk(level=RiskLevel.MEDIUM, category="b", description="b"),
            Risk(level=RiskLevel.LOW, category="c", description="c"),
            Risk(level=RiskLevel.INFO, category="d", description="d"),
        ]
        result = AnalysisResult(filename="f.pdf", summary="s", risks=risks)
        score = result.risk_score
        assert 0.0 < score < 1.0

    def test_analysis_result_empty_to_dict(self) -> None:
        result = AnalysisResult(filename="empty.txt", summary="Nothing found.")
        d = result.to_dict()
        assert d["clauses"] == []
        assert d["entities"] == []
        assert d["risks"] == []
        assert d["risk_score"] == 0.0

    def test_clause_type_enum_values(self) -> None:
        """All enum members should have lowercase-snake values."""
        for ct in ClauseType:
            assert ct.value == ct.value.lower()
            assert " " not in ct.value

    def test_risk_level_enum_completeness(self) -> None:
        expected = {"high", "medium", "low", "info"}
        actual = {rl.value for rl in RiskLevel}
        assert expected == actual

    def test_entity_type_enum_completeness(self) -> None:
        assert len(EntityType) >= 7


# ---------------------------------------------------------------------------
# Parsers edge cases
# ---------------------------------------------------------------------------


class TestParsersEdgeCases:
    """Edge-case tests for document parsers."""

    def test_text_parser_empty_file(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import TextParser

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        parser = TextParser()
        doc = parser.parse(f)
        assert doc.page_count == 1
        assert doc.full_text == ""

    def test_text_parser_form_feed_pages(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import TextParser

        f = tmp_path / "multi.txt"
        f.write_text("Page one\fPage two\fPage three", encoding="utf-8")
        parser = TextParser()
        doc = parser.parse(f)
        assert doc.page_count == 3

    def test_text_parser_nonexistent_file(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import TextParser

        parser = TextParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nope.txt")

    def test_get_parser_unsupported_ext(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import get_parser

        f = tmp_path / "data.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="No parser available"):
            get_parser(f)

    def test_get_parser_txt(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import TextParser, get_parser

        f = tmp_path / "doc.txt"
        f.write_text("hello")
        parser = get_parser(f)
        assert isinstance(parser, TextParser)

    def test_parsed_document_char_offset(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import TextParser

        f = tmp_path / "contract.txt"
        f.write_text("Page 1 text\fPage 2 text", encoding="utf-8")
        parser = TextParser()
        doc = parser.parse(f)
        # Offset 0 should be page 1
        assert doc.get_page_for_char_offset(0) == 1
        # Offset past page 1 should be page 2
        assert doc.get_page_for_char_offset(20) == 2

    def test_html_parser_strips_tags(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import HTMLParser

        f = tmp_path / "doc.html"
        f.write_text(
            "<html><body><h1>Title</h1><p>Content here.</p></body></html>",
            encoding="utf-8",
        )
        parser = HTMLParser()
        doc = parser.parse(f)
        assert "Title" in doc.full_text
        assert "<h1>" not in doc.full_text

    def test_html_parser_removes_script(self, tmp_path) -> None:
        from legal_doc_analyzer.parsers import HTMLParser

        f = tmp_path / "doc.html"
        f.write_text(
            "<html><body><script>alert('x')</script><p>Safe text</p></body></html>",
            encoding="utf-8",
        )
        parser = HTMLParser()
        doc = parser.parse(f)
        assert "alert" not in doc.full_text
        assert "Safe text" in doc.full_text
