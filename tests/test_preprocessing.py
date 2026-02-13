"""Tests for the text preprocessing and readability analysis module."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.preprocessing import (
    ComparisonResult,
    ReadabilityResult,
    TextPreprocessor,
    compare_documents,
    count_syllables,
    LEGAL_JARGON,
    STOP_WORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Default preprocessor with all features enabled."""
    return TextPreprocessor(fix_ocr=True, expand_abbreviations=False, normalize_unicode=True)


@pytest.fixture
def expanding_preprocessor() -> TextPreprocessor:
    """Preprocessor with abbreviation expansion enabled."""
    return TextPreprocessor(fix_ocr=True, expand_abbreviations=True, normalize_unicode=True)


SAMPLE_CONTRACT = """
SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of January 15, 2024,
by and between Acme Corporation, a Delaware corporation ("Company"), and
John Smith ("Contractor").

1. SCOPE OF SERVICES

The Contractor shall provide software development services as described in
Exhibit A attached hereto. The Contractor shall use reasonable efforts to
complete all deliverables within the timeframes specified therein.

2. COMPENSATION

The Company shall pay the Contractor a fee of $150,000.00 per annum, payable
in monthly installments of $12,500.00. Payment shall be due within thirty (30)
days of receipt of a valid invoice. Late payments shall accrue interest at a
rate of 1.5% per month.

3. CONFIDENTIALITY

Each party hereby agrees to maintain the confidentiality of all proprietary
information disclosed by the other party. Confidential Information shall
include, but not be limited to, trade secrets, business plans, customer lists,
and technical specifications.

4. TERM AND TERMINATION

This Agreement shall commence on the Effective Date and continue for a period
of twelve (12) months, unless earlier terminated. Either party may terminate
this Agreement upon thirty (30) days written notice to the other party.
Notwithstanding the foregoing, Company may terminate this Agreement immediately
upon material breach by Contractor.

5. LIMITATION OF LIABILITY

In no event shall either party be liable for any indirect, incidental,
consequential, special, or punitive damages arising out of this Agreement.
The aggregate liability of either party shall not exceed the total fees paid
or payable under this Agreement during the twelve (12) month period preceding
the claim.

6. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws
of the State of Delaware, without regard to its conflict of law provisions.
"""

SIMPLE_TEXT = """
The cat sat on the mat. The dog played in the yard.
Birds were singing in the trees. It was a beautiful day.
"""


# ---------------------------------------------------------------------------
# count_syllables tests
# ---------------------------------------------------------------------------

class TestCountSyllables:
    """Tests for the syllable counting heuristic."""

    def test_one_syllable_words(self):
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1
        assert count_syllables("the") == 1
        assert count_syllables("court") == 1

    def test_two_syllable_words(self):
        assert count_syllables("party") == 2
        assert count_syllables("notice") == 2
        assert count_syllables("payment") == 2

    def test_three_syllable_words(self):
        assert count_syllables("agreement") == 3
        assert count_syllables("terminate") == 3
        assert count_syllables("defendant") == 3

    def test_multi_syllable_legal_terms(self):
        # These should be >= 4
        assert count_syllables("indemnification") >= 4
        assert count_syllables("confidentiality") >= 4
        assert count_syllables("notwithstanding") >= 4

    def test_empty_and_short(self):
        assert count_syllables("") == 0
        assert count_syllables("a") == 1
        assert count_syllables("an") == 1
        assert count_syllables("the") == 1

    def test_silent_e(self):
        # "terminate" ends in -e but has "ate" vowel group
        result = count_syllables("terminate")
        assert result >= 3

    def test_ed_suffix(self):
        # "signed" should be 1 syllable (not 2)
        assert count_syllables("signed") == 1
        # "created" should be 2-3
        assert count_syllables("created") >= 2


# ---------------------------------------------------------------------------
# TextPreprocessor.clean tests
# ---------------------------------------------------------------------------

class TestClean:
    """Tests for text cleaning and normalization."""

    def test_unicode_normalization(self, preprocessor: TextPreprocessor):
        text = "\u201cHello\u201d \u2013 world\u2026"
        result = preprocessor.clean(text)
        assert '"Hello"' in result
        assert "-" in result
        assert "..." in result

    def test_smart_quotes_replaced(self, preprocessor: TextPreprocessor):
        text = "\u2018single\u2019 and \u201cdouble\u201d"
        result = preprocessor.clean(text)
        assert "'" in result
        assert '"' in result

    def test_ocr_pipe_removal(self, preprocessor: TextPreprocessor):
        text = "Column A | Column B | Column C"
        result = preprocessor.clean(text)
        assert "|" not in result

    def test_multiple_spaces_collapsed(self, preprocessor: TextPreprocessor):
        text = "Too   many    spaces   here"
        result = preprocessor.clean(text)
        assert "  " not in result
        assert "Too many spaces here" == result

    def test_broken_hyphenation(self, preprocessor: TextPreprocessor):
        text = "indemni-\nfication"
        result = preprocessor.clean(text)
        assert "indemnification" in result

    def test_multiple_blank_lines(self, preprocessor: TextPreprocessor):
        text = "Line 1\n\n\n\n\nLine 2"
        result = preprocessor.clean(text)
        assert "\n\n\n" not in result
        assert "Line 1\n\nLine 2" == result

    def test_empty_input(self, preprocessor: TextPreprocessor):
        assert preprocessor.clean("") == ""
        assert preprocessor.clean("   ") == ""

    def test_space_before_punctuation(self, preprocessor: TextPreprocessor):
        text = "Hello , World . How are you ?"
        result = preprocessor.clean(text)
        assert "Hello, World. How are you?" == result

    def test_abbreviation_expansion(self, expanding_preprocessor: TextPreprocessor):
        text = "Pursuant to Sec. 4.2 and Art. 3 of the agreement."
        result = expanding_preprocessor.clean(text)
        assert "section" in result.lower()
        assert "article" in result.lower()

    def test_no_expansion_by_default(self, preprocessor: TextPreprocessor):
        text = "See Sec. 4.2 for details."
        result = preprocessor.clean(text)
        assert "Sec." in result  # Should not expand


# ---------------------------------------------------------------------------
# TextPreprocessor.segment_sentences tests
# ---------------------------------------------------------------------------

class TestSegmentSentences:
    """Tests for legal-aware sentence segmentation."""

    def test_basic_sentences(self, preprocessor: TextPreprocessor):
        text = "First sentence. Second sentence. Third sentence."
        sentences = preprocessor.segment_sentences(text)
        assert len(sentences) == 3

    def test_abbreviations_not_split(self, preprocessor: TextPreprocessor):
        text = "Dr. Smith and Mrs. Jones signed the contract."
        sentences = preprocessor.segment_sentences(text)
        assert len(sentences) == 1

    def test_legal_citations_not_split(self, preprocessor: TextPreprocessor):
        text = "As stated in Sec. 4.2 of the agreement. The parties agree."
        sentences = preprocessor.segment_sentences(text)
        # "Sec." should not cause a split
        assert any("Sec." in s or "sec." in s.lower() for s in sentences)

    def test_multiline_input(self, preprocessor: TextPreprocessor):
        text = "First sentence on\nmultiple lines. Second sentence."
        sentences = preprocessor.segment_sentences(text)
        assert len(sentences) == 2

    def test_empty_input(self, preprocessor: TextPreprocessor):
        assert preprocessor.segment_sentences("") == []
        assert preprocessor.segment_sentences("   ") == []

    def test_contract_text(self, preprocessor: TextPreprocessor):
        sentences = preprocessor.segment_sentences(SAMPLE_CONTRACT)
        # Should detect multiple sentences from the contract
        assert len(sentences) >= 10
        # Each sentence should be non-empty
        assert all(len(s.strip()) > 0 for s in sentences)

    def test_vs_not_split(self, preprocessor: TextPreprocessor):
        text = "In Smith vs. Jones, the court ruled. The decision stood."
        sentences = preprocessor.segment_sentences(text)
        assert len(sentences) == 2


# ---------------------------------------------------------------------------
# TextPreprocessor.tokenize and term_frequencies tests
# ---------------------------------------------------------------------------

class TestTokenize:
    """Tests for tokenization and term frequency analysis."""

    def test_basic_tokenization(self, preprocessor: TextPreprocessor):
        text = "The quick brown fox jumps."
        tokens = preprocessor.tokenize(text)
        assert tokens == ["the", "quick", "brown", "fox", "jumps"]

    def test_handles_punctuation(self, preprocessor: TextPreprocessor):
        text = "Hello, world! How's it going?"
        tokens = preprocessor.tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "how's" in tokens

    def test_hyphenated_words(self, preprocessor: TextPreprocessor):
        text = "non-disclosure and well-known"
        tokens = preprocessor.tokenize(text)
        assert "non-disclosure" in tokens
        assert "well-known" in tokens

    def test_term_frequencies_basic(self, preprocessor: TextPreprocessor):
        text = "cat dog cat bird cat dog"
        freqs = preprocessor.term_frequencies(text, exclude_stopwords=False, min_length=0)
        assert freqs["cat"] == 3
        assert freqs["dog"] == 2
        assert freqs["bird"] == 1

    def test_term_frequencies_excludes_stopwords(self, preprocessor: TextPreprocessor):
        text = "the cat and the dog are in the house"
        freqs = preprocessor.term_frequencies(text, exclude_stopwords=True)
        assert "the" not in freqs
        assert "and" not in freqs
        assert "cat" in freqs

    def test_term_frequencies_min_length(self, preprocessor: TextPreprocessor):
        text = "a an the cat dog agreement"
        freqs = preprocessor.term_frequencies(text, exclude_stopwords=False, min_length=4)
        assert "a" not in freqs
        assert "an" not in freqs
        assert "the" not in freqs
        assert "agreement" in freqs


# ---------------------------------------------------------------------------
# TextPreprocessor.analyze_readability tests
# ---------------------------------------------------------------------------

class TestReadability:
    """Tests for readability analysis."""

    def test_simple_text_low_grade(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SIMPLE_TEXT)
        assert isinstance(result, ReadabilityResult)
        assert result.word_count > 0
        assert result.sentence_count > 0
        # Simple text should have low grade level
        assert result.flesch_kincaid_grade < 10

    def test_legal_text_higher_grade(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        assert result.word_count > 100
        assert result.sentence_count >= 10
        # Legal text typically scores higher
        assert result.flesch_kincaid_grade > 5
        # Should detect some jargon
        assert result.jargon_density > 0

    def test_complex_word_ratio(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        # Legal text has many multi-syllable words
        assert result.complex_word_ratio > 0.1

    def test_vocabulary_richness(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        # Should have reasonable vocabulary diversity
        assert 0 < result.vocabulary_richness <= 1.0

    def test_top_terms_populated(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        assert len(result.top_terms) > 0
        # Each term should be a (word, count) tuple
        for term, count in result.top_terms:
            assert isinstance(term, str)
            assert isinstance(count, int)
            assert count > 0

    def test_empty_text_returns_defaults(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability("")
        assert result.word_count == 0
        assert result.sentence_count == 0
        assert result.flesch_kincaid_grade == 0.0

    def test_grade_label_mapping(self):
        r = ReadabilityResult(flesch_kincaid_grade=22)
        assert "post-graduate" in r.grade_label.lower()

        r = ReadabilityResult(flesch_kincaid_grade=17)
        assert "graduate" in r.grade_label.lower()

        r = ReadabilityResult(flesch_kincaid_grade=14)
        assert "college" in r.grade_label.lower()

        r = ReadabilityResult(flesch_kincaid_grade=5)
        assert "accessible" in r.grade_label.lower() or "elementary" in r.grade_label.lower()

    def test_to_dict(self, preprocessor: TextPreprocessor):
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        d = result.to_dict()
        assert "flesch_kincaid_grade" in d
        assert "coleman_liau_index" in d
        assert "ari" in d
        assert "grade_label" in d
        assert "top_terms" in d
        assert isinstance(d["top_terms"], list)

    def test_all_metrics_finite(self, preprocessor: TextPreprocessor):
        """Ensure no NaN or Inf values in results."""
        result = preprocessor.analyze_readability(SAMPLE_CONTRACT)
        import math
        assert math.isfinite(result.flesch_kincaid_grade)
        assert math.isfinite(result.coleman_liau_index)
        assert math.isfinite(result.ari)
        assert math.isfinite(result.avg_sentence_length)
        assert math.isfinite(result.avg_word_length)
        assert math.isfinite(result.avg_syllables_per_word)
        assert math.isfinite(result.jargon_density)
        assert math.isfinite(result.complex_word_ratio)
        assert math.isfinite(result.vocabulary_richness)


# ---------------------------------------------------------------------------
# Document comparison tests
# ---------------------------------------------------------------------------

class TestCompareDocuments:
    """Tests for document comparison functionality."""

    def test_identical_documents(self):
        result = compare_documents(SAMPLE_CONTRACT, SAMPLE_CONTRACT)
        assert result.cosine_similarity > 0.99
        assert result.jaccard_similarity == 1.0
        assert len(result.unique_to_a) == 0
        assert len(result.unique_to_b) == 0

    def test_completely_different_documents(self):
        text_a = "The quick brown fox jumps over the lazy dog repeatedly."
        text_b = "Quantum mechanics describes nature at atomic subatomic particle scales."
        result = compare_documents(text_a, text_b)
        assert result.cosine_similarity < 0.5
        assert result.jaccard_similarity < 0.3

    def test_similar_documents(self):
        text_a = SAMPLE_CONTRACT
        # Slightly modified version
        text_b = SAMPLE_CONTRACT.replace("Acme Corporation", "Beta Industries")
        text_b = text_b.replace("$150,000.00", "$200,000.00")
        result = compare_documents(text_a, text_b)
        # Should be very similar but not identical
        assert result.cosine_similarity > 0.9
        assert result.jaccard_similarity > 0.8

    def test_empty_documents(self):
        result = compare_documents("", "")
        assert result.cosine_similarity == 0.0
        assert result.jaccard_similarity == 0.0

    def test_one_empty_document(self):
        result = compare_documents(SAMPLE_CONTRACT, "")
        assert result.cosine_similarity == 0.0

    def test_shared_terms_populated(self):
        text_a = "The agreement shall terminate upon notice. Notice shall be written."
        text_b = "The agreement shall be terminated. Written notice is required."
        result = compare_documents(text_a, text_b)
        assert len(result.shared_terms) > 0
        # shared_terms are (term, count_a, count_b) tuples
        for term, ca, cb in result.shared_terms:
            assert isinstance(term, str)
            assert ca > 0
            assert cb > 0

    def test_unique_terms_correct(self):
        text_a = "alpha beta gamma delta"
        text_b = "alpha beta epsilon zeta"
        result = compare_documents(text_a, text_b)
        assert "gamma" in result.unique_to_a
        assert "delta" in result.unique_to_a
        assert "epsilon" in result.unique_to_b
        assert "zeta" in result.unique_to_b

    def test_length_ratio(self):
        text_a = "Short text with few words."
        text_b = SAMPLE_CONTRACT
        result = compare_documents(text_a, text_b)
        assert 0 < result.length_ratio < 0.5  # text_a is much shorter

    def test_to_dict(self):
        result = compare_documents(SAMPLE_CONTRACT, SAMPLE_CONTRACT)
        d = result.to_dict()
        assert "cosine_similarity" in d
        assert "jaccard_similarity" in d
        assert "shared_terms_count" in d
        assert "top_shared_terms" in d


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for the legal jargon and stopword sets."""

    def test_jargon_set_not_empty(self):
        assert len(LEGAL_JARGON) > 50

    def test_jargon_all_lowercase(self):
        for term in LEGAL_JARGON:
            assert term == term.lower(), f"Jargon term not lowercase: {term}"

    def test_stopwords_not_empty(self):
        assert len(STOP_WORDS) > 30

    def test_common_stopwords_present(self):
        for word in ["the", "and", "or", "is", "in", "of"]:
            assert word in STOP_WORDS

    def test_jargon_contains_key_legal_terms(self):
        key_terms = [
            "herein", "whereas", "notwithstanding", "indemnify",
            "jurisdiction", "covenant", "waiver", "estoppel",
        ]
        for term in key_terms:
            assert term in LEGAL_JARGON, f"Missing key legal term: {term}"

    def test_no_overlap_jargon_stopwords(self):
        """Legal jargon and stopwords should not overlap significantly."""
        overlap = LEGAL_JARGON & STOP_WORDS
        # Some minor overlap is acceptable (e.g., "all", "between")
        # but there shouldn't be many
        assert len(overlap) < 5, f"Too much overlap: {overlap}"
