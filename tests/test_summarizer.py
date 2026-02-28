"""Tests for the LegalSummarizer and TF-IDF utilities."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.extractors import ClauseExtractor
from legal_doc_analyzer.summarizer import (
    LegalSummarizer,
    SummaryResult,
    _cosine_similarity,
    _idf,
    _split_sentences,
    _tf,
    _tokenize,
    _top_terms,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def summarizer() -> LegalSummarizer:
    return LegalSummarizer(max_sentences=3)


@pytest.fixture
def long_contract() -> str:
    return (
        "This Agreement is entered into between Alpha Corp and Beta Services LLC "
        "on the 1st day of January 2024. "
        "Both parties agree to maintain strict confidentiality of all proprietary "
        "information disclosed during the course of this Agreement. "
        "The Provider shall indemnify, defend, and hold harmless the Client from "
        "any claims, damages, or liabilities arising from the Provider's services. "
        "Either party may terminate this Agreement upon thirty days written notice "
        "if the other party commits a material breach and fails to cure within fifteen days. "
        "The aggregate liability of either party shall not exceed the total fees paid "
        "under this Agreement in the preceding twelve months. "
        "This Agreement shall be governed by the laws of the State of New York, "
        "without regard to conflict of law principles. "
        "All disputes shall be resolved through binding arbitration before a neutral "
        "arbitrator under the rules of the American Arbitration Association. "
        "Neither party shall assign this Agreement without the prior written consent "
        "of the other party, which consent shall not be unreasonably withheld. "
        "If any provision of this Agreement is held invalid or unenforceable, "
        "the remaining provisions shall remain in full force and effect. "
        "This Agreement constitutes the entire agreement between the parties and "
        "supersedes all prior negotiations, representations, and agreements."
    )


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_includes_legal_terms(self) -> None:
        tokens = _tokenize("The party shall indemnify and defend.")
        assert "party" in tokens
        assert "indemnify" in tokens
        assert "defend" in tokens

    def test_excludes_stop_words(self) -> None:
        tokens = _tokenize("the and or but in on at to for")
        assert tokens == []

    def test_lowercases_all(self) -> None:
        tokens = _tokenize("Party SHALL Indemnify")
        assert "party" in tokens
        assert "indemnify" in tokens

    def test_short_words_removed(self) -> None:
        tokens = _tokenize("go is it an ok be")
        assert all(len(t) > 2 for t in tokens)

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenize("") == []

    def test_non_alpha_removed(self) -> None:
        tokens = _tokenize("clause 1.2 section 4.3 payment $5,000")
        # Numbers should not appear; only alpha sequences
        assert all(t.isalpha() for t in tokens)


# ---------------------------------------------------------------------------
# _tf
# ---------------------------------------------------------------------------


class TestTf:
    def test_max_term_has_tf_one(self) -> None:
        tf = _tf(["legal", "legal", "contract"])
        assert tf["legal"] == pytest.approx(1.0)

    def test_less_frequent_term_below_one(self) -> None:
        tf = _tf(["legal", "legal", "contract"])
        assert tf["contract"] < 1.0

    def test_empty_tokens_returns_empty(self) -> None:
        assert _tf([]) == {}

    def test_single_token(self) -> None:
        tf = _tf(["only"])
        assert tf["only"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _idf
# ---------------------------------------------------------------------------


class TestIdf:
    def test_rare_term_gets_higher_idf(self) -> None:
        sentences = [["apple", "banana"], ["apple", "cherry"], ["cherry", "date"]]
        vocab = {"apple", "banana", "cherry", "date"}
        idf = _idf(sentences, vocab)
        # "banana" appears in 1 doc, "apple" in 2 -> banana has higher IDF
        assert idf["banana"] > idf["apple"]

    def test_ubiquitous_term_gets_minimum_idf(self) -> None:
        sentences = [["common"], ["common"], ["common"]]
        vocab = {"common"}
        idf = _idf(sentences, vocab)
        # log((1+3)/(1+3)) + 1 = log(1) + 1 = 1.0
        assert idf["common"] == pytest.approx(1.0)

    def test_empty_sentences(self) -> None:
        result = _idf([], set())
        assert result == {}


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        v = {"alpha": 1.0, "beta": 2.0}
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        assert _cosine_similarity({"x": 1.0}, {"y": 1.0}) == pytest.approx(0.0)

    def test_empty_vector_returns_zero(self) -> None:
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0
        assert _cosine_similarity({"a": 1.0}, {}) == 0.0
        assert _cosine_similarity({}, {}) == 0.0

    def test_partial_overlap(self) -> None:
        a = {"x": 1.0, "y": 1.0}
        b = {"x": 1.0, "z": 1.0}
        sim = _cosine_similarity(a, b)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_splits_on_period_capital(self) -> None:
        text = "First sentence ends here. Second begins here. Third follows."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_very_short_fragments_excluded(self) -> None:
        text = "Ok. This sentence is definitely long enough to pass the filter."
        sentences = _split_sentences(text)
        assert any("long enough" in s for s in sentences)
        assert not any(s == "Ok." for s in sentences)

    def test_empty_string_returns_empty(self) -> None:
        assert _split_sentences("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert _split_sentences("   \n\t  ") == []

    def test_single_sentence_returned_as_is(self) -> None:
        text = "This is one complete sentence with no trailing period"
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text


# ---------------------------------------------------------------------------
# _top_terms
# ---------------------------------------------------------------------------


class TestTopTerms:
    def test_returns_correct_count(self) -> None:
        vec = {f"term{i}": float(i) for i in range(20)}
        result = _top_terms(vec, n=5)
        assert len(result) == 5

    def test_sorted_descending(self) -> None:
        vec = {"a": 0.1, "b": 0.9, "c": 0.5}
        result = _top_terms(vec, n=3)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_vector(self) -> None:
        assert _top_terms({}, n=5) == []


# ---------------------------------------------------------------------------
# LegalSummarizer initialisation
# ---------------------------------------------------------------------------


class TestLegalSummarizerInit:
    def test_default_params(self) -> None:
        s = LegalSummarizer()
        assert s.max_sentences == 5
        assert s.legal_boost == pytest.approx(0.2)
        assert s.position_weight == pytest.approx(0.1)

    def test_custom_params(self) -> None:
        s = LegalSummarizer(max_sentences=3, legal_boost=0.3, position_weight=0.05)
        assert s.max_sentences == 3

    def test_invalid_max_sentences_raises(self) -> None:
        with pytest.raises(ValueError, match="max_sentences"):
            LegalSummarizer(max_sentences=0)

    def test_negative_max_sentences_raises(self) -> None:
        with pytest.raises(ValueError, match="max_sentences"):
            LegalSummarizer(max_sentences=-5)

    def test_legal_boost_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="legal_boost"):
            LegalSummarizer(legal_boost=1.5)

    def test_legal_boost_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="legal_boost"):
            LegalSummarizer(legal_boost=-0.1)

    def test_position_weight_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="position_weight"):
            LegalSummarizer(position_weight=1.1)

    def test_position_weight_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="position_weight"):
            LegalSummarizer(position_weight=-0.5)


# ---------------------------------------------------------------------------
# LegalSummarizer.summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_returns_summary_result_type(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        assert isinstance(result, SummaryResult)

    def test_summary_is_non_empty(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        result = summarizer.summarize(long_contract)
        assert len(result.summary.strip()) > 0

    def test_respects_max_sentences(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        result = summarizer.summarize(long_contract)
        assert len(result.key_sentences) <= summarizer.max_sentences

    def test_override_max_sentences(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        result = summarizer.summarize(long_contract, max_sentences=2)
        assert len(result.key_sentences) <= 2

    def test_max_sentences_one(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        result = summarizer.summarize(long_contract, max_sentences=1)
        assert len(result.key_sentences) == 1

    def test_compression_ratio_less_than_one(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        assert result.compression_ratio < 1.0

    def test_compression_ratio_positive(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        assert result.compression_ratio > 0.0

    def test_original_word_count_matches(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        assert result.original_word_count == len(long_contract.split())

    def test_word_count_matches_summary(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        assert result.word_count == len(result.summary.split())

    def test_key_sentences_in_original(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        for ks in result.key_sentences:
            assert ks.text in long_contract

    def test_key_sentences_sorted_by_position(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        positions = [s.position for s in result.key_sentences]
        assert positions == sorted(positions)

    def test_all_key_sentences_flagged(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        for ks in result.key_sentences:
            assert ks.is_key_sentence is True

    def test_top_terms_present(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        result = summarizer.summarize(long_contract)
        assert len(result.top_terms) > 0
        for term, score in result.top_terms:
            assert isinstance(term, str) and len(term) > 0
            assert score >= 0.0

    def test_legal_terms_appear_in_summary(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        """Legal keyword boost should surface legally-significant sentences."""
        result = summarizer.summarize(long_contract)
        summary_lower = result.summary.lower()
        legal_stems = ["indemnif", "terminat", "confiden", "liabilit", "govern"]
        matched = sum(1 for stem in legal_stems if stem in summary_lower)
        assert matched >= 2

    def test_short_text_single_sentence(self, summarizer: LegalSummarizer) -> None:
        text = "This is a single complete sentence in the document."
        result = summarizer.summarize(text)
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0

    def test_empty_string_handled_gracefully(self, summarizer: LegalSummarizer) -> None:
        result = summarizer.summarize("")
        assert isinstance(result, SummaryResult)
        assert result.original_word_count == 0

    def test_whitespace_only_handled_gracefully(self, summarizer: LegalSummarizer) -> None:
        result = summarizer.summarize("   \n  ")
        assert isinstance(result, SummaryResult)


# ---------------------------------------------------------------------------
# LegalSummarizer.summarize_clauses
# ---------------------------------------------------------------------------


class TestSummarizeClauses:
    def test_summarize_from_extracted_clauses(
        self, summarizer: LegalSummarizer, short_legal_text: str
    ) -> None:
        extractor = ClauseExtractor()
        clauses = extractor.extract(short_legal_text)
        if clauses:
            result = summarizer.summarize_clauses(clauses, max_sentences=3)
            assert isinstance(result, SummaryResult)
            assert len(result.summary) > 0

    def test_empty_clause_list(self, summarizer: LegalSummarizer) -> None:
        result = summarizer.summarize_clauses([])
        assert isinstance(result, SummaryResult)

    def test_objects_without_text_attribute_skipped(self, summarizer: LegalSummarizer) -> None:
        class _Dummy:
            pass

        result = summarizer.summarize_clauses([_Dummy(), _Dummy()])
        assert isinstance(result, SummaryResult)


# ---------------------------------------------------------------------------
# SummaryResult.to_dict
# ---------------------------------------------------------------------------


class TestSummaryResultToDict:
    def test_required_keys_present(self, summarizer: LegalSummarizer, long_contract: str) -> None:
        d = summarizer.summarize(long_contract).to_dict()
        for key in (
            "summary",
            "compression_ratio",
            "word_count",
            "original_word_count",
            "top_terms",
            "key_sentence_count",
        ):
            assert key in d

    def test_compression_rounded_to_three_decimals(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        result = summarizer.summarize(long_contract)
        d = result.to_dict()
        assert d["compression_ratio"] == round(result.compression_ratio, 3)

    def test_top_terms_are_dicts_with_term_and_score(
        self, summarizer: LegalSummarizer, long_contract: str
    ) -> None:
        d = summarizer.summarize(long_contract).to_dict()
        for entry in d["top_terms"]:
            assert "term" in entry
            assert "score" in entry
