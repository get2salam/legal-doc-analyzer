"""Extractive document summarizer for legal documents.

Uses TF-IDF sentence scoring with legal keyword boosting to produce
concise summaries of contract text without requiring any external ML models.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Legal domain constants
# ---------------------------------------------------------------------------

#: Stem fragments that indicate legally significant sentences (for score boost).
_LEGAL_BOOST_STEMS: frozenset[str] = frozenset(
    {
        "indemnif",
        "terminat",
        "confidential",
        "liabilit",
        "breach",
        "govern",
        "jurisdict",
        "warrant",
        "represent",
        "covenant",
        "intellectual property",
        "force majeure",
        "arbitrat",
        "dispute",
        "payment",
        "compensat",
        "assignment",
        "obligat",
    }
)

#: Common English stop words to exclude from TF-IDF computation.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "such",
        "any",
        "all",
        "each",
        "both",
        "either",
        "neither",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "if",
        "then",
        "than",
        "there",
        "here",
        "where",
        "when",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "how",
        "also",
        "other",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "upon",
        "under",
        "over",
    }
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SentenceScore:
    """A scored sentence with position metadata."""

    text: str
    position: int
    score: float
    is_key_sentence: bool = False


@dataclass
class SummaryResult:
    """Result of an extractive summarization pass."""

    summary: str
    key_sentences: list[SentenceScore]
    compression_ratio: float
    word_count: int
    original_word_count: int
    top_terms: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "summary": self.summary,
            "compression_ratio": round(self.compression_ratio, 3),
            "word_count": self.word_count,
            "original_word_count": self.original_word_count,
            "top_terms": [
                {"term": term, "score": round(score, 4)} for term, score in self.top_terms
            ],
            "key_sentence_count": len(self.key_sentences),
        }


# ---------------------------------------------------------------------------
# TF-IDF utilities (pure Python, zero external deps)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* into lowercase words, excluding stop words and short tokens."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute augmented term frequency (normalized by max count)."""
    if not tokens:
        return {}
    counts: Counter[str] = Counter(tokens)
    max_count = max(counts.values())
    return {term: count / max_count for term, count in counts.items()}


def _idf(sentences: list[list[str]], vocab: set[str]) -> dict[str, float]:
    """Compute inverse document frequency for *vocab* across *sentences*."""
    n = len(sentences)
    result: dict[str, float] = {}
    for term in vocab:
        doc_count = sum(1 for sent_tokens in sentences if term in sent_tokens)
        # Smooth IDF: log((1+N)/(1+df)) + 1
        result[term] = math.log((1 + n) / (1 + doc_count)) + 1.0
    return result


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Build a TF-IDF vector for the given token list."""
    tf = _tf(tokens)
    return {term: tf[term] * idf.get(term, 1.0) for term in tf}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, preserving clause boundaries.

    Uses punctuation + capitalisation heuristics -- no NLTK required.
    """
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])", text)
    return [s.strip() for s in raw if len(s.strip()) >= 15]


# ---------------------------------------------------------------------------
# LegalSummarizer
# ---------------------------------------------------------------------------


class LegalSummarizer:
    """Extractive text summarizer optimized for legal documents.

    Scores sentences using TF-IDF cosine similarity to the full document,
    with optional boosts for sentences that contain legal keywords and for
    sentences near the document boundaries.

    No external ML models are required -- runs entirely on the Python
    standard library.

    Example::

        summarizer = LegalSummarizer(max_sentences=5)
        result = summarizer.summarize(contract_text)
        print(result.summary)
        print(f"Compression: {result.compression_ratio:.1%}")
    """

    def __init__(
        self,
        max_sentences: int = 5,
        legal_boost: float = 0.2,
        position_weight: float = 0.1,
    ) -> None:
        """Create a LegalSummarizer.

        Args:
            max_sentences: Default maximum number of sentences in the output
                summary.  Can be overridden per call via :meth:`summarize`.
            legal_boost: Extra score added to sentences containing at least one
                legal keyword stem. Must be in [0.0, 1.0].
            position_weight: Extra score for sentences in the first or last 15%
                of the document (preamble / conclusion carry high info density
                in contracts). Must be in [0.0, 1.0].

        Raises:
            ValueError: If any parameter is out of range.
        """
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        if not (0.0 <= legal_boost <= 1.0):
            raise ValueError("legal_boost must be between 0.0 and 1.0")
        if not (0.0 <= position_weight <= 1.0):
            raise ValueError("position_weight must be between 0.0 and 1.0")

        self.max_sentences = max_sentences
        self.legal_boost = legal_boost
        self.position_weight = position_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        max_sentences: int | None = None,
    ) -> SummaryResult:
        """Produce an extractive summary of *text*.

        Args:
            text: Full document text to summarize.
            max_sentences: Override the instance-level :attr:`max_sentences`
                for this call.

        Returns:
            :class:`SummaryResult` containing the summary string and metadata.
        """
        n = max(1, max_sentences if max_sentences is not None else self.max_sentences)
        orig_word_count = len(text.split())

        sentences = _split_sentences(text)
        if not sentences:
            return SummaryResult(
                summary=text.strip(),
                key_sentences=[],
                compression_ratio=1.0,
                word_count=orig_word_count,
                original_word_count=orig_word_count,
                top_terms=[],
            )

        tokenized = [_tokenize(s) for s in sentences]

        # Build vocabulary and IDF from all sentences
        vocab: set[str] = {t for tokens in tokenized for t in tokens}
        idf = _idf(tokenized, vocab)

        # Document-level TF-IDF vector (all tokens concatenated)
        all_tokens: list[str] = [t for tokens in tokenized for t in tokens]
        doc_vec = _tfidf_vector(all_tokens, idf)

        # Score every sentence
        scored = self._score_sentences(sentences, tokenized, doc_vec, idf)

        # Select top-N by score, then restore original order
        top_n = min(n, len(scored))
        top_sentences = sorted(scored, key=lambda s: s.score, reverse=True)[:top_n]
        for sent in top_sentences:
            sent.is_key_sentence = True
        top_sentences.sort(key=lambda s: s.position)

        summary = " ".join(s.text for s in top_sentences)
        summary_words = len(summary.split())
        compression = summary_words / max(orig_word_count, 1)
        top_terms = _top_terms(doc_vec, n=10)

        return SummaryResult(
            summary=summary,
            key_sentences=top_sentences,
            compression_ratio=compression,
            word_count=summary_words,
            original_word_count=orig_word_count,
            top_terms=top_terms,
        )

    def summarize_clauses(
        self,
        clauses: Sequence,
        max_sentences: int | None = None,
    ) -> SummaryResult:
        """Summarize a sequence of clause objects by joining their texts.

        Args:
            clauses: Any sequence whose elements expose a ``.text`` attribute
                (e.g. :class:`~legal_doc_analyzer.models.Clause` objects).
            max_sentences: Override the instance-level :attr:`max_sentences`.

        Returns:
            :class:`SummaryResult` for the combined clause text.
        """
        combined = " ".join(c.text for c in clauses if hasattr(c, "text") and c.text.strip())
        return self.summarize(combined, max_sentences=max_sentences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_sentences(
        self,
        sentences: list[str],
        tokenized: list[list[str]],
        doc_vec: dict[str, float],
        idf: dict[str, float],
    ) -> list[SentenceScore]:
        """Score each sentence and return a :class:`SentenceScore` list."""
        n = len(sentences)
        scored: list[SentenceScore] = []

        for i, (sentence, tokens) in enumerate(zip(sentences, tokenized, strict=False)):
            if not tokens:
                scored.append(SentenceScore(text=sentence, position=i, score=0.0))
                continue

            # Base score: TF-IDF cosine similarity to the full document
            sent_vec = _tfidf_vector(tokens, idf)
            base_score = _cosine_similarity(sent_vec, doc_vec)

            # Legal keyword boost
            sentence_lower = sentence.lower()
            legal_boost = (
                self.legal_boost
                if any(stem in sentence_lower for stem in _LEGAL_BOOST_STEMS)
                else 0.0
            )

            # Position bonus for first / last 15% of the document
            pos_ratio = i / max(n - 1, 1)
            position_bonus = self.position_weight if pos_ratio <= 0.15 or pos_ratio >= 0.85 else 0.0

            scored.append(
                SentenceScore(
                    text=sentence,
                    position=i,
                    score=base_score + legal_boost + position_bonus,
                )
            )

        return scored


# ---------------------------------------------------------------------------
# Module-level helper (also exported for testing)
# ---------------------------------------------------------------------------


def _top_terms(vec: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    """Return the *n* highest-scoring terms from a TF-IDF vector."""
    return sorted(vec.items(), key=lambda x: x[1], reverse=True)[:n]
