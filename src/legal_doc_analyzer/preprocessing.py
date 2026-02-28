"""Text preprocessing and readability analysis for legal documents.

Provides cleaning, normalization, sentence segmentation, token analysis,
and readability scoring tailored to legal text. No external NLP libraries
required — uses pure Python with regex-based processing.

Readability metrics include:
- Flesch-Kincaid Grade Level
- Coleman-Liau Index
- Automated Readability Index (ARI)
- Legal jargon density
- Sentence complexity score

These metrics help assess document accessibility and flag overly
complex or ambiguous language patterns.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Readability Result
# ---------------------------------------------------------------------------


@dataclass
class ReadabilityResult:
    """Readability analysis output for a document or section.

    Attributes:
        flesch_kincaid_grade: Estimated US school grade level needed to
            comprehend the text. Legal documents typically score 14-20+.
        coleman_liau_index: Grade level estimate based on character counts.
        ari: Automated Readability Index.
        avg_sentence_length: Mean number of words per sentence.
        avg_word_length: Mean number of characters per word.
        avg_syllables_per_word: Estimated mean syllable count per word.
        jargon_density: Fraction of words identified as legal jargon (0-1).
        complex_word_ratio: Fraction of words with 3+ syllables.
        sentence_count: Total sentences detected.
        word_count: Total words detected.
        vocabulary_richness: Type-token ratio (unique words / total words).
        top_terms: Most frequent non-stopword terms with counts.
        grade_label: Human-readable interpretation of the grade level.
    """

    flesch_kincaid_grade: float = 0.0
    coleman_liau_index: float = 0.0
    ari: float = 0.0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    avg_syllables_per_word: float = 0.0
    jargon_density: float = 0.0
    complex_word_ratio: float = 0.0
    sentence_count: int = 0
    word_count: int = 0
    vocabulary_richness: float = 0.0
    top_terms: list[tuple[str, int]] = field(default_factory=list)

    @property
    def grade_label(self) -> str:
        """Human-readable label for the Flesch-Kincaid grade level."""
        grade = self.flesch_kincaid_grade
        if grade >= 20:
            return "Extremely complex (post-graduate level)"
        elif grade >= 16:
            return "Very complex (graduate level)"
        elif grade >= 13:
            return "Complex (college level)"
        elif grade >= 10:
            return "Moderately complex (high school)"
        elif grade >= 7:
            return "Moderate (middle school)"
        else:
            return "Accessible (elementary)"

    def to_dict(self) -> dict:
        return {
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 2),
            "coleman_liau_index": round(self.coleman_liau_index, 2),
            "ari": round(self.ari, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_syllables_per_word": round(self.avg_syllables_per_word, 2),
            "jargon_density": round(self.jargon_density, 4),
            "complex_word_ratio": round(self.complex_word_ratio, 4),
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "vocabulary_richness": round(self.vocabulary_richness, 4),
            "top_terms": self.top_terms[:20],
            "grade_label": self.grade_label,
        }


# ---------------------------------------------------------------------------
# Legal-specific constants
# ---------------------------------------------------------------------------

# Common legal jargon terms used to compute jargon density
LEGAL_JARGON: frozenset[str] = frozenset(
    {
        "herein",
        "hereinafter",
        "hereinbefore",
        "hereinabove",
        "hereinbelow",
        "hereof",
        "hereto",
        "hereunder",
        "hereby",
        "herewith",
        "thereof",
        "therein",
        "thereto",
        "thereunder",
        "thereafter",
        "thereby",
        "therewith",
        "therefrom",
        "whereas",
        "wherefore",
        "wherein",
        "whereby",
        "notwithstanding",
        "aforementioned",
        "aforesaid",
        "foregoing",
        "forthwith",
        "indemnify",
        "indemnification",
        "indemnified",
        "liquidated",
        "stipulated",
        "adjudicated",
        "tort",
        "tortious",
        "tortfeasor",
        "plaintiff",
        "defendant",
        "appellant",
        "respondent",
        "petitioner",
        "claimant",
        "complainant",
        "estoppel",
        "laches",
        "subpoena",
        "mandamus",
        "certiorari",
        "habeas",
        "corpus",
        "prima",
        "facie",
        "jurisdiction",
        "jurisdictional",
        "adjudication",
        "pursuant",
        "preamble",
        "proviso",
        "codicil",
        "rescission",
        "rescind",
        "revocation",
        "revoke",
        "assignee",
        "assignor",
        "mortgagee",
        "mortgagor",
        "lessee",
        "lessor",
        "licensee",
        "licensor",
        "obligor",
        "obligee",
        "surety",
        "guarantor",
        "abatement",
        "acquittal",
        "adjournment",
        "affidavit",
        "allegation",
        "arbitration",
        "bailment",
        "chattel",
        "cognizable",
        "counterclaim",
        "decedent",
        "deposition",
        "easement",
        "encumbrance",
        "escrow",
        "fiduciary",
        "garnishment",
        "hereditament",
        "injunction",
        "interpleader",
        "lien",
        "malfeasance",
        "misfeasance",
        "nonfeasance",
        "novation",
        "pendente",
        "lite",
        "quorum",
        "recusal",
        "remand",
        "replevin",
        "stipulation",
        "subrogation",
        "usufruct",
        "venue",
        "waiver",
        "severability",
        "severance",
        "covenant",
        "covenants",
        "inure",
        "supersede",
        "supersedes",
        "counterpart",
        "counterparts",
        "mutatis",
        "mutandis",
        "inter",
        "alia",
        "ipso",
        "facto",
        "bona",
        "fide",
        "mala",
        "fides",
        "pro",
        "rata",
        "quantum",
        "meruit",
    }
)

# English stopwords for term frequency analysis
STOP_WORDS: frozenset[str] = frozenset(
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
        "can",
        "must",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "his",
        "her",
        "our",
        "your",
        "we",
        "you",
        "who",
        "whom",
        "which",
        "what",
        "where",
        "when",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "any",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "also",
        "because",
        "before",
        "between",
        "during",
        "into",
        "through",
        "under",
        "until",
        "up",
        "out",
        "over",
        "here",
        "there",
    }
)

# Regex patterns for OCR artifact cleanup
_OCR_ARTIFACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Stray pipe characters often from table borders
    (re.compile(r"\|"), " "),
    # Repeated dots (table of contents leaders)
    (re.compile(r"\.{4,}"), "..."),
    # Spaces before punctuation
    (re.compile(r"\s+([.,;:!?])"), r"\1"),
    # Multiple spaces
    (re.compile(r"[ \t]{2,}"), " "),
    # Broken hyphenation across lines
    (re.compile(r"(\w)-\s*\n\s*(\w)"), r"\1\2"),
]

# Legal abbreviation expansions
_LEGAL_ABBREVIATIONS: dict[str, str] = {
    r"\bw/o\b": "without",
    r"\bw/\b": "with",
    r"\bincl\.": "including",
    r"\bexcl\.": "excluding",
    r"\bapprox\.": "approximately",
    r"\bet\s+al\.?\b": "et al.",
    r"\bi\.e\.": "i.e.",
    r"\be\.g\.": "e.g.",
    r"\bvs\.?\b": "versus",
    r"\bv\.": "versus",
    r"\bno\.": "number",
    r"\bpara\.": "paragraph",
    r"\bsec\.": "section",
    r"\bart\.": "article",
    r"\bcl\.": "clause",
    r"\bsched\.": "schedule",
    r"\bex\.": "exhibit",
    r"\bapp\.": "appendix",
    r"\bamdt\.": "amendment",
    r"\baff't\b": "affidavit",
    r"\bdist\.": "district",
    r"\bct\.": "court",
    r"\bjudg\.": "judgment",
}


# ---------------------------------------------------------------------------
# Syllable Counter
# ---------------------------------------------------------------------------


def count_syllables(word: str) -> int:
    """Estimate syllable count for an English word.

    Uses a heuristic approach: counts vowel groups with adjustments
    for silent-e, common suffixes, and diphthongs. Accuracy is
    typically within ±1 syllable for legal English.

    Args:
        word: A single English word (case-insensitive).

    Returns:
        Estimated number of syllables (minimum 1).
    """
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    # Remove trailing silent-e (but not "le" endings like "table")
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]

    # Count vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)

    # Adjustments for common patterns
    # -ed endings usually don't add a syllable (unless preceded by t/d)
    if word.endswith("ed") and len(word) > 3 and word[-3] not in "td":
        count = max(1, count - 1)

    # -es endings: usually silent unless preceded by certain consonant clusters
    if word.endswith("es") and len(word) > 3:
        if word[-3] in "sz" or word[-4:-2] in ("sh", "ch", "th"):
            pass  # -es IS a syllable here
        else:
            count = max(1, count - 1)

    # Common suffixes that add syllables
    for suffix in ("tion", "sion", "cian", "tial", "cial", "tious", "cious"):
        if word.endswith(suffix):
            count = max(1, count)
            break

    # -ious, -eous add a syllable
    if re.search(r"[^aeiou]ious$|[^aeiou]eous$", word):
        count += 1

    return max(1, count)


# ---------------------------------------------------------------------------
# Text Preprocessor
# ---------------------------------------------------------------------------


class TextPreprocessor:
    """Clean and normalize legal document text.

    Handles Unicode normalization, OCR artifact removal, whitespace
    cleanup, legal abbreviation expansion, and sentence segmentation
    tuned for legal text patterns.

    Example::

        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean("  Whereas , the\\nParty...  ")
        sentences = preprocessor.segment_sentences(cleaned)
        readability = preprocessor.analyze_readability(cleaned)
    """

    # Legal-aware sentence boundary detection
    # Handles: numbered lists, abbreviations, citations, decimal numbers
    _SENTENCE_BOUNDARY_RE = re.compile(
        r"""
        (?<=[.!?])       # After sentence-ending punctuation
        (?<!\b[A-Z]\.)   # Not after single-letter abbreviation (e.g., "U.S.")
        (?<!\bNo\.)      # Not after "No."
        (?<!\bMr\.)      # Not after "Mr."
        (?<!\bMs\.)      # Not after "Ms."
        (?<!\bMrs\.)     # Not after "Mrs."
        (?<!\bDr\.)      # Not after "Dr."
        (?<!\bJr\.)      # Not after "Jr."
        (?<!\bSr\.)      # Not after "Sr."
        (?<!\bSt\.)      # Not after "St."
        (?<!\bCo\.)      # Not after "Co."
        (?<!\bInc\.)     # Not after "Inc."
        (?<!\bLtd\.)     # Not after "Ltd."
        (?<!\bCorp\.)    # Not after "Corp."
        (?<!\bvs\.)      # Not after "vs."
        (?<!\bSec\.)     # Not after "Sec."
        (?<!\bArt\.)     # Not after "Art."
        (?<!\bPar\.)     # Not after "Par."
        (?<!\bCl\.)      # Not after "Cl."
        (?<!\bEt\.)      # Not after "Et."
        (?<!\bAl\.)      # Not after "Al."
        (?<!\be\.g\.)    # Not after "e.g."
        (?<!\bi\.e\.)    # Not after "i.e."
        (?<!\d\.)        # Not after a digit (decimal numbers, section refs)
        \s+              # Whitespace after punctuation
        (?=[A-Z("])      # Next char is uppercase, open paren, or quote
        """,
        re.VERBOSE,
    )

    # Tokenizer: splits on whitespace and punctuation boundaries
    _WORD_RE = re.compile(r"\b[a-zA-Z'-]+\b")

    def __init__(
        self,
        fix_ocr: bool = True,
        expand_abbreviations: bool = False,
        normalize_unicode: bool = True,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            fix_ocr: Remove common OCR artifacts (pipes, broken hyphens).
            expand_abbreviations: Expand legal abbreviations to full words.
            normalize_unicode: Apply NFC Unicode normalization.
        """
        self.fix_ocr = fix_ocr
        self.expand_abbreviations = expand_abbreviations
        self.normalize_unicode = normalize_unicode

    def clean(self, text: str) -> str:
        """Apply all configured cleaning steps to the text.

        Processing order:
        1. Unicode normalization (NFC)
        2. OCR artifact removal
        3. Legal abbreviation expansion (if enabled)
        4. Whitespace normalization
        5. Line break cleanup

        Args:
            text: Raw document text.

        Returns:
            Cleaned text ready for analysis.
        """
        if not text:
            return ""

        # 1. Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
            # Replace common Unicode quotation marks with ASCII
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            text = text.replace("\u2013", "-").replace("\u2014", "--")
            text = text.replace("\u2026", "...")
            text = text.replace("\xa0", " ")  # Non-breaking space

        # 2. OCR artifact removal
        if self.fix_ocr:
            for pattern, replacement in _OCR_ARTIFACT_PATTERNS:
                text = pattern.sub(replacement, text)

        # 3. Legal abbreviation expansion
        if self.expand_abbreviations:
            for abbr_pattern, expansion in _LEGAL_ABBREVIATIONS.items():
                text = re.sub(abbr_pattern, expansion, text, flags=re.IGNORECASE)

        # 4. Whitespace normalization
        # Collapse multiple blank lines into at most two (preserve paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize spaces within lines
        text = re.sub(r"[ \t]+", " ", text)
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def segment_sentences(self, text: str) -> list[str]:
        """Split text into sentences using legal-aware rules.

        Handles common legal abbreviations, section references, and
        citation formats that contain periods but are not sentence
        boundaries.

        Args:
            text: Cleaned text (run ``clean()`` first for best results).

        Returns:
            List of sentence strings.
        """
        if not text:
            return []

        # Flatten to single line for splitting (preserve paragraph info)
        flat = re.sub(r"\n+", " ", text)
        flat = re.sub(r"\s+", " ", flat).strip()

        # Split on sentence boundaries
        raw_sentences = self._SENTENCE_BOUNDARY_RE.split(flat)

        # Post-process: merge very short fragments with previous sentence
        sentences: list[str] = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            # Very short fragment (likely a continuation) — merge
            if len(s) < 15 and sentences and not s[0].isupper():
                sentences[-1] = sentences[-1] + " " + s
            else:
                sentences.append(s)

        return sentences

    def tokenize(self, text: str) -> list[str]:
        """Extract word tokens from text.

        Returns lowercase tokens with only alphabetic characters and
        common punctuation (hyphens, apostrophes).

        Args:
            text: Input text.

        Returns:
            List of lowercase word tokens.
        """
        return [m.group().lower() for m in self._WORD_RE.finditer(text)]

    def term_frequencies(
        self,
        text: str,
        exclude_stopwords: bool = True,
        min_length: int = 3,
    ) -> Counter:
        """Compute term frequency counts.

        Args:
            text: Input text.
            exclude_stopwords: Remove common English stopwords.
            min_length: Minimum word length to include.

        Returns:
            Counter of {term: count} sorted by frequency.
        """
        tokens = self.tokenize(text)
        if exclude_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        if min_length > 0:
            tokens = [t for t in tokens if len(t) >= min_length]
        return Counter(tokens)

    def analyze_readability(self, text: str) -> ReadabilityResult:
        """Compute comprehensive readability metrics for the text.

        Runs sentence segmentation, syllable counting, and multiple
        readability formulas to assess text complexity.

        Args:
            text: Document text (cleaned or raw — cleaning is applied internally).

        Returns:
            ReadabilityResult with all computed metrics.
        """
        cleaned = self.clean(text) if text else ""
        sentences = self.segment_sentences(cleaned)
        tokens = self.tokenize(cleaned)

        if not tokens or not sentences:
            return ReadabilityResult()

        total_words = len(tokens)
        total_sentences = len(sentences)
        total_chars = sum(len(t) for t in tokens)
        total_syllables = sum(count_syllables(t) for t in tokens)
        complex_words = sum(1 for t in tokens if count_syllables(t) >= 3)
        unique_words = len(set(tokens))

        # Jargon density: fraction of tokens that are legal jargon
        jargon_count = sum(1 for t in tokens if t in LEGAL_JARGON)

        avg_sentence_length = total_words / total_sentences
        avg_word_length = total_chars / total_words
        avg_syllables = total_syllables / total_words

        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59

        # Coleman-Liau Index
        # L = avg letters per 100 words, S = avg sentences per 100 words
        l_score = (total_chars / total_words) * 100
        s_score = (total_sentences / total_words) * 100
        coleman_liau = 0.0588 * l_score - 0.296 * s_score - 15.8

        # Automated Readability Index
        ari = 4.71 * (total_chars / total_words) + 0.5 * (total_words / total_sentences) - 21.43

        # Term frequencies (top 20 non-stopword terms)
        freqs = self.term_frequencies(cleaned)
        top_terms = freqs.most_common(20)

        return ReadabilityResult(
            flesch_kincaid_grade=round(fk_grade, 2),
            coleman_liau_index=round(coleman_liau, 2),
            ari=round(ari, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            avg_word_length=round(avg_word_length, 2),
            avg_syllables_per_word=round(avg_syllables, 2),
            jargon_density=round(jargon_count / total_words, 4) if total_words else 0,
            complex_word_ratio=round(complex_words / total_words, 4) if total_words else 0,
            sentence_count=total_sentences,
            word_count=total_words,
            vocabulary_richness=round(unique_words / total_words, 4) if total_words else 0,
            top_terms=top_terms,
        )


# ---------------------------------------------------------------------------
# Document Comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing two document texts.

    Attributes:
        cosine_similarity: TF-IDF cosine similarity (0-1).
        jaccard_similarity: Jaccard coefficient on unique term sets (0-1).
        shared_terms: Terms appearing in both documents with counts.
        unique_to_a: Terms only in document A.
        unique_to_b: Terms only in document B.
        length_ratio: Ratio of shorter to longer document (0-1).
    """

    cosine_similarity: float = 0.0
    jaccard_similarity: float = 0.0
    shared_terms: list[tuple[str, int, int]] = field(default_factory=list)
    unique_to_a: list[str] = field(default_factory=list)
    unique_to_b: list[str] = field(default_factory=list)
    length_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cosine_similarity": round(self.cosine_similarity, 4),
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "shared_terms_count": len(self.shared_terms),
            "unique_to_a_count": len(self.unique_to_a),
            "unique_to_b_count": len(self.unique_to_b),
            "length_ratio": round(self.length_ratio, 4),
            "top_shared_terms": self.shared_terms[:15],
        }


def compare_documents(
    text_a: str,
    text_b: str,
    preprocessor: TextPreprocessor | None = None,
) -> ComparisonResult:
    """Compare two documents using TF-IDF cosine similarity and Jaccard.

    Computes a lightweight bag-of-words comparison without requiring
    numpy or scikit-learn. Useful for detecting document versions,
    finding similar clauses, or deduplication.

    Args:
        text_a: First document text.
        text_b: Second document text.
        preprocessor: Optional preprocessor instance (creates default if None).

    Returns:
        ComparisonResult with similarity scores and term analysis.
    """
    pp = preprocessor or TextPreprocessor()

    freq_a = pp.term_frequencies(text_a)
    freq_b = pp.term_frequencies(text_b)

    if not freq_a or not freq_b:
        return ComparisonResult()

    # Build vocabulary
    vocab = set(freq_a.keys()) | set(freq_b.keys())

    # Jaccard similarity on term sets
    intersection = set(freq_a.keys()) & set(freq_b.keys())
    union = set(freq_a.keys()) | set(freq_b.keys())
    jaccard = len(intersection) / len(union) if union else 0

    # TF-IDF inspired cosine similarity
    # Use log-scaled term frequency (no IDF since we only have 2 docs)
    def _log_tf(count: int) -> float:
        return 1 + math.log(count) if count > 0 else 0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term in vocab:
        a_val = _log_tf(freq_a.get(term, 0))
        b_val = _log_tf(freq_b.get(term, 0))
        dot_product += a_val * b_val
        norm_a += a_val**2
        norm_b += b_val**2

    cosine = 0.0
    if norm_a > 0 and norm_b > 0:
        cosine = dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # Shared and unique terms
    shared = sorted(
        [(term, freq_a[term], freq_b[term]) for term in intersection],
        key=lambda x: x[1] + x[2],
        reverse=True,
    )
    unique_a = sorted(set(freq_a.keys()) - intersection)
    unique_b = sorted(set(freq_b.keys()) - intersection)

    # Length ratio
    words_a = sum(freq_a.values())
    words_b = sum(freq_b.values())
    length_ratio = min(words_a, words_b) / max(words_a, words_b) if max(words_a, words_b) > 0 else 0

    return ComparisonResult(
        cosine_similarity=cosine,
        jaccard_similarity=jaccard,
        shared_terms=shared[:50],
        unique_to_a=unique_a[:50],
        unique_to_b=unique_b[:50],
        length_ratio=length_ratio,
    )
