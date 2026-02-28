"""Regex-based Named Entity Recognition (NER) for legal documents.

Extracts structured entities from legal text without requiring external NLP
libraries. Recognises the following entity types:

- DATE: Calendar dates in various formats (ISO, US, UK, written-out)
- MONETARY: Currency amounts with optional symbols and suffixes
- CITATION: Common-law case citations (e.g. "Smith v. Jones [2020] 1 WLR 100")
- STATUTE: Statute and regulation references (e.g. "Companies Act 2006")
- ORGANIZATION: Corporate entities, government bodies, and legal firms
- PERSON: Human names — basic heuristic using title prefixes

All patterns are intentionally jurisdiction-neutral; they work with
documents from any common-law jurisdiction without hard-coding country-
specific names or abbreviations.

Example::

    from legal_doc_analyzer.ner import extract_entities, EntityType

    text = "On 12 January 2024, Acme Corp. agreed to pay $50,000."
    entities = extract_entities(text)
    for e in entities:
        print(e.entity_type, e.text, e.start)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    # Enums & dataclasses
    "EntityType",
    "Entity",
    # Extractors
    "extract_dates",
    "extract_monetary",
    "extract_citations",
    "extract_statutes",
    "extract_organizations",
    "extract_persons",
    "extract_entities",
]

# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Types of named entities extractable from legal documents."""

    DATE = "date"
    MONETARY = "monetary"
    CITATION = "citation"
    STATUTE = "statute"
    ORGANIZATION = "organization"
    PERSON = "person"


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A named entity extracted from legal document text.

    Attributes:
        text: Raw matched text as it appears in the source.
        entity_type: Category of the entity (e.g. DATE, MONETARY).
        start: Character offset of the first character of the match.
        end: Character offset one past the last character of the match.
        confidence: Extraction confidence in the range [0, 1].
        normalized: Optional normalised/canonical form of the value.
        metadata: Arbitrary extra key-value pairs (e.g. currency symbol).
    """

    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for serialisation."""
        return {
            "entity_type": self.entity_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "normalized": self.normalized,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# --- Dates ---

# ISO 8601:  2024-01-31
_RE_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

# US numeric:  01/31/2024  or  1/31/24
_RE_DATE_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

# UK/European numeric:  31.01.2024  or  31-01-2024
_RE_DATE_EU = re.compile(r"\b(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})\b")

# Written-out long form:  January 31, 2024  /  31 January 2024  /  31st January 2024
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_RE_DATE_LONG_MDY = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    re.IGNORECASE,
)
_RE_DATE_LONG_DMY = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\s+\d{{4}}\b",
    re.IGNORECASE,
)

# Year only with context: "in 2024", "of 1999"
_RE_DATE_YEAR_ONLY = re.compile(
    r"\b(?:in|of|since|until|before|after|by)\s+(\d{4})\b",
    re.IGNORECASE,
)

# --- Monetary amounts ---

# Symbol or ISO-code prefixed amounts.
# NOTE: \b does NOT work before non-word characters like $, £, €.
# We use (?<!\w) (negative lookbehind for word char) instead.
#
# Handles:
#   $1,234.56     £50,000     €1.5 million
#   USD 50,000    GBP 1,250.00
_RE_MONEY_SYMBOL_OR_CODE = re.compile(
    r"(?<!\w)"
    r"(?:USD|GBP|EUR|CAD|AUD|CHF|JPY|CNY)[\s]*"
    r"\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"(?:\s*(?:million|billion|trillion|thousand|k|m|bn)\b)?",
    re.IGNORECASE,
)

_RE_MONEY_CURRENCY_SYMBOL = re.compile(
    r"(?<!\w)"
    r"[\$\u00a3\u20ac\u00a5\u20b9]"  # $  £  €  ¥  ₹
    r"\s*"
    r"\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"(?:\s*(?:million|billion|trillion|thousand|k|m|bn)\b)?",
    re.IGNORECASE,
)

# Suffix form:  50,000 dollars  /  1.5 million pounds  /  300 euros
_RE_MONEY_SUFFIX = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
    r"(?:\s+(?:million|billion|trillion|thousand))?"
    r"\s+(?:dollars?|pounds?\s+sterling|pounds?|euros?|sterling|cents?)\b",
    re.IGNORECASE,
)

# --- Citations ---

# "Smith v Jones [2020] 1 WLR 100"  /  "Smith v. Jones (2020) 123 F.3d 456"
_RE_CASE_CITATION = re.compile(
    r"\b[A-Z][A-Za-z\s&',.()-]{1,50}\s+v\.?\s+[A-Z][A-Za-z\s&',.()-]{1,50}"
    r"(?:\s*[\[(]\d{4}[\])])?"
    r"(?:\s+\d+\s+[A-Z][A-Za-z.]+\s+\d+)?",
)

# Neutral citation: "[2024] EWCA Civ 123" / "[2020] UKSC 45"
_RE_NEUTRAL_CITATION = re.compile(
    r"\[\d{4}\]\s+[A-Z]{2,6}(?:\s+[A-Za-z]+)?\s+\d+",
)

# US Code citation: "42 U.S.C. § 1983"
_RE_US_CODE_CITATION = re.compile(
    r"\b\d+\s+U\.S\.C\.?\s+§+\s*\d+(?:\.\d+)*\b",
)

# US reporter: "123 F.3d 456" / "456 U.S. 789"
_RE_US_REPORTER = re.compile(
    r"\b\d+\s+(?:[A-Z][A-Za-z.]*\s+){1,3}\d+\b",
)

# --- Statute references ---

# "Companies Act 2006" / "Civil Rights Act of 1964"
_RE_STATUTE_ACT = re.compile(
    r"\b(?:[A-Z][A-Za-z\s&()]{2,60})\s+Act(?:\s+of)?\s+\d{4}\b",
)

# US code section: "18 U.S.C. § 1030"
_RE_STATUTE_CODE = re.compile(
    r"\b\d+\s+[A-Z][A-Za-z.]+\s+§+\s*\d+(?:\.\d+)*\b",
)

# EU/UK Regulation/Directive: "Regulation (EU) 2016/679"
_RE_STATUTE_REGULATION = re.compile(
    r"\b(?:Regulation|Directive|Order|Rule)\s+(?:\([A-Z]+\)\s+)?\d{4}/\d+\b",
    re.IGNORECASE,
)

# Section / Article / § references
_RE_STATUTE_SECTION = re.compile(
    r"\b(?:[Ss]ection|[Aa]rticle|§)\s*\d+(?:\([a-zA-Z0-9]+\))*"
    r"(?:\s+of\s+the\s+[A-Z][A-Za-z\s]+Act(?:\s+of)?\s+\d{4})?\b",
)

# --- Organizations ---

# Explicit suffix forms: "Acme Corporation", "Smith & Jones LLP"
# Use (?=\s|$|[^A-Za-z]) instead of \b after dot-suffixes like Inc.
_ORG_SUFFIXES = (
    r"(?:Corporation|Corp\.|Incorporated|Inc\.|Limited|Ltd\.?|"
    r"LLC|LLP|LP|PLC|plc|GmbH|S\.A\.|N\.V\.|B\.V\.|"
    r"Association|Foundation|Institute|Authority|Commission|"
    r"Department|Ministry|Office|Agency|Bureau|Board|Council|"
    r"Group|Holdings|Partners?|Trustees?)"
)
_RE_ORG = re.compile(
    rf"\b(?:[A-Z][A-Za-z&',.]*(?:\s+[A-Za-z&',.]+){{0,8}})\s+{_ORG_SUFFIXES}"
    r"(?=\s|$|[^A-Za-z])",
)

# "The [Name] Company/Bank/Fund/Trust" pattern
_RE_ORG_THE = re.compile(
    r"\b(?:the\s+)?[A-Z][A-Za-z\s&',.-]{1,50}"
    r"\s+(?:Company|Bank|Fund|Trust|Firm|Society|Union|Club|Court|University|College)\b",
    re.IGNORECASE,
)

# --- Person names ---

# Title + name: "Mr. John Smith" / "Dr Jane Doe" / "Justice Williams"
_TITLES = (
    r"(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Sir|Dame|Lord|Lady|"
    r"Justice|Judge|Honourable|Honorable|Hon\.?|Counsel|Barrister|"
    r"Solicitor|Attorney|Advocate)"
)
_RE_PERSON = re.compile(
    rf"\b{_TITLES}\s+(?:[A-Z][a-z]+\s+)*[A-Z][a-z]+(?:\s+(?:Jr\.?|Sr\.?|III?|IV))?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove fully overlapping entities, keeping the longer match.

    When two entities have overlapping character spans the one with the
    wider span is preferred.  Entities with identical spans keep only
    the first.
    """
    if not entities:
        return []

    # Sort by start position, then by descending length (longest first)
    sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

    result: list[Entity] = []
    max_end = -1
    for ent in sorted_ents:
        if ent.start >= max_end:
            result.append(ent)
            max_end = ent.end
        # If ent.start < max_end the entity overlaps with a prior match — skip it
    return result


def _make_entities(
    pattern: re.Pattern,
    text: str,
    entity_type: EntityType,
    confidence: float = 1.0,
    normalizer=None,
    metadata_fn=None,
) -> list[Entity]:
    """Find all non-overlapping matches and return Entity objects."""
    entities: list[Entity] = []
    for m in pattern.finditer(text):
        raw = m.group().strip()
        if not raw:
            continue
        normalized = normalizer(raw) if normalizer else None
        meta = metadata_fn(raw) if metadata_fn else {}
        entities.append(
            Entity(
                text=raw,
                entity_type=entity_type,
                start=m.start(),
                end=m.start() + len(raw),
                confidence=confidence,
                normalized=normalized,
                metadata=meta,
            )
        )
    return entities


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def extract_dates(text: str) -> list[Entity]:
    """Extract date entities from *text*.

    Recognises ISO (YYYY-MM-DD), US numeric (MM/DD/YYYY), European
    numeric (DD.MM.YYYY), written-out long-form dates, and year-only
    references with contextual prepositions.

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with ``entity_type=EntityType.DATE``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(_make_entities(_RE_DATE_ISO, text, EntityType.DATE, confidence=0.95))
    raw.extend(_make_entities(_RE_DATE_US, text, EntityType.DATE, confidence=0.85))
    raw.extend(_make_entities(_RE_DATE_EU, text, EntityType.DATE, confidence=0.85))
    raw.extend(_make_entities(_RE_DATE_LONG_MDY, text, EntityType.DATE, confidence=0.98))
    raw.extend(_make_entities(_RE_DATE_LONG_DMY, text, EntityType.DATE, confidence=0.98))
    raw.extend(_make_entities(_RE_DATE_YEAR_ONLY, text, EntityType.DATE, confidence=0.70))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_monetary(text: str) -> list[Entity]:
    """Extract monetary amount entities from *text*.

    Recognises currency-symbol-prefixed amounts (``$``, ``£``, ``€``
    etc.), ISO currency code prefixed amounts (``USD``, ``GBP`` …),
    and suffix forms (``50,000 dollars``).

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with ``entity_type=EntityType.MONETARY``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(
        _make_entities(_RE_MONEY_CURRENCY_SYMBOL, text, EntityType.MONETARY, confidence=0.97)
    )
    raw.extend(_make_entities(_RE_MONEY_SYMBOL_OR_CODE, text, EntityType.MONETARY, confidence=0.96))
    raw.extend(_make_entities(_RE_MONEY_SUFFIX, text, EntityType.MONETARY, confidence=0.88))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_citations(text: str) -> list[Entity]:
    """Extract case citation entities from *text*.

    Handles:
    - Party-v-party citations with optional reporter references
    - Neutral citations (``[2024] EWCA Civ 123``)
    - US Code citations (``42 U.S.C. § 1983``)
    - US reporter citations (``123 F.3d 456``)

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with
        ``entity_type=EntityType.CITATION``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(_make_entities(_RE_CASE_CITATION, text, EntityType.CITATION, confidence=0.85))
    raw.extend(_make_entities(_RE_NEUTRAL_CITATION, text, EntityType.CITATION, confidence=0.95))
    raw.extend(_make_entities(_RE_US_CODE_CITATION, text, EntityType.CITATION, confidence=0.92))
    raw.extend(_make_entities(_RE_US_REPORTER, text, EntityType.CITATION, confidence=0.70))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_statutes(text: str) -> list[Entity]:
    """Extract statute and regulation references from *text*.

    Handles:
    - "<Name> Act [of] <year>" forms
    - US code section references (``18 U.S.C. § 1030``)
    - EU/UK Regulation/Directive references
    - Section / Article / § references with optional parent-act context

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with ``entity_type=EntityType.STATUTE``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(_make_entities(_RE_STATUTE_ACT, text, EntityType.STATUTE, confidence=0.92))
    raw.extend(_make_entities(_RE_STATUTE_CODE, text, EntityType.STATUTE, confidence=0.90))
    raw.extend(_make_entities(_RE_STATUTE_REGULATION, text, EntityType.STATUTE, confidence=0.90))
    raw.extend(_make_entities(_RE_STATUTE_SECTION, text, EntityType.STATUTE, confidence=0.80))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_organizations(text: str) -> list[Entity]:
    """Extract organization name entities from *text*.

    Looks for capitalised phrases followed by standard corporate/
    governmental suffixes (``Inc.``, ``LLP``, ``Authority``, etc.).

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with
        ``entity_type=EntityType.ORGANIZATION``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(_make_entities(_RE_ORG, text, EntityType.ORGANIZATION, confidence=0.88))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_persons(text: str) -> list[Entity]:
    """Extract person name entities from *text*.

    Uses title-based heuristics: a name is recognised when preceded by
    a known honorific or professional title (``Mr.``, ``Dr.``,
    ``Justice``, etc.).

    Args:
        text: Raw document text.

    Returns:
        List of :class:`Entity` objects with ``entity_type=EntityType.PERSON``.
    """
    if not text:
        return []

    raw: list[Entity] = []
    raw.extend(_make_entities(_RE_PERSON, text, EntityType.PERSON, confidence=0.90))

    return _deduplicate(sorted(raw, key=lambda e: (e.start, -(e.end - e.start))))


def extract_entities(text: str) -> list[Entity]:
    """Extract all supported entity types from *text*.

    Runs all specialised extractors and returns a single deduplicated,
    position-sorted list of :class:`Entity` objects.

    Args:
        text: Raw legal document text.

    Returns:
        List of :class:`Entity` objects sorted by start offset.
    """
    if not text:
        return []

    all_entities: list[Entity] = []
    all_entities.extend(extract_dates(text))
    all_entities.extend(extract_monetary(text))
    all_entities.extend(extract_citations(text))
    all_entities.extend(extract_statutes(text))
    all_entities.extend(extract_organizations(text))
    all_entities.extend(extract_persons(text))

    return _deduplicate(sorted(all_entities, key=lambda e: (e.start, -(e.end - e.start))))
