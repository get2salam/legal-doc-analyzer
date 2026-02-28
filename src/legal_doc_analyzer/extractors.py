"""NLP extraction engines for legal documents.

Provides clause extraction, entity extraction, and risk detection using
regex and keyword patterns. No heavy ML dependencies required for the
basic version â€” works out of the box on any legal text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import (
    Clause,
    ClauseType,
    Entity,
    EntityType,
    Risk,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# Clause Extractor
# ---------------------------------------------------------------------------


@dataclass
class _ClausePattern:
    """Internal pattern definition for clause detection."""

    clause_type: ClauseType
    # Regex patterns that indicate this clause type (case-insensitive)
    heading_patterns: list[str] = field(default_factory=list)
    # Keywords that appear in the clause body
    body_keywords: list[str] = field(default_factory=list)
    # Minimum keyword matches required
    min_keyword_hits: int = 2


# Master pattern library â€” each entry defines how to detect a clause type
_CLAUSE_PATTERNS: list[_ClausePattern] = [
    _ClausePattern(
        clause_type=ClauseType.INDEMNIFICATION,
        heading_patterns=[
            r"indemnif(?:y|ication|ied)",
            r"hold\s+harmless",
        ],
        body_keywords=[
            "indemnify",
            "indemnification",
            "hold harmless",
            "defend",
            "losses",
            "damages",
            "claims",
            "third party",
            "liabilities",
            "costs and expenses",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.TERMINATION,
        heading_patterns=[
            r"termination",
            r"term\s+and\s+termination",
        ],
        body_keywords=[
            "terminate",
            "termination",
            "expiry",
            "expiration",
            "upon written notice",
            "days notice",
            "right to terminate",
            "material breach",
            "effective date of termination",
            "surviving provisions",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.CONFIDENTIALITY,
        heading_patterns=[
            r"confidential(?:ity)?",
            r"non[\-\s]?disclosure",
            r"nda",
        ],
        body_keywords=[
            "confidential",
            "confidentiality",
            "non-disclosure",
            "proprietary information",
            "trade secret",
            "disclose",
            "receiving party",
            "disclosing party",
            "confidential information",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.NON_COMPETE,
        heading_patterns=[
            r"non[\-\s]?compete",
            r"non[\-\s]?competition",
            r"restrictive\s+covenant",
        ],
        body_keywords=[
            "non-compete",
            "non-competition",
            "compete",
            "competitive",
            "restricted period",
            "restricted area",
            "competing business",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.NON_SOLICITATION,
        heading_patterns=[
            r"non[\-\s]?solicitation",
        ],
        body_keywords=[
            "non-solicitation",
            "solicit",
            "hire",
            "recruit",
            "employees",
            "customers",
            "clients",
            "personnel",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.GOVERNING_LAW,
        heading_patterns=[
            r"governing\s+law",
            r"choice\s+of\s+law",
            r"applicable\s+law",
        ],
        body_keywords=[
            "governed by",
            "governing law",
            "laws of the state",
            "laws of",
            "applicable law",
            "without regard to",
            "conflict of law",
            "choice of law",
        ],
        min_keyword_hits=1,
    ),
    _ClausePattern(
        clause_type=ClauseType.JURISDICTION,
        heading_patterns=[
            r"jurisdiction",
            r"venue",
        ],
        body_keywords=[
            "jurisdiction",
            "venue",
            "courts of",
            "submit to",
            "exclusive jurisdiction",
            "competent court",
        ],
        min_keyword_hits=1,
    ),
    _ClausePattern(
        clause_type=ClauseType.FORCE_MAJEURE,
        heading_patterns=[
            r"force\s+majeure",
        ],
        body_keywords=[
            "force majeure",
            "act of god",
            "natural disaster",
            "beyond reasonable control",
            "epidemic",
            "pandemic",
            "war",
            "terrorism",
            "flood",
            "earthquake",
            "unforeseeable",
            "unavoidable",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.PAYMENT,
        heading_patterns=[
            r"payment",
            r"fees?\s+and\s+payment",
            r"compensation",
            r"pricing",
        ],
        body_keywords=[
            "payment",
            "pay",
            "invoice",
            "net 30",
            "net 60",
            "due date",
            "late fee",
            "interest",
            "compensation",
            "fee",
            "pricing",
            "amount due",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.LIABILITY,
        heading_patterns=[
            r"limitation\s+of\s+liability",
            r"liability",
            r"cap\s+on\s+liability",
        ],
        body_keywords=[
            "limitation of liability",
            "aggregate liability",
            "shall not exceed",
            "indirect damages",
            "consequential damages",
            "incidental damages",
            "special damages",
            "punitive damages",
            "in no event",
            "cap on liability",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.IP_RIGHTS,
        heading_patterns=[
            r"intellectual\s+property",
            r"ip\s+rights",
            r"ownership\s+of\s+work",
        ],
        body_keywords=[
            "intellectual property",
            "patent",
            "copyright",
            "trademark",
            "trade secret",
            "license",
            "ownership",
            "work product",
            "proprietary rights",
            "ip rights",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.WARRANTY,
        heading_patterns=[
            r"warrant(?:y|ies)",
            r"disclaimer\s+of\s+warrant",
        ],
        body_keywords=[
            "warranty",
            "warranties",
            "warrants",
            "as is",
            "merchantability",
            "fitness for a particular purpose",
            "disclaimer",
            "no warranty",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.REPRESENTATIONS,
        heading_patterns=[
            r"representations?\s+and\s+warranties",
            r"representations",
        ],
        body_keywords=[
            "represents",
            "representation",
            "warrants",
            "represents and warrants",
            "covenants",
            "acknowledges",
            "duly authorized",
            "validly existing",
            "good standing",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.DISPUTE_RESOLUTION,
        heading_patterns=[
            r"dispute\s+resolution",
            r"arbitration",
            r"mediation",
        ],
        body_keywords=[
            "dispute",
            "arbitration",
            "mediation",
            "arbitrator",
            "dispute resolution",
            "good faith",
            "negotiate",
            "binding arbitration",
            "adr",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.ASSIGNMENT,
        heading_patterns=[
            r"assignment",
        ],
        body_keywords=[
            "assign",
            "assignment",
            "transfer",
            "delegate",
            "without prior written consent",
            "assignee",
            "successor",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.SEVERABILITY,
        heading_patterns=[
            r"severability",
        ],
        body_keywords=[
            "severability",
            "severable",
            "invalid",
            "unenforceable",
            "remaining provisions",
            "shall remain in full force",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.ENTIRE_AGREEMENT,
        heading_patterns=[
            r"entire\s+agreement",
            r"integration",
            r"merger\s+clause",
        ],
        body_keywords=[
            "entire agreement",
            "whole agreement",
            "supersedes",
            "prior agreements",
            "prior negotiations",
            "constitutes the entire",
            "oral or written",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.AMENDMENT,
        heading_patterns=[
            r"amendment",
            r"modification",
        ],
        body_keywords=[
            "amendment",
            "modify",
            "modification",
            "amended",
            "in writing",
            "signed by both parties",
            "mutual consent",
            "written agreement",
        ],
        min_keyword_hits=2,
    ),
    _ClausePattern(
        clause_type=ClauseType.NOTICE,
        heading_patterns=[
            r"notice",
            r"notices",
        ],
        body_keywords=[
            "notice",
            "written notice",
            "deliver",
            "certified mail",
            "email notification",
            "deemed received",
            "sent to",
            "address",
            "attention",
        ],
        min_keyword_hits=2,
    ),
]


class ClauseExtractor:
    """Extract and classify legal clauses from document text.

    Uses a combination of section heading detection and keyword matching
    to identify clause types. Each detected clause is scored with a
    confidence value based on how many patterns matched.

    Example::

        extractor = ClauseExtractor()
        clauses = extractor.extract(document_text)
        for clause in clauses:
            print(f"{clause.type.value}: {clause.text[:80]}...")
    """

    # Regex to split text into sections by numbered headings or ALL-CAPS lines
    _SECTION_SPLIT_RE = re.compile(
        r"""
        (?:^|\n)                            # start of text or newline
        (?:                                  # heading indicators:
            (?:\d{1,3}[\.\)]\s*)             #   "1. " or "1) "
            |(?:[A-Z][A-Z\s]{3,}[A-Z]\.?\s*\n)  #   ALL-CAPS LINE
            |(?:(?:Section|Article|Clause)\s+\d+[\.\:]?\s*)  # "Section 1:"
        )
        """,
        re.MULTILINE | re.VERBOSE,
    )

    def __init__(self, patterns: list[_ClausePattern] | None = None) -> None:
        """Initialize the clause extractor.

        Args:
            patterns: Custom clause patterns. Uses built-in library if None.
        """
        self.patterns = patterns or _CLAUSE_PATTERNS

    def extract(self, text: str, page_hint: int | None = None) -> list[Clause]:
        """Extract all detectable clauses from the given text.

        Args:
            text: Full document text (or a single page).
            page_hint: If provided, all clauses get this page number.

        Returns:
            List of Clause objects sorted by position in text.
        """
        sections = self._split_sections(text)
        clauses: list[Clause] = []
        seen_offsets: set[int] = set()

        for start, end, section_text in sections:
            if len(section_text.strip()) < 20:
                continue
            for pattern in self.patterns:
                confidence = self._score_section(section_text, pattern)
                if confidence > 0 and start not in seen_offsets:
                    clauses.append(
                        Clause(
                            type=pattern.clause_type,
                            text=section_text.strip(),
                            confidence=min(confidence, 1.0),
                            page=page_hint,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    seen_offsets.add(start)

        # Sort by position
        clauses.sort(key=lambda c: c.start_char or 0)
        return clauses

    def extract_by_type(self, text: str, clause_types: list[ClauseType]) -> list[Clause]:
        """Extract only the specified clause types.

        Args:
            text: Document text.
            clause_types: List of ClauseType values to look for.

        Returns:
            Filtered list of Clause objects.
        """
        all_clauses = self.extract(text)
        return [c for c in all_clauses if c.type in clause_types]

    def _split_sections(self, text: str) -> list[tuple[int, int, str]]:
        """Split text into (start, end, text) tuples for each logical section.

        Falls back to paragraph-based splitting when no clear section
        headings are found.
        """
        # Try splitting by section headings first
        matches = list(self._SECTION_SPLIT_RE.finditer(text))
        if len(matches) >= 2:
            sections: list[tuple[int, int, str]] = []
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((start, end, text[start:end]))
            return sections

        # Fallback: split by double-newlines (paragraph blocks)
        paragraphs: list[tuple[int, int, str]] = []
        for match in re.finditer(r"(?:(?!\n\n).)+", text, re.DOTALL):
            paragraphs.append((match.start(), match.end(), match.group()))

        # Merge very short paragraphs with the next one
        merged: list[tuple[int, int, str]] = []
        buffer_start: int | None = None
        buffer_parts: list[str] = []
        for start, end, para in paragraphs:
            if buffer_start is None:
                buffer_start = start
            buffer_parts.append(para)
            combined = "\n\n".join(buffer_parts)
            if len(combined) >= 100 or para == paragraphs[-1][2]:
                merged.append((buffer_start, end, combined))
                buffer_start = None
                buffer_parts = []

        return merged if merged else [(0, len(text), text)]

    def _score_section(self, section: str, pattern: _ClausePattern) -> float:
        """Score how likely a section matches a clause pattern.

        Returns a confidence between 0.0 (no match) and 1.0 (strong match).
        """
        lower = section.lower()
        score = 0.0

        # Check heading patterns (strong signal â€” worth 0.5)
        first_line = section.strip().split("\n")[0].lower()
        for hp in pattern.heading_patterns:
            if re.search(hp, first_line, re.IGNORECASE):
                score += 0.5
                break

        # Check body keywords
        keyword_hits = sum(1 for kw in pattern.body_keywords if kw.lower() in lower)
        keyword_ratio = keyword_hits / max(len(pattern.body_keywords), 1)

        if keyword_hits >= pattern.min_keyword_hits:
            score += 0.3 + (keyword_ratio * 0.3)

        # Bonus for strong keyword density
        if keyword_hits >= pattern.min_keyword_hits + 2:
            score += 0.1

        return score


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extract named entities from legal text using regex patterns.

    Detects parties, dates, monetary values, obligations, durations,
    and legal references without requiring spaCy or any ML model.

    Example::

        extractor = EntityExtractor()
        entities = extractor.extract("This Agreement is between Acme Corp...")
        for entity in entities:
            print(f"{entity.type.value}: {entity.text}")
    """

    # Date patterns â€” covers most legal date formats
    _DATE_PATTERNS: list[re.Pattern] = [
        # "January 1, 2024" / "Jan 1, 2024"
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\.?\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        # "1 January 2024" / "01 Jan 2024"
        re.compile(
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\.?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        # "2024-01-15" / "01/15/2024" / "15/01/2024"
        re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),
        re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    ]

    # Monetary values
    _MONEY_PATTERNS: list[re.Pattern] = [
        # "$1,000,000.00" / "USD 500" / "Â£1,000"
        re.compile(
            r"(?:[\$Â£â‚¬Â¥]|USD|GBP|EUR)\s*\d[\d,]*(?:\.\d{1,2})?"
            r"(?:\s*(?:million|billion|thousand|hundred))?",
            re.IGNORECASE,
        ),
        # "1,000 dollars" / "500 pounds"
        re.compile(
            r"\b\d[\d,]*(?:\.\d{1,2})?\s*(?:dollars?|pounds?|euros?|USD|GBP|EUR)\b",
            re.IGNORECASE,
        ),
    ]

    # Party patterns â€” detect "between X and Y" or "by and between" constructs
    _PARTY_PATTERNS: list[re.Pattern] = [
        # "by and between Acme Corp. ("Company") and John Smith ("Contractor")"
        re.compile(
            r"(?:by\s+and\s+)?between\s+"
            r"((?:[A-Z][A-Za-z&,.\'\- ]+(?:Ltd|LLC|Inc|Corp|LLP|GmbH|Pvt|PLC)?\.?))"
            r"\s*(?:\([\"\u201c]?\w+[\"\u201d]?\))?"
            r"\s+and\s+"
            r"((?:[A-Z][A-Za-z&,.\'\- ]+(?:Ltd|LLC|Inc|Corp|LLP|GmbH|Pvt|PLC)?\.?))",
            re.MULTILINE,
        ),
        # "Acme Corp. (the "Company")" or 'Acme Corp. (hereinafter "Company")'
        re.compile(
            r"([A-Z][A-Za-z&,.\'\- ]+"
            r"(?:Ltd|LLC|Inc|Corp|LLP|GmbH|Pvt|PLC)?\.?)"
            r"\s*\((?:the\s+|hereinafter\s+)?[\"\u201c](\w+)[\"\u201d]\)",
        ),
    ]

    # Duration patterns â€” "12 months", "two (2) years"
    _DURATION_PATTERN = re.compile(
        r"\b(?:\w+\s*\(\s*)?(\d+)\s*\)?\s*(days?|weeks?|months?|years?|business\s+days?)\b",
        re.IGNORECASE,
    )

    # Legal reference patterns â€” "Section 4.2", "Article III"
    _REFERENCE_PATTERN = re.compile(
        r"\b(?:Section|Article|Clause|Paragraph|Schedule|Exhibit|Appendix|Annex)"
        r"\s+[\dIVXivx]+(?:\.\d+)*(?:\([a-z]\))?",
        re.IGNORECASE,
    )

    # Obligation keywords
    _OBLIGATION_PATTERN = re.compile(
        r"(?:shall|must|agrees?\s+to|is\s+obligated\s+to|undertakes?\s+to|"
        r"covenants?\s+to|will\s+be\s+required\s+to)"
        r"\s+(.{10,120}?)(?:\.|;|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    def extract(self, text: str) -> list[Entity]:
        """Extract all entities from the given text.

        Args:
            text: Document text to analyze.

        Returns:
            List of Entity objects.
        """
        entities: list[Entity] = []
        entities.extend(self._extract_dates(text))
        entities.extend(self._extract_money(text))
        entities.extend(self._extract_parties(text))
        entities.extend(self._extract_durations(text))
        entities.extend(self._extract_references(text))
        entities.extend(self._extract_obligations(text))
        return entities

    def _extract_dates(self, text: str) -> list[Entity]:
        """Extract date entities."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for pattern in self._DATE_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group().strip()
                if raw not in seen:
                    seen.add(raw)
                    entities.append(
                        Entity(
                            type=EntityType.DATE,
                            text=raw,
                            start_char=match.start(),
                            end_char=match.end(),
                            confidence=0.9,
                        )
                    )
        return entities

    def _extract_money(self, text: str) -> list[Entity]:
        """Extract monetary value entities."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for pattern in self._MONEY_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group().strip()
                if raw not in seen:
                    seen.add(raw)
                    entities.append(
                        Entity(
                            type=EntityType.MONEY,
                            text=raw,
                            start_char=match.start(),
                            end_char=match.end(),
                            confidence=0.95,
                        )
                    )
        return entities

    def _extract_parties(self, text: str) -> list[Entity]:
        """Extract party name entities."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for pattern in self._PARTY_PATTERNS:
            for match in pattern.finditer(text):
                for group_text in match.groups():
                    if group_text and len(group_text) > 2:
                        cleaned = group_text.strip().rstrip(",. ")
                        if cleaned not in seen and len(cleaned) > 2:
                            seen.add(cleaned)
                            entities.append(
                                Entity(
                                    type=EntityType.PARTY,
                                    text=cleaned,
                                    start_char=match.start(),
                                    end_char=match.end(),
                                    confidence=0.8,
                                )
                            )
        return entities

    def _extract_durations(self, text: str) -> list[Entity]:
        """Extract duration entities (e.g., '12 months')."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for match in self._DURATION_PATTERN.finditer(text):
            raw = match.group().strip()
            if raw not in seen:
                seen.add(raw)
                entities.append(
                    Entity(
                        type=EntityType.DURATION,
                        text=raw,
                        normalized=f"{match.group(1)} {match.group(2).lower()}",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.85,
                    )
                )
        return entities

    def _extract_references(self, text: str) -> list[Entity]:
        """Extract legal reference entities (Section X, Article Y)."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for match in self._REFERENCE_PATTERN.finditer(text):
            raw = match.group().strip()
            if raw not in seen:
                seen.add(raw)
                entities.append(
                    Entity(
                        type=EntityType.REFERENCE,
                        text=raw,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.95,
                    )
                )
        return entities

    def _extract_obligations(self, text: str) -> list[Entity]:
        """Extract obligation entities (shall, must, agrees to...)."""
        entities: list[Entity] = []
        seen: set[str] = set()
        for match in self._OBLIGATION_PATTERN.finditer(text):
            full = match.group().strip()
            obligation = match.group(1).strip() if match.group(1) else full
            if obligation not in seen and len(obligation) > 15:
                seen.add(obligation)
                entities.append(
                    Entity(
                        type=EntityType.OBLIGATION,
                        text=full,
                        normalized=obligation,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.7,
                    )
                )
        return entities


# ---------------------------------------------------------------------------
# Risk Detector
# ---------------------------------------------------------------------------


@dataclass
class _RiskRule:
    """Internal rule definition for risk detection."""

    category: str
    level: RiskLevel
    description: str
    suggestion: str
    # Function that checks if this risk applies
    check: str  # method name on RiskDetector


class RiskDetector:
    """Analyze clauses and document text for legal risks.

    Uses rule-based analysis to flag missing clauses, one-sided terms,
    unlimited liability, and other common contract risks.

    Example::

        detector = RiskDetector()
        risks = detector.analyze(clauses=clauses, text=full_text)
        for risk in risks:
            print(f"[{risk.level.value}] {risk.description}")
    """

    # Clause types that should be present in most contracts
    ESSENTIAL_CLAUSES: set[ClauseType] = {
        ClauseType.TERMINATION,
        ClauseType.LIABILITY,
        ClauseType.GOVERNING_LAW,
        ClauseType.CONFIDENTIALITY,
    }

    # Clause types that are recommended but not essential
    RECOMMENDED_CLAUSES: set[ClauseType] = {
        ClauseType.INDEMNIFICATION,
        ClauseType.DISPUTE_RESOLUTION,
        ClauseType.FORCE_MAJEURE,
        ClauseType.SEVERABILITY,
        ClauseType.NOTICE,
    }

    def analyze(self, clauses: list[Clause], text: str) -> list[Risk]:
        """Run all risk detection rules against the document.

        Args:
            clauses: List of extracted Clause objects.
            text: Full document text for additional analysis.

        Returns:
            List of Risk objects, sorted by severity (high first).
        """
        risks: list[Risk] = []

        risks.extend(self._check_missing_essential_clauses(clauses))
        risks.extend(self._check_missing_recommended_clauses(clauses))
        risks.extend(self._check_unlimited_liability(clauses, text))
        risks.extend(self._check_one_sided_indemnity(clauses, text))
        risks.extend(self._check_auto_renewal(text))
        risks.extend(self._check_unilateral_termination(text))
        risks.extend(self._check_broad_ip_assignment(text))
        risks.extend(self._check_non_compete_scope(clauses, text))
        risks.extend(self._check_ambiguous_language(text))

        # Sort: HIGH first, then MEDIUM, LOW, INFO
        severity_order = {
            RiskLevel.HIGH: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.LOW: 2,
            RiskLevel.INFO: 3,
        }
        risks.sort(key=lambda r: severity_order.get(r.level, 99))

        # Annotate clause risk levels based on detected risks
        self._annotate_clauses(clauses, risks)

        return risks

    def _check_missing_essential_clauses(self, clauses: list[Clause]) -> list[Risk]:
        """Flag missing essential clauses as HIGH risk."""
        found_types = {c.type for c in clauses}
        risks: list[Risk] = []
        for clause_type in self.ESSENTIAL_CLAUSES:
            if clause_type not in found_types:
                risks.append(
                    Risk(
                        level=RiskLevel.HIGH,
                        category="missing_clause",
                        description=(
                            "Missing essential clause: "
                            f"{clause_type.value.replace('_', ' ').title()}. "
                            f"This clause is typically required in commercial contracts."
                        ),
                        suggestion=(
                            f"Add a {clause_type.value.replace('_', ' ')} clause to the agreement."
                        ),
                    )
                )
        return risks

    def _check_missing_recommended_clauses(self, clauses: list[Clause]) -> list[Risk]:
        """Flag missing recommended clauses as LOW risk."""
        found_types = {c.type for c in clauses}
        risks: list[Risk] = []
        for clause_type in self.RECOMMENDED_CLAUSES:
            if clause_type not in found_types:
                risks.append(
                    Risk(
                        level=RiskLevel.LOW,
                        category="missing_clause",
                        description=(
                            "Missing recommended clause: "
                            f"{clause_type.value.replace('_', ' ').title()}."
                        ),
                        suggestion=(
                            f"Consider adding a {clause_type.value.replace('_', ' ')} clause."
                        ),
                    )
                )
        return risks

    def _check_unlimited_liability(self, clauses: list[Clause], text: str) -> list[Risk]:
        """Check for unlimited or uncapped liability."""
        risks: list[Risk] = []
        liability_clauses = [c for c in clauses if c.type == ClauseType.LIABILITY]
        if liability_clauses:
            for clause in liability_clauses:
                clause_lower = clause.text.lower()
                # Check if there's a cap
                has_cap = any(
                    phrase in clause_lower
                    for phrase in [
                        "shall not exceed",
                        "limited to",
                        "aggregate",
                        "maximum",
                        "cap",
                    ]
                )
                if not has_cap:
                    risks.append(
                        Risk(
                            level=RiskLevel.HIGH,
                            category="unlimited_liability",
                            description="Liability clause found but no cap or limit is specified.",
                            clause=clause,
                            suggestion=(
                                "Add a specific liability cap (e.g., 'not to exceed"
                                " the total fees paid')."
                            ),
                        )
                    )
        return risks

    def _check_one_sided_indemnity(self, clauses: list[Clause], text: str) -> list[Risk]:
        """Check for one-sided indemnification provisions."""
        risks: list[Risk] = []
        indemnity_clauses = [c for c in clauses if c.type == ClauseType.INDEMNIFICATION]

        for clause in indemnity_clauses:
            lower = clause.text.lower()
            # Look for mutual indemnification
            is_mutual = any(
                phrase in lower for phrase in ["mutual", "each party", "both parties", "reciprocal"]
            )
            if not is_mutual:
                risks.append(
                    Risk(
                        level=RiskLevel.MEDIUM,
                        category="one_sided_indemnity",
                        description="Indemnification appears to be one-sided rather than mutual.",
                        clause=clause,
                        suggestion="Consider negotiating mutual indemnification provisions.",
                    )
                )
        return risks

    def _check_auto_renewal(self, text: str) -> list[Risk]:
        """Check for automatic renewal clauses that might lock parties in."""
        risks: list[Risk] = []
        lower = text.lower()

        auto_renewal_patterns = [
            r"automatically\s+renew",
            r"auto[\-\s]?renew",
            r"shall\s+renew\s+for\s+successive",
            r"renewal\s+term",
        ]

        for pattern in auto_renewal_patterns:
            if re.search(pattern, lower):
                risks.append(
                    Risk(
                        level=RiskLevel.MEDIUM,
                        category="auto_renewal",
                        description="Contract contains automatic renewal provisions.",
                        suggestion=(
                            "Ensure adequate notice period for non-renewal"
                            " and clear opt-out mechanism."
                        ),
                    )
                )
                break
        return risks

    def _check_unilateral_termination(self, text: str) -> list[Risk]:
        """Check for termination rights that favor one party."""
        risks: list[Risk] = []
        lower = text.lower()

        patterns = [
            r"(?:may|can)\s+terminate\s+(?:this\s+agreement\s+)?(?:at\s+any\s+time\s+)?(?:without\s+cause|for\s+convenience|for\s+any\s+reason)",
        ]

        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                risks.append(
                    Risk(
                        level=RiskLevel.MEDIUM,
                        category="unilateral_termination",
                        description="One party may terminate without cause or for convenience.",
                        suggestion=(
                            "Ensure termination for convenience is mutual"
                            " or negotiate adequate notice period."
                        ),
                    )
                )
                break
        return risks

    def _check_broad_ip_assignment(self, text: str) -> list[Risk]:
        """Check for overly broad IP assignment clauses."""
        risks: list[Risk] = []
        lower = text.lower()

        broad_ip_phrases = [
            "all intellectual property",
            "all rights, title, and interest",
            "work product shall belong",
            "assigns all rights",
            "hereby assigns",
        ]

        if any(phrase in lower for phrase in broad_ip_phrases):
            # Check if there's a limitation
            has_limitation = any(
                phrase in lower
                for phrase in [
                    "limited to",
                    "scope of",
                    "related to the project",
                    "arising from",
                    "in connection with",
                ]
            )
            if not has_limitation:
                risks.append(
                    Risk(
                        level=RiskLevel.HIGH,
                        category="broad_ip_assignment",
                        description="Broad IP assignment without clear scope limitation.",
                        suggestion=(
                            "Limit IP assignment to work product created"
                            " specifically under this agreement."
                        ),
                    )
                )
        return risks

    def _check_non_compete_scope(self, clauses: list[Clause], text: str) -> list[Risk]:
        """Check for overly broad non-compete clauses."""
        risks: list[Risk] = []
        nc_clauses = [c for c in clauses if c.type == ClauseType.NON_COMPETE]

        for clause in nc_clauses:
            lower = clause.text.lower()
            # Check for geographic or temporal limitations
            has_limits = any(
                phrase in lower
                for phrase in [
                    "within",
                    "mile",
                    "radius",
                    "geographic",
                    "months",
                    "years",
                    "period of",
                ]
            )
            if not has_limits:
                risks.append(
                    Risk(
                        level=RiskLevel.HIGH,
                        category="broad_non_compete",
                        description=(
                            "Non-compete clause may lack geographic or temporal boundaries."
                        ),
                        clause=clause,
                        suggestion=(
                            "Ensure non-compete has reasonable geographic"
                            " scope and time limitation."
                        ),
                    )
                )
        return risks

    def _check_ambiguous_language(self, text: str) -> list[Risk]:
        """Flag potentially ambiguous or vague language."""
        risks: list[Risk] = []
        lower = text.lower()

        ambiguous_phrases = [
            (
                "reasonable efforts",
                "Define what constitutes 'reasonable efforts' with specific criteria.",
            ),
            ("best efforts", "Define specific benchmarks for 'best efforts' obligations."),
            ("as soon as practicable", "Replace with specific timeframes."),
            ("material adverse", "Define 'material adverse' with specific thresholds."),
        ]

        found_count = 0
        for phrase, _suggestion in ambiguous_phrases:
            if phrase in lower:
                found_count += 1

        if found_count >= 2:
            risks.append(
                Risk(
                    level=RiskLevel.LOW,
                    category="ambiguous_language",
                    description=(
                        f"Document contains {found_count} potentially ambiguous phrases "
                        f"(e.g., 'reasonable efforts', 'best efforts')."
                    ),
                    suggestion=(
                        "Consider defining ambiguous terms or replacing with specific criteria."
                    ),
                )
            )
        return risks

    def _annotate_clauses(self, clauses: list[Clause], risks: list[Risk]) -> None:
        """Update clause objects with risk information from detected risks."""
        for risk in risks:
            if risk.clause:
                risk.clause.risk_level = risk.level
                risk.clause.risk_reason = risk.description
