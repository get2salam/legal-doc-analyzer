"""Data models for legal document analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClauseType(str, Enum):
    """Standard contract clause types."""

    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    NON_COMPETE = "non_compete"
    NON_SOLICITATION = "non_solicitation"
    GOVERNING_LAW = "governing_law"
    JURISDICTION = "jurisdiction"
    FORCE_MAJEURE = "force_majeure"
    PAYMENT = "payment"
    LIABILITY = "liability"
    IP_RIGHTS = "intellectual_property"
    WARRANTY = "warranty"
    REPRESENTATIONS = "representations"
    DISPUTE_RESOLUTION = "dispute_resolution"
    ASSIGNMENT = "assignment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    AMENDMENT = "amendment"
    NOTICE = "notice"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk severity levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EntityType(str, Enum):
    """Legal entity types."""

    PARTY = "party"
    DATE = "date"
    MONEY = "money"
    OBLIGATION = "obligation"
    REFERENCE = "legal_reference"
    JURISDICTION = "jurisdiction"
    DURATION = "duration"


@dataclass
class Clause:
    """A single extracted clause from a legal document."""

    type: ClauseType
    text: str
    confidence: float
    page: Optional[int] = None
    paragraph: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    risk_level: RiskLevel = RiskLevel.INFO
    risk_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_risky(self) -> bool:
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "page": self.page,
            "paragraph": self.paragraph,
            "risk_level": self.risk_level.value,
            "risk_reason": self.risk_reason,
        }


@dataclass
class Entity:
    """An extracted entity from a legal document."""

    type: EntityType
    text: str
    normalized: Optional[str] = None
    confidence: float = 1.0
    page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "text": self.text,
            "normalized": self.normalized,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Risk:
    """A detected risk in the document."""

    level: RiskLevel
    category: str
    description: str
    clause: Optional[Clause] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "category": self.category,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result for a document."""

    filename: str
    summary: str
    clauses: list[Clause] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw_text: str = ""

    @property
    def risk_score(self) -> float:
        """Overall risk score (0-1)."""
        if not self.risks:
            return 0.0
        weights = {RiskLevel.HIGH: 1.0, RiskLevel.MEDIUM: 0.5, RiskLevel.LOW: 0.2, RiskLevel.INFO: 0.0}
        total = sum(weights.get(r.level, 0) for r in self.risks)
        return min(1.0, total / max(len(self.risks), 1))

    @property
    def high_risks(self) -> list[Risk]:
        return [r for r in self.risks if r.level == RiskLevel.HIGH]

    @property
    def clause_types_found(self) -> set[ClauseType]:
        return {c.type for c in self.clauses}

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "summary": self.summary,
            "risk_score": round(self.risk_score, 3),
            "clauses": [c.to_dict() for c in self.clauses],
            "entities": [e.to_dict() for e in self.entities],
            "risks": [r.to_dict() for r in self.risks],
            "metadata": self.metadata,
        }
