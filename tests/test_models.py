"""Tests for data models — Clause, Entity, Risk, AnalysisResult."""

from __future__ import annotations

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
# Clause tests
# ---------------------------------------------------------------------------


class TestClause:
    """Tests for the Clause dataclass."""

    def test_create_clause(self) -> None:
        clause = Clause(
            type=ClauseType.TERMINATION,
            text="Either party may terminate this agreement.",
            confidence=0.85,
            page=2,
            paragraph=5,
        )
        assert clause.type == ClauseType.TERMINATION
        assert clause.confidence == 0.85
        assert clause.page == 2
        assert clause.paragraph == 5

    def test_default_risk_is_info(self) -> None:
        clause = Clause(type=ClauseType.PAYMENT, text="Payment terms.", confidence=0.7)
        assert clause.risk_level == RiskLevel.INFO
        assert clause.is_risky is False

    def test_is_risky_high(self) -> None:
        clause = Clause(
            type=ClauseType.LIABILITY,
            text="Unlimited liability.",
            confidence=0.9,
            risk_level=RiskLevel.HIGH,
        )
        assert clause.is_risky is True

    def test_is_risky_medium(self) -> None:
        clause = Clause(
            type=ClauseType.INDEMNIFICATION,
            text="One-sided indemnity.",
            confidence=0.8,
            risk_level=RiskLevel.MEDIUM,
        )
        assert clause.is_risky is True

    def test_is_risky_low(self) -> None:
        clause = Clause(
            type=ClauseType.SEVERABILITY,
            text="Severability.",
            confidence=0.9,
            risk_level=RiskLevel.LOW,
        )
        assert clause.is_risky is False

    def test_to_dict(self) -> None:
        clause = Clause(
            type=ClauseType.CONFIDENTIALITY,
            text="Keep it secret.",
            confidence=0.9123,
            page=1,
            risk_level=RiskLevel.LOW,
            risk_reason="Standard clause",
        )
        d = clause.to_dict()
        assert d["type"] == "confidentiality"
        assert d["confidence"] == 0.912
        assert d["page"] == 1
        assert d["risk_level"] == "low"
        assert d["risk_reason"] == "Standard clause"

    def test_metadata_default(self) -> None:
        clause = Clause(type=ClauseType.OTHER, text="text", confidence=0.5)
        assert clause.metadata == {}

    def test_all_clause_types_exist(self) -> None:
        """Ensure the enum has at least 15 types as advertised."""
        assert len(ClauseType) >= 15


# ---------------------------------------------------------------------------
# Entity tests
# ---------------------------------------------------------------------------


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_create_entity(self) -> None:
        entity = Entity(type=EntityType.PARTY, text="Acme Corp.")
        assert entity.type == EntityType.PARTY
        assert entity.text == "Acme Corp."
        assert entity.confidence == 1.0

    def test_entity_with_normalized(self) -> None:
        entity = Entity(
            type=EntityType.DATE,
            text="January 1, 2024",
            normalized="2024-01-01",
            confidence=0.9,
        )
        assert entity.normalized == "2024-01-01"

    def test_to_dict(self) -> None:
        entity = Entity(
            type=EntityType.MONEY,
            text="$50,000",
            normalized="50000 USD",
            confidence=0.95,
        )
        d = entity.to_dict()
        assert d["type"] == "money"
        assert d["text"] == "$50,000"
        assert d["confidence"] == 0.95

    def test_all_entity_types_exist(self) -> None:
        expected = {
            "party",
            "date",
            "money",
            "obligation",
            "legal_reference",
            "jurisdiction",
            "duration",
        }
        actual = {e.value for e in EntityType}
        assert expected.issubset(actual)


# ---------------------------------------------------------------------------
# Risk tests
# ---------------------------------------------------------------------------


class TestRisk:
    """Tests for the Risk dataclass."""

    def test_create_risk(self) -> None:
        risk = Risk(
            level=RiskLevel.HIGH,
            category="missing_clause",
            description="Missing termination clause.",
        )
        assert risk.level == RiskLevel.HIGH
        assert risk.category == "missing_clause"
        assert risk.clause is None

    def test_risk_with_clause(self) -> None:
        clause = Clause(type=ClauseType.LIABILITY, text="Unlimited.", confidence=0.9)
        risk = Risk(
            level=RiskLevel.HIGH,
            category="unlimited_liability",
            description="No cap.",
            clause=clause,
            suggestion="Add a cap.",
        )
        assert risk.clause is clause
        assert risk.suggestion == "Add a cap."

    def test_to_dict(self) -> None:
        risk = Risk(
            level=RiskLevel.MEDIUM,
            category="one_sided",
            description="One-sided indemnity.",
            suggestion="Make it mutual.",
        )
        d = risk.to_dict()
        assert d["level"] == "medium"
        assert d["suggestion"] == "Make it mutual."


# ---------------------------------------------------------------------------
# AnalysisResult tests
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass."""

    def _make_result(self, **kwargs) -> AnalysisResult:
        defaults = {
            "filename": "test.pdf",
            "summary": "Test summary",
            "clauses": [],
            "entities": [],
            "risks": [],
        }
        defaults.update(kwargs)
        return AnalysisResult(**defaults)

    def test_empty_result(self) -> None:
        result = self._make_result()
        assert result.risk_score == 0.0
        assert result.high_risks == []
        assert result.clause_types_found == set()

    def test_risk_score_with_high_risk(self) -> None:
        risks = [
            Risk(level=RiskLevel.HIGH, category="a", description="bad"),
            Risk(level=RiskLevel.LOW, category="b", description="minor"),
        ]
        result = self._make_result(risks=risks)
        assert result.risk_score > 0.0
        assert result.risk_score <= 1.0

    def test_risk_score_capped_at_one(self) -> None:
        """Many high risks should still cap at 1.0."""
        risks = [
            Risk(level=RiskLevel.HIGH, category=f"cat{i}", description=f"risk {i}")
            for i in range(10)
        ]
        result = self._make_result(risks=risks)
        assert result.risk_score == 1.0

    def test_high_risks_filter(self) -> None:
        risks = [
            Risk(level=RiskLevel.HIGH, category="a", description="a"),
            Risk(level=RiskLevel.MEDIUM, category="b", description="b"),
            Risk(level=RiskLevel.HIGH, category="c", description="c"),
        ]
        result = self._make_result(risks=risks)
        assert len(result.high_risks) == 2

    def test_clause_types_found(self) -> None:
        clauses = [
            Clause(type=ClauseType.TERMINATION, text="t", confidence=0.8),
            Clause(type=ClauseType.PAYMENT, text="p", confidence=0.7),
            Clause(type=ClauseType.TERMINATION, text="t2", confidence=0.6),
        ]
        result = self._make_result(clauses=clauses)
        assert result.clause_types_found == {ClauseType.TERMINATION, ClauseType.PAYMENT}

    def test_to_dict(self) -> None:
        result = self._make_result(
            clauses=[Clause(type=ClauseType.PAYMENT, text="pay", confidence=0.9)],
            entities=[Entity(type=EntityType.MONEY, text="$100")],
            risks=[Risk(level=RiskLevel.LOW, category="x", description="y")],
        )
        d = result.to_dict()
        assert d["filename"] == "test.pdf"
        assert len(d["clauses"]) == 1
        assert len(d["entities"]) == 1
        assert len(d["risks"]) == 1
        assert "risk_score" in d
