"""Tests for clause extraction, entity extraction, and risk detection."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.extractors import ClauseExtractor, EntityExtractor, RiskDetector
from legal_doc_analyzer.models import (
    Clause,
    ClauseType,
    EntityType,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# ClauseExtractor tests
# ---------------------------------------------------------------------------

class TestClauseExtractor:
    """Tests for the ClauseExtractor."""

    @pytest.fixture
    def extractor(self) -> ClauseExtractor:
        return ClauseExtractor()

    def test_extract_from_sample_contract(
        self, extractor: ClauseExtractor, sample_contract_text: str
    ) -> None:
        """Should find multiple clauses in the sample contract."""
        clauses = extractor.extract(sample_contract_text)
        assert len(clauses) > 0
        types_found = {c.type for c in clauses}
        # The sample contract definitely has these
        assert ClauseType.TERMINATION in types_found
        assert ClauseType.CONFIDENTIALITY in types_found

    def test_extract_termination(self, extractor: ClauseExtractor) -> None:
        text = (
            "1. TERMINATION\n"
            "Either party may terminate this Agreement upon thirty (30) days "
            "written notice if the other party commits a material breach of "
            "this Agreement and fails to cure such breach."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.TERMINATION in types

    def test_extract_confidentiality(self, extractor: ClauseExtractor) -> None:
        text = (
            "CONFIDENTIALITY\n"
            "Each party shall maintain the confidentiality of all proprietary "
            "information and trade secrets. The receiving party shall not disclose "
            "confidential information to any third party without prior written consent "
            "of the disclosing party."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.CONFIDENTIALITY in types

    def test_extract_indemnification(self, extractor: ClauseExtractor) -> None:
        text = (
            "5. INDEMNIFICATION\n"
            "The Contractor shall indemnify, defend, and hold harmless the Company "
            "from and against any and all claims, damages, losses, liabilities, "
            "costs and expenses arising out of the Contractor's breach of this Agreement."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.INDEMNIFICATION in types

    def test_extract_governing_law(self, extractor: ClauseExtractor) -> None:
        text = (
            "10. GOVERNING LAW\n"
            "This Agreement shall be governed by and construed in accordance with "
            "the laws of the State of California, without regard to its conflict "
            "of law principles."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.GOVERNING_LAW in types

    def test_extract_payment(self, extractor: ClauseExtractor) -> None:
        text = (
            "3. PAYMENT\n"
            "The Client shall pay the Provider $50,000 upon execution. "
            "All invoices are due within thirty (30) days of receipt. "
            "Late payments shall accrue interest at a rate of 1.5% per month."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.PAYMENT in types

    def test_extract_force_majeure(self, extractor: ClauseExtractor) -> None:
        text = (
            "FORCE MAJEURE\n"
            "Neither party shall be liable for any failure or delay resulting "
            "from circumstances beyond reasonable control, including acts of God, "
            "natural disaster, war, terrorism, epidemic, pandemic, or flood."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.FORCE_MAJEURE in types

    def test_extract_liability(self, extractor: ClauseExtractor) -> None:
        text = (
            "6. LIMITATION OF LIABILITY\n"
            "In no event shall either party be liable for any indirect, incidental, "
            "special, consequential, or punitive damages. The aggregate liability "
            "shall not exceed the total fees paid under this Agreement."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.LIABILITY in types

    def test_extract_ip_rights(self, extractor: ClauseExtractor) -> None:
        text = (
            "INTELLECTUAL PROPERTY\n"
            "All work product shall be the sole property of the Company. "
            "The Contractor hereby assigns all copyright, patent, and trademark "
            "rights in the work product. The Company shall own all proprietary rights "
            "and intellectual property created under this Agreement."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.IP_RIGHTS in types

    def test_extract_warranty(self, extractor: ClauseExtractor) -> None:
        text = (
            "WARRANTY\n"
            "The Contractor warrants that the services shall be performed in a "
            "professional manner. All warranties of merchantability and fitness "
            "for a particular purpose are hereby disclaimed."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.WARRANTY in types

    def test_extract_severability(self, extractor: ClauseExtractor) -> None:
        text = (
            "SEVERABILITY\n"
            "If any provision of this Agreement is held to be invalid or "
            "unenforceable, the remaining provisions shall remain in full force "
            "and effect."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.SEVERABILITY in types

    def test_extract_entire_agreement(self, extractor: ClauseExtractor) -> None:
        text = (
            "ENTIRE AGREEMENT\n"
            "This Agreement constitutes the entire agreement between the parties "
            "and supersedes all prior agreements, understandings, and negotiations, "
            "whether oral or written."
        )
        clauses = extractor.extract(text)
        types = {c.type for c in clauses}
        assert ClauseType.ENTIRE_AGREEMENT in types

    def test_extract_by_type(self, extractor: ClauseExtractor, short_legal_text: str) -> None:
        clauses = extractor.extract_by_type(
            short_legal_text, [ClauseType.CONFIDENTIALITY, ClauseType.PAYMENT]
        )
        for c in clauses:
            assert c.type in {ClauseType.CONFIDENTIALITY, ClauseType.PAYMENT}

    def test_confidence_range(self, extractor: ClauseExtractor, sample_contract_text: str) -> None:
        clauses = extractor.extract(sample_contract_text)
        for clause in clauses:
            assert 0.0 < clause.confidence <= 1.0

    def test_empty_text(self, extractor: ClauseExtractor) -> None:
        clauses = extractor.extract("")
        assert clauses == []

    def test_no_clauses_in_random_text(self, extractor: ClauseExtractor, minimal_text: str) -> None:
        clauses = extractor.extract(minimal_text)
        assert len(clauses) == 0


# ---------------------------------------------------------------------------
# EntityExtractor tests
# ---------------------------------------------------------------------------

class TestEntityExtractor:
    """Tests for the EntityExtractor."""

    @pytest.fixture
    def extractor(self) -> EntityExtractor:
        return EntityExtractor()

    def test_extract_dates(self, extractor: EntityExtractor) -> None:
        text = "This Agreement is dated January 15, 2024 and expires on December 31, 2025."
        entities = extractor.extract(text)
        dates = [e for e in entities if e.type == EntityType.DATE]
        assert len(dates) >= 2
        date_texts = [d.text for d in dates]
        assert any("January 15, 2024" in t for t in date_texts)
        assert any("December 31, 2025" in t for t in date_texts)

    def test_extract_dates_iso_format(self, extractor: EntityExtractor) -> None:
        text = "Effective date: 2024-03-15. Deadline: 2024-12-31."
        entities = extractor.extract(text)
        dates = [e for e in entities if e.type == EntityType.DATE]
        assert len(dates) >= 2

    def test_extract_money_dollar(self, extractor: EntityExtractor) -> None:
        text = "The fee shall be $150,000 per year. Expenses up to $10,000 per quarter."
        entities = extractor.extract(text)
        money = [e for e in entities if e.type == EntityType.MONEY]
        assert len(money) >= 2
        money_texts = [m.text for m in money]
        assert any("150,000" in t for t in money_texts)
        assert any("10,000" in t for t in money_texts)

    def test_extract_money_other_currencies(self, extractor: EntityExtractor) -> None:
        text = "The price is £5,000 or €4,500."
        entities = extractor.extract(text)
        money = [e for e in entities if e.type == EntityType.MONEY]
        assert len(money) >= 2

    def test_extract_parties(self, extractor: EntityExtractor) -> None:
        text = 'by and between Acme Technologies Inc. ("Company") and Global Solutions Ltd. ("Contractor")'
        entities = extractor.extract(text)
        parties = [e for e in entities if e.type == EntityType.PARTY]
        assert len(parties) >= 2
        party_texts = " ".join(p.text for p in parties)
        assert "Acme" in party_texts
        assert "Global" in party_texts

    def test_extract_durations(self, extractor: EntityExtractor) -> None:
        text = "The term is twelve (12) months. Notice period: thirty (30) days."
        entities = extractor.extract(text)
        durations = [e for e in entities if e.type == EntityType.DURATION]
        assert len(durations) >= 2

    def test_extract_references(self, extractor: EntityExtractor) -> None:
        text = "As described in Section 4.2 and Article III, per Exhibit A."
        entities = extractor.extract(text)
        refs = [e for e in entities if e.type == EntityType.REFERENCE]
        assert len(refs) >= 2
        ref_texts = [r.text for r in refs]
        assert any("Section 4.2" in t for t in ref_texts)

    def test_extract_obligations(self, extractor: EntityExtractor) -> None:
        text = (
            "The Contractor shall deliver the work product within 30 days. "
            "The Company must provide access to all relevant systems."
        )
        entities = extractor.extract(text)
        obligations = [e for e in entities if e.type == EntityType.OBLIGATION]
        assert len(obligations) >= 1

    def test_from_sample_contract(
        self, extractor: EntityExtractor, sample_contract_text: str
    ) -> None:
        entities = extractor.extract(sample_contract_text)
        types_found = {e.type for e in entities}
        assert EntityType.DATE in types_found
        assert EntityType.MONEY in types_found
        assert EntityType.PARTY in types_found

    def test_empty_text(self, extractor: EntityExtractor) -> None:
        entities = extractor.extract("")
        assert entities == []


# ---------------------------------------------------------------------------
# RiskDetector tests
# ---------------------------------------------------------------------------

class TestRiskDetector:
    """Tests for the RiskDetector."""

    @pytest.fixture
    def detector(self) -> RiskDetector:
        return RiskDetector()

    def test_missing_essential_clauses(self, detector: RiskDetector) -> None:
        """Empty clause list should flag missing essential clauses."""
        risks = detector.analyze(clauses=[], text="Hello world.")
        high_risks = [r for r in risks if r.level == RiskLevel.HIGH]
        assert len(high_risks) >= 1
        assert any("missing" in r.description.lower() for r in high_risks)

    def test_no_missing_with_all_essentials(self, detector: RiskDetector) -> None:
        """Should not flag missing essentials if they're all present."""
        clauses = [
            Clause(type=ct, text=f"Clause for {ct.value}", confidence=0.8)
            for ct in RiskDetector.ESSENTIAL_CLAUSES
        ]
        risks = detector.analyze(clauses=clauses, text="Some contract text.")
        missing_essential = [
            r for r in risks
            if r.level == RiskLevel.HIGH and r.category == "missing_clause"
        ]
        assert len(missing_essential) == 0

    def test_unlimited_liability_flagged(self, detector: RiskDetector) -> None:
        """Liability clause without a cap should trigger a risk."""
        clauses = [
            Clause(
                type=ClauseType.LIABILITY,
                text="Each party is liable for damages arising under this Agreement.",
                confidence=0.8,
            )
        ]
        risks = detector.analyze(clauses=clauses, text="Some text.")
        liability_risks = [r for r in risks if r.category == "unlimited_liability"]
        assert len(liability_risks) >= 1
        assert liability_risks[0].level == RiskLevel.HIGH

    def test_capped_liability_no_flag(self, detector: RiskDetector) -> None:
        """Liability clause with a cap should not trigger unlimited liability risk."""
        clauses = [
            Clause(
                type=ClauseType.LIABILITY,
                text="Aggregate liability shall not exceed the total fees paid.",
                confidence=0.8,
            )
        ]
        risks = detector.analyze(clauses=clauses, text="Some text.")
        liability_risks = [r for r in risks if r.category == "unlimited_liability"]
        assert len(liability_risks) == 0

    def test_one_sided_indemnity(self, detector: RiskDetector) -> None:
        clauses = [
            Clause(
                type=ClauseType.INDEMNIFICATION,
                text="The Contractor shall indemnify and hold harmless the Company from all claims and damages.",
                confidence=0.8,
            )
        ]
        risks = detector.analyze(clauses=clauses, text="Some text.")
        indemnity_risks = [r for r in risks if r.category == "one_sided_indemnity"]
        assert len(indemnity_risks) >= 1

    def test_mutual_indemnity_no_flag(self, detector: RiskDetector) -> None:
        clauses = [
            Clause(
                type=ClauseType.INDEMNIFICATION,
                text="Each party shall indemnify and hold harmless the other party from all claims.",
                confidence=0.8,
            )
        ]
        risks = detector.analyze(clauses=clauses, text="Some text.")
        indemnity_risks = [r for r in risks if r.category == "one_sided_indemnity"]
        assert len(indemnity_risks) == 0

    def test_auto_renewal_detected(self, detector: RiskDetector) -> None:
        text = "This Agreement shall automatically renew for successive one-year periods."
        risks = detector.analyze(clauses=[], text=text)
        auto_risks = [r for r in risks if r.category == "auto_renewal"]
        assert len(auto_risks) >= 1

    def test_risks_sorted_by_severity(self, detector: RiskDetector) -> None:
        """Risks should be sorted HIGH first."""
        risks = detector.analyze(clauses=[], text="Some text with automatically renew clause.")
        if len(risks) >= 2:
            severity = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2, RiskLevel.INFO: 3}
            for i in range(len(risks) - 1):
                assert severity[risks[i].level] <= severity[risks[i + 1].level]

    def test_full_sample_contract(
        self, detector: RiskDetector, sample_contract_text: str
    ) -> None:
        """Run risk detection on the full sample contract."""
        clause_extractor = ClauseExtractor()
        clauses = clause_extractor.extract(sample_contract_text)
        risks = detector.analyze(clauses=clauses, text=sample_contract_text)
        # Should find some risks (e.g., one-sided indemnity, auto-renewal, unilateral termination)
        assert len(risks) > 0
        # Every risk should have a description
        for risk in risks:
            assert risk.description
            assert risk.level in RiskLevel
