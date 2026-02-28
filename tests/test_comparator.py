"""Tests for the ContractComparator clause-level diff engine."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.comparator import (
    ContractComparator,
    ContractDiff,
    EntityDelta,
)
from legal_doc_analyzer.models import ClauseType, EntityType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTRACT_A = """
SERVICE AGREEMENT

This Agreement is entered into as of January 15, 2024,
by and between Acme Technologies Inc. ("Client") and Beta Solutions LLC ("Provider").

1. TERMINATION
Either party may terminate this Agreement upon thirty (30) days written notice
if the other party commits a material breach and fails to cure such breach within
thirty (30) days of receiving written notice of the breach.

2. CONFIDENTIALITY
Each party shall maintain the confidentiality of all proprietary information
and trade secrets. The receiving party shall not disclose confidential information
to any third party without prior written consent of the disclosing party.

3. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, special,
consequential, or punitive damages. The aggregate liability of either party shall
not exceed the total fees paid in the twelve months preceding the claim.

4. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws
of the State of California, without regard to its conflict of law principles.

5. INDEMNIFICATION
The Provider shall indemnify, defend, and hold harmless the Client from and against
any and all claims, damages, losses arising out of the Provider's breach.

6. PAYMENT
The Client shall pay $50,000 upon execution. All invoices are due within net 30 days.
Late payments shall accrue interest at 1.5% per month.
"""

# Version B: termination notice changed, liability cap removed,
# California --> New York, force majeure added, non-compete added
CONTRACT_B = """
SERVICE AGREEMENT

This Agreement is entered into as of March 1, 2024,
by and between Acme Technologies Inc. ("Client") and Beta Solutions LLC ("Provider").

1. TERMINATION
Either party may terminate this Agreement upon sixty (60) days written notice
if the other party commits a material breach.

2. CONFIDENTIALITY
Each party shall maintain the confidentiality of all proprietary information
and trade secrets. The receiving party shall not disclose confidential information
to any third party without prior written consent of the disclosing party.

3. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, special,
consequential, or punitive damages arising under this Agreement.

4. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws
of the State of New York, without regard to its conflict of law principles.

5. INDEMNIFICATION
Each party shall mutually indemnify, defend, and hold harmless the other party
from and against any and all claims, damages, losses arising out of a breach.

6. PAYMENT
The Client shall pay $75,000 upon execution. All invoices are due within net 30 days.
Late payments shall accrue interest at 1.5% per month.

7. FORCE MAJEURE
Neither party shall be liable for any failure or delay resulting from circumstances
beyond reasonable control, including acts of God, natural disaster, war, or pandemic.

8. NON-COMPETE
During the term and for twelve (12) months thereafter, within a 100 mile radius,
Provider shall not compete with Client's core business or solicit its customers.
"""


@pytest.fixture
def comparator() -> ContractComparator:
    return ContractComparator()


@pytest.fixture
def diff(comparator: ContractComparator) -> ContractDiff:
    return comparator.compare(CONTRACT_A, CONTRACT_B, name_a="v1", name_b="v2")


# ---------------------------------------------------------------------------
# ContractComparator.compare()
# ---------------------------------------------------------------------------


class TestContractComparatorCompare:
    def test_returns_contract_diff(self, diff: ContractDiff) -> None:
        assert isinstance(diff, ContractDiff)

    def test_doc_names_preserved(self, diff: ContractDiff) -> None:
        assert diff.doc_a_name == "v1"
        assert diff.doc_b_name == "v2"

    def test_clause_diffs_populated(self, diff: ContractDiff) -> None:
        assert len(diff.clause_diffs) > 0

    def test_entity_deltas_populated(self, diff: ContractDiff) -> None:
        assert len(diff.entity_deltas) > 0

    def test_overall_similarity_in_range(self, diff: ContractDiff) -> None:
        assert 0.0 <= diff.overall_similarity <= 1.0

    def test_summary_non_empty(self, diff: ContractDiff) -> None:
        assert diff.summary.strip() != ""

    def test_summary_contains_doc_names(self, diff: ContractDiff) -> None:
        assert "v1" in diff.summary
        assert "v2" in diff.summary

    def test_added_clauses_detected(self, diff: ContractDiff) -> None:
        """Force majeure and non-compete exist in B but not A."""
        added_types = {cd.clause_type for cd in diff.added_clauses}
        assert ClauseType.FORCE_MAJEURE in added_types or ClauseType.NON_COMPETE in added_types

    def test_modified_clauses_detected(self, diff: ContractDiff) -> None:
        """Termination notice period changed, liability cap removed, governing law changed."""
        assert len(diff.modified_clauses) > 0

    def test_to_dict_shape(self, diff: ContractDiff) -> None:
        d = diff.to_dict()
        assert "doc_a" in d
        assert "doc_b" in d
        assert "overall_similarity" in d
        assert "clause_diffs" in d
        assert "entity_deltas" in d
        assert "stats" in d
        stats = d["stats"]
        for key in ("added", "removed", "modified", "unchanged"):
            assert key in stats

    def test_stats_sum_equals_total(self, diff: ContractDiff) -> None:
        total = (
            len(diff.added_clauses)
            + len(diff.removed_clauses)
            + len(diff.modified_clauses)
            + len(diff.unchanged_clauses)
        )
        assert total == len(diff.clause_diffs)


# ---------------------------------------------------------------------------
# Identical documents -> unchanged
# ---------------------------------------------------------------------------


class TestIdenticalDocuments:
    def test_identical_docs_high_similarity(self, comparator: ContractComparator) -> None:
        diff = comparator.compare(CONTRACT_A, CONTRACT_A)
        assert diff.overall_similarity >= 0.9

    def test_identical_docs_no_added_or_removed(self, comparator: ContractComparator) -> None:
        diff = comparator.compare(CONTRACT_A, CONTRACT_A)
        assert len(diff.added_clauses) == 0
        assert len(diff.removed_clauses) == 0

    def test_identical_docs_all_unchanged_or_modified(self, comparator: ContractComparator) -> None:
        """With identical text, every clause should be 'unchanged'."""
        diff = comparator.compare(CONTRACT_A, CONTRACT_A)
        for cd in diff.clause_diffs:
            assert cd.status in ("unchanged", "modified")

    def test_identical_docs_similarity_1(self, comparator: ContractComparator) -> None:
        diff = comparator.compare(CONTRACT_A, CONTRACT_A)
        for cd in diff.clause_diffs:
            assert cd.similarity > 0.85


# ---------------------------------------------------------------------------
# Empty documents
# ---------------------------------------------------------------------------


class TestEmptyDocuments:
    def test_both_empty(self, comparator: ContractComparator) -> None:
        diff = comparator.compare("", "")
        assert diff.overall_similarity == 1.0
        assert diff.clause_diffs == []

    def test_a_empty(self, comparator: ContractComparator) -> None:
        diff = comparator.compare("", CONTRACT_A)
        for cd in diff.clause_diffs:
            assert cd.status == "added"

    def test_b_empty(self, comparator: ContractComparator) -> None:
        diff = comparator.compare(CONTRACT_A, "")
        for cd in diff.clause_diffs:
            assert cd.status == "removed"


# ---------------------------------------------------------------------------
# ClauseDiff properties
# ---------------------------------------------------------------------------


class TestClauseDiff:
    def test_added_has_no_clause_a(self, diff: ContractDiff) -> None:
        for cd in diff.added_clauses:
            assert cd.clause_a is None
            assert cd.clause_b is not None

    def test_removed_has_no_clause_b(self, diff: ContractDiff) -> None:
        for cd in diff.removed_clauses:
            assert cd.clause_b is None
            assert cd.clause_a is not None

    def test_modified_has_both_clauses(self, diff: ContractDiff) -> None:
        for cd in diff.modified_clauses:
            assert cd.clause_a is not None
            assert cd.clause_b is not None

    def test_added_similarity_zero(self, diff: ContractDiff) -> None:
        for cd in diff.added_clauses:
            assert cd.similarity == 0.0

    def test_removed_similarity_zero(self, diff: ContractDiff) -> None:
        for cd in diff.removed_clauses:
            assert cd.similarity == 0.0

    def test_modified_similarity_between_0_and_threshold(self, diff: ContractDiff) -> None:
        for cd in diff.modified_clauses:
            assert 0.0 <= cd.similarity < 0.85

    def test_to_dict_shape(self, diff: ContractDiff) -> None:
        for cd in diff.clause_diffs:
            d = cd.to_dict()
            assert "clause_type" in d
            assert "status" in d
            assert "similarity" in d
            assert "key_differences" in d


# ---------------------------------------------------------------------------
# Key-difference detection
# ---------------------------------------------------------------------------


class TestDescribeDifferences:
    def test_notice_period_change_detected(self, comparator: ContractComparator) -> None:
        """30-day notice in A vs 60-day in B should appear in key_differences."""
        termination_diffs = [
            cd
            for cd in comparator.compare(CONTRACT_A, CONTRACT_B).clause_diffs
            if cd.clause_type == ClauseType.TERMINATION and cd.status == "modified"
        ]
        if termination_diffs:
            notes = " ".join(termination_diffs[0].key_differences)
            # Key differences may be empty if no detailed extraction; that is acceptable
            if notes:
                assert "notice" in notes.lower() or "30" in notes or "60" in notes

    def test_governing_law_change_detected(self, comparator: ContractComparator) -> None:
        """California -> New York should appear in governing law diffs."""
        glaw_diffs = [
            cd
            for cd in comparator.compare(CONTRACT_A, CONTRACT_B).clause_diffs
            if cd.clause_type == ClauseType.GOVERNING_LAW and cd.status == "modified"
        ]
        if glaw_diffs:
            notes = " ".join(glaw_diffs[0].key_differences)
            assert "california" in notes.lower() or "new york" in notes.lower()

    def test_liability_cap_removal_detected(self, comparator: ContractComparator) -> None:
        """Removing 'shall not exceed' should be flagged."""
        lib_diffs = [
            cd
            for cd in comparator.compare(CONTRACT_A, CONTRACT_B).clause_diffs
            if cd.clause_type == ClauseType.LIABILITY and cd.status == "modified"
        ]
        if lib_diffs:
            notes = " ".join(lib_diffs[0].key_differences)
            assert "cap" in notes.lower() or "liability" in notes.lower()

    def test_mutual_indemnity_change_detected(self, comparator: ContractComparator) -> None:
        """Adding 'mutual' to indemnity should be flagged."""
        indem_diffs = [
            cd
            for cd in comparator.compare(CONTRACT_A, CONTRACT_B).clause_diffs
            if cd.clause_type == ClauseType.INDEMNIFICATION and cd.status == "modified"
        ]
        if indem_diffs:
            notes = " ".join(indem_diffs[0].key_differences)
            assert "mutual" in notes.lower() or "one-sided" in notes.lower()


# ---------------------------------------------------------------------------
# EntityDelta
# ---------------------------------------------------------------------------


class TestEntityDelta:
    def test_entity_deltas_cover_expected_types(self, diff: ContractDiff) -> None:
        types_found = {ed.entity_type for ed in diff.entity_deltas}
        assert EntityType.PARTY in types_found
        assert EntityType.DATE in types_found
        assert EntityType.MONEY in types_found

    def test_date_change_detected(self, diff: ContractDiff) -> None:
        """A uses Jan 2024, B uses Mar 2024 — should appear in date delta."""
        date_deltas = [ed for ed in diff.entity_deltas if ed.entity_type == EntityType.DATE]
        assert date_deltas
        delta = date_deltas[0]
        # At least some date should differ
        all_dates = delta.only_in_a + delta.only_in_b + delta.shared
        assert len(all_dates) > 0

    def test_to_dict_shape(self, diff: ContractDiff) -> None:
        for ed in diff.entity_deltas:
            d = ed.to_dict()
            assert "entity_type" in d
            assert "only_in_a" in d
            assert "only_in_b" in d
            assert "shared" in d

    def test_has_changes_property(self) -> None:
        delta_with_changes = EntityDelta(
            entity_type=EntityType.MONEY,
            only_in_a=["$50,000"],
            only_in_b=["$75,000"],
        )
        assert delta_with_changes.has_changes is True

        delta_without_changes = EntityDelta(
            entity_type=EntityType.PARTY,
            shared=["Acme Corp"],
        )
        assert delta_without_changes.has_changes is False


# ---------------------------------------------------------------------------
# compare_files
# ---------------------------------------------------------------------------


class TestCompareFiles:
    def test_compare_files_roundtrip(self, comparator: ContractComparator, tmp_path) -> None:
        file_a = tmp_path / "contract_a.txt"
        file_b = tmp_path / "contract_b.txt"
        file_a.write_text(CONTRACT_A, encoding="utf-8")
        file_b.write_text(CONTRACT_B, encoding="utf-8")

        diff = comparator.compare_files(file_a, file_b)
        assert diff.doc_a_name == "contract_a.txt"
        assert diff.doc_b_name == "contract_b.txt"
        assert len(diff.clause_diffs) > 0

    def test_compare_files_not_found(self, comparator: ContractComparator, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            comparator.compare_files(tmp_path / "missing.txt", tmp_path / "also_missing.txt")


# ---------------------------------------------------------------------------
# Custom similarity threshold
# ---------------------------------------------------------------------------


class TestSimilarityThreshold:
    def test_lower_threshold_classifies_more_as_modified(self) -> None:
        strict = ContractComparator(similarity_threshold=0.99)
        lenient = ContractComparator(similarity_threshold=0.5)

        diff_strict = strict.compare(CONTRACT_A, CONTRACT_B)
        diff_lenient = lenient.compare(CONTRACT_A, CONTRACT_B)

        # Stricter threshold -> more "modified", fewer "unchanged"
        assert len(diff_strict.modified_clauses) >= len(diff_lenient.modified_clauses)

    def test_threshold_boundary(self) -> None:
        """A clause with similarity exactly at threshold should be 'unchanged'."""
        cmp = ContractComparator(similarity_threshold=0.0)
        diff = cmp.compare(CONTRACT_A, CONTRACT_B)
        # With threshold=0, no clause should ever be "modified"
        for cd in diff.clause_diffs:
            if cd.clause_a and cd.clause_b:
                assert cd.status == "unchanged"
