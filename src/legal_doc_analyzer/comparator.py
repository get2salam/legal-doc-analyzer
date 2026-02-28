"""Contract comparison engine for legal document diffing.

Provides clause-level comparison between two legal documents, identifying
added, removed, and modified clauses, entity changes (parties, dates,
amounts), and an overall structural similarity score.

Typical usage::

    comparator = ContractComparator()
    diff = comparator.compare(text_v1, text_v2, name_a="v1.txt", name_b="v2.txt")

    print(f"Overall similarity: {diff.overall_similarity:.0%}")
    for cd in diff.clause_diffs:
        print(f"  [{cd.status}] {cd.clause_type.value}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .extractors import ClauseExtractor, EntityExtractor
from .models import Clause, ClauseType, Entity, EntityType

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class ClauseDiff:
    """Comparison result for a single clause type across two documents.

    Attributes:
        clause_type: The type of clause being compared.
        status: One of ``"added"``, ``"removed"``, ``"modified"``, or
            ``"unchanged"``.
        clause_a: The clause instance from document A (None if absent).
        clause_b: The clause instance from document B (None if absent).
        similarity: Jaccard word-overlap similarity for ``"modified"``
            clauses (0.0 -- 1.0).  1.0 for ``"unchanged"``, 0.0 for
            ``"added"`` or ``"removed"``.
        key_differences: Human-readable notes highlighting observed
            differences between the two clause texts.
    """

    clause_type: ClauseType
    status: Literal["added", "removed", "modified", "unchanged"]
    clause_a: Clause | None
    clause_b: Clause | None
    similarity: float = 0.0
    key_differences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clause_type": self.clause_type.value,
            "status": self.status,
            "similarity": round(self.similarity, 4),
            "key_differences": self.key_differences,
            "text_a": self.clause_a.text[:200] if self.clause_a else None,
            "text_b": self.clause_b.text[:200] if self.clause_b else None,
        }


@dataclass
class EntityDelta:
    """Change in a specific entity type between two documents.

    Attributes:
        entity_type: The kind of entity (PARTY, DATE, MONEY, ...).
        only_in_a: Entity texts present in doc A but not doc B.
        only_in_b: Entity texts present in doc B but not doc A.
        shared: Entity texts present in both documents.
    """

    entity_type: EntityType
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    shared: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.only_in_a or self.only_in_b)

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type.value,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "shared": self.shared,
        }


@dataclass
class ContractDiff:
    """Full comparison result between two legal documents.

    Attributes:
        doc_a_name: Label for the first document.
        doc_b_name: Label for the second document.
        clause_diffs: Per-clause comparison results.
        entity_deltas: Per-entity-type comparison results.
        overall_similarity: Weighted clause-level similarity (0.0 -- 1.0).
        risk_delta: Change in total risk count (positive = more risks in B).
        summary: Human-readable summary of major differences.
    """

    doc_a_name: str
    doc_b_name: str
    clause_diffs: list[ClauseDiff] = field(default_factory=list)
    entity_deltas: list[EntityDelta] = field(default_factory=list)
    overall_similarity: float = 0.0
    risk_delta: int = 0
    summary: str = ""

    @property
    def added_clauses(self) -> list[ClauseDiff]:
        """Clauses present in B but absent in A."""
        return [cd for cd in self.clause_diffs if cd.status == "added"]

    @property
    def removed_clauses(self) -> list[ClauseDiff]:
        """Clauses present in A but absent in B."""
        return [cd for cd in self.clause_diffs if cd.status == "removed"]

    @property
    def modified_clauses(self) -> list[ClauseDiff]:
        """Clauses present in both but with different language."""
        return [cd for cd in self.clause_diffs if cd.status == "modified"]

    @property
    def unchanged_clauses(self) -> list[ClauseDiff]:
        """Clauses with effectively identical language in both documents."""
        return [cd for cd in self.clause_diffs if cd.status == "unchanged"]

    def to_dict(self) -> dict:
        return {
            "doc_a": self.doc_a_name,
            "doc_b": self.doc_b_name,
            "overall_similarity": round(self.overall_similarity, 4),
            "risk_delta": self.risk_delta,
            "summary": self.summary,
            "clause_diffs": [cd.to_dict() for cd in self.clause_diffs],
            "entity_deltas": [ed.to_dict() for ed in self.entity_deltas],
            "stats": {
                "added": len(self.added_clauses),
                "removed": len(self.removed_clauses),
                "modified": len(self.modified_clauses),
                "unchanged": len(self.unchanged_clauses),
            },
        }


# ---------------------------------------------------------------------------
# Core comparator
# ---------------------------------------------------------------------------


class ContractComparator:
    """Compare two legal contracts at the clause and entity level.

    Extracts clauses and entities from both documents, then aligns them
    by type to produce a structured diff.  No ML models required -- uses
    keyword-based extraction and word-overlap similarity.

    Args:
        clause_extractor: Custom extractor (uses default if ``None``).
        entity_extractor: Custom extractor (uses default if ``None``).
        similarity_threshold: Clause similarity at or above this value is
            reported as ``"unchanged"``; below is ``"modified"``.
            Default ``0.85``.

    Example::

        cmp = ContractComparator()
        diff = cmp.compare(text_a, text_b)
        print(diff.summary)
    """

    def __init__(
        self,
        clause_extractor: ClauseExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._clause_ext = clause_extractor or ClauseExtractor()
        self._entity_ext = entity_extractor or EntityExtractor()
        self._sim_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        text_a: str,
        text_b: str,
        name_a: str = "Document A",
        name_b: str = "Document B",
    ) -> ContractDiff:
        """Compare two document texts and return a structured diff.

        Args:
            text_a: Full text of the first document.
            text_b: Full text of the second document.
            name_a: Human-readable label for document A.
            name_b: Human-readable label for document B.

        Returns:
            :class:`ContractDiff` with clause and entity comparison results.
        """
        clauses_a = self._clause_ext.extract(text_a)
        clauses_b = self._clause_ext.extract(text_b)
        entities_a = self._entity_ext.extract(text_a)
        entities_b = self._entity_ext.extract(text_b)

        clause_diffs = self._diff_clauses(clauses_a, clauses_b)
        entity_deltas = self._diff_entities(entities_a, entities_b)
        overall_sim = self._compute_overall_similarity(clause_diffs)
        summary = self._generate_summary(name_a, name_b, clause_diffs, entity_deltas, overall_sim)

        return ContractDiff(
            doc_a_name=name_a,
            doc_b_name=name_b,
            clause_diffs=clause_diffs,
            entity_deltas=entity_deltas,
            overall_similarity=overall_sim,
            summary=summary,
        )

    def compare_files(
        self,
        path_a: str | Path,
        path_b: str | Path,
    ) -> ContractDiff:
        """Compare two plain-text contract files.

        Args:
            path_a: Path to the first document (UTF-8 text file).
            path_b: Path to the second document (UTF-8 text file).

        Returns:
            :class:`ContractDiff` with comparison results.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If a file cannot be read as UTF-8 text.
        """
        p_a = Path(path_a)
        p_b = Path(path_b)

        if not p_a.exists():
            raise FileNotFoundError(f"File not found: {p_a}")
        if not p_b.exists():
            raise FileNotFoundError(f"File not found: {p_b}")

        try:
            text_a = p_a.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot read {p_a} as UTF-8: {exc}") from exc

        try:
            text_b = p_b.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot read {p_b} as UTF-8: {exc}") from exc

        return self.compare(text_a, text_b, name_a=p_a.name, name_b=p_b.name)

    # ------------------------------------------------------------------
    # Clause diffing
    # ------------------------------------------------------------------

    def _diff_clauses(
        self,
        clauses_a: list[Clause],
        clauses_b: list[Clause],
    ) -> list[ClauseDiff]:
        """Align clauses by type and compute per-type diffs."""
        map_a: dict[ClauseType, Clause] = self._best_clause_per_type(clauses_a)
        map_b: dict[ClauseType, Clause] = self._best_clause_per_type(clauses_b)

        all_types = sorted(
            set(map_a.keys()) | set(map_b.keys()),
            key=lambda t: t.value,
        )

        diffs: list[ClauseDiff] = []
        for ctype in all_types:
            ca = map_a.get(ctype)
            cb = map_b.get(ctype)

            if ca is None:
                diffs.append(
                    ClauseDiff(
                        clause_type=ctype,
                        status="added",
                        clause_a=None,
                        clause_b=cb,
                        similarity=0.0,
                    )
                )
            elif cb is None:
                diffs.append(
                    ClauseDiff(
                        clause_type=ctype,
                        status="removed",
                        clause_a=ca,
                        clause_b=None,
                        similarity=0.0,
                    )
                )
            else:
                sim = self._text_similarity(ca.text, cb.text)
                if sim >= self._sim_threshold:
                    status: Literal["modified", "unchanged"] = "unchanged"
                    diffs.append(
                        ClauseDiff(
                            clause_type=ctype,
                            status=status,
                            clause_a=ca,
                            clause_b=cb,
                            similarity=sim,
                        )
                    )
                else:
                    diffs.append(
                        ClauseDiff(
                            clause_type=ctype,
                            status="modified",
                            clause_a=ca,
                            clause_b=cb,
                            similarity=sim,
                            key_differences=self._describe_differences(ca.text, cb.text),
                        )
                    )

        return diffs

    @staticmethod
    def _best_clause_per_type(clauses: list[Clause]) -> dict[ClauseType, Clause]:
        """Return the highest-confidence clause for each type."""
        best: dict[ClauseType, Clause] = {}
        for clause in clauses:
            existing = best.get(clause.type)
            if existing is None or clause.confidence > existing.confidence:
                best[clause.type] = clause
        return best

    # ------------------------------------------------------------------
    # Entity diffing
    # ------------------------------------------------------------------

    def _diff_entities(
        self,
        entities_a: list[Entity],
        entities_b: list[Entity],
    ) -> list[EntityDelta]:
        """Compare entity sets grouped by entity type."""
        types_to_check = [
            EntityType.PARTY,
            EntityType.DATE,
            EntityType.MONEY,
            EntityType.DURATION,
        ]
        deltas: list[EntityDelta] = []

        for etype in types_to_check:
            texts_a = {e.text.strip() for e in entities_a if e.type == etype}
            texts_b = {e.text.strip() for e in entities_b if e.type == etype}

            only_a = sorted(texts_a - texts_b)
            only_b = sorted(texts_b - texts_a)
            shared = sorted(texts_a & texts_b)

            deltas.append(
                EntityDelta(
                    entity_type=etype,
                    only_in_a=only_a,
                    only_in_b=only_b,
                    shared=shared,
                )
            )

        return deltas

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set:
        """Return lowercase word token set."""
        return set(re.findall(r"\b[a-z'-]+\b", text.lower()))

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Jaccard word-overlap similarity between two texts (0.0 -- 1.0)."""
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0

        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _describe_differences(text_a: str, text_b: str) -> list[str]:
        """Generate human-readable notes about differences between texts."""
        notes: list[str] = []
        lower_a = text_a.lower()
        lower_b = text_b.lower()

        # Liability cap language
        cap_terms = ["shall not exceed", "aggregate liability", "limited to", "maximum liability"]
        had_cap = any(t in lower_a for t in cap_terms)
        has_cap = any(t in lower_b for t in cap_terms)
        if had_cap and not has_cap:
            notes.append("Liability cap language removed in new version.")
        elif not had_cap and has_cap:
            notes.append("Liability cap language added in new version.")

        # Mutual vs one-sided
        mutual_terms = ["mutual", "each party", "both parties", "reciprocal"]
        was_mutual = any(t in lower_a for t in mutual_terms)
        is_mutual = any(t in lower_b for t in mutual_terms)
        if was_mutual and not is_mutual:
            notes.append("Provision changed from mutual to one-sided.")
        elif not was_mutual and is_mutual:
            notes.append("Provision changed from one-sided to mutual.")

        # Notice period
        notice_a = re.findall(r"(\d+)\s*(?:calendar\s+)?days?\s+(?:written\s+)?notice", lower_a)
        notice_b = re.findall(r"(\d+)\s*(?:calendar\s+)?days?\s+(?:written\s+)?notice", lower_b)
        if notice_a and notice_b and set(notice_a) != set(notice_b):
            notes.append(
                f"Notice period changed: {', '.join(notice_a)} days --> {', '.join(notice_b)} days."
            )

        # Governing jurisdiction
        def _extract_jurisdiction(text: str) -> str | None:
            m = re.search(
                r"(?:laws? of (?:the )?(?:State of )?|governed by )([A-Z][A-Za-z\s]+)",
                text,
            )
            return m.group(1).strip() if m else None

        jur_a = _extract_jurisdiction(text_a)
        jur_b = _extract_jurisdiction(text_b)
        if jur_a and jur_b and jur_a.lower() != jur_b.lower():
            notes.append(f"Governing jurisdiction changed: '{jur_a}' --> '{jur_b}'.")

        # Auto-renewal
        auto_a = bool(re.search(r"auto(?:matically)?\s+renew", lower_a))
        auto_b = bool(re.search(r"auto(?:matically)?\s+renew", lower_b))
        if auto_a and not auto_b:
            notes.append("Auto-renewal provision removed.")
        elif not auto_a and auto_b:
            notes.append("Auto-renewal provision added.")

        # Generic length change
        len_diff = abs(len(text_a) - len(text_b))
        if not notes and len_diff > 200:
            direction = "expanded" if len(text_b) > len(text_a) else "condensed"
            notes.append(
                f"Clause text {direction} significantly ({len_diff:,} character difference)."
            )

        return notes

    # ------------------------------------------------------------------
    # Overall similarity and summary
    # ------------------------------------------------------------------

    def _compute_overall_similarity(self, clause_diffs: list[ClauseDiff]) -> float:
        """Compute average similarity across all clause types."""
        if not clause_diffs:
            return 1.0
        return sum(cd.similarity for cd in clause_diffs) / len(clause_diffs)

    @staticmethod
    def _generate_summary(
        name_a: str,
        name_b: str,
        clause_diffs: list[ClauseDiff],
        entity_deltas: list[EntityDelta],
        overall_sim: float,
    ) -> str:
        """Build a plain-English summary of the comparison results."""
        added = [cd for cd in clause_diffs if cd.status == "added"]
        removed = [cd for cd in clause_diffs if cd.status == "removed"]
        modified = [cd for cd in clause_diffs if cd.status == "modified"]
        unchanged = [cd for cd in clause_diffs if cd.status == "unchanged"]

        parts: list[str] = []
        parts.append(f"Contract Comparison: {name_a} vs {name_b}")
        parts.append(f"Overall similarity: {overall_sim:.0%}")
        parts.append("")
        parts.append(
            f"Clauses: {len(unchanged)} unchanged, {len(modified)} modified, "
            f"{len(added)} added, {len(removed)} removed."
        )

        if removed:
            labels = ", ".join(cd.clause_type.value.replace("_", " ") for cd in removed)
            parts.append(f"  Removed clauses: {labels}")

        if added:
            labels = ", ".join(cd.clause_type.value.replace("_", " ") for cd in added)
            parts.append(f"  Added clauses: {labels}")

        if modified:
            for cd in modified:
                label = cd.clause_type.value.replace("_", " ").title()
                parts.append(f"  Modified [{label}] -- similarity {cd.similarity:.0%}")
                for note in cd.key_differences:
                    parts.append(f"    -> {note}")

        for delta in entity_deltas:
            if not delta.has_changes:
                continue
            label = delta.entity_type.value.title()
            if delta.only_in_a:
                parts.append(f"  {label} removed: {', '.join(delta.only_in_a[:3])}")
            if delta.only_in_b:
                parts.append(f"  {label} added: {', '.join(delta.only_in_b[:3])}")

        return "\n".join(parts)
