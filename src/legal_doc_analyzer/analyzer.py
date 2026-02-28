"""Main analyzer orchestrating document parsing, extraction, and risk detection.

The ``LegalAnalyzer`` class is the primary entry point for users. It accepts
a file path, parses the document, runs clause/entity extraction and risk
detection, and returns a comprehensive ``AnalysisResult``.
"""

from __future__ import annotations

from pathlib import Path

from .extractors import ClauseExtractor, EntityExtractor, RiskDetector
from .models import AnalysisResult, Clause, ClauseType, Entity, Risk
from .parsers import ParsedDocument, get_parser
from .preprocessing import TextPreprocessor


class LegalAnalyzer:
    """High-level legal document analyzer.

    Orchestrates parsing, clause extraction, entity extraction, and risk
    detection into a single ``analyze()`` call. Can also run individual
    extraction steps independently.

    Example::

        analyzer = LegalAnalyzer()
        result = analyzer.analyze("contract.pdf")

        print(result.summary)
        print(f"Found {len(result.clauses)} clauses")
        print(f"Risk score: {result.risk_score:.0%}")

    Args:
        clause_extractor: Custom ClauseExtractor instance (optional).
        entity_extractor: Custom EntityExtractor instance (optional).
        risk_detector: Custom RiskDetector instance (optional).
    """

    def __init__(
        self,
        clause_extractor: ClauseExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        risk_detector: RiskDetector | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        self._clause_extractor = clause_extractor or ClauseExtractor()
        self._entity_extractor = entity_extractor or EntityExtractor()
        self._risk_detector = risk_detector or RiskDetector()
        self._preprocessor = preprocessor or TextPreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, file_path: str | Path) -> AnalysisResult:
        """Run full analysis on a document file.

        Parses the document, extracts clauses and entities, runs risk
        detection, and generates a summary.

        Args:
            file_path: Path to the document (PDF, DOCX, TXT, or HTML).

        Returns:
            Complete AnalysisResult with clauses, entities, risks, and summary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        path = Path(file_path)
        parsed = self._parse_document(path)
        full_text = parsed.full_text

        # Extract clauses page by page for accurate page references
        clauses = self._extract_clauses_from_parsed(parsed)

        # Extract entities from full text
        entities = self._entity_extractor.extract(full_text)

        # Annotate entities with page numbers
        for entity in entities:
            if entity.start_char is not None:
                entity.page = parsed.get_page_for_char_offset(entity.start_char)

        # Run risk detection
        risks = self._risk_detector.analyze(clauses, full_text)

        # Run readability analysis via preprocessor
        readability = self._preprocessor.analyze_readability(full_text)

        # Generate summary
        summary = self._generate_summary(parsed, clauses, entities, risks)

        return AnalysisResult(
            filename=path.name,
            summary=summary,
            clauses=clauses,
            entities=entities,
            risks=risks,
            metadata={
                "page_count": parsed.page_count,
                "format": parsed.metadata.get("format", "unknown"),
                "char_count": len(full_text),
                "clause_count": len(clauses),
                "entity_count": len(entities),
                "risk_count": len(risks),
                "readability": readability.to_dict(),
            },
            raw_text=full_text,
        )

    def extract_clauses(
        self,
        file_path: str | Path,
        types: list[str] | None = None,
    ) -> list[Clause]:
        """Extract clauses from a document, optionally filtered by type.

        Args:
            file_path: Path to the document.
            types: Optional list of clause type names to filter
                   (e.g., ``["indemnification", "termination"]``).

        Returns:
            List of extracted Clause objects.
        """
        path = Path(file_path)
        parsed = self._parse_document(path)
        clauses = self._extract_clauses_from_parsed(parsed)

        if types:
            requested_types: set[ClauseType] = set()
            for t in types:
                try:
                    requested_types.add(ClauseType(t.lower().strip()))
                except ValueError:
                    continue  # Silently skip unknown types
            if requested_types:
                clauses = [c for c in clauses if c.type in requested_types]

        return clauses

    def extract_entities(self, file_path: str | Path) -> list[Entity]:
        """Extract entities from a document.

        Args:
            file_path: Path to the document.

        Returns:
            List of extracted Entity objects.
        """
        path = Path(file_path)
        parsed = self._parse_document(path)
        return self._entity_extractor.extract(parsed.full_text)

    def detect_risks(self, file_path: str | Path) -> list[Risk]:
        """Detect risks in a document.

        Args:
            file_path: Path to the document.

        Returns:
            List of detected Risk objects.
        """
        path = Path(file_path)
        parsed = self._parse_document(path)
        clauses = self._extract_clauses_from_parsed(parsed)
        return self._risk_detector.analyze(clauses, parsed.full_text)

    def summarize(self, file_path: str | Path) -> str:
        """Generate a plain-English summary of the document.

        Args:
            file_path: Path to the document.

        Returns:
            Multi-line summary string.
        """
        result = self.analyze(file_path)
        return result.summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_document(self, path: Path) -> ParsedDocument:
        """Parse a document file using the appropriate parser."""
        parser = get_parser(path)
        return parser.parse(path)

    def _extract_clauses_from_parsed(self, parsed: ParsedDocument) -> list[Clause]:
        """Extract clauses page-by-page for accurate page references."""
        all_clauses: list[Clause] = []

        if parsed.page_count <= 1:
            # Single page: just extract from full text
            return self._clause_extractor.extract(parsed.full_text, page_hint=1)

        # Multi-page: extract per page, then deduplicate
        offset = 0
        for page in parsed.pages:
            page_clauses = self._clause_extractor.extract(page.text, page_hint=page.page_number)
            # Adjust character offsets to be document-relative
            for clause in page_clauses:
                if clause.start_char is not None:
                    clause.start_char += offset
                if clause.end_char is not None:
                    clause.end_char += offset
            all_clauses.extend(page_clauses)
            offset += len(page.text) + 2  # +2 for \n\n separator

        # Also run extraction on full text for clauses spanning page boundaries
        full_clauses = self._clause_extractor.extract(parsed.full_text)
        existing_types = {(c.type, c.start_char) for c in all_clauses}
        for clause in full_clauses:
            if (clause.type, clause.start_char) not in existing_types:
                # Try to assign a page number
                if clause.start_char is not None:
                    clause.page = parsed.get_page_for_char_offset(clause.start_char)
                all_clauses.append(clause)

        # Sort by position
        all_clauses.sort(key=lambda c: c.start_char or 0)
        return all_clauses

    def _generate_summary(
        self,
        parsed: ParsedDocument,
        clauses: list[Clause],
        entities: list[Entity],
        risks: list[Risk],
    ) -> str:
        """Generate a structured summary of the analysis results.

        This is a rule-based summary generator. For production use, this
        could be replaced with an LLM-based summarizer.
        """
        from .models import EntityType, RiskLevel

        parts: list[str] = []

        # Document overview
        parts.append(f"Document: {parsed.filename} ({parsed.page_count} page(s))")
        parts.append("")

        # Parties
        parties = [e for e in entities if e.type == EntityType.PARTY]
        if parties:
            party_names = ", ".join(p.text for p in parties[:5])
            parts.append(f"Parties: {party_names}")

        # Key dates
        dates = [e for e in entities if e.type == EntityType.DATE]
        if dates:
            date_strs = ", ".join(d.text for d in dates[:5])
            parts.append(f"Key Dates: {date_strs}")

        # Monetary values
        money = [e for e in entities if e.type == EntityType.MONEY]
        if money:
            money_strs = ", ".join(m.text for m in money[:5])
            parts.append(f"Values: {money_strs}")

        parts.append("")

        # Clauses found
        clause_types = sorted({c.type.value for c in clauses})
        if clause_types:
            parts.append(f"Clauses Identified ({len(clauses)} total):")
            for ct in clause_types:
                count = sum(1 for c in clauses if c.type.value == ct)
                label = ct.replace("_", " ").title()
                parts.append(f"  • {label}" + (f" (×{count})" if count > 1 else ""))
        else:
            parts.append("No standard clauses identified.")

        parts.append("")

        # Risk summary
        high_risks = [r for r in risks if r.level == RiskLevel.HIGH]
        medium_risks = [r for r in risks if r.level == RiskLevel.MEDIUM]
        low_risks = [r for r in risks if r.level == RiskLevel.LOW]

        if high_risks or medium_risks:
            parts.append("Risk Assessment:")
            for risk in high_risks:
                parts.append(f"  🔴 HIGH: {risk.description}")
            for risk in medium_risks:
                parts.append(f"  🟡 MEDIUM: {risk.description}")
            if low_risks:
                parts.append(f"  🟢 {len(low_risks)} low-risk item(s)")
        else:
            parts.append("Risk Assessment: No significant risks detected.")

        return "\n".join(parts)
