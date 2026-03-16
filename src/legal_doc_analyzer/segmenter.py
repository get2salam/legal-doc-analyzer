"""Hierarchical document section segmenter for legal documents.

Parses free-form legal text into a tree of :class:`Section` nodes, preserving
the original heading hierarchy (Part → Article → Section → Clause →
Subsection → Paragraph).  Works with common legal numbering styles:

- Decimal outline:  ``1.``, ``1.1``, ``1.1.1``, …
- Roman numeral parts/articles:  ``I.``, ``II.``, ``Article III``
- Alphabetic subsections:  ``(a)``, ``(b)`` / ``a.``, ``b.``
- Symbolic:  ``§ 4``, ``§ 4.2``
- Prose headings:  ``SECTION 5 — TERM AND TERMINATION``

The segmenter is intentionally dependency-free and uses only the Python
standard library so it runs in any environment.

Example::

    from legal_doc_analyzer.segmenter import DocumentSegmenter, segment_document

    structure = segment_document(contract_text)
    print(structure.title)
    for section in structure.sections:
        print(section.number, section.heading)
        for child in section.children:
            print("  ", child.number, child.heading)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "SectionLevel",
    "Section",
    "DocumentStructure",
    "DocumentSegmenter",
    "segment_document",
]

# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class SectionLevel(str, Enum):
    """Hierarchy level of a detected section."""

    PART = "part"
    ARTICLE = "article"
    SECTION = "section"
    CLAUSE = "clause"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    DEFINITION = "definition"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """A single section node in the document tree.

    Attributes:
        level: Hierarchy level (PART, ARTICLE, SECTION …).
        number: The raw numbering string, e.g. ``"3.1"`` or ``"(a)"``.
        heading: Heading text stripped of the number, uppercased.
        content: Body text directly under this heading (before any child).
        children: Ordered list of child :class:`Section` nodes.
        start_char: Character offset in the original document text.
        end_char: Character offset of the last character (exclusive).
        depth: Nesting depth (0 = top-level).
        metadata: Arbitrary extra data attached by specialised passes.
    """

    level: SectionLevel
    number: str
    heading: str
    content: str = ""
    children: list[Section] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    depth: int = 0
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def full_heading(self) -> str:
        """Number and heading joined with a space."""
        parts = [p for p in (self.number, self.heading) if p]
        return " ".join(parts)

    @property
    def word_count(self) -> int:
        """Total words in the section body (does not recurse into children)."""
        return len(self.content.split())

    @property
    def has_children(self) -> bool:
        return bool(self.children)

    def iter_all(self) -> Iterator[Section]:
        """Depth-first iterator over this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.iter_all()

    def to_dict(self, include_content: bool = True) -> dict:
        """Serialize to a plain dict (recursively)."""
        d: dict = {
            "level": self.level.value,
            "number": self.number,
            "heading": self.heading,
            "depth": self.depth,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "children": [c.to_dict(include_content=include_content) for c in self.children],
        }
        if include_content:
            d["content"] = self.content
        return d


@dataclass
class DocumentStructure:
    """Top-level result of segmenting a document.

    Attributes:
        title: Detected document title (first non-blank line or empty string).
        preamble: Text before the first heading.
        sections: Top-level :class:`Section` nodes in document order.
        definitions: Mapping of defined terms to their definitions, if a
            definitions section was found and parsed.
        metadata: Arbitrary extra data (e.g. detected party names).
    """

    title: str
    preamble: str = ""
    sections: list[Section] = field(default_factory=list)
    definitions: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def section_count(self) -> int:
        """Total number of top-level sections."""
        return len(self.sections)

    @property
    def total_sections(self) -> int:
        """Total section nodes including all descendants."""
        return sum(1 for _ in self.iter_all_sections())

    def iter_all_sections(self) -> Iterator[Section]:
        """Depth-first iterator over every section in the document."""
        for s in self.sections:
            yield from s.iter_all()

    def find_by_heading(self, keyword: str) -> list[Section]:
        """Return all sections whose heading contains *keyword* (case-insensitive)."""
        kw = keyword.lower()
        return [s for s in self.iter_all_sections() if kw in s.heading.lower()]

    def find_by_number(self, number: str) -> Section | None:
        """Return the first section whose :attr:`~Section.number` equals *number*."""
        for s in self.iter_all_sections():
            if s.number == number:
                return s
        return None

    def to_dict(self, include_content: bool = True) -> dict:
        return {
            "title": self.title,
            "preamble": self.preamble if include_content else "",
            "section_count": self.section_count,
            "total_sections": self.total_sections,
            "sections": [s.to_dict(include_content=include_content) for s in self.sections],
            "definitions": self.definitions,
        }


# ---------------------------------------------------------------------------
# Internal heading patterns
# ---------------------------------------------------------------------------

# Each pattern is a compiled regex that matches a heading line.  Groups:
#   group(1) = raw number/label
#   group(2) = heading text (may be empty)
#
# Patterns are tested in priority order; the first match wins.


@dataclass
class _HeadingPattern:
    regex: re.Pattern
    level: SectionLevel
    depth_fn: object  # callable(match) -> int, used to compute nesting depth


def _decimal_depth(number: str) -> int:
    """Depth derived from decimal notation: "1" -> 0, "1.1" -> 1, "1.1.2" -> 2."""
    return number.count(".")


_HEADING_PATTERNS: list[_HeadingPattern] = [
    # PART I / PART 1 / PART ONE
    _HeadingPattern(
        regex=re.compile(
            r"^(?P<num>PART\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN))"
            r"[\s:\-—]*(?P<heading>.*)$",
            re.IGNORECASE,
        ),
        level=SectionLevel.PART,
        depth_fn=lambda m: 0,
    ),
    # ARTICLE I / ARTICLE 1
    _HeadingPattern(
        regex=re.compile(
            r"^(?P<num>ARTICLE\s+(?:[IVXLCDM]+|\d+))"
            r"[\s:\-—]*(?P<heading>.*)$",
            re.IGNORECASE,
        ),
        level=SectionLevel.ARTICLE,
        depth_fn=lambda m: 0,
    ),
    # SECTION 1 / SECTION 1.2
    _HeadingPattern(
        regex=re.compile(
            r"^(?P<num>(?:SECTION|SEC\.?)\s+\d+(?:\.\d+)*)"
            r"[\s:\-—.]*(?P<heading>.*)$",
            re.IGNORECASE,
        ),
        level=SectionLevel.SECTION,
        depth_fn=lambda m: (
            m.group("num").upper().lstrip("SECTION").lstrip("SEC.").strip().count(".")
        ),
    ),
    # § 4 / § 4.2 / §4
    _HeadingPattern(
        regex=re.compile(
            r"^(?P<num>§+\s*\d+(?:\.\d+)*)"
            r"[\s:\-—.]*(?P<heading>.*)$",
        ),
        level=SectionLevel.SECTION,
        depth_fn=lambda m: m.group("num").count("."),
    ),
    # Decimal outline: 1. / 1.1 / 1.1.2  (heading on same line)
    _HeadingPattern(
        regex=re.compile(r"^(?P<num>\d+(?:\.\d+)+)[\s:\-—.]+(?P<heading>[A-Z].*)$"),
        level=SectionLevel.CLAUSE,
        depth_fn=lambda m: _decimal_depth(m.group("num")),
    ),
    # Top-level decimal: "1. HEADING" (must have uppercase heading to avoid matching lists)
    _HeadingPattern(
        regex=re.compile(r"^(?P<num>\d+)[\.\)]\s+(?P<heading>[A-Z][A-Z ]{2,}.*)$"),
        level=SectionLevel.SECTION,
        depth_fn=lambda m: 0,
    ),
    # Alphabetic subsection: (a) / (b) / a. / b.
    _HeadingPattern(
        regex=re.compile(r"^(?P<num>\([a-z]{1,2}\)|[a-z]\.)[\s]+(?P<heading>.+)$"),
        level=SectionLevel.SUBSECTION,
        depth_fn=lambda m: 2,
    ),
    # Roman numeral top-level: I. / II. / III.
    _HeadingPattern(
        regex=re.compile(r"^(?P<num>[IVXLCDM]{1,6})\.\s+(?P<heading>[A-Z].*)$"),
        level=SectionLevel.SECTION,
        depth_fn=lambda m: 0,
    ),
    # ALL-CAPS standalone heading line (no number)
    _HeadingPattern(
        regex=re.compile(r"^(?P<num>)(?P<heading>[A-Z][A-Z\s\-&'/]{4,})$"),
        level=SectionLevel.SECTION,
        depth_fn=lambda m: 0,
    ),
]

# Definition line: "  'Confidential Information' means …" or "  "Term" means"
_RE_DEFINITION = re.compile(
    r"""^[\s"'\u201c\u2018]*(?P<term>[A-Z][A-Za-z\s]{2,40})["'\u201d\u2019]*\s+(?:means?|shall\s+mean|refers?\s+to)\s+(?P<defn>.+)$"""
)

# Detect preamble lines (recitals / whereas)
_RE_PREAMBLE_MARKER = re.compile(
    r"\b(?:WHEREAS|RECITALS?|BACKGROUND|NOW,?\s+THEREFORE|THIS\s+AGREEMENT)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DocumentSegmenter
# ---------------------------------------------------------------------------


class DocumentSegmenter:
    """Parses legal text into a hierarchical :class:`DocumentStructure`.

    The segmenter is entirely regex-based and requires no external models.
    It operates line by line, detecting heading patterns and aggregating
    body text between them.

    Args:
        extract_definitions: If ``True`` (default), look for a definitions
            section and parse ``term means definition`` entries.
        min_heading_chars: Minimum character length for a heading line to
            be considered valid. Very short lines (e.g. ``"a."`` alone) are
            skipped.
        strip_page_numbers: If ``True`` (default), attempt to strip common
            page-number artifacts (lines matching ``"- N -"`` or just a
            digit).
    """

    def __init__(
        self,
        extract_definitions: bool = True,
        min_heading_chars: int = 3,
        strip_page_numbers: bool = True,
    ) -> None:
        self.extract_definitions = extract_definitions
        self.min_heading_chars = min_heading_chars
        self.strip_page_numbers = strip_page_numbers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, text: str) -> DocumentStructure:
        """Segment *text* into a :class:`DocumentStructure`.

        Args:
            text: Raw document text (plain text or lightly pre-processed).

        Returns:
            :class:`DocumentStructure` with populated sections tree.
        """
        if not text or not text.strip():
            return DocumentStructure(title="")

        lines = self._preprocess(text)
        title = self._detect_title(lines)
        spans = self._split_by_headings(text, lines)
        sections = self._build_tree(spans)
        preamble = self._extract_preamble(text)
        definitions: dict[str, str] = {}
        if self.extract_definitions:
            definitions = self._extract_definitions(sections)

        return DocumentStructure(
            title=title,
            preamble=preamble,
            sections=sections,
            definitions=definitions,
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> list[str]:
        """Split text into lines and optionally strip page artifacts."""
        lines = text.splitlines()
        if self.strip_page_numbers:
            lines = [ln for ln in lines if not re.match(r"^\s*-?\s*\d+\s*-?\s*$", ln)]
        return lines

    # ------------------------------------------------------------------
    # Title detection
    # ------------------------------------------------------------------

    def _detect_title(self, lines: list[str]) -> str:
        """Return the first non-blank line as the document title."""
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                return stripped
        return ""

    # ------------------------------------------------------------------
    # Heading detection helpers
    # ------------------------------------------------------------------

    def _match_heading(self, line: str) -> tuple[_HeadingPattern, re.Match] | None:
        """Return the first pattern match for *line*, or ``None``."""
        stripped = line.strip()
        if len(stripped) < self.min_heading_chars:
            return None
        for pattern in _HEADING_PATTERNS:
            m = pattern.regex.match(stripped)
            if m:
                return pattern, m
        return None

    # ------------------------------------------------------------------
    # Split text by headings
    # ------------------------------------------------------------------

    @dataclass
    class _Span:
        """Raw data for a single heading span before tree assembly."""

        level: SectionLevel
        number: str
        heading: str
        content: str
        start_char: int
        end_char: int
        depth: int

    def _split_by_headings(self, text: str, lines: list[str]) -> list[_Span]:
        """Walk *lines* and split at heading boundaries."""
        spans: list[DocumentSegmenter._Span] = []
        current_num = ""
        current_heading = ""
        current_level = SectionLevel.UNKNOWN
        current_depth = 0
        current_start = 0
        body_lines: list[str] = []
        offset = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for the stripped newline
            result = self._match_heading(line)
            if result is not None:
                pat, m = result
                # Save previous span
                if current_heading or current_num:
                    content = "\n".join(body_lines).strip()
                    spans.append(
                        self._Span(
                            level=current_level,
                            number=current_num,
                            heading=current_heading,
                            content=content,
                            start_char=current_start,
                            end_char=offset,
                            depth=current_depth,
                        )
                    )
                num_raw = m.group("num").strip() if m.group("num") else ""
                heading_raw = m.group("heading").strip() if m.group("heading") else ""
                try:
                    depth = pat.depth_fn(m)
                except Exception:  # noqa: BLE001
                    depth = 0
                current_num = num_raw
                current_heading = heading_raw.upper() if heading_raw else ""
                current_level = pat.level
                current_depth = depth
                current_start = offset
                body_lines = []
            else:
                body_lines.append(line)
            offset += line_len

        # Flush last span
        if current_heading or current_num:
            content = "\n".join(body_lines).strip()
            spans.append(
                self._Span(
                    level=current_level,
                    number=current_num,
                    heading=current_heading,
                    content=content,
                    start_char=current_start,
                    end_char=offset,
                    depth=current_depth,
                )
            )

        return spans

    # ------------------------------------------------------------------
    # Tree assembly
    # ------------------------------------------------------------------

    def _build_tree(self, spans: list[_Span]) -> list[Section]:
        """Convert a flat list of :class:`_Span` objects into a section tree."""
        roots: list[Section] = []
        # Stack holds (depth, section) pairs
        stack: list[tuple[int, Section]] = []

        for span in spans:
            node = Section(
                level=span.level,
                number=span.number,
                heading=span.heading,
                content=span.content,
                start_char=span.start_char,
                end_char=span.end_char,
                depth=span.depth,
            )
            # Pop stack until we find a suitable parent
            while stack and stack[-1][0] >= span.depth:
                stack.pop()

            if stack:
                parent = stack[-1][1]
                node.depth = parent.depth + 1
                parent.children.append(node)
            else:
                node.depth = 0
                roots.append(node)

            stack.append((span.depth, node))

        return roots

    # ------------------------------------------------------------------
    # Preamble extraction
    # ------------------------------------------------------------------

    def _extract_preamble(self, text: str) -> str:
        """Return text before the first heading, truncated at 1 000 chars."""
        lines = text.splitlines()
        preamble_lines: list[str] = []
        for line in lines:
            if self._match_heading(line):
                break
            preamble_lines.append(line)
        preamble = "\n".join(preamble_lines).strip()
        return preamble[:1000]

    # ------------------------------------------------------------------
    # Definitions extraction
    # ------------------------------------------------------------------

    def _extract_definitions(self, sections: list[Section]) -> dict[str, str]:
        """Scan the sections tree for a definitions block and parse it."""
        defs: dict[str, str] = {}

        def _scan(node: Section) -> None:
            if "DEFINIT" in node.heading.upper() or "DEFINED TERMS" in node.heading.upper():
                for line in node.content.splitlines():
                    m = _RE_DEFINITION.match(line.strip())
                    if m:
                        term = m.group("term").strip()
                        defn = m.group("defn").strip().rstrip(".")
                        defs[term] = defn
                # Also check children
                for child in node.children:
                    m2 = _RE_DEFINITION.match(child.full_heading.strip())
                    if m2:
                        defs[m2.group("term").strip()] = m2.group("defn").strip().rstrip(".")
                    else:
                        _scan(child)
            else:
                for child in node.children:
                    _scan(child)

        for s in sections:
            _scan(s)
        return defs


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def segment_document(
    text: str,
    *,
    extract_definitions: bool = True,
    strip_page_numbers: bool = True,
) -> DocumentStructure:
    """Convenience wrapper: segment *text* and return a :class:`DocumentStructure`.

    Creates a :class:`DocumentSegmenter` with the given options and calls
    :meth:`~DocumentSegmenter.segment`.

    Args:
        text: Raw document text.
        extract_definitions: Parse a definitions section if present.
        strip_page_numbers: Remove page-number artifacts from input lines.

    Returns:
        :class:`DocumentStructure` containing the parsed section tree.

    Example::

        structure = segment_document(contract_text)
        for section in structure.sections:
            print(section.number, section.heading)
    """
    segmenter = DocumentSegmenter(
        extract_definitions=extract_definitions,
        strip_page_numbers=strip_page_numbers,
    )
    return segmenter.segment(text)
