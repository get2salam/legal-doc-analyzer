"""Contract timeline extractor for legal documents.

Extracts key dates, deadlines, and temporal obligations from contract text
and assembles them into a chronological timeline.  Understands common
contract date patterns (effective dates, payment schedules, notice
periods, expiry / renewal windows) without requiring any external NLP
library.

Typical usage::

    from legal_doc_analyzer.timeline import TimelineExtractor

    extractor = TimelineExtractor()
    events = extractor.extract(contract_text)
    timeline = extractor.build_timeline(events)

    for event in timeline.events:
        print(f"{event.label}: {event.date_text}  ({event.event_type.value})")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "EventType",
    "TimelineEvent",
    "ContractTimeline",
    "TimelineExtractor",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Semantic category for a timeline event."""

    EXECUTION = "execution"
    EFFECTIVE = "effective"
    COMMENCEMENT = "commencement"
    EXPIRY = "expiry"
    RENEWAL = "renewal"
    PAYMENT = "payment"
    NOTICE = "notice"
    DEADLINE = "deadline"
    REVIEW = "review"
    MILESTONE = "milestone"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TimelineEvent:
    """A single temporal event extracted from a contract.

    Attributes:
        event_type:   Semantic category (e.g. PAYMENT, DEADLINE).
        date_text:    Raw date string as it appeared in the document
                      (e.g. ``"January 1, 2025"``).
        label:        Short human-readable label for the event.
        context:      The sentence or fragment surrounding the date.
        confidence:   Extraction confidence in [0, 1].
        start_char:   Character offset of the date in the source text.
        end_char:     Character offset of the end of the date span.
        relative_days: Integer day offset relative to the effective/execution
                       date if both are present; ``None`` when not computable.
    """

    event_type: EventType
    date_text: str
    label: str
    context: str = ""
    confidence: float = 0.8
    start_char: int | None = None
    end_char: int | None = None
    relative_days: int | None = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "date_text": self.date_text,
            "label": self.label,
            "context": self.context,
            "confidence": round(self.confidence, 3),
            "relative_days": self.relative_days,
        }


@dataclass
class ContractTimeline:
    """Ordered collection of timeline events.

    Attributes:
        events:         Events sorted chronologically (or by document order
                        when dates cannot be parsed for comparison).
        anchor_date:    The first reliable anchor date (effective / execution).
        total_events:   Number of events found.
        has_expiry:     ``True`` if at least one EXPIRY event was found.
        has_renewals:   ``True`` if at least one RENEWAL event was found.
    """

    events: list[TimelineEvent] = field(default_factory=list)
    anchor_date: str | None = None
    has_expiry: bool = False
    has_renewals: bool = False

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def payment_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.event_type == EventType.PAYMENT]

    @property
    def deadline_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.event_type == EventType.DEADLINE]

    def to_dict(self) -> dict:
        return {
            "anchor_date": self.anchor_date,
            "total_events": self.total_events,
            "has_expiry": self.has_expiry,
            "has_renewals": self.has_renewals,
            "events": [e.to_dict() for e in self.events],
        }


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

# Canonical date patterns reused across rules
_MONTH_NAMES = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?"
)
_DATE_LONG = rf"(?:{_MONTH_NAMES}\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+{_MONTH_NAMES}\s+\d{{4}})"
_DATE_ISO = r"\d{4}[-/]\d{2}[-/]\d{2}"
_DATE_US = r"\d{1,2}/\d{1,2}/\d{4}"
_DATE_ANY = rf"(?:{_DATE_LONG}|{_DATE_ISO}|{_DATE_US})"

# Event-specific trigger patterns → (EventType, label, confidence)
_EVENT_RULES: list[tuple[re.Pattern, EventType, str, float]] = [
    # Execution / signing
    (
        re.compile(
            rf"(?:executed|signed|dated|entered\s+into)\s+(?:as\s+of\s+|on\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.EXECUTION,
        "Contract executed",
        0.95,
    ),
    (
        re.compile(
            rf"(?:as\s+of|dated)\s+({_DATE_ANY})\s*,?\s*(?:by\s+and\s+between|between)",
            re.IGNORECASE,
        ),
        EventType.EXECUTION,
        "Agreement date",
        0.9,
    ),
    # Effective date
    (
        re.compile(
            rf"effective\s+(?:date\s+(?:of\s+)?(?:this\s+agreement\s+)?shall\s+be\s+|as\s+of\s+|date(?:[:]\s+)?)({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.EFFECTIVE,
        "Effective date",
        0.95,
    ),
    (
        re.compile(
            rf"({_DATE_ANY})\s+\(the\s+[\"']?effective\s+date[\"']?\)",
            re.IGNORECASE,
        ),
        EventType.EFFECTIVE,
        "Effective date",
        0.95,
    ),
    # Commencement / start
    (
        re.compile(
            rf"(?:commence[sd]?|begins?|start[sd]?)\s+(?:on\s+|as\s+of\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.COMMENCEMENT,
        "Term commencement",
        0.85,
    ),
    # Expiry / end
    (
        re.compile(
            rf"(?:expir(?:es?|ation\s+date)|terminat(?:es?|ion\s+date)|end\s+(?:date|of\s+term))\s*(?:is\s+|[:]\s*|shall\s+be\s+|on\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.EXPIRY,
        "Contract expiry",
        0.9,
    ),
    (
        re.compile(
            rf"(?:through\s+and\s+including|until|through)\s+({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.EXPIRY,
        "Term end",
        0.75,
    ),
    # Renewal
    (
        re.compile(
            rf"(?:renew(?:al|s|ed)?|extended?)\s+(?:for|until|on|through)\s+(?:a\s+further\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.RENEWAL,
        "Renewal date",
        0.8,
    ),
    (
        re.compile(
            rf"(?:auto(?:matic(?:ally)?)?|shall\s+automatically)\s+renew\s+(?:on|as\s+of|from)\s+({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.RENEWAL,
        "Auto-renewal date",
        0.85,
    ),
    # Payment / invoice
    (
        re.compile(
            rf"(?:due(?:\s+date)?|payable|paid|invoiced?)\s+(?:on\s+|by\s+|no\s+later\s+than\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.PAYMENT,
        "Payment due",
        0.85,
    ),
    (
        re.compile(
            rf"(?:first|initial|second|third|final)\s+(?:payment|instalment|installment)\s+(?:due\s+)?(?:on\s+|by\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.PAYMENT,
        "Instalment due",
        0.85,
    ),
    # Notice deadlines
    (
        re.compile(
            rf"(?:notice|notification)\s+(?:shall\s+be\s+|must\s+be\s+)?(?:given|provided|delivered|sent)\s+(?:by\s+|no\s+later\s+than\s+|on\s+or\s+before\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.NOTICE,
        "Notice deadline",
        0.85,
    ),
    (
        re.compile(
            rf"(?:non[\-\s]?renewal|cancellation|opt[\-\s]?out)\s+notice\s+(?:by|before|on)\s+({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.NOTICE,
        "Non-renewal notice",
        0.9,
    ),
    # Review / audit
    (
        re.compile(
            rf"(?:review(?:ed)?|audit(?:ed)?|assessed?)\s+(?:on\s+|by\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.REVIEW,
        "Review date",
        0.7,
    ),
    # Milestones / deliverables
    (
        re.compile(
            rf"(?:deliverable|milestone|deliverd?|complet(?:ed?|ion))\s+(?:by\s+|on\s+|no\s+later\s+than\s+)?({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.MILESTONE,
        "Milestone / deliverable",
        0.8,
    ),
    # Generic deadline
    (
        re.compile(
            rf"(?:deadline|no\s+later\s+than|on\s+or\s+before|by)\s+({_DATE_ANY})",
            re.IGNORECASE,
        ),
        EventType.DEADLINE,
        "Deadline",
        0.7,
    ),
]

# Fallback: bare date inside a sentence → OTHER (lowest confidence)
_BARE_DATE_RE = re.compile(_DATE_ANY, re.IGNORECASE)

# Sentence splitter (lightweight)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")


def _extract_context(text: str, start: int, end: int, window: int = 200) -> str:
    """Return up to *window* characters of context around [start:end]."""
    ctx_start = max(0, start - window // 2)
    ctx_end = min(len(text), end + window // 2)
    snippet = text[ctx_start:ctx_end].replace("\n", " ").strip()
    # Trim to sentence boundaries when possible
    if ctx_start > 0 and ". " in snippet[:30]:
        snippet = snippet[snippet.index(". ") + 2 :]
    if ctx_end < len(text) and ". " in snippet[-30:]:
        snippet = snippet[: snippet.rindex(". ") + 1]
    return snippet


# ---------------------------------------------------------------------------
# TimelineExtractor
# ---------------------------------------------------------------------------


class TimelineExtractor:
    """Extract and organise temporal events from contract text.

    Works entirely on plain text (no ML, no external libraries).  Run
    ``extract()`` to get a flat list of :class:`TimelineEvent` objects,
    or ``build_timeline()`` to get a sorted :class:`ContractTimeline`.

    Args:
        min_confidence: Discard events below this confidence threshold.
            Default ``0.65``.
        deduplicate: When ``True`` (default), suppress duplicate dates that
            resolve to the same text within the same event category.

    Example::

        extractor = TimelineExtractor()
        timeline = extractor.build_timeline(contract_text)
        print(f"Found {timeline.total_events} events")
        for event in timeline.events:
            print(f"  [{event.event_type.value}] {event.date_text}: {event.label}")
    """

    def __init__(
        self,
        min_confidence: float = 0.65,
        deduplicate: bool = True,
    ) -> None:
        self.min_confidence = min_confidence
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[TimelineEvent]:
        """Extract all timeline events from *text*.

        Applies rule-based patterns in priority order, then falls back
        to bare-date extraction for any dates not yet captured.

        Args:
            text: Full contract text.

        Returns:
            Unsorted list of :class:`TimelineEvent` objects whose
            confidence is >= ``self.min_confidence``.
        """
        if not text or not text.strip():
            return []

        events: list[TimelineEvent] = []
        seen: set[tuple[str, str]] = set()  # (date_text_lower, event_type) pairs

        # Rule-based extraction
        for pattern, event_type, label, base_confidence in _EVENT_RULES:
            for match in pattern.finditer(text):
                date_text = match.group(1).strip()
                if not date_text:
                    continue
                key = (date_text.lower(), event_type.value)
                if self.deduplicate and key in seen:
                    continue
                seen.add(key)

                ctx = _extract_context(text, match.start(), match.end())
                event = TimelineEvent(
                    event_type=event_type,
                    date_text=date_text,
                    label=label,
                    context=ctx,
                    confidence=base_confidence,
                    start_char=match.start(1),
                    end_char=match.end(1),
                )
                if event.confidence >= self.min_confidence:
                    events.append(event)

        # Fallback: bare dates not yet captured
        covered = {(e.start_char, e.end_char) for e in events}
        for match in _BARE_DATE_RE.finditer(text):
            if any(s is not None and e is not None and s <= match.start() < e for s, e in covered):
                continue
            date_text = match.group().strip()
            key = (date_text.lower(), EventType.OTHER.value)
            if self.deduplicate and key in seen:
                continue
            seen.add(key)

            ctx = _extract_context(text, match.start(), match.end())
            # Skip very short matches that are probably false positives
            if len(date_text) < 6:
                continue
            event = TimelineEvent(
                event_type=EventType.OTHER,
                date_text=date_text,
                label="Date mentioned",
                context=ctx,
                confidence=0.65,
                start_char=match.start(),
                end_char=match.end(),
            )
            if event.confidence >= self.min_confidence:
                events.append(event)

        return events

    def build_timeline(self, text: str) -> ContractTimeline:
        """Extract events and return a sorted :class:`ContractTimeline`.

        Events are ordered by document position (character offset).
        Anchor detection identifies the first EFFECTIVE or EXECUTION event
        as the reference point.

        Args:
            text: Full contract text.

        Returns:
            :class:`ContractTimeline` instance.
        """
        events = self.extract(text)

        # Sort by document position
        events.sort(key=lambda e: e.start_char if e.start_char is not None else 0)

        # Identify anchor (first effective or execution date)
        anchor_date: str | None = None
        for event in events:
            if event.event_type in (EventType.EFFECTIVE, EventType.EXECUTION):
                anchor_date = event.date_text
                break

        has_expiry = any(e.event_type == EventType.EXPIRY for e in events)
        has_renewals = any(e.event_type == EventType.RENEWAL for e in events)

        return ContractTimeline(
            events=events,
            anchor_date=anchor_date,
            has_expiry=has_expiry,
            has_renewals=has_renewals,
        )

    def summary(self, text: str) -> dict:
        """Return a concise summary dict of the timeline.

        Suitable for API responses or logging without the full event list.

        Args:
            text: Full contract text.

        Returns:
            Dictionary with counts, anchor date, and top-level flags.
        """
        timeline = self.build_timeline(text)
        type_counts: dict[str, int] = {}
        for event in timeline.events:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1

        return {
            "anchor_date": timeline.anchor_date,
            "total_events": timeline.total_events,
            "has_expiry": timeline.has_expiry,
            "has_renewals": timeline.has_renewals,
            "event_type_counts": type_counts,
        }
