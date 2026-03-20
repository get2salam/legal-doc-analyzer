"""Tests for the contract timeline extractor."""

from __future__ import annotations

import pytest

from legal_doc_analyzer.timeline import (
    ContractTimeline,
    EventType,
    TimelineExtractor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> TimelineExtractor:
    return TimelineExtractor()


@pytest.fixture
def full_contract() -> str:
    return (
        "SERVICE AGREEMENT\n\n"
        "This Agreement is entered into as of March 1, 2025, "
        'by and between Acme Technologies Inc. ("Company") '
        'and Digital Solutions Ltd. ("Provider").\n\n'
        "1. EFFECTIVE DATE\n"
        "This Agreement shall be effective as of March 1, 2025 "
        '(the "Effective Date") and shall commence on April 1, 2025.\n\n'
        "2. TERM\n"
        "The initial term of this Agreement shall expire on February 28, 2026. "
        "The Agreement will automatically renew for successive one-year periods "
        "unless either party provides non-renewal notice by January 31, 2026.\n\n"
        "3. PAYMENT\n"
        "The Company shall pay the Provider $50,000. "
        "The first payment is due on April 15, 2025. "
        "The final payment shall be payable by December 31, 2025.\n\n"
        "4. DELIVERABLES\n"
        "The Provider shall complete the initial deliverable by June 30, 2025. "
        "A progress review will be conducted on September 1, 2025.\n\n"
        "5. NOTICE\n"
        "Any notice shall be given no later than October 1, 2025 "
        "to be effective under this Agreement.\n\n"
        "IN WITNESS WHEREOF the parties have signed this Agreement "
        "on the date first written above."
    )


@pytest.fixture
def minimal_contract() -> str:
    return "This Agreement is dated January 15, 2024. The term expires on January 14, 2025."


# ---------------------------------------------------------------------------
# TimelineExtractor.extract()
# ---------------------------------------------------------------------------


class TestTimelineExtractorExtract:
    def test_empty_text_returns_empty(self, extractor: TimelineExtractor) -> None:
        assert extractor.extract("") == []

    def test_whitespace_only_returns_empty(self, extractor: TimelineExtractor) -> None:
        assert extractor.extract("   \n\n   ") == []

    def test_no_dates_returns_empty(self, extractor: TimelineExtractor) -> None:
        result = extractor.extract("No dates are mentioned in this text at all.")
        assert result == []

    def test_extracts_execution_date(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement is entered into as of March 1, 2025, by and between the parties."
        events = extractor.extract(text)
        execution = [e for e in events if e.event_type == EventType.EXECUTION]
        assert len(execution) >= 1
        assert "March 1, 2025" in execution[0].date_text

    def test_extracts_effective_date(self, extractor: TimelineExtractor) -> None:
        text = 'This Agreement shall be effective as of April 1, 2025 (the "Effective Date").'
        events = extractor.extract(text)
        effective = [e for e in events if e.event_type == EventType.EFFECTIVE]
        assert len(effective) >= 1

    def test_extracts_effective_date_parenthetical(self, extractor: TimelineExtractor) -> None:
        text = 'Commencing on June 1, 2025 (the "Effective Date").'
        events = extractor.extract(text)
        effective = [e for e in events if e.event_type == EventType.EFFECTIVE]
        assert len(effective) >= 1

    def test_extracts_expiry_date(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement shall expire on December 31, 2025."
        events = extractor.extract(text)
        expiry = [e for e in events if e.event_type == EventType.EXPIRY]
        assert len(expiry) >= 1
        assert "December 31, 2025" in expiry[0].date_text

    def test_extracts_payment_due(self, extractor: TimelineExtractor) -> None:
        text = "Payment is due on March 15, 2025."
        events = extractor.extract(text)
        payment = [e for e in events if e.event_type == EventType.PAYMENT]
        assert len(payment) >= 1

    def test_extracts_notice_deadline(self, extractor: TimelineExtractor) -> None:
        text = "Notice must be given no later than November 1, 2025."
        events = extractor.extract(text)
        notice = [e for e in events if e.event_type == EventType.NOTICE]
        assert len(notice) >= 1

    def test_extracts_milestone(self, extractor: TimelineExtractor) -> None:
        text = "The deliverable must be completed by June 30, 2025."
        events = extractor.extract(text)
        milestones = [e for e in events if e.event_type == EventType.MILESTONE]
        assert len(milestones) >= 1

    def test_extracts_review_date(self, extractor: TimelineExtractor) -> None:
        text = "The contract terms shall be reviewed on September 1, 2025."
        events = extractor.extract(text)
        reviews = [e for e in events if e.event_type == EventType.REVIEW]
        assert len(reviews) >= 1

    def test_extracts_commencement(self, extractor: TimelineExtractor) -> None:
        text = "The services shall commence on May 1, 2025."
        events = extractor.extract(text)
        starts = [e for e in events if e.event_type == EventType.COMMENCEMENT]
        assert len(starts) >= 1

    def test_extracts_renewal(self, extractor: TimelineExtractor) -> None:
        text = "The Agreement shall automatically renew on January 1, 2026."
        events = extractor.extract(text)
        renewals = [e for e in events if e.event_type == EventType.RENEWAL]
        assert len(renewals) >= 1

    def test_extracts_generic_deadline(self, extractor: TimelineExtractor) -> None:
        text = "The report must be submitted no later than August 31, 2025."
        events = extractor.extract(text)
        deadlines = [e for e in events if e.event_type == EventType.DEADLINE]
        assert len(deadlines) >= 1

    def test_iso_date_format(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement expires on 2025-12-31."
        events = extractor.extract(text)
        assert len(events) >= 1

    def test_us_slash_date_format(self, extractor: TimelineExtractor) -> None:
        text = "Payment is due on 12/31/2025."
        events = extractor.extract(text)
        assert len(events) >= 1

    def test_uk_date_format(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement commences on 1 April 2025."
        events = extractor.extract(text)
        starts = [e for e in events if e.event_type == EventType.COMMENCEMENT]
        assert len(starts) >= 1

    def test_confidence_in_range(self, extractor: TimelineExtractor) -> None:
        text = (
            "This Agreement is effective as of January 1, 2025. "
            "Payment is due on February 28, 2025."
        )
        events = extractor.extract(text)
        for event in events:
            assert 0.0 <= event.confidence <= 1.0

    def test_context_is_non_empty(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement is effective as of January 1, 2025."
        events = extractor.extract(text)
        for event in events:
            assert event.context  # context should be non-empty

    def test_start_end_char_set(self, extractor: TimelineExtractor) -> None:
        text = "Payment is due on March 15, 2025 under this Agreement."
        events = extractor.extract(text)
        for event in events:
            if event.start_char is not None:
                assert event.start_char >= 0
                assert event.end_char is not None
                assert event.end_char > event.start_char

    def test_deduplication_enabled(self, extractor: TimelineExtractor) -> None:
        text = "The effective date is January 1, 2025. Effective as of January 1, 2025."
        extractor_dedup = TimelineExtractor(deduplicate=True)
        events = extractor_dedup.extract(text)
        effective_dates = [e for e in events if e.event_type == EventType.EFFECTIVE]
        # With dedup, same date+type should only appear once
        date_texts = [e.date_text.lower() for e in effective_dates]
        assert len(date_texts) == len(set(date_texts))

    def test_deduplication_disabled(self) -> None:
        extractor_no_dedup = TimelineExtractor(deduplicate=False)
        text = "The effective date is January 1, 2025. Also effective as of January 1, 2025."
        events = extractor_no_dedup.extract(text)
        effective = [e for e in events if e.event_type == EventType.EFFECTIVE]
        # Without dedup, same date may appear twice
        assert len(effective) >= 1  # at minimum one found

    def test_min_confidence_filters_low_events(self) -> None:
        extractor_strict = TimelineExtractor(min_confidence=0.9)
        text = (
            "This Agreement is entered into as of March 1, 2025. "
            "The report deadline is no later than June 30, 2025."
        )
        events_strict = extractor_strict.extract(text)
        extractor_loose = TimelineExtractor(min_confidence=0.5)
        events_loose = extractor_loose.extract(text)
        # Strict extractor should return fewer or equal events
        assert len(events_strict) <= len(events_loose)

    def test_full_contract_events(self, extractor: TimelineExtractor, full_contract: str) -> None:
        events = extractor.extract(full_contract)
        assert len(events) >= 5
        types_found = {e.event_type for e in events}
        assert EventType.EXECUTION in types_found or EventType.EFFECTIVE in types_found
        assert EventType.EXPIRY in types_found
        assert EventType.PAYMENT in types_found

    def test_non_renewal_notice(self, extractor: TimelineExtractor) -> None:
        text = (
            "Either party may provide non-renewal notice by January 31, 2026 "
            "to prevent automatic renewal."
        )
        events = extractor.extract(text)
        notice = [e for e in events if e.event_type == EventType.NOTICE]
        assert len(notice) >= 1

    def test_instalment_payment(self, extractor: TimelineExtractor) -> None:
        text = "The first instalment is due on April 15, 2025."
        events = extractor.extract(text)
        payments = [e for e in events if e.event_type == EventType.PAYMENT]
        assert len(payments) >= 1

    def test_termination_date_as_expiry(self, extractor: TimelineExtractor) -> None:
        text = "The termination date shall be June 30, 2025."
        events = extractor.extract(text)
        expiry = [e for e in events if e.event_type == EventType.EXPIRY]
        assert len(expiry) >= 1


# ---------------------------------------------------------------------------
# TimelineExtractor.build_timeline()
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    def test_returns_contract_timeline(self, extractor: TimelineExtractor) -> None:
        result = extractor.build_timeline("This Agreement expires on December 31, 2025.")
        assert isinstance(result, ContractTimeline)

    def test_empty_text_empty_timeline(self, extractor: TimelineExtractor) -> None:
        timeline = extractor.build_timeline("")
        assert timeline.total_events == 0
        assert timeline.anchor_date is None

    def test_anchor_date_from_effective(self, extractor: TimelineExtractor) -> None:
        text = (
            'This Agreement is effective as of January 1, 2025 (the "Effective Date"). '
            "Payment is due on February 1, 2025."
        )
        timeline = extractor.build_timeline(text)
        assert timeline.anchor_date is not None
        assert "2025" in timeline.anchor_date

    def test_anchor_date_from_execution_fallback(self, extractor: TimelineExtractor) -> None:
        text = (
            "This Agreement is entered into as of March 1, 2025. Payment is due on April 1, 2025."
        )
        timeline = extractor.build_timeline(text)
        assert timeline.anchor_date is not None

    def test_has_expiry_flag(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement expires on December 31, 2025."
        timeline = extractor.build_timeline(text)
        assert timeline.has_expiry is True

    def test_no_expiry_flag(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement is effective as of January 1, 2025."
        timeline = extractor.build_timeline(text)
        assert timeline.has_expiry is False

    def test_has_renewals_flag(self, extractor: TimelineExtractor) -> None:
        text = "The Agreement shall automatically renew on January 1, 2026."
        timeline = extractor.build_timeline(text)
        assert timeline.has_renewals is True

    def test_events_ordered_by_position(self, extractor: TimelineExtractor) -> None:
        text = (
            "This Agreement is effective as of January 1, 2025. "
            "Payment is due on February 28, 2025. "
            "The term expires on December 31, 2025."
        )
        timeline = extractor.build_timeline(text)
        positions = [e.start_char for e in timeline.events if e.start_char is not None]
        assert positions == sorted(positions)

    def test_total_events_property(self, extractor: TimelineExtractor) -> None:
        text = "Effective as of January 1, 2025. Expires on December 31, 2025."
        timeline = extractor.build_timeline(text)
        assert timeline.total_events == len(timeline.events)

    def test_payment_events_filter(self, extractor: TimelineExtractor) -> None:
        text = (
            "Effective as of January 1, 2025. "
            "First payment due on February 1, 2025. "
            "Second payment due on March 1, 2025."
        )
        timeline = extractor.build_timeline(text)
        assert all(e.event_type == EventType.PAYMENT for e in timeline.payment_events)

    def test_deadline_events_filter(self, extractor: TimelineExtractor) -> None:
        text = "Submit the report no later than August 31, 2025."
        timeline = extractor.build_timeline(text)
        assert all(e.event_type == EventType.DEADLINE for e in timeline.deadline_events)

    def test_full_contract_timeline(self, extractor: TimelineExtractor, full_contract: str) -> None:
        timeline = extractor.build_timeline(full_contract)
        assert timeline.total_events >= 5
        assert timeline.anchor_date is not None
        assert timeline.has_expiry is True

    def test_to_dict_structure(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement is effective as of January 1, 2025. Expires on December 31, 2025."
        timeline = extractor.build_timeline(text)
        d = timeline.to_dict()
        assert "anchor_date" in d
        assert "total_events" in d
        assert "has_expiry" in d
        assert "has_renewals" in d
        assert "events" in d
        assert isinstance(d["events"], list)

    def test_event_to_dict_keys(self, extractor: TimelineExtractor) -> None:
        text = "This Agreement expires on December 31, 2025."
        timeline = extractor.build_timeline(text)
        for event in timeline.events:
            d = event.to_dict()
            assert "event_type" in d
            assert "date_text" in d
            assert "label" in d
            assert "context" in d
            assert "confidence" in d


# ---------------------------------------------------------------------------
# TimelineExtractor.summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_dict(self, extractor: TimelineExtractor) -> None:
        text = "Effective as of January 1, 2025. Expires December 31, 2025."
        result = extractor.summary(text)
        assert isinstance(result, dict)

    def test_summary_has_required_keys(self, extractor: TimelineExtractor) -> None:
        text = "Effective as of January 1, 2025."
        result = extractor.summary(text)
        assert "anchor_date" in result
        assert "total_events" in result
        assert "has_expiry" in result
        assert "has_renewals" in result
        assert "event_type_counts" in result

    def test_summary_event_type_counts(self, extractor: TimelineExtractor) -> None:
        text = (
            "Effective as of January 1, 2025. "
            "Expires December 31, 2025. "
            "Payment due February 1, 2025."
        )
        result = extractor.summary(text)
        counts = result["event_type_counts"]
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == result["total_events"]

    def test_summary_empty_text(self, extractor: TimelineExtractor) -> None:
        result = extractor.summary("")
        assert result["total_events"] == 0
        assert result["anchor_date"] is None
