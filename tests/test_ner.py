"""Tests for the regex-based Named Entity Recognition (NER) module.

Covers all entity types, edge cases, overlapping entities, empty inputs,
and combined extraction. At least 30 tests in total.
"""

from __future__ import annotations

import pytest

from legal_doc_analyzer.ner import (
    Entity,
    EntityType,
    extract_citations,
    extract_dates,
    extract_entities,
    extract_monetary,
    extract_organizations,
    extract_persons,
    extract_statutes,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def texts(entities: list[Entity]) -> list[str]:
    """Return just the matched text strings for easier assertion."""
    return [e.text for e in entities]


# ---------------------------------------------------------------------------
# __all__ / public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_exports_available(self):
        import legal_doc_analyzer.ner as ner_module

        for name in ner_module.__all__:
            assert hasattr(ner_module, name), f"Missing export: {name}"

    def test_all_is_list(self):
        import legal_doc_analyzer.ner as ner_module

        assert isinstance(ner_module.__all__, list)


# ---------------------------------------------------------------------------
# EntityType enum
# ---------------------------------------------------------------------------


class TestEntityType:
    def test_all_expected_types_present(self):
        expected = {"date", "monetary", "citation", "statute", "organization", "person"}
        actual = {et.value for et in EntityType}
        assert expected == actual

    def test_is_string_enum(self):
        assert EntityType.DATE == "date"
        assert EntityType.MONETARY == "monetary"
        assert EntityType.CITATION == "citation"
        assert EntityType.STATUTE == "statute"
        assert EntityType.ORGANIZATION == "organization"
        assert EntityType.PERSON == "person"

    def test_enum_count(self):
        assert len(EntityType) == 6


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


class TestEntityDataclass:
    def test_basic_creation(self):
        e = Entity(text="1 January 2024", entity_type=EntityType.DATE, start=0, end=14)
        assert e.entity_type == EntityType.DATE
        assert e.text == "1 January 2024"
        assert e.start == 0
        assert e.end == 14
        assert e.confidence == 1.0
        assert e.normalized is None
        assert e.metadata == {}

    def test_confidence_field(self):
        e = Entity(text="$500", entity_type=EntityType.MONETARY, start=0, end=4, confidence=0.97)
        assert e.confidence == pytest.approx(0.97)

    def test_to_dict_keys(self):
        e = Entity(
            text="$50,000",
            entity_type=EntityType.MONETARY,
            start=5,
            end=12,
            normalized="50000 USD",
            confidence=0.97,
        )
        d = e.to_dict()
        assert set(d.keys()) >= {"entity_type", "text", "start", "end", "confidence"}

    def test_to_dict_values(self):
        e = Entity(text="$50,000", entity_type=EntityType.MONETARY, start=5, end=12)
        d = e.to_dict()
        assert d["entity_type"] == "monetary"
        assert d["text"] == "$50,000"
        assert d["start"] == 5
        assert d["end"] == 12

    def test_metadata_stored(self):
        e = Entity(
            text="Dr. Jane Doe",
            entity_type=EntityType.PERSON,
            start=0,
            end=12,
            metadata={"title": "Dr."},
        )
        assert e.metadata["title"] == "Dr."

    def test_span_ordering(self):
        e = Entity(text="foo", entity_type=EntityType.DATE, start=3, end=6)
        assert e.start < e.end


# ---------------------------------------------------------------------------
# extract_dates
# ---------------------------------------------------------------------------


class TestExtractDates:
    def test_iso_format(self):
        result = extract_dates("The contract was signed on 2024-01-15.")
        assert any("2024-01-15" in e.text for e in result)

    def test_us_numeric_format(self):
        result = extract_dates("Effective date: 01/31/2024.")
        assert any("01/31/2024" in e.text for e in result)

    def test_long_form_mdy(self):
        result = extract_dates("Signed on January 31, 2024 by both parties.")
        assert any("January 31, 2024" in e.text for e in result)

    def test_long_form_dmy(self):
        result = extract_dates("This Agreement dated 15 March 2023.")
        assert any("15 March 2023" in e.text for e in result)

    def test_ordinal_day(self):
        result = extract_dates("Commencing on the 1st January 2025.")
        assert len(result) >= 1

    def test_abbreviated_month(self):
        result = extract_dates("Due by Jan 15, 2024.")
        assert len(result) >= 1

    def test_year_only_with_preposition(self):
        result = extract_dates("In force since 2020.")
        assert any("2020" in e.text for e in result)

    def test_multiple_dates_in_text(self):
        text = "From 2024-01-01 to 2024-12-31 the agreement is in effect."
        result = extract_dates(text)
        assert len(result) >= 2

    def test_empty_text_returns_empty(self):
        assert extract_dates("") == []

    def test_none_like_empty_string(self):
        assert extract_dates("   ") == []

    def test_entity_type_is_date(self):
        for e in extract_dates("Signed 12 February 2020."):
            assert e.entity_type == EntityType.DATE

    def test_confidence_below_one_for_year_only(self):
        result = extract_dates("Since 2019 the policy has applied.")
        year_ents = [e for e in result if "2019" in e.text]
        assert all(e.confidence < 1.0 for e in year_ents)

    def test_sorted_by_start(self):
        text = "On 2023-01-01 and later 2024-06-15."
        result = extract_dates(text)
        starts = [e.start for e in result]
        assert starts == sorted(starts)

    def test_span_within_text(self):
        text = "Dated 15th March 2024 and effective immediately."
        for e in extract_dates(text):
            assert e.start >= 0
            assert e.end <= len(text)
            assert e.start < e.end


# ---------------------------------------------------------------------------
# extract_monetary
# ---------------------------------------------------------------------------


class TestExtractMonetary:
    def test_dollar_amount(self):
        result = extract_monetary("The fee is $10,000.00 per month.")
        assert any("$10,000.00" in e.text for e in result)

    def test_pound_amount(self):
        result = extract_monetary("Damages of \u00a3250,000 were awarded.")
        assert any("\u00a3250,000" in e.text for e in result)

    def test_euro_amount(self):
        result = extract_monetary("A fine of \u20ac500,000 was imposed.")
        assert any("\u20ac500,000" in e.text for e in result)

    def test_iso_code_prefix(self):
        result = extract_monetary("Payment of USD 75,000 is due.")
        assert any("USD 75,000" in e.text for e in result)

    def test_gbp_iso_code(self):
        result = extract_monetary("Settlement amount: GBP 100,000.")
        assert any("GBP 100,000" in e.text for e in result)

    def test_suffix_dollars(self):
        result = extract_monetary("Compensation of 1,500,000 dollars was agreed.")
        assert len(result) >= 1

    def test_million_suffix_with_symbol(self):
        result = extract_monetary("The fund is valued at $1.5 million.")
        assert len(result) >= 1

    def test_multiple_amounts(self):
        text = "Pay $1,000 upfront and \u00a32,000 on completion."
        result = extract_monetary(text)
        assert len(result) >= 2

    def test_empty_text_returns_empty(self):
        assert extract_monetary("") == []

    def test_entity_type_is_monetary(self):
        for e in extract_monetary("Invoice total: $99.99."):
            assert e.entity_type == EntityType.MONETARY

    def test_no_false_positive_plain_numbers(self):
        text = "There were 500 attendees at the conference."
        result = extract_monetary(text)
        for e in result:
            has_currency = any(
                sym in e.text for sym in ["$", "\u00a3", "\u20ac", "\u00a5", "USD", "GBP", "EUR"]
            ) or any(word in e.text.lower() for word in ["dollar", "pound", "euro"])
            assert has_currency

    def test_span_within_text(self):
        text = "Agreed price is $5,000.00 inclusive of tax."
        for e in extract_monetary(text):
            assert e.start >= 0
            assert e.end <= len(text)


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_simple_party_v_party(self):
        result = extract_citations("See Smith v. Jones for the applicable principle.")
        assert len(result) >= 1

    def test_neutral_citation(self):
        result = extract_citations("As decided in [2021] UKSC 45.")
        assert any("[2021] UKSC 45" in e.text for e in result)

    def test_neutral_citation_with_court_division(self):
        result = extract_citations("Followed in [2019] EWCA Civ 100.")
        assert len(result) >= 1

    def test_citation_with_year_in_brackets(self):
        result = extract_citations("Brown v. Board [1954] was landmark.")
        assert len(result) >= 1

    def test_us_code_citation(self):
        result = extract_citations("Violated 42 U.S.C. § 1983 rights.")
        assert len(result) >= 1

    def test_empty_text_returns_empty(self):
        assert extract_citations("") == []

    def test_entity_type_is_citation(self):
        for e in extract_citations("Re: [2020] EWHC 1234."):
            assert e.entity_type == EntityType.CITATION

    def test_citation_spans_are_valid(self):
        text = "Refer to [2022] EWCA Civ 567 for context."
        for e in extract_citations(text):
            assert e.start >= 0
            assert e.end <= len(text)
            assert e.start < e.end

    def test_confidence_set(self):
        result = extract_citations("[2023] UKSC 10 is the leading authority.")
        for e in result:
            assert 0.0 < e.confidence <= 1.0


# ---------------------------------------------------------------------------
# extract_statutes
# ---------------------------------------------------------------------------


class TestExtractStatutes:
    def test_act_with_year(self):
        result = extract_statutes("Subject to the Companies Act 2006.")
        assert any("Companies Act 2006" in e.text for e in result)

    def test_act_of_year(self):
        result = extract_statutes("Prohibited by the Civil Rights Act of 1964.")
        assert len(result) >= 1

    def test_eu_regulation(self):
        result = extract_statutes("Governed by Regulation (EU) 2016/679.")
        assert len(result) >= 1

    def test_us_code_section(self):
        result = extract_statutes("Under 18 U.S.C. § 1030 it is an offence.")
        assert len(result) >= 1

    def test_section_reference(self):
        result = extract_statutes("Pursuant to Section 12(3) of this Agreement.")
        assert len(result) >= 1

    def test_article_reference(self):
        result = extract_statutes("As required by Article 3 of the Convention.")
        assert len(result) >= 1

    def test_empty_text_returns_empty(self):
        assert extract_statutes("") == []

    def test_entity_type_is_statute(self):
        for e in extract_statutes("The Employment Rights Act 1996 applies."):
            assert e.entity_type == EntityType.STATUTE

    def test_span_within_text(self):
        text = "See the Consumer Rights Act 2015 for remedies."
        for e in extract_statutes(text):
            assert e.start >= 0
            assert e.end <= len(text)


# ---------------------------------------------------------------------------
# extract_organizations
# ---------------------------------------------------------------------------


class TestExtractOrganizations:
    def test_corporation_suffix(self):
        result = extract_organizations("Acme Corporation entered into the agreement.")
        assert any("Acme Corporation" in e.text for e in result)

    def test_llp_suffix(self):
        result = extract_organizations("Represented by Smith Jones LLP.")
        assert any("LLP" in e.text for e in result)

    def test_inc_suffix(self):
        result = extract_organizations("TechStart Inc. filed the complaint.")
        assert len(result) >= 1

    def test_authority_suffix(self):
        result = extract_organizations("The Financial Conduct Authority issued guidance.")
        assert any("Authority" in e.text for e in result)

    def test_limited_suffix(self):
        result = extract_organizations("GlobalTech Limited was acquired.")
        assert len(result) >= 1

    def test_empty_text_returns_empty(self):
        assert extract_organizations("") == []

    def test_entity_type_is_organization(self):
        for e in extract_organizations("Registered with Global Holdings Ltd."):
            assert e.entity_type == EntityType.ORGANIZATION


# ---------------------------------------------------------------------------
# extract_persons
# ---------------------------------------------------------------------------


class TestExtractPersons:
    def test_mr_title(self):
        result = extract_persons("Mr. John Smith signed the deed.")
        assert any("John Smith" in e.text for e in result)

    def test_dr_title(self):
        result = extract_persons("Dr. Emily Carter provided expert testimony.")
        assert any("Emily Carter" in e.text for e in result)

    def test_judge_title(self):
        result = extract_persons("Justice Williams delivered the judgment.")
        assert any("Williams" in e.text for e in result)

    def test_mrs_title(self):
        result = extract_persons("Mrs. Anne Brown was the respondent.")
        assert len(result) >= 1

    def test_professor_title(self):
        result = extract_persons("Prof. Alan Turing testified as an expert.")
        assert len(result) >= 1

    def test_empty_text_returns_empty(self):
        assert extract_persons("") == []

    def test_entity_type_is_person(self):
        for e in extract_persons("Prof. Alan Turing was cited."):
            assert e.entity_type == EntityType.PERSON

    def test_span_within_text(self):
        text = "Sir Robert Wilson chaired the tribunal."
        for e in extract_persons(text):
            assert e.start >= 0
            assert e.end <= len(text)


# ---------------------------------------------------------------------------
# extract_entities (combined)
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_returns_multiple_types(self):
        text = (
            "On 15 March 2023, Acme Corporation agreed to pay $50,000 "
            "to Mr. John Smith under the Employment Rights Act 1996."
        )
        result = extract_entities(text)
        types_found = {e.entity_type for e in result}
        assert EntityType.DATE in types_found
        assert EntityType.MONETARY in types_found

    def test_empty_text_returns_empty(self):
        assert extract_entities("") == []

    def test_sorted_by_start_offset(self):
        text = "On 2024-01-01, pay $500 to Dr. Jane Doe."
        result = extract_entities(text)
        starts = [e.start for e in result]
        assert starts == sorted(starts)

    def test_no_duplicate_spans(self):
        text = "See [2020] EWCA Civ 100. Pay $1,000 by 31 December 2024."
        result = extract_entities(text)
        spans = [(e.start, e.end) for e in result]
        assert len(spans) == len(set(spans))

    def test_to_dict_on_all_entities(self):
        text = "Mr. Smith paid $500 on 1 Jan 2024."
        result = extract_entities(text)
        for e in result:
            d = e.to_dict()
            assert "entity_type" in d
            assert "text" in d
            assert "start" in d
            assert "end" in d

    def test_all_confidences_valid(self):
        text = (
            "On 2024-01-15, USD 10,000 was paid to Smith v. Jones "
            "under the Companies Act 2006 by Acme Ltd. to Mr. Brown."
        )
        result = extract_entities(text)
        for e in result:
            assert 0.0 <= e.confidence <= 1.0

    def test_entity_texts_match_source(self):
        text = "Judgment dated 12 February 2022 awarded EUR 75,000."
        result = extract_entities(text)
        for e in result:
            # The entity text must appear in the source at the reported offsets
            assert text[e.start : e.end].strip() == e.text

    def test_whitespace_only_returns_empty(self):
        assert extract_entities("   \n\t  ") == []

    def test_mixed_citation_and_statute(self):
        text = "Smith v. Jones [2019] UKSC 5 interpreted the Companies Act 2006."
        result = extract_entities(text)
        types_found = {e.entity_type for e in result}
        assert EntityType.CITATION in types_found or EntityType.STATUTE in types_found
