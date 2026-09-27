import json
import sqlite3
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import Settings
from app.workflow import (
    Draft,
    build_workflow,
    capped_contribution,
    named_countries,
    numeric_values,
    parse_query,
    validate_draft,
)

AVAILABLE = {("Canada", 2024), ("Canada", 2025), ("Germany", 2024), ("Germany", 2025)}
EVIDENCE = dict(
    citation_id="S1",
    chunk_id="abc",
    document_id="canada-2025",
    document_name="Canada TaxBEN 2025",
    country="Canada",
    policy_year=2025,
    page_number=32,
    source="https://www.oecd.org/source.pdf",
    text="Employees outside Quebec contribute 1.64% on annual earnings up to CAD 65 700.",
)
DRAFT = dict(
    claims=[
        dict(
            text="Employees outside Quebec contribute 1.64% on earnings up to CAD 65 700.",
            citation_ids=["S1"],
        )
    ],
)


def test_clarification_preserves_income_and_year():
    state = parse_query(
        {"message": "I earn 80,000 a year. What would my tax situation be in 2025?"}, AVAILABLE
    )
    assert state["status"] == "needs_clarification"
    assert state["answer"] == "Which country are you asking about?"
    assert state["numeric_values"] == ["80000"]
    original = state["original_query"]
    state = parse_query({**state, "message": "Canada"}, AVAILABLE)
    assert state["original_query"] == original
    assert state["numeric_values"] == ["80000"]
    assert state["scopes"] == [("Canada", 2025)]
    assert state["status"] == "ready"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Compare Canada and Germany in 2025", {("Canada", 2025), ("Germany", 2025)}),
        ("How did Canadian policy change from 2024 to 2025?", {("Canada", 2024), ("Canada", 2025)}),
        ("Compare Canada 2024 with Germany 2025", {("Canada", 2024), ("Germany", 2025)}),
    ],
)
def test_comparison_scopes(query, expected):
    assert set(parse_query({"message": query}, AVAILABLE)["scopes"]) == expected


def test_new_question_clears_previous_scope_but_followup_retains_it():
    state = parse_query({"message": "Canada income tax in 2025"}, AVAILABLE)
    followup = parse_query({**state, "message": "And Germany?"}, AVAILABLE)
    assert followup["scopes"] == [("Germany", 2025)]
    country_only = parse_query({**state, "status": "answered", "message": "Germany"}, AVAILABLE)
    assert country_only["scopes"] == [("Germany", 2025)]
    assert country_only["original_query"] == state["original_query"]
    new = parse_query({**state, "message": "What are the child benefit rules?"}, AVAILABLE)
    assert new["status"] == "needs_clarification"
    assert not new["scopes"]
    foreign = parse_query({**state, "message": "What about France?"}, AVAILABLE)
    assert foreign["status"] == "insufficient_evidence"
    assert foreign["countries"] == ["France"]


def test_unknown_year_does_not_fall_back():
    state = parse_query({"message": "Canada tax in 2030"}, AVAILABLE)
    assert state["status"] == "insufficient_evidence"
    assert state["scopes"] == [("Canada", 2030)]


def test_literal_numeric_values():
    assert numeric_values("CAD 80k in 2025") == ["80000"]
    assert numeric_values("income 80,000.50") == ["80000.50"]
    assert numeric_values("income -100.") == ["-100"]


def test_unknown_followup_does_not_reuse_country():
    state = parse_query({"message": "Canada tax in 2025"}, AVAILABLE)
    unknown = parse_query({**state, "message": "What about Nigeria?"}, AVAILABLE)
    assert unknown["status"] == "needs_clarification"
    assert unknown["countries"] == []
    assert unknown["requested_years"] == [2025]


@pytest.fixture
def answer_state():
    return {
        "scopes": [("Canada", 2025)],
        "countries": ["Canada"],
        "requested_years": [2025],
        "retrieved_evidence": [EVIDENCE.copy()],
        "numeric_values": ["80000"],
        "currency": "CAD",
        "current_query": (
            "Calculate my annual EI contribution outside Quebec, Canada in 2025 on CAD 80000."
        ),
    }


def test_citation_provenance_and_numeric_support(answer_state):
    draft = Draft.model_validate(DRAFT)
    assert validate_draft(draft, answer_state)
    draft.claims[0].text = "The contribution is 99%."
    assert not validate_draft(draft, answer_state)


@pytest.mark.parametrize(
    "field,value", [("country", "Germany"), ("policy_year", 2024), ("citation_id", "S9")]
)
def test_wrong_scope_and_unknown_citations_rejected(answer_state, field, value):
    answer_state["retrieved_evidence"][0][field] = value
    assert not validate_draft(Draft.model_validate(DRAFT), answer_state)


def test_claim_without_citation_rejected(answer_state):
    draft = Draft.model_validate(DRAFT)
    draft.claims[0].citation_ids = ["unretrieved-source"]
    assert not validate_draft(draft, answer_state)


def test_comparison_needs_citations_from_both_scopes(answer_state):
    answer_state["scopes"].append(("Canada", 2024))
    assert not validate_draft(Draft.model_validate(DRAFT), answer_state)


def test_overview_does_not_satisfy_exact_liability_request(answer_state):
    answer_state["current_query"] = "What is my exact total income tax liability in Canada in 2025?"
    draft = Draft.model_validate({**DRAFT, "limitation": "personal_liability"})
    assert not validate_draft(draft, answer_state)


def test_calculation_is_decimal_and_capped(answer_state):
    assert capped_contribution(Decimal("80000"), Decimal("65700"), Decimal("1.64")) == Decimal(
        "1077.48"
    )
    assert capped_contribution(Decimal("20000"), Decimal("65700"), Decimal("1.64")) == Decimal(
        "328.00"
    )
    assert capped_contribution(Decimal("0"), Decimal("65700"), Decimal("1.64")) == 0
    for income in [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")]:
        with pytest.raises(ValueError):
            capped_contribution(income, Decimal("65700"), Decimal("1.64"))
    draft = Draft.model_validate(
        {
            **DRAFT,
            "calculation": dict(
                label="EI premium",
                rate_percent="1.64",
                annual_ceiling="65700",
                currency="CAD",
                citation_ids=["S1"],
            ),
        }
    )
    assert validate_draft(draft, answer_state)
    answer_state["currency"] = None
    assert not validate_draft(draft, answer_state)


def test_checkpoint_restores_clarification_after_reopen(tmp_path, monkeypatch):
    class Retriever:
        groups = {scope: [] for scope in AVAILABLE}

        def search(self, query, scopes):
            assert "80000" in query
            assert scopes == {("Canada", 2025)}
            return [], {"timing_ms": {}}

    path = str(tmp_path / "sessions.sqlite")
    config = {"configurable": {"thread_id": "session-a"}}
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_workflow(Retriever(), Settings(), saver)
        first = graph.invoke({"message": "I earn 80000. What tax applies in 2025?"}, config)
        assert first["status"] == "needs_clarification"
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_workflow(Retriever(), Settings(), saver)
        second = graph.invoke({"message": "Canada"}, config)
        assert second["numeric_values"] == ["80000"]
        assert second["status"] == "insufficient_evidence"
        separate = graph.invoke({"message": "Canada"}, {"configurable": {"thread_id": "session-b"}})
        assert separate["status"] == "needs_clarification"


@pytest.mark.parametrize("supported,status", [(True, "answered"), (False, "insufficient_evidence")])
def test_semantic_guard_fails_closed(monkeypatch, supported, status):
    class Retriever:
        groups = {scope: [] for scope in AVAILABLE}

        def search(self, query, scopes):
            return [EVIDENCE.copy()], {"timing_ms": {}}

    class Ollama:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            value = (
                {"supported": supported} if "supported" in kwargs["format"]["properties"] else DRAFT
            )
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))

    monkeypatch.setattr("app.workflow.Client", Ollama)
    with SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)).conn as conn:
        graph = build_workflow(Retriever(), Settings(), SqliteSaver(conn))
        result = graph.invoke(
            {"message": "Canada tax in 2025"}, {"configurable": {"thread_id": "test"}}
        )
        assert result["status"] == status
        assert bool(result["citations"]) == supported


def test_country_names_use_longest_location_match():
    assert named_countries("England, Wales, and Northern Ireland") == ["United Kingdom"]
    assert named_countries("Northern Ireland and Ireland") == ["United Kingdom", "Ireland"]
    assert named_countries("U.S. and U.K.") == ["United States", "United Kingdom"]
    assert named_countries("Tell us about taxes") == []
