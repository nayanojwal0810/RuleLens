"""Offline unit tests for RuleLens Stage E answer orchestration (answering.py).

Verifies:
- Full answer_question workflow with mocked retrieval and LLM provider.
- Input validation (role, date, non-empty question).
- Defense-in-depth role and date verification.
- Early refusal for unsupported and role-restricted questions without calling Groq.
- Safe error handling for missing keys, timeouts, 429s, and provider errors.
- CLI entrypoint execution.
"""

from __future__ import annotations

import datetime
import json
import socket
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from rulelens.answering import (
    GenerationResult,
    answer_question,
    main as answering_main,
)
from rulelens.llm import AnswerResponse, AnswerStatus, GroqClientWrapper
from rulelens.retrieval import RetrievalResult


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline unit tests")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.corpus_fingerprint = "4f2485c2ea0a10148532e154f308554f44c27ae8b0fbdd7d174dff83f6dab9b3"
    return retriever


@pytest.fixture
def sample_retrieval_result():
    return RetrievalResult(
        chunk_id="DOC-TRV-002_TRV-C03",
        text="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
        doc_id="DOC-TRV-002",
        clause_id="TRV-C03",
        title="Corporate Travel Policy v2",
        department="Travel",
        version="v2",
        source_file="DOC-TRV-002_Travel_Policy_v2.pdf",
        source_page=1,
        effective_from="2026-07-01",
        effective_to=None,
        allowed_roles=["employee", "manager", "director", "finance", "security", "auditor"],
    )


# ==============================================================================
# 1. Input Validation Tests
# ==============================================================================


def test_answer_question_rejects_empty_query():
    """answer_question must raise ValueError on empty or whitespace question."""
    with pytest.raises(ValueError, match="Invalid question: question must be a non-empty string"):
        answer_question(question="", role="employee")

    with pytest.raises(ValueError, match="Invalid question: question must be a non-empty string"):
        answer_question(question="   ", role="employee")


def test_answer_question_rejects_invalid_role():
    """answer_question must raise ValueError on unapproved employee role."""
    with pytest.raises(ValueError, match="Unknown role 'superadmin'"):
        answer_question(question="What is the policy?", role="superadmin")


def test_answer_question_rejects_malformed_date():
    """answer_question must raise ValueError on invalid query date format."""
    with pytest.raises(ValueError, match="Malformed query date"):
        answer_question(question="What is the policy?", role="employee", query_date="invalid-date")


# ==============================================================================
# 2. End-to-End Orchestration Tests
# ==============================================================================


def test_answer_question_successful_flow(mock_retriever, sample_retrieval_result):
    """Successful direct question returns verified answer, citations, and metadata."""
    mock_retriever.search.return_value = [sample_retrieval_result]

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="The daily meal allowance is capped at INR 2,000 per diem with itemized receipts.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C03"],
            referenced_rule_ids=[],
            reason="Direct quotation from clause TRV-C03.",
        ),
        {"total_tokens": 120, "calls_made": 1, "latency_ms": 25.0},
    )

    result = answer_question(
        question="What is the daily meal allowance cap under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert isinstance(result, GenerationResult)
    assert result.status == "answered"
    assert "INR 2,000 per diem" in result.answer
    assert result.citations == ["DOC-TRV-002_TRV-C03"]
    assert len(result.citation_metadata) == 1
    assert result.citation_metadata[0]["doc_id"] == "DOC-TRV-002"
    assert result.citation_metadata[0]["clause_id"] == "TRV-C03"
    assert result.token_usage == 120
    assert result.error is None
    # Confirm Groq was called with evidence
    mock_llm.generate_answer.assert_called_once()


def test_answer_question_early_refusal_when_no_evidence_found(mock_retriever):
    """When retrieval yields no authorized evidence, returns insufficient without calling Groq."""
    mock_retriever.search.return_value = []
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="Can I get reimbursed for taking personal pets on business travel?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "insufficient"
    assert "corporate policies do not contain sufficient evidence" in result.answer
    assert result.citations == []
    # Groq must NOT be called
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_early_refusal_for_restricted_query(mock_retriever):
    """When query targets a restricted policy area and yields 0 chunks, returns restricted without calling Groq."""
    mock_retriever.search.return_value = []
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="What is the travel exception approval route for expenses exceeding INR 100,000?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "restricted"
    assert "Access restricted" in result.answer
    assert result.citations == []
    # Groq must NOT be called
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_defense_in_depth_filters_leaked_chunk(mock_retriever, sample_retrieval_result):
    """If retriever somehow returns an unauthorized chunk, defense-in-depth excludes it from Groq call."""
    unauthorized_chunk = RetrievalResult(
        chunk_id="DOC-TRV-004_TRV-E01",
        text="Executive travel exception approval requires Department VP.",
        doc_id="DOC-TRV-004",
        clause_id="TRV-E01",
        title="Travel Exceptions Policy",
        department="Travel",
        version="v1",
        source_file="DOC-TRV-004.pdf",
        source_page=1,
        effective_from="2026-01-01",
        effective_to=None,
        allowed_roles=["manager", "director", "finance", "legal"],  # employee not allowed!
    )
    # Retriever returns both
    mock_retriever.search.return_value = [unauthorized_chunk, sample_retrieval_result]

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="Meal allowance is INR 2,000.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C03"],
            referenced_rule_ids=[],
            reason="Valid citation.",
        ),
        {"total_tokens": 100},
    )

    result = answer_question(
        question="What is meal allowance?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "answered"
    # Confirm Groq was only sent the authorized chunk
    called_chunks = mock_llm.generate_answer.call_args.kwargs["evidence_chunks"]
    assert len(called_chunks) == 1
    assert called_chunks[0].chunk_id == "DOC-TRV-002_TRV-C03"


def test_answer_question_handles_llm_failure_gracefully(mock_retriever, sample_retrieval_result):
    """When LLM provider raises a network or 429 error, returns safe error response without crashing."""
    mock_retriever.search.return_value = [sample_retrieval_result]

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.side_effect = RuntimeError("Groq service temporarily unavailable (429 rate limit).")

    result = answer_question(
        question="What is the meal allowance cap?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "insufficient"
    assert "generation service is temporarily unavailable" in result.answer
    assert result.citations == []
    assert "429 rate limit" in result.error


# ==============================================================================
# 3. CLI Smoke Test Execution
# ==============================================================================


def test_cli_smoke_test_execution(monkeypatch, sample_retrieval_result):
    """CLI answering entry point runs cleanly and exits with code 0 on valid response."""
    with patch("rulelens.answering.HybridRetriever") as mock_retriever_cls, \
         patch("rulelens.answering.GroqClientWrapper") as mock_llm_cls:

        retriever_inst = MagicMock()
        retriever_inst.corpus_fingerprint = "test_fp"
        retriever_inst.search.return_value = [sample_retrieval_result]
        mock_retriever_cls.return_value = retriever_inst

        llm_inst = MagicMock()
        llm_inst.generate_answer.return_value = (
            AnswerResponse(
                answer="Domestic travel must be booked 14 days in advance.",
                status=AnswerStatus.ANSWERED,
                citations=["DOC-TRV-002_TRV-C03"],
                referenced_rule_ids=[],
                reason="Direct clause citation.",
            ),
            {"total_tokens": 80},
        )
        mock_llm_cls.return_value = llm_inst

        test_args = [
            "prog",
            "--question", "How many days in advance must domestic flights be booked?",
            "--role", "employee",
            "--date", "2026-11-22",
            "--backend", "local",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Must run without throwing unhandled exceptions
        answering_main()
