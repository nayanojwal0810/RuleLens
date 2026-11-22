"""Offline unit tests for RuleLens Stage E answer orchestration (answering.py).

Verifies:
- Full answer_question workflow with mocked retrieval and LLM provider.
- Input validation (role, date, non-empty question).
- Evidence integrity validation against canonical corpus records.
- Policy metadata-based restricted document detection (no hardcoded keywords).
- Conservative evidence sufficiency and safe refusals.
- Mapped operational error handling (no secrets leaked, no raw stack traces).
- CLI entrypoint execution and nonzero exit on operational error.
"""

from __future__ import annotations

import datetime
import io
import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from rulelens.answering import (
    GenerationResult,
    answer_question,
    main as answering_main,
)
from rulelens.llm import (
    AnswerResponse,
    AnswerStatus,
    CitationValidationError,
    GroqClientWrapper,
    SanitizedProviderError,
)
from rulelens.retrieval import RetrievalResult, load_validated_corpus


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
def canonical_corpus(repo_root: Path) -> List[Dict[str, Any]]:
    return load_validated_corpus(repo_root)


@pytest.fixture
def canonical_chunks_by_id(canonical_corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["chunk_id"]: c for c in canonical_corpus}


@pytest.fixture
def mock_retriever(canonical_corpus: List[Dict[str, Any]], canonical_chunks_by_id: Dict[str, Dict[str, Any]]):
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "4f2485c2ea0a10148532e154f308554f44c27ae8b0fbdd7d174dff83f6dab9b3"}
    # Use real BM25 index on canonical chunks for search
    from rulelens.retrieval import BM25Index
    retriever.get_bm25_index.side_effect = lambda chunks: BM25Index(chunks)
    return retriever


@pytest.fixture
def sample_retrieval_result(canonical_chunks_by_id: Dict[str, Dict[str, Any]]) -> RetrievalResult:
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C03"]
    return RetrievalResult(
        chunk_id=c["chunk_id"],
        text=c["text"],
        doc_id=c["doc_id"],
        clause_id=c["clause_id"],
        title=c["title"],
        department=c["department"],
        version=c["version"],
        source_file=c["source_file"],
        source_page=int(c["source_page"]),
        effective_from=c["effective_from"],
        effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
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
        question="What is the daily meal allowance under Travel Policy v2?",
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
        question="What is the reimbursement policy for space travel?",
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


def test_answer_question_restricted_policy_detection_without_keywords(mock_retriever):
    """Bug regression: Restriction must be determined by policy metadata and permissions, not hardcoded keywords."""
    mock_llm = MagicMock(spec=GroqClientWrapper)

    # Query targeting executive travel exceptions (DOC-TRV-004), restricted for employee
    result = answer_question(
        question="What executive approval route is required when extraordinary travel exception expenses exceed INR 100,000 under the travel exceptions policy?",
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


def test_check_restricted_policy_access_never_creates_bm25_or_reads_chunk_text(mock_retriever):
    """Security verification: check_restricted_policy_access must NEVER instantiate BM25 or index chunk text."""
    from rulelens.answering import check_restricted_policy_access
    mock_retriever.get_bm25_index.reset_mock()

    is_restr = check_restricted_policy_access(
        retriever=mock_retriever,
        query="What is the travel exceptions policy?",
        role="employee",
        query_date=datetime.date(2026, 11, 22),
    )
    assert is_restr is True
    # BM25 index over chunk bodies must NEVER be called
    mock_retriever.get_bm25_index.assert_not_called()


def test_check_restricted_policy_access_across_roles_and_dates(mock_retriever):
    """Security verification: Verify restriction detection across all 6 roles and dates."""
    from rulelens.answering import check_restricted_policy_access

    # DOC-TRV-004: Travel Exceptions (allowed: auditor, finance, manager; NOT employee, hr, security)
    q_trv004 = "What are the rules under the travel exceptions policy?"
    curr_date = datetime.date(2026, 11, 22)
    assert check_restricted_policy_access(mock_retriever, q_trv004, "employee", curr_date) is True
    assert check_restricted_policy_access(mock_retriever, q_trv004, "hr", curr_date) is True
    assert check_restricted_policy_access(mock_retriever, q_trv004, "security", curr_date) is True
    assert check_restricted_policy_access(mock_retriever, q_trv004, "manager", curr_date) is False
    assert check_restricted_policy_access(mock_retriever, q_trv004, "finance", curr_date) is False
    assert check_restricted_policy_access(mock_retriever, q_trv004, "auditor", curr_date) is False

    # DOC-PPO-004: Employee Transfers (allowed: hr, manager; NOT employee, finance, auditor, security)
    q_ppo004 = "What is the policy for employee transfers?"
    assert check_restricted_policy_access(mock_retriever, q_ppo004, "employee", curr_date) is True
    assert check_restricted_policy_access(mock_retriever, q_ppo004, "auditor", curr_date) is True
    assert check_restricted_policy_access(mock_retriever, q_ppo004, "hr", curr_date) is False
    assert check_restricted_policy_access(mock_retriever, q_ppo004, "manager", curr_date) is False


def test_authorized_document_not_wrongly_blocked_by_restricted_detection(mock_retriever, sample_retrieval_result):
    """Security verification: An authorized query must not be blocked because a restricted policy shares words."""
    from rulelens.answering import check_restricted_policy_access
    mock_retriever.search.return_value = [sample_retrieval_result]
    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="Meal allowance is INR 2,000.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C03"],
            referenced_rule_ids=[],
            reason="Grounded in evidence.",
        ),
        {"total_tokens": 100},
    )

    # Travel Policy v2 is authorized for employee
    is_restr = check_restricted_policy_access(
        retriever=mock_retriever,
        query="What is the daily meal allowance under Travel Policy v2?",
        role="employee",
        query_date=datetime.date(2026, 11, 22),
    )
    assert is_restr is False

    result = answer_question(
        question="What is the daily meal allowance under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )
    assert result.status == "answered"


def test_answer_question_rule_unit_error_fails_safely(mock_retriever, canonical_chunks_by_id):
    """Security verification: Incompatible rule unit input must fail safely with insufficient without calling Groq."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_result = RetrievalResult(
        chunk_id=c["chunk_id"],
        text=c["text"],
        doc_id=c["doc_id"],
        clause_id=c["clause_id"],
        title=c["title"],
        department=c["department"],
        version=c["version"],
        source_file=c["source_file"],
        source_page=int(c["source_page"]),
        effective_from=c["effective_from"],
        effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_result]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    # Mock extract_conditions_from_query to simulate an incompatible unit (e.g. days instead of INR for expense amount)
    with patch("rulelens.answering.extract_conditions_from_query") as mock_extract:
        mock_extract.return_value = [{"field": "expense_amount", "value": 50000, "unit": "days"}]
        result = answer_question(
            question="What approval is required for travel expenses exceeding 50000 days under Travel Policy v2?",
            role="employee",
            query_date="2026-11-22",
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

    assert result.status == "insufficient"
    assert result.error_category == "invalid_unit"
    assert "rule evaluation error" in result.reason.lower()
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_rejects_forged_evidence_metadata(mock_retriever, sample_retrieval_result):
    """Bug regression: Adversarial candidate with correct ID but altered permissions or text must be rejected."""
    forged_chunk = RetrievalResult(
        chunk_id=sample_retrieval_result.chunk_id,
        text="FORGED TEXT: All expenses are 100% approved without limits.",
        doc_id=sample_retrieval_result.doc_id,
        clause_id=sample_retrieval_result.clause_id,
        title=sample_retrieval_result.title,
        department=sample_retrieval_result.department,
        version=sample_retrieval_result.version,
        source_file=sample_retrieval_result.source_file,
        source_page=sample_retrieval_result.source_page,
        effective_from=sample_retrieval_result.effective_from,
        effective_to=sample_retrieval_result.effective_to,
        allowed_roles=sample_retrieval_result.allowed_roles,
    )
    mock_retriever.search.return_value = [forged_chunk]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="What is the meal allowance cap?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "operational_error"
    assert result.error_category == "service_unavailable"
    assert "integrity" in result.error.lower()
    # Provider must NEVER be called with forged evidence
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_unsupported_query_safely_refused(mock_retriever, sample_retrieval_result):
    """Bug regression: Questions asking for unsupported topics (e.g. pets) must safely refuse with insufficient."""
    mock_retriever.search.return_value = [sample_retrieval_result]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="How much reimbursement can an employee claim for taking personal domestic pets on business travel?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "insufficient"
    assert "corporate policies do not contain sufficient evidence" in result.answer
    assert result.citations == []
    # Groq must NOT be called for unsupported topic with zero entity overlap
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_handles_provider_error_as_operational_error(mock_retriever, sample_retrieval_result):
    """Bug regression: Provider errors must NOT be classified as insufficient evidence."""
    mock_retriever.search.return_value = [sample_retrieval_result]

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.side_effect = SanitizedProviderError(
        category="rate_limited",
        safe_message="Generation service is currently rate limited. Please retry later.",
    )

    result = answer_question(
        question="What is the meal allowance under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    # Must be operational_error, NOT insufficient!
    assert result.status == "operational_error"
    assert result.error_category == "rate_limited"
    assert "rate limited" in result.error
    assert result.citations == []


def test_answer_question_leaks_no_secrets_in_errors(mock_retriever, sample_retrieval_result):
    """Bug regression: Exceptions with fake API keys or secret strings must be completely sanitized."""
    mock_retriever.search.return_value = [sample_retrieval_result]

    secret_leak_exc = RuntimeError("Fatal crash in request: Authorization: Bearer gsk_SUPER_SECRET_KEY_12345")
    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.side_effect = secret_leak_exc

    result = answer_question(
        question="What is the meal allowance cap under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "operational_error"
    # Ensure no part of the secret string leaked into any result field
    res_json = json.dumps(result.to_dict())
    assert "SUPER_SECRET" not in res_json
    assert "gsk_" not in res_json
    assert "Bearer" not in res_json


# ==============================================================================
# 3. CLI Smoke Test Execution
# ==============================================================================


def test_cli_smoke_test_execution(monkeypatch, sample_retrieval_result, canonical_corpus, canonical_chunks_by_id):
    """CLI answering entry point runs cleanly and exits with code 0 on valid response."""
    with patch("rulelens.answering.HybridRetriever") as mock_retriever_cls, \
         patch("rulelens.answering.GroqClientWrapper") as mock_llm_cls:

        from rulelens.retrieval import BM25Index
        retriever_inst = MagicMock()
        retriever_inst.chunks = canonical_corpus
        retriever_inst.chunks_by_id = canonical_chunks_by_id
        retriever_inst.corpus_manifest = {"processed_fingerprint": "test_fp"}
        retriever_inst.get_bm25_index.side_effect = lambda c_list: BM25Index(c_list)
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
            "--question", "What is the daily meal allowance under Travel Policy v2?",
            "--role", "employee",
            "--date", "2026-11-22",
            "--backend", "local",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Must run without throwing unhandled exceptions
        answering_main()


def test_cli_exits_nonzero_on_operational_error(monkeypatch, sample_retrieval_result, canonical_corpus, canonical_chunks_by_id):
    """CLI answering entry point must exit with code 1 when an operational error occurs."""
    with patch("rulelens.answering.HybridRetriever") as mock_retriever_cls, \
         patch("rulelens.answering.GroqClientWrapper") as mock_llm_cls:

        from rulelens.retrieval import BM25Index
        retriever_inst = MagicMock()
        retriever_inst.chunks = canonical_corpus
        retriever_inst.chunks_by_id = canonical_chunks_by_id
        retriever_inst.corpus_manifest = {"processed_fingerprint": "test_fp"}
        retriever_inst.get_bm25_index.side_effect = lambda c_list: BM25Index(c_list)
        retriever_inst.search.return_value = [sample_retrieval_result]
        mock_retriever_cls.return_value = retriever_inst

        llm_inst = MagicMock()
        llm_inst.generate_answer.side_effect = SanitizedProviderError(
            category="authentication_error",
            safe_message="Provider authentication failed. Please verify credentials.",
        )
        mock_llm_cls.return_value = llm_inst

        test_args = [
            "prog",
            "--question", "What is the daily meal allowance under Travel Policy v2?",
            "--role", "employee",
            "--date", "2026-11-22",
            "--backend", "local",
        ]
        monkeypatch.setattr("sys.argv", test_args)

        with pytest.raises(SystemExit) as excinfo:
            answering_main()
        assert excinfo.value.code == 1


# ==============================================================================
# 4. Strict Evidence Limits, Forged Department, and Ambiguous Input Tests
# ==============================================================================


def test_answer_question_rejects_invalid_top_k(mock_retriever):
    """top_k must be strictly validated as an int between 1 and 4 inclusive."""
    for invalid_val in (0, 5, True, False, "4", None):
        with pytest.raises(ValueError, match="Invalid top_k"):
            answer_question(
                question="What is the meal allowance?",
                role="employee",
                top_k=invalid_val,  # type: ignore
                retriever=mock_retriever,
            )


def test_answer_question_rejects_forged_department(mock_retriever, sample_retrieval_result):
    """validate_candidate_evidence must reject candidate whose department does not match canonical."""
    forged_dept_chunk = RetrievalResult(
        chunk_id=sample_retrieval_result.chunk_id,
        text=sample_retrieval_result.text,
        doc_id=sample_retrieval_result.doc_id,
        clause_id=sample_retrieval_result.clause_id,
        title=sample_retrieval_result.title,
        department="Finance",  # Canonical is "Travel"
        version=sample_retrieval_result.version,
        source_file=sample_retrieval_result.source_file,
        source_page=sample_retrieval_result.source_page,
        effective_from=sample_retrieval_result.effective_from,
        effective_to=sample_retrieval_result.effective_to,
        allowed_roles=sample_retrieval_result.allowed_roles,
    )
    mock_retriever.search.return_value = [forged_dept_chunk]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="What is the meal allowance?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "operational_error"
    assert result.error_category == "service_unavailable"
    assert "department" in result.error.lower()
    assert result.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_ambiguous_amount_safely_refuses(mock_retriever, canonical_chunks_by_id):
    """When query specifies conflicting/ambiguous amounts, safe refusal status insufficient is returned."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"],
        text=c["text"],
        doc_id=c["doc_id"],
        clause_id=c["clause_id"],
        title=c["title"],
        department=c["department"],
        version=c["version"],
        source_file=c["source_file"],
        source_page=int(c["source_page"]),
        effective_from=c["effective_from"],
        effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="What approval is required for travel expenses of INR 40,000 or INR 60,000 under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "insufficient"
    assert result.error_category == "ambiguous_condition"
    assert result.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_answer_question_retrieval_failure_returns_operational_error(mock_retriever):
    """When retriever.search raises an unexpected exception, answer_question returns operational_error."""
    mock_retriever.search.side_effect = RuntimeError("Vector index disk I/O error")
    mock_llm = MagicMock(spec=GroqClientWrapper)

    result = answer_question(
        question="What is the meal allowance?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert result.status == "operational_error"
    assert result.error_category == "service_unavailable"
    assert result.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()
