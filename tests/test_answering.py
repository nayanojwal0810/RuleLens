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
from rulelens.retrieval import HybridRetriever, RetrievalResult, load_validated_corpus


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
    assert result.error_category in ("incompatible_unit", "invalid_unit")
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
    assert result.error_category == "evidence_integrity_error"
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
    assert result.error_category == "evidence_integrity_error"
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


def test_missing_credentials_and_rule_loading_errors_return_sanitized(mock_retriever, sample_retrieval_result, monkeypatch):
    """Test 9: Missing credentials and rule-loading errors return sanitized operational_error without exposing secrets or paths."""
    mock_retriever.search.return_value = [sample_retrieval_result]

    # Subtest A: Missing / invalid API credentials during client initialization
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("rulelens.answering.GroqClientWrapper", side_effect=ValueError("GROQ_API_KEY is not set or empty: gsk_secret1234567890")):
        res_auth = answer_question(
            question="What is the meal allowance?",
            role="employee",
            query_date="2026-11-22",
            retriever=mock_retriever,
            llm_client=None,
        )
        assert res_auth.status == "operational_error"
        assert res_auth.error_category in ("authentication_error", "service_unavailable")
        assert "gsk_" not in str(res_auth.error)
        assert "gsk_" not in str(res_auth.answer)
        assert res_auth.provider_calls_attempted == 0

    # Subtest B: Unexpected rule-loading infrastructure error
    with patch("rulelens.answering.resolve_rules_for_evidence", side_effect=RuntimeError("Corrupt CSV at /var/secrets/rules/private_rules.csv")):
        res_rule = answer_question(
            question="What is the meal allowance?",
            role="employee",
            query_date="2026-11-22",
            retriever=mock_retriever,
            llm_client=MagicMock(),
        )
        assert res_rule.status == "operational_error"
        assert res_rule.error_category == "rule_loading_error"
        assert "/var/secrets" not in str(res_rule.error)
        assert "/var/secrets" not in str(res_rule.answer)
        assert res_rule.provider_calls_attempted == 0


# ==============================================================================
# 4. Correction A: Restricted Topic Detection (SEC-009) & Diagnostic Coverage
# ==============================================================================


def test_sec_009_restricted_topic_detection_across_roles(canonical_corpus, canonical_chunks_by_id):
    """SEC-009 (storage drive sanitization) returns restricted for finance and employee, but permits security and auditor."""
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    retriever.search.return_value = []
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "What standards and physical methods must be used when sanitizing decommissioned storage drives before disposal?"

    # Role: finance (unauthorized -> restricted)
    res_fin = answer_question(
        question=q,
        role="finance",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res_fin.status == "restricted"
    assert res_fin.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()

    # Role: employee (unauthorized -> restricted)
    res_emp = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res_emp.status == "restricted"
    assert res_emp.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()

    # Role: security (authorized -> not restricted, proceeds to retrieval)
    res_sec = answer_question(
        question=q,
        role="security",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res_sec.status != "restricted"

    # Role: auditor (authorized -> not restricted, proceeds to retrieval)
    res_aud = answer_question(
        question=q,
        role="auditor",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res_aud.status != "restricted"


@pytest.mark.parametrize(
    "query",
    [
        "What standards and physical methods must be used when sanitizing decommissioned storage drives before disposal?",
        "Sanitizing decommissioned storage drives.",
        "Secure disposal of retired storage media.",
        "Drive sanitization.",
        "Disk wiping before disposal.",
        "Secure erasure of decommissioned disks.",
    ],
)
def test_drive_sanitization_paraphrases_detected_as_restricted(canonical_corpus, canonical_chunks_by_id, query: str):
    """Multiple paraphrases of drive sanitization are recognized and refused as restricted for unauthorized roles."""
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}

    res = answer_question(
        question=query,
        role="finance",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=MagicMock(),
    )
    assert res.status == "restricted"
    assert res.provider_calls_attempted == 0


@pytest.mark.parametrize(
    "query",
    [
        "What is the company storage policy?",
        "How is data storage managed across teams?",
        "General corporate policy overview.",
    ],
)
def test_generic_terms_do_not_trigger_restrictions(canonical_corpus, canonical_chunks_by_id, query: str):
    """Generic terms like 'storage', 'data', or 'policy' alone do not trigger restricted classification."""
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    retriever.search.return_value = []

    res = answer_question(
        question=query,
        role="finance",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=MagicMock(),
    )
    assert res.status != "restricted"


def test_all_five_restricted_diagnostics_refuse_correctly(repo_root: Path, canonical_corpus, canonical_chunks_by_id):
    """All 5 restricted diagnostic queries from dev_questions.json produce status 'restricted'."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    dev_questions = json.loads(dev_path.read_text(encoding="utf-8"))["questions"]
    q_lookup = {q["question_id"]: q for q in dev_questions}
    restricted_ids = ["TRV-009", "FIN-009", "PPO-009", "PRC-009", "SEC-009"]

    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    mock_llm = MagicMock(spec=GroqClientWrapper)

    for qid in restricted_ids:
        item = q_lookup[qid]
        res = answer_question(
            question=item["question"],
            role=item["role"],
            query_date=item["query_date"],
            retriever=retriever,
            llm_client=mock_llm,
        )
        assert res.status == "restricted", f"Expected restricted for {qid}: {item['question']}"
        assert res.provider_calls_attempted == 0
        assert res.citations == []
        mock_llm.generate_answer.assert_not_called()


def test_all_five_unsupported_diagnostics_refuse_as_insufficient(repo_root: Path):
    """All 5 unsupported diagnostic queries from dev_questions.json produce safe status 'insufficient' with real BM25 retrieval."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    dev_questions = json.loads(dev_path.read_text(encoding="utf-8"))["questions"]
    q_lookup = {q["question_id"]: q for q in dev_questions}
    unsupported_ids = ["TRV-010", "FIN-010", "PPO-010", "PRC-010", "SEC-010"]

    real_retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = MagicMock(spec=GroqClientWrapper)

    for qid in unsupported_ids:
        item = q_lookup[qid]
        res = answer_question(
            question=item["question"],
            role=item["role"],
            query_date=item["query_date"],
            retriever=real_retriever,
            mode="bm25",
            llm_client=mock_llm,
        )
        assert res.status == "insufficient", f"Expected insufficient for {qid}: {item['question']}"
        assert res.citations == []
        assert res.provider_calls_attempted == 0
        mock_llm.generate_answer.assert_not_called()


def test_adversarial_spa_treatment_not_approved_by_travel_threshold(canonical_corpus, canonical_chunks_by_id):
    """Adversarial query about personal spa treatment (INR 50,000) using actual Travel Policy v2 clause must NOT approve."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04 = RetrievalResult(
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
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    retriever.search.return_value = [c04]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = answer_question(
        question="Can employees expense personal spa treatments costing INR 50,000 during business trips?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert "approved" not in res.answer.lower() or "not contain sufficient evidence" in res.answer.lower()
    assert res.citations == []
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_personal_travel_not_approved_and_groq_not_called(canonical_corpus, canonical_chunks_by_id):
    """Query 'Can employees spend INR 50,000 on personal travel?' must NOT approve personal travel and must NOT call Groq."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04 = RetrievalResult(
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
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    retriever.search.return_value = [c04]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = answer_question(
        question="Can employees spend INR 50,000 on personal travel?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert "approved" not in res.answer.lower() or "not contain sufficient evidence" in res.answer.lower()
    assert res.citations == []
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_personal_travel_with_real_retriever_refuses_without_groq(repo_root):
    """End-to-end verification that 'Can employees spend INR 50,000 on personal travel?' with real retriever refuses without Groq."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    retriever = HybridRetriever(base_dir=repo_root)
    res = answer_question(
        question="Can employees spend INR 50,000 on personal travel?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        mode="bm25",
        llm_client=mock_llm,
        base_dir=repo_root,
    )
    assert res.status == "insufficient"
    assert "approved" not in res.answer.lower() or "not contain sufficient evidence" in res.answer.lower()
    assert res.citations == []
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_business_travel_with_real_retriever_approves_deterministically(repo_root):
    """End-to-end verification that legitimate business travel approval query succeeds deterministically with real retriever."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    retriever = HybridRetriever(base_dir=repo_root)
    res = answer_question(
        question="Can a direct manager approve an expense of INR 50,000 under Travel Policy v2?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        mode="bm25",
        llm_client=mock_llm,
        base_dir=repo_root,
    )
    assert res.status == "answered"
    assert "Standard manager approval is permitted" in res.answer
    assert res.citations == ["DOC-TRV-002_TRV-C04"]
    assert res.referenced_rule_ids == ["R-TRV-003"]
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_deterministic_booking_notice_query_bypasses_groq_with_none_schema(canonical_corpus, canonical_chunks_by_id):
    """Booking notice query (14 days) succeeds authoritatively with R-TRV-005, citation DOC-TRV-003_TRV-B02, and schema_valid=None."""
    c = canonical_chunks_by_id["DOC-TRV-003_TRV-B02"]
    b02 = RetrievalResult(
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
    retriever = MagicMock()
    retriever.chunks = canonical_corpus
    retriever.chunks_by_id = canonical_chunks_by_id
    retriever.corpus_manifest = {"processed_fingerprint": "test_fp"}
    retriever.search.return_value = [b02]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = answer_question(
        question="Is 14 days notice enough for booking?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
    )
    assert res.status == "answered"
    assert res.schema_valid is None
    assert res.provider_calls_attempted == 0
    assert res.citations == ["DOC-TRV-003_TRV-B02"]
    assert "R-TRV-005" in res.referenced_rule_ids
    assert "14-day policy requirement" in res.answer
    mock_llm.generate_answer.assert_not_called()


# ==============================================================================
# 5. Correction B: Authoritative Rule Decision Precedence (Adversarial Groq Test)
# ==============================================================================


def test_adversarial_wrong_groq_approval_cannot_override_rule_decision(mock_retriever, canonical_chunks_by_id):
    """An intentionally wrong mocked Groq approval cannot override an authoritative rule decision."""
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

    # Adversarial mock LLM that would falsely claim manager can approve INR 50,001
    adversarial_llm = MagicMock(spec=GroqClientWrapper)
    adversarial_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="Yes, your direct manager has full authority to approve INR 50,001 without escalation.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C04"],
            referenced_rule_ids=["R-TRV-004"],
            reason="Adversarial false approval.",
        ),
        {"total_tokens": 100},
    )

    q = "Can a direct manager approve an expense of INR 50,001 under Travel Policy v2?"
    res = answer_question(
        question=q,
        role="manager",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=adversarial_llm,
    )

    # Must be answered deterministically using authoritative rule R-TRV-004
    assert res.status == "answered"
    assert "Additional approval is required" in res.answer
    assert "not sufficient" in res.answer
    assert res.citations == ["DOC-TRV-002_TRV-C04"]
    assert res.referenced_rule_ids == ["R-TRV-004"]
    assert res.provider_calls_attempted == 0
    # Groq was bypassed completely!
    adversarial_llm.generate_answer.assert_not_called()


# ==============================================================================
# 6. Correction E: Evidence Support Behavior Tests
# ==============================================================================


def test_check_evidence_support_direct_paraphrase_and_unsupported(canonical_chunks_by_id):
    """Test check_evidence_support on direct questions, paraphrases, and unsupported queries."""
    from rulelens.answering import check_evidence_support

    c04 = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    chunks = [c04]

    # Direct question
    assert check_evidence_support("What is the expense approval threshold for managers under Travel Policy v2?", chunks) is True

    # Strong paraphrase
    assert check_evidence_support("What approval threshold applies to direct reporting managers?", chunks) is True

    # Relevant question with different vocabulary
    assert check_evidence_support("When does an expense require approval from the department director?", chunks) is True

    # Unsupported query
    assert check_evidence_support("Can employees expense personal cryptocurrency mining rigs?", chunks) is False

    # Unrelated query with superficial corporate words
    assert check_evidence_support("What are the employee wellness leave policies for family illness?", chunks) is False


# ==============================================================================
# 7. Correction F: Strict Error Sanitization (Arbitrary Secrets & Paths)
# ==============================================================================


@pytest.mark.parametrize(
    "secret_string",
    [
        "gsk_abcdef1234567890_VERY_SECRET_GROQ_KEY",
        "qdrant_sec_998877665544332211",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload.sig",
        "C:\\Users\\admin\\.ssh\\id_rsa",
        "/var/run/secrets/kubernetes.io/serviceaccount/token",
        "postgres://user:super_secret_pw@db.internal:5432/asterline",
    ],
)
def test_operational_error_never_leaks_arbitrary_secrets_or_paths(
    mock_retriever, sample_retrieval_result, secret_string: str
):
    """Injected arbitrary secrets, bearer tokens, and private paths must never appear in error outputs."""
    mock_retriever.search.return_value = [sample_retrieval_result]
    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.side_effect = RuntimeError(f"Connection failed to endpoint: {secret_string}")

    res = answer_question(
        question="What is the meal allowance?",
        role="employee",
        query_date="2026-11-22",
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res.status == "operational_error"
    res_str = json.dumps(res.to_dict())
    assert secret_string not in res_str
    assert "super_secret" not in res_str.lower()
    assert "id_rsa" not in res_str


# ==============================================================================
# 8. Numerical Comparison Directional Safeguard Integration Tests (Section 2.5)
# ==============================================================================


def test_integration_exceeding_threshold_deterministic_additional_approval(mock_retriever, canonical_chunks_by_id):
    """Exceeding INR 50,000 deterministically triggers additional approval with 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a manager approve travel expenses exceeding INR 50,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "answered"
    assert "Additional approval is required" in res.answer
    assert res.referenced_rule_ids == ["R-TRV-004"]
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_at_least_threshold_safe_insufficient_refusal(mock_retriever, canonical_chunks_by_id):
    """At least INR 50,000 crosses approval boundaries; must refuse safely with insufficient and 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a manager approve travel expenses of at least INR 50,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert "unconditional decision cannot be determined" in res.answer or "sufficient evidence" in res.answer
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_range_crossing_boundary_safe_insufficient_refusal(mock_retriever, canonical_chunks_by_id):
    """Range crossing INR 50,000 threshold must refuse safely with insufficient and 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a manager approve travel expenses between INR 40,000 and 60,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_up_to_threshold_deterministic_standard_approval(mock_retriever, canonical_chunks_by_id):
    """Up to INR 50,000 deterministically triggers standard approval with 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a direct manager approve travel expenses up to INR 50,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "answered"
    assert "Standard manager approval is permitted" in res.answer
    assert res.referenced_rule_ids == ["R-TRV-003"]
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_below_threshold_deterministic_standard_approval(mock_retriever, canonical_chunks_by_id):
    """Below INR 50,000 deterministically triggers standard approval with 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a direct manager approve travel expenses below INR 50,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "answered"
    assert "Standard manager approval is permitted" in res.answer
    assert res.referenced_rule_ids == ["R-TRV-003"]
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_foreign_currency_incompatible_unit_safe_refusal(mock_retriever, canonical_chunks_by_id):
    """USD 50,000 cannot produce INR approval decision; fails closed safely with 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can a direct manager approve travel expenses of USD 50,000 under Travel Policy v2?"
    res = answer_question(
        question=q, role="manager", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert res.error_category == "incompatible_unit"
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()


def test_integration_personal_travel_unsupported_term_safe_refusal(mock_retriever, canonical_chunks_by_id):
    """Personal travel unsupported requirement refuses safely with 0 Groq calls."""
    c = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    c04_res = RetrievalResult(
        chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"], clause_id=c["clause_id"],
        title=c["title"], department=c["department"], version=c["version"], source_file=c["source_file"],
        source_page=int(c["source_page"]), effective_from=c["effective_from"], effective_to=c.get("effective_to"),
        allowed_roles=list(c["allowed_roles"]),
    )
    mock_retriever.search.return_value = [c04_res]
    mock_llm = MagicMock(spec=GroqClientWrapper)

    q = "Can employees spend INR 50,000 on personal travel?"
    res = answer_question(
        question=q, role="employee", query_date="2026-11-22",
        retriever=mock_retriever, llm_client=mock_llm,
    )
    assert res.status == "insufficient"
    assert res.provider_calls_attempted == 0
    mock_llm.generate_answer.assert_not_called()
