"""Offline unit tests for RuleLens Stage E generation evaluation (generation_eval.py).

Verifies:
- Evaluating answerable questions using known canonical gold evidence.
- Evaluating diagnostic questions without leaking forbidden targets or reading gold labels to produce answers.
- 30-question development subset structure and balance across all 5 departments.
- Dynamic calculation of schema validity, citation accuracy, and refusal outcomes (no hardcoded 100%).
- Markdown summary generation.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from rulelens.generation_eval import (
    DEV_SUBSET_QUESTION_IDS,
    evaluate_single_question,
    format_rate,
    generate_summary_markdown,
    load_evaluation_dataset,
    run_generation_evaluation,
)
from rulelens.llm import AnswerResponse, AnswerStatus, GroqClientWrapper
from rulelens.retrieval import HybridRetriever, load_validated_corpus


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
def dev_dataset(repo_root: Path) -> Dict[str, Any]:
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    return load_evaluation_dataset(dataset_path)


@pytest.fixture
def mock_retriever(repo_root: Path):
    retriever = MagicMock()
    chunks = load_validated_corpus(repo_root)
    retriever.chunks = chunks
    retriever.chunks_by_id = {c["chunk_id"]: c for c in chunks}
    retriever.corpus_manifest = {"processed_fingerprint": "4f2485c2ea0a10148532e154f308554f44c27ae8b0fbdd7d174dff83f6dab9b3"}
    from rulelens.retrieval import BM25Index
    retriever.get_bm25_index.side_effect = lambda c_list: BM25Index(c_list)
    return retriever


# ==============================================================================
# 1. Dataset Subset Structure Tests
# ==============================================================================


def test_dev_subset_has_exactly_30_questions_and_balanced_departments(dev_dataset: Dict[str, Any]):
    """DEV_SUBSET_QUESTION_IDS must contain exactly 30 questions, 6 per department."""
    assert len(DEV_SUBSET_QUESTION_IDS) == 30
    q_map = {q["question_id"]: q for q in dev_dataset["questions"]}

    dept_counts: Dict[str, int] = {}
    ans_count = 0
    restr_count = 0
    unsupp_count = 0

    for qid in DEV_SUBSET_QUESTION_IDS:
        assert qid in q_map, f"Question {qid} not in dev_questions.json"
        q = q_map[qid]
        dept = q["department"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

        if q["answerability"]:
            ans_count += 1
        elif q.get("restricted_target_chunk_ids"):
            restr_count += 1
        else:
            unsupp_count += 1

    assert len(dept_counts) == 5
    for dept, count in dept_counts.items():
        assert count == 6, f"Department {dept} has {count} questions, expected 6."

    assert ans_count == 20
    assert restr_count == 5
    assert unsupp_count == 5


# ==============================================================================
# 2. Single Question Evaluation with Mock LLM
# ==============================================================================


def test_evaluate_single_answerable_question(dev_dataset: Dict[str, Any], mock_retriever):
    """Answerable question evaluation provides gold evidence to mock LLM and validates output."""
    q_trv001 = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-001")

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="Daily meal allowance cap is INR 2,000 per diem.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C03"],
            referenced_rule_ids=[],
            reason="Verified from clause TRV-C03.",
        ),
        {"total_tokens": 110, "calls_made": 1, "latency_ms": 25.0},
    )

    res = evaluate_single_question(
        question_data=q_trv001,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-001"
    assert res["status"] == "answered"
    assert res["schema_valid"] is True
    assert res["citation_valid"] is True
    assert res["role_violations"] == 0
    assert res["version_violations"] == 0
    assert res["citations"] == ["DOC-TRV-002_TRV-C03"]


def test_evaluate_single_diagnostic_restricted_question(dev_dataset: Dict[str, Any], mock_retriever):
    """Diagnostic restricted question evaluation safely refuses without leaking forbidden evidence."""
    q_restr = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-009")

    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = evaluate_single_question(
        question_data=q_restr,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-009"
    assert res["status"] == "restricted"
    assert res["citations"] == []
    assert res["refusal_correct"] is True
    assert res["role_violations"] == 0
    # Model should not have been called
    mock_llm.generate_answer.assert_not_called()


def test_evaluate_single_diagnostic_unsupported_question(dev_dataset: Dict[str, Any], mock_retriever):
    """Diagnostic unsupported question evaluation safely refuses with status insufficient."""
    q_unsupp = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-010")

    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = evaluate_single_question(
        question_data=q_unsupp,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-010"
    assert res["status"] == "insufficient"
    assert res["citations"] == []
    assert res["refusal_correct"] is True
    # Model should not have been called
    mock_llm.generate_answer.assert_not_called()


def test_evaluator_does_not_use_gold_labels_to_produce_answer(dev_dataset: Dict[str, Any], mock_retriever):
    """Bug regression: Evaluator must not read expected labels to construct diagnostic refusals."""
    q_restr = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-009").copy()
    # If someone tampers with the question's gold labels:
    q_restr["restricted_target_chunk_ids"] = []
    q_restr["answerability"] = False

    mock_llm = MagicMock(spec=GroqClientWrapper)
    # The real answer pipeline should STILL detect that this query targets restricted DOC-TRV-004
    res = evaluate_single_question(
        question_data=q_restr,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )
    # Status comes from real pipeline, not from q_restr["restricted_target_chunk_ids"]
    assert res["status"] == "restricted"


def test_metrics_calculated_dynamically_without_hardcoded_100_percent():
    """Bug regression: Summary rates must be dynamically calculated and handle zero denominators cleanly."""
    assert format_rate(0, 0) == "N/A"
    assert format_rate(5, 10) == "5 / 10 (50.0%)"
    assert format_rate(3, 4) == "3 / 4 (75.0%)"

    sample_summary = {
        "timestamp": "2026-11-22T00:00:00Z",
        "model": "openai/gpt-oss-20b",
        "evaluation_state": "partial",
        "stop_reason": "Rate limit HTTP 429 encountered",
        "planned_questions": 30,
        "completed_questions": 10,
        "generations_attempted": 10,
        "schema_valid_generations": 8,
        "citation_valid_generations": 6,
        "restricted_refusal_successes": 2,
        "unsupported_refusal_successes": 1,
        "provider_errors": 1,
        "median_e2e_latency_ms": 500.0,
        "median_provider_latency_ms": 400.0,
        "median_diagnostic_latency_ms": 10.0,
        "tokens_used": 1500,
        "per_question_results": [
            {
                "question_id": "TRV-001",
                "role": "employee",
                "query_date": "2026-11-22",
                "status": "answered",
                "citations": ["DOC-TRV-002_TRV-C03"],
                "referenced_rule_ids": [],
                "latency_ms": 500.0,
                "answer": "Test answer.",
                "answerability": True,
            }
        ],
    }
    md = generate_summary_markdown(sample_summary)
    # Must NOT say 100% for schema validity when 8/10
    assert "8 / 10 (80.0%)" in md
    assert "PARTIAL" in md


# ==============================================================================
# 3. Full Run with Mocked Provider & Markdown Summary
# ==============================================================================


def test_run_generation_evaluation_offline(repo_root: Path, mock_retriever):
    """run_generation_evaluation executes cleanly with mocked LLM across a capped subset."""
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"

    mock_llm = MagicMock(spec=GroqClientWrapper)
    def mock_generate(question, role, evidence_chunks, **kwargs):
        cid = evidence_chunks[0].get("chunk_id") if isinstance(evidence_chunks[0], dict) else evidence_chunks[0].chunk_id
        return (
            AnswerResponse(
                answer="Policy mock answer.",
                status=AnswerStatus.ANSWERED,
                citations=[cid],
                referenced_rule_ids=[],
                reason="Mock reason.",
            ),
            {"total_tokens": 75, "latency_ms": 20.0},
        )

    mock_llm.generate_answer.side_effect = mock_generate

    summary = run_generation_evaluation(
        dataset_path=dataset_path,
        base_dir=repo_root,
        max_questions=5,
        llm_client=mock_llm,
        retriever=mock_retriever,
    )

    assert summary["completed_questions"] == 5
    # In first 5 questions (TRV-001, 002, 003, 007, 009), 4 are answerable and 1 is restricted diagnostic
    assert summary["generations_attempted"] == 4
    assert summary["schema_valid_generations"] == 4
    assert summary["restricted_refusal_successes"] == 1

    # Markdown generation
    md = generate_summary_markdown(summary)
    assert "# RuleLens — Stage E Development Answer Generation Summary" in md
    assert "Schema Validity (Groq)" in md


# ==============================================================================
# 4. Adversarial Gold Evidence Fail-Closed Tests
# ==============================================================================


def test_adversarial_gold_chunk_unauthorized_role_fails_closed(mock_retriever):
    """Security verification: Unauthorized role in gold evidence fails closed immediately without Groq call."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    # Adversarial question: employee requesting DOC-TRV-004 chunk (restricted for employee)
    adv_q = {
        "question_id": "ADV-ROLE-001",
        "question": "What is the exception route?",
        "role": "employee",
        "query_date": "2026-11-22",
        "answerability": True,
        "gold_chunk_ids": ["DOC-TRV-004_TRV-E01"],
        "gold_evidence": [{"chunk_id": "DOC-TRV-004_TRV-E01", "quote": "Adversarial test quote"}],
    }

    res = evaluate_single_question(
        question_data=adv_q,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["status"] == "operational_error"
    assert res["provider_error"] is False
    assert res["evidence_validation_failure"] is True
    assert res["role_violations"] == 1
    assert res["version_violations"] == 0
    assert res["provider_calls_attempted"] == 0
    assert res["provider_latency_ms"] is None
    # Model must NEVER be called with unauthorized gold chunks!
    mock_llm.generate_answer.assert_not_called()


def test_adversarial_gold_chunk_expired_date_fails_closed(mock_retriever):
    """Security verification: Expired date in gold evidence fails closed immediately without Groq call."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    # Adversarial question: query date 2026-11-22 requesting DOC-TRV-001 (expired 2026-06-30)
    adv_q = {
        "question_id": "ADV-DATE-001",
        "question": "What was the v1 travel allowance?",
        "role": "employee",
        "query_date": "2026-11-22",
        "answerability": True,
        "gold_chunk_ids": ["DOC-TRV-001_TRV-C01"],
        "gold_evidence": [{"chunk_id": "DOC-TRV-001_TRV-C01", "quote": "Expired test quote"}],
    }

    res = evaluate_single_question(
        question_data=adv_q,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["status"] == "operational_error"
    assert res["provider_error"] is False
    assert res["evidence_validation_failure"] is True
    assert res["version_violations"] == 1
    assert res["role_violations"] == 0
    assert res["provider_calls_attempted"] == 0
    assert res["provider_latency_ms"] is None
    # Model must NEVER be called with expired gold chunks!
    mock_llm.generate_answer.assert_not_called()


# ==============================================================================
# 5. Backend Selection and Typed Rate Limit Tests
# ==============================================================================


def test_backend_selection_honored(repo_root: Path, mock_retriever):
    """Verification: Backend selection is honored and passed through to answer_question."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    mock_qdrant = MagicMock()

    with patch("rulelens.generation_eval.evaluate_single_question") as mock_eval:
        mock_eval.return_value = {
            "question_id": "TRV-001",
            "answerability": True,
            "status": "answered",
            "provider_error": False,
            "latency_ms": 100.0,
            "provider_latency_ms": 80.0,
            "token_usage": 50,
            "schema_valid": True,
            "citation_valid": True,
            "refusal_correct": False,
            "provider_calls_attempted": 1,
        }

        summary = run_generation_evaluation(
            dataset_path=dataset_path,
            base_dir=repo_root,
            max_questions=1,
            llm_client=mock_llm,
            retriever=mock_retriever,
            dense_backend="qdrant",
            qdrant_client=mock_qdrant,
        )

        mock_eval.assert_called_once()
        _, kwargs = mock_eval.call_args
        assert kwargs.get("dense_backend") == "qdrant"
        assert kwargs.get("qdrant_client") is mock_qdrant
        assert summary["retrieval_backend"] == "qdrant"


def test_backend_qdrant_fails_when_unavailable(repo_root: Path, mock_retriever):
    """Verification: When dense_backend='qdrant' and Qdrant fails, raise RuntimeError (no silent fallback)."""
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    mock_llm = MagicMock(spec=GroqClientWrapper)

    with patch("rulelens.qdrant_store.get_qdrant_client", side_effect=ConnectionError("Cannot connect to Qdrant Cloud")):
        with pytest.raises(RuntimeError, match="Qdrant client initialization failed"):
            run_generation_evaluation(
                dataset_path=dataset_path,
                base_dir=repo_root,
                max_questions=1,
                llm_client=mock_llm,
                retriever=mock_retriever,
                dense_backend="qdrant",
                qdrant_client=None,
            )


def test_evaluation_stops_on_typed_rate_limit(repo_root: Path, mock_retriever):
    """Verification: Evaluation stops immediately on typed 'rate_limited' error category."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"

    with patch("rulelens.generation_eval.evaluate_single_question") as mock_eval:
        mock_eval.return_value = {
            "question_id": "TRV-001",
            "answerability": True,
            "status": "operational_error",
            "provider_error": True,
            "error_category": "rate_limited",
            "error": "Rate limit exceeded (HTTP 429)",
            "latency_ms": 50.0,
            "provider_latency_ms": None,
            "token_usage": None,
            "schema_valid": False,
            "citation_valid": False,
            "refusal_correct": False,
            "provider_calls_attempted": 1,
        }

        summary = run_generation_evaluation(
            dataset_path=dataset_path,
            base_dir=repo_root,
            max_questions=5,
            llm_client=mock_llm,
            retriever=mock_retriever,
        )

        # Evaluator should have stopped after question 1
        assert mock_eval.call_count == 1
        assert summary["evaluation_state"] == "partial"
        assert "Rate limit" in summary["stop_reason"]
        assert summary["completed_questions"] == 1


# ==============================================================================
# 6. Additional Fail-Closed Gold Evidence & Bounds Tests
# ==============================================================================


def test_adversarial_gold_chunks_exceeds_limit_fails_closed(mock_retriever):
    """Gold chunks count > 4 fails closed immediately without Groq call."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    adv_q = {
        "question_id": "ADV-5CHUNKS-001",
        "question": "What are all the travel rules?",
        "role": "employee",
        "query_date": "2026-11-22",
        "answerability": True,
        "gold_chunk_ids": [
            "DOC-TRV-002_TRV-C01", "DOC-TRV-002_TRV-C02", "DOC-TRV-002_TRV-C03",
            "DOC-TRV-002_TRV-C04", "DOC-TRV-003_TRV-C01",
        ],
        "gold_evidence": [],
    }

    res = evaluate_single_question(
        question_data=adv_q,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["status"] == "operational_error"
    assert res["provider_error"] is False
    assert res["evidence_validation_failure"] is True
    assert res["provider_calls_attempted"] == 0
    mock_llm.generate_answer.assert_not_called()


def test_adversarial_gold_chunks_duplicate_cids_fails_closed(mock_retriever):
    """Duplicate chunk IDs in gold_chunk_ids fails closed immediately without Groq call."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    adv_q = {
        "question_id": "ADV-DUP-001",
        "question": "What is the meal allowance?",
        "role": "employee",
        "query_date": "2026-11-22",
        "answerability": True,
        "gold_chunk_ids": ["DOC-TRV-002_TRV-C03", "DOC-TRV-002_TRV-C03"],
        "gold_evidence": [],
    }

    res = evaluate_single_question(
        question_data=adv_q,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["status"] == "operational_error"
    assert res["provider_error"] is False
    assert res["evidence_validation_failure"] is True
    assert res["provider_calls_attempted"] == 0
    mock_llm.generate_answer.assert_not_called()


def test_adversarial_gold_quote_mismatch_fails_closed(mock_retriever):
    """Evidence quote that is not a verbatim substring of chunk text fails closed immediately."""
    mock_llm = MagicMock(spec=GroqClientWrapper)
    adv_q = {
        "question_id": "ADV-QUOTE-001",
        "question": "What is the meal allowance?",
        "role": "employee",
        "query_date": "2026-11-22",
        "answerability": True,
        "gold_chunk_ids": ["DOC-TRV-002_TRV-C03"],
        "gold_evidence": [
            {
                "chunk_id": "DOC-TRV-002_TRV-C03",
                "quote": "FABRICATED QUOTE NOT IN POLICY TEXT: Free caviar for all employees.",
            }
        ],
    }

    res = evaluate_single_question(
        question_data=adv_q,
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    assert res["status"] == "operational_error"
    assert res["provider_error"] is False
    assert res["evidence_validation_failure"] is True
    assert res["provider_calls_attempted"] == 0
    mock_llm.generate_answer.assert_not_called()


def test_run_generation_evaluation_rejects_invalid_max_questions(repo_root: Path):
    """run_generation_evaluation rejects max_questions < 1 or > 50 with ValueError."""
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    mock_llm = MagicMock(spec=GroqClientWrapper)

    for invalid_val in (0, 51, -1, 100):
        with pytest.raises(ValueError, match="max_questions must be an integer between 1 and 50"):
            run_generation_evaluation(
                dataset_path=dataset_path,
                base_dir=repo_root,
                max_questions=invalid_val,
                llm_client=mock_llm,
            )
