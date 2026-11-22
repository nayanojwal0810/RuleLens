"""Offline unit tests for RuleLens Stage E generation evaluation (generation_eval.py).

Verifies:
- Evaluating answerable questions using known canonical gold evidence.
- Evaluating diagnostic questions without leaking forbidden targets.
- 30-question development subset structure and balance across all 5 departments.
- Aggregation of schema validity, citation accuracy, and refusal outcomes.
- Markdown summary generation.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from rulelens.generation_eval import (
    DEV_SUBSET_QUESTION_IDS,
    evaluate_single_question,
    generate_summary_markdown,
    load_evaluation_dataset,
    run_generation_evaluation,
)
from rulelens.llm import AnswerResponse, AnswerStatus, GroqClientWrapper
from rulelens.retrieval import load_validated_corpus


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
def corpus_chunks(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    chunks = load_validated_corpus(repo_root)
    return {c["chunk_id"]: c for c in chunks}


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


def test_evaluate_single_answerable_question(dev_dataset: Dict[str, Any], corpus_chunks: Dict[str, Dict[str, Any]]):
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
        {"total_tokens": 110, "calls_made": 1},
    )

    res = evaluate_single_question(
        question_data=q_trv001,
        corpus_chunks_by_id=corpus_chunks,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-001"
    assert res["status"] == "answered"
    assert res["schema_valid"] is True
    assert res["citation_valid"] is True
    assert res["role_violations"] == 0
    assert res["version_violations"] == 0
    assert res["citations"] == ["DOC-TRV-002_TRV-C03"]


def test_evaluate_single_diagnostic_restricted_question(dev_dataset: Dict[str, Any], corpus_chunks: Dict[str, Dict[str, Any]]):
    """Diagnostic restricted question evaluation safely refuses without leaking forbidden evidence."""
    q_restr = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-009")

    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = evaluate_single_question(
        question_data=q_restr,
        corpus_chunks_by_id=corpus_chunks,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-009"
    assert res["status"] == "restricted"
    assert res["citations"] == []
    assert res["refusal_correct"] is True
    assert res["role_violations"] == 0
    # Model should not have been called
    mock_llm.generate_answer.assert_not_called()


def test_evaluate_single_diagnostic_unsupported_question(dev_dataset: Dict[str, Any], corpus_chunks: Dict[str, Dict[str, Any]]):
    """Diagnostic unsupported question evaluation safely refuses with status insufficient."""
    q_unsupp = next(q for q in dev_dataset["questions"] if q["question_id"] == "TRV-010")

    mock_llm = MagicMock(spec=GroqClientWrapper)

    res = evaluate_single_question(
        question_data=q_unsupp,
        corpus_chunks_by_id=corpus_chunks,
        llm_client=mock_llm,
    )

    assert res["question_id"] == "TRV-010"
    assert res["status"] == "insufficient"
    assert res["citations"] == []
    assert res["refusal_correct"] is True
    # Model should not have been called
    mock_llm.generate_answer.assert_not_called()


# ==============================================================================
# 3. Full Run with Mocked Provider & Markdown Summary
# ==============================================================================


def test_run_generation_evaluation_offline(repo_root: Path):
    """run_generation_evaluation executes cleanly with mocked LLM across a capped subset."""
    dataset_path = repo_root / "data" / "evaluation" / "dev_questions.json"

    mock_llm = MagicMock(spec=GroqClientWrapper)
    mock_llm.generate_answer.return_value = (
        AnswerResponse(
            answer="Policy mock answer.",
            status=AnswerStatus.ANSWERED,
            citations=["DOC-TRV-002_TRV-C03"],
            referenced_rule_ids=[],
            reason="Mock reason.",
        ),
        {"total_tokens": 75},
    )

    summary = run_generation_evaluation(
        dataset_path=dataset_path,
        base_dir=repo_root,
        max_questions=5,
        llm_client=mock_llm,
    )

    assert summary["total_evaluated"] == 5
    assert summary["schema_valid_count"] == 5
    assert summary["role_violations"] == 0
    assert summary["version_violations"] == 0

    # Markdown generation
    md = generate_summary_markdown(summary)
    assert "# RuleLens — Stage E Development Answer Generation Summary" in md
    assert "Schema Validity" in md
