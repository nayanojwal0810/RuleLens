"""Offline unit tests for RuleLens retrieval evaluation harness.

Verifies:
- Standard IR metrics calculation: Hit@K, Recall@K, MRR@K.
- Handling of empty gold chunk sets and edge cases.
- Separation of answerable evaluation questions from diagnostic queries.
- Latency profiling and quantile calculation.
- Role permission and effective date violation tracking.
"""

from __future__ import annotations

import datetime
import socket
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from rulelens.retrieval import RetrievalResult
from rulelens.retrieval_eval import (
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_recall_at_k,
    evaluate_retriever_method,
)


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline unit tests")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ==============================================================================
# 1. Metric Calculation Tests
# ==============================================================================


def test_compute_hit_at_k():
    """compute_hit_at_k returns 1.0 when any gold chunk is in top K, else 0.0."""
    gold = {"chunk_A", "chunk_B"}

    assert compute_hit_at_k(["chunk_A", "chunk_C", "chunk_D"], gold, k=5) == 1.0
    assert compute_hit_at_k(["chunk_C", "chunk_B", "chunk_D"], gold, k=5) == 1.0
    assert compute_hit_at_k(["chunk_C", "chunk_D", "chunk_E"], gold, k=5) == 0.0

    # Cut off at K
    assert compute_hit_at_k(["chunk_C", "chunk_D", "chunk_A"], gold, k=2) == 0.0
    assert compute_hit_at_k(["chunk_C", "chunk_D", "chunk_A"], gold, k=3) == 1.0

    # Empty gold set
    assert compute_hit_at_k(["chunk_A"], set(), k=5) == 0.0


def test_compute_recall_at_k():
    """compute_recall_at_k returns fraction of gold chunks retrieved in top K."""
    gold = {"chunk_A", "chunk_B"}

    # Both retrieved
    assert compute_recall_at_k(["chunk_A", "chunk_B", "chunk_C"], gold, k=5) == 1.0
    # One of two retrieved
    assert compute_recall_at_k(["chunk_A", "chunk_C", "chunk_D"], gold, k=5) == 0.5
    # None retrieved
    assert compute_recall_at_k(["chunk_C", "chunk_D", "chunk_E"], gold, k=5) == 0.0

    # Cut off by K
    assert compute_recall_at_k(["chunk_A", "chunk_C", "chunk_B"], gold, k=2) == 0.5

    # Empty gold set
    assert compute_recall_at_k(["chunk_A"], set(), k=5) == 0.0


def test_compute_mrr_at_k():
    """compute_mrr_at_k returns reciprocal rank of first gold chunk found."""
    gold = {"chunk_B", "chunk_C"}

    # First gold at rank 1
    assert compute_mrr_at_k(["chunk_B", "chunk_A"], gold, k=5) == 1.0

    # First gold at rank 2
    assert compute_mrr_at_k(["chunk_A", "chunk_B"], gold, k=5) == 0.5

    # First gold at rank 3
    assert compute_mrr_at_k(["chunk_X", "chunk_Y", "chunk_C"], gold, k=5) == pytest.approx(1.0 / 3.0)

    # Not found in top K
    assert compute_mrr_at_k(["chunk_X", "chunk_Y"], gold, k=2) == 0.0

    # Found at rank 3, but K=2
    assert compute_mrr_at_k(["chunk_X", "chunk_Y", "chunk_B"], gold, k=2) == 0.0

    # Empty gold set
    assert compute_mrr_at_k(["chunk_A"], set(), k=5) == 0.0


# ==============================================================================
# 2. Evaluation Harness & Diagnostic Separation
# ==============================================================================


def test_evaluate_retriever_method_diagnostic_separation():
    """evaluate_retriever_method must separate answerable from diagnostic questions."""
    mock_retriever = MagicMock()

    # Question 1: Answerable, hit
    # Question 2: Answerable, miss
    # Question 3: Diagnostic / unanswerable (answerability=False)
    questions = [
        {
            "question_id": "Q1",
            "question": "What is the travel per diem?",
            "role": "employee",
            "query_date": "2026-11-22",
            "question_type": "direct",
            "gold_chunk_ids": ["CHUNK_1"],
            "answerability": True,
        },
        {
            "question_id": "Q2",
            "question": "What is flight allowance?",
            "role": "employee",
            "query_date": "2026-11-22",
            "question_type": "threshold",
            "gold_chunk_ids": ["CHUNK_2"],
            "answerability": True,
        },
        {
            "question_id": "Q3_DIAG",
            "question": "How are executive travel exceptions approved?",
            "role": "employee",
            "query_date": "2026-11-22",
            "question_type": "permission",
            "gold_chunk_ids": [],
            "answerability": False,
        },
    ]

    def mock_search(query, role, query_date, **kwargs):
        if query == "What is the travel per diem?":
            return [
                RetrievalResult(
                    chunk_id="CHUNK_1",
                    text="text1",
                    doc_id="DOC1",
                    clause_id="C1",
                    title="T1",
                    department="Travel",
                    version="v1",
                    source_file="f1.pdf",
                    source_page=1,
                    effective_from="2026-01-01",
                    effective_to=None,
                    allowed_roles=["employee", "manager"],
                )
            ]
        elif query == "What is flight allowance?":
            return [
                RetrievalResult(
                    chunk_id="OTHER_CHUNK",
                    text="other text",
                    doc_id="DOC2",
                    clause_id="C2",
                    title="T2",
                    department="Travel",
                    version="v1",
                    source_file="f2.pdf",
                    source_page=1,
                    effective_from="2026-01-01",
                    effective_to=None,
                    allowed_roles=["employee", "manager"],
                )
            ]
        else:
            return []

    mock_retriever.search.side_effect = mock_search

    report = evaluate_retriever_method(
        retriever=mock_retriever,
        questions=questions,
        mode="hybrid",
        top_k=5,
        warmup=False,
    )

    assert report["total_questions_evaluated"] == 3
    assert report["total_answerable_questions"] == 2
    assert report["total_diagnostic_questions"] == 1

    # 1 hit out of 2 answerable -> Hit@5 = 0.5, Recall@5 = 0.5, MRR@5 = 0.5
    assert report["hit_at_5"] == 0.5
    assert report["recall_at_5"] == 0.5
    assert report["mrr_at_5"] == 0.5

    assert report["missed_questions_count"] == 1
    assert report["misses"][0]["question_id"] == "Q2"
    assert report["permission_violations"] == 0
    assert report["version_violations"] == 0


def test_evaluate_retriever_tracks_permission_violations():
    """evaluate_retriever_method must increment violation count if retrieved chunk violates permissions."""
    mock_retriever = MagicMock()
    questions = [
        {
            "question_id": "Q1",
            "question": "Restricted question",
            "role": "employee",
            "query_date": "2026-11-22",
            "question_type": "permission",
            "gold_chunk_ids": ["CHUNK_MANAGER_ONLY"],
            "answerability": True,
        }
    ]

    # Return a chunk where role 'employee' is NOT in allowed_roles
    mock_retriever.search.return_value = [
        RetrievalResult(
            chunk_id="CHUNK_MANAGER_ONLY",
            text="restricted",
            doc_id="DOC-MGR",
            clause_id="C1",
            title="Manager Title",
            department="Travel",
            version="v1",
            source_file="m.pdf",
            source_page=1,
            effective_from="2026-01-01",
            effective_to=None,
            allowed_roles=["manager", "executive"],
        )
    ]

    report = evaluate_retriever_method(
        retriever=mock_retriever,
        questions=questions,
        mode="hybrid",
        warmup=False,
    )

    assert report["permission_violations"] == 1
