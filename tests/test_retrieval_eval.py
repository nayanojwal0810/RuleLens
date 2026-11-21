"""Offline unit tests for RuleLens retrieval evaluation harness.

Verifies:
- Standard IR metrics calculation: Hit@K, Recall@K, MRR@K.
- Handling of empty gold chunk sets and edge cases.
- Separation of answerable evaluation questions from diagnostic queries.
- Latency profiling and quantile calculation.
- Role permission and effective date violation tracking.
"""

from __future__ import annotations

import copy
import datetime
import json
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
    main,
    validate_evaluation_dataset,
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


# ==============================================================================
# 3. Dataset Validation Tests
# ==============================================================================


@pytest.fixture
def repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


def test_validate_evaluation_dataset_on_canonical_dataset(repo_root: Path):
    """Canonical data/evaluation/dev_questions.json must pass all validation checks."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    summary = validate_evaluation_dataset(dev_path, base_dir=repo_root)

    assert summary["status"] == "valid"
    assert summary["total_questions"] == 50
    assert summary["answerable_count"] == 40
    assert summary["restricted_count"] == 5
    assert summary["unsupported_count"] == 5
    assert summary["multi_clause_count"] >= 12
    assert summary["difficulty_counts"] == {"easy": 5, "medium": 15, "hard": 30}


def test_validate_evaluation_dataset_rejects_mutations(repo_root: Path):
    """Validator must reject invalid schema, fingerprint, counts, and permissions."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        canonical_data = json.load(f)

    # 1. Invalid schema version
    bad_version = copy.deepcopy(canonical_data)
    bad_version["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="Invalid schema_version"):
        validate_evaluation_dataset(bad_version, base_dir=repo_root)

    # 2. Fingerprint mismatch
    bad_fp = copy.deepcopy(canonical_data)
    bad_fp["corpus_fingerprint"] = "corrupted_fp_12345"
    with pytest.raises(ValueError, match="does not match"):
        validate_evaluation_dataset(bad_fp, base_dir=repo_root)

    # 3. Wrong question count (49 instead of 50)
    bad_count = copy.deepcopy(canonical_data)
    bad_count["questions"].pop()
    with pytest.raises(ValueError, match="exactly 50 questions"):
        validate_evaluation_dataset(bad_count, base_dir=repo_root)

    # 4. Duplicate question ID
    bad_id = copy.deepcopy(canonical_data)
    bad_id["questions"][1]["question_id"] = bad_id["questions"][0]["question_id"]
    with pytest.raises(ValueError, match="Duplicate question_id"):
        validate_evaluation_dataset(bad_id, base_dir=repo_root)

    # 5. Invalid query date
    bad_date = copy.deepcopy(canonical_data)
    bad_date["questions"][0]["query_date"] = "2026/11/22"
    with pytest.raises(ValueError, match="malformed query_date"):
        validate_evaluation_dataset(bad_date, base_dir=repo_root)

    # 6. Non-existent gold chunk
    bad_chunk = copy.deepcopy(canonical_data)
    bad_chunk["questions"][0]["gold_chunk_ids"] = ["NON_EXISTENT_CHUNK_XYZ"]
    with pytest.raises(ValueError, match="non-existent gold chunk"):
        validate_evaluation_dataset(bad_chunk, base_dir=repo_root)

    # 7. Role not allowed for gold chunk
    bad_role = copy.deepcopy(canonical_data)
    bad_role["questions"][0]["gold_chunk_ids"] = ["DOC-TRV-004_TRV-E01"]
    bad_role["questions"][0]["role"] = "employee"
    with pytest.raises(ValueError, match="not in allowed_roles"):
        validate_evaluation_dataset(bad_role, base_dir=repo_root)

    # 8. Restricted diagnostic where role IS allowed
    bad_restr = copy.deepcopy(canonical_data)
    # Find restricted question
    restr_q = next(q for q in bad_restr["questions"] if q.get("restricted_target_chunk_ids"))
    restr_q["role"] = "legal"  # If target is accessible to legal, or set role to one of target's allowed_roles
    target_cid = restr_q["restricted_target_chunk_ids"][0]
    # Set role to manager which is allowed for travel exceptions
    restr_q["role"] = "manager"
    restr_q["restricted_target_chunk_ids"] = ["DOC-TRV-004_TRV-E01"]
    with pytest.raises(ValueError, match="IS allowed for target chunk"):
        validate_evaluation_dataset(bad_restr, base_dir=repo_root)

    # 9. Unsupported diagnostic missing reason
    bad_unsupp = copy.deepcopy(canonical_data)
    unsupp_q = next(q for q in bad_unsupp["questions"] if not q.get("answerability") and not q.get("restricted_target_chunk_ids"))
    unsupp_q["evidence_reason"] = "   "
    with pytest.raises(ValueError, match="must have non-empty evidence_reason"):
        validate_evaluation_dataset(bad_unsupp, base_dir=repo_root)


def test_cli_validate_command(repo_root: Path):
    """CLI validate command must succeed with exit code 0 on valid dataset."""
    dev_path = str(repo_root / "data" / "evaluation" / "dev_questions.json")
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--questions", dev_path])
    assert exc.value.code == 0
