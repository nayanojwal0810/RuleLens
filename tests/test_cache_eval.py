"""Offline unit tests for RuleLens Stage F cache evaluation harness (cache_eval.py).

Verifies:
- Dataset validation (schema, required counts, canonical roles, dates, rejection reasons).
- Evaluation of pairs across Modes A, B, and C.
- Contrast between Mode B (similarity-only unsafe hits) and Mode C (guarded 100% precision).
- Threshold sweep execution.
- Markdown summary report generation.
- CLI execution and error handling.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from rulelens.cache import ensure_cache_collection
from rulelens.cache_eval import (
    DEFAULT_CACHE_DEV_PAIRS_PATH,
    EXPECTED_SCENARIO_COUNTS,
    create_mock_llm_for_eval,
    evaluate_cache_pair,
    format_cache_eval_summary,
    main as cache_eval_main,
    run_cache_evaluation,
    sweep_candidate_thresholds,
    validate_cache_dev_dataset,
)
from rulelens.llm import AnswerResponse, AnswerStatus
from rulelens.retrieval import HybridRetriever, load_validated_corpus
from tests.test_cache import DeterministicTestEmbedder


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Ensure evaluation unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline cache eval test!")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def canonical_chunks_by_id(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    corpus = load_validated_corpus(repo_root)
    return {c["chunk_id"]: c for c in corpus}


@pytest.fixture
def mock_embedder() -> DeterministicTestEmbedder:
    return DeterministicTestEmbedder()


def test_validate_cache_dev_dataset_success():
    """Verify that the official 40-pair dataset validates cleanly."""
    res = validate_cache_dev_dataset(DEFAULT_CACHE_DEV_PAIRS_PATH)
    assert res["status"] == "valid"
    assert res["total_pairs"] == 40
    assert res["scenarios"] == EXPECTED_SCENARIO_COUNTS


def test_validate_cache_dev_dataset_invalid_role(tmp_path):
    """Verify validation fails if an invalid non-canonical role is present."""
    data = json.loads(Path(DEFAULT_CACHE_DEV_PAIRS_PATH).read_text(encoding="utf-8"))
    data["pairs"][0]["candidate_role"] = "superadmin"
    bad_file = tmp_path / "bad_pairs.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid candidate_role"):
        validate_cache_dev_dataset(bad_file)


def test_validate_cache_dev_dataset_missing_field(tmp_path):
    """Verify validation fails if a required field is missing."""
    data = json.loads(Path(DEFAULT_CACHE_DEV_PAIRS_PATH).read_text(encoding="utf-8"))
    del data["pairs"][0]["source_question"]
    bad_file = tmp_path / "bad_pairs.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        validate_cache_dev_dataset(bad_file)


def test_validate_cache_dev_dataset_count_mismatch(tmp_path):
    """Verify validation fails if pair count is not exactly 40."""
    data = json.loads(Path(DEFAULT_CACHE_DEV_PAIRS_PATH).read_text(encoding="utf-8"))
    data["pairs"] = data["pairs"][:39]
    bad_file = tmp_path / "bad_pairs.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 40 pairs"):
        validate_cache_dev_dataset(bad_file)


def test_create_mock_llm_for_eval():
    """Verify that create_mock_llm_for_eval returns structured AnswerResponse."""
    mock_llm = create_mock_llm_for_eval()
    dummy_chunk = {"chunk_id": "TEST_CHUNK_001"}
    resp, meta = mock_llm.generate_answer(
        question="What is the travel policy?",
        role="employee",
        evidence_chunks=[dummy_chunk],
    )
    assert isinstance(resp, AnswerResponse)
    assert resp.status == AnswerStatus.ANSWERED
    assert resp.citations == ["TEST_CHUNK_001"]
    assert meta["calls_made"] == 1


def test_evaluate_cache_pair_modes(repo_root, canonical_chunks_by_id, mock_embedder):
    """Verify evaluation of a pair across Modes A (No cache), B (Similarity-only), and C (Guarded)."""
    retriever = HybridRetriever(base_dir=repo_root)

    # Valid paraphrase pair
    valid_pair = {
        "pair_id": "PAIR-001",
        "scenario": "valid_paraphrase",
        "policy_subject": "Travel Expense Approvals",
        "source_question": "Under Travel Policy v2, what is the maximum travel expense amount a reporting manager can approve on their own authority?",
        "candidate_question": "Under Travel Policy v2, what is the maximum travel expense amount a direct manager can approve on their own authority?",
        "source_role": "employee",
        "candidate_role": "employee",
        "source_date": "2026-11-22",
        "candidate_date": "2026-11-22",
        "expected_decision": "safe_hit",
        "rejection_reason": None,
    }

    # Mode A: No Cache
    res_a = evaluate_cache_pair(
        pair=valid_pair,
        retriever=retriever,
        embedder=mock_embedder,
        qdrant_client=QdrantClient(":memory:"),
        canonical_chunks_by_id=canonical_chunks_by_id,
        mode="no_cache",
        base_dir=repo_root,
    )
    assert res_a["cache_hit"] is False
    assert res_a["decision"] == "miss"
    assert res_a["provider_calls"] >= 1

    # Mode B: Similarity-Only
    client_b = QdrantClient(":memory:")
    ensure_cache_collection(client_b)
    res_b = evaluate_cache_pair(
        pair=valid_pair,
        retriever=retriever,
        embedder=mock_embedder,
        qdrant_client=client_b,
        canonical_chunks_by_id=canonical_chunks_by_id,
        mode="similarity_only",
        similarity_threshold=0.85,
        base_dir=repo_root,
    )
    assert res_b["cache_hit"] is True
    assert res_b["decision"] == "correct_hit"
    assert res_b["provider_calls"] == 0

    # Mode C: Guarded
    client_c = QdrantClient(":memory:")
    ensure_cache_collection(client_c)
    res_c = evaluate_cache_pair(
        pair=valid_pair,
        retriever=retriever,
        embedder=mock_embedder,
        qdrant_client=client_c,
        canonical_chunks_by_id=canonical_chunks_by_id,
        mode="guarded",
        similarity_threshold=0.85,
        base_dir=repo_root,
    )
    assert res_c["cache_hit"] is True
    assert res_c["decision"] == "correct_hit"
    assert res_c["provider_calls"] == 0


def test_evaluate_cache_pair_contrast_on_unsafe_candidate(repo_root, canonical_chunks_by_id, mock_embedder):
    """Verify that Mode B blindly hits on negative candidate while Mode C rejects it safely."""
    retriever = HybridRetriever(base_dir=repo_root)

    # Negative pair: numerical amount changed from 50,000 to 75,000
    negative_pair = {
        "pair_id": "PAIR-013",
        "scenario": "changed_numerical_condition",
        "policy_subject": "Travel Expense Approvals",
        "source_question": "Can an employee claim an expense of INR 50,000 under Travel Policy v2?",
        "candidate_question": "Can an employee claim an expense of INR 75,000 under Travel Policy v2?",
        "source_role": "employee",
        "candidate_role": "employee",
        "source_date": "2026-11-22",
        "candidate_date": "2026-11-22",
        "expected_decision": "reject",
        "rejection_reason": "numerical_condition_mismatch",
    }

    # Mode B: Blind similarity-only hits (unsafe!)
    client_b = QdrantClient(":memory:")
    ensure_cache_collection(client_b)
    res_b = evaluate_cache_pair(
        pair=negative_pair,
        retriever=retriever,
        embedder=mock_embedder,
        qdrant_client=client_b,
        canonical_chunks_by_id=canonical_chunks_by_id,
        mode="similarity_only",
        similarity_threshold=0.85,
        base_dir=repo_root,
    )
    assert res_b["cache_hit"] is True
    assert res_b["decision"] == "unsafe_hit"

    # Mode C: Guarded cache catches the mismatch and rejects safely
    client_c = QdrantClient(":memory:")
    ensure_cache_collection(client_c)
    res_c = evaluate_cache_pair(
        pair=negative_pair,
        retriever=retriever,
        embedder=mock_embedder,
        qdrant_client=client_c,
        canonical_chunks_by_id=canonical_chunks_by_id,
        mode="guarded",
        similarity_threshold=0.85,
        base_dir=repo_root,
    )
    assert res_c["cache_hit"] is False
    assert res_c["decision"] == "safe_rejection"


def test_run_cache_evaluation_offline(repo_root, mock_embedder):
    """Verify complete run_cache_evaluation execution and metrics aggregation."""
    eval_data = run_cache_evaluation(
        dataset_path=DEFAULT_CACHE_DEV_PAIRS_PATH,
        modes=("no_cache", "guarded"),
        similarity_threshold=0.88,
        base_dir=repo_root,
        embedder=mock_embedder,
    )
    assert "results_by_mode" in eval_data
    assert "no_cache" in eval_data["results_by_mode"]
    assert "guarded" in eval_data["results_by_mode"]

    grd = eval_data["results_by_mode"]["guarded"]
    # Guarded semantic cache must maintain 100% precision with 0 unsafe hits
    assert grd["unsafe_hits"] == 0
    assert grd["safe_reuse_precision"] == 1.0


def test_sweep_candidate_thresholds_offline(repo_root, mock_embedder):
    """Verify sweep_candidate_thresholds executes across multiple thresholds."""
    sweep_results = sweep_candidate_thresholds(
        dataset_path=DEFAULT_CACHE_DEV_PAIRS_PATH,
        thresholds=(0.85, 0.90),
        base_dir=repo_root,
        embedder=mock_embedder,
    )
    assert len(sweep_results) == 2
    for row in sweep_results:
        assert row["guarded_unsafe_hits"] == 0
        assert row["guarded_precision"] == 1.0


def test_format_cache_eval_summary():
    """Verify Markdown report formatting."""
    eval_data = {
        "timestamp": "2026-11-22T00:00:00Z",
        "similarity_threshold": 0.88,
        "results_by_mode": {
            "no_cache": {"total_pairs": 40, "total_hits": 0, "correct_hits": 0, "unsafe_hits": 0, "safe_rejections": 0, "false_rejections": 0},
            "similarity_only": {"total_pairs": 40, "total_hits": 30, "correct_hits": 12, "unsafe_hits": 18, "safe_rejections": 10, "false_rejections": 0, "safe_reuse_precision": 0.40, "valid_reuse_rate": 1.0, "cache_hit_rate": 0.75, "provider_requests_avoided": 16, "tokens_avoided": 4000, "latency_median_ms": 25.0, "latency_p95_ms": 35.0},
            "guarded": {"total_pairs": 40, "total_hits": 12, "correct_hits": 12, "unsafe_hits": 0, "safe_rejections": 28, "false_rejections": 0, "safe_reuse_precision": 1.0, "valid_reuse_rate": 1.0, "cache_hit_rate": 0.30, "provider_requests_avoided": 12, "tokens_avoided": 3000, "latency_median_ms": 28.0, "latency_p95_ms": 65.0, "scenario_breakdown": {}, "rejection_reasons": {}},
        },
    }
    report = format_cache_eval_summary(eval_data)
    assert "# RuleLens — Stage F Guarded Semantic Cache Evaluation Summary" in report
    assert "Mode C: Guarded Semantic Cache" in report
    assert "**0 (Zero Unsafe)**" in report
    assert "**100.0%**" in report


def test_cli_validate(capsys):
    """Verify CLI validate command runs and outputs SUCCESS."""
    with patch.object(sys, "argv", ["cache_eval.py", "validate"]):
        cache_eval_main()
    captured = capsys.readouterr()
    assert "RuleLens Cache Development Dataset Validation: SUCCESS" in captured.out
