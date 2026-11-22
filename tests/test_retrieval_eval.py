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

from rulelens.retrieval import RetrievalResult, load_validated_corpus
from rulelens.retrieval_eval import (
    compute_full_evidence_at_k,
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_recall_at_k,
    evaluate_retriever_method,
    generate_summary_markdown,
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


def test_compute_full_evidence_at_k():
    """compute_full_evidence_at_k returns 1.0 when all gold chunks are in top K, else 0.0."""
    gold = {"chunk_A", "chunk_B"}

    # Both retrieved in top K
    assert compute_full_evidence_at_k(["chunk_A", "chunk_B", "chunk_C"], gold, k=5) == 1.0
    assert compute_full_evidence_at_k(["chunk_B", "chunk_A"], gold, k=5) == 1.0

    # Only one retrieved
    assert compute_full_evidence_at_k(["chunk_A", "chunk_C", "chunk_D"], gold, k=5) == 0.0
    # None retrieved
    assert compute_full_evidence_at_k(["chunk_C", "chunk_D", "chunk_E"], gold, k=5) == 0.0

    # Cut off at K (second gold at index 2, K=2)
    assert compute_full_evidence_at_k(["chunk_A", "chunk_C", "chunk_B"], gold, k=2) == 0.0
    assert compute_full_evidence_at_k(["chunk_A", "chunk_C", "chunk_B"], gold, k=3) == 1.0

    # Empty gold set
    assert compute_full_evidence_at_k(["chunk_A"], set(), k=5) == 0.0


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
    assert report["full_evidence_at_5"] == 0.5
    assert len(report["per_question_details"]) == 3

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
    with pytest.raises(ValueError, match="must have non-empty 'evidence_reason'"):
        validate_evaluation_dataset(bad_unsupp, base_dir=repo_root)

    # 10. Missing required field (difficulty_reason)
    bad_missing = copy.deepcopy(canonical_data)
    del bad_missing["questions"][0]["difficulty_reason"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_evaluation_dataset(bad_missing, base_dir=repo_root)

    # 11. Empty difficulty_reason
    bad_diff_reason = copy.deepcopy(canonical_data)
    bad_diff_reason["questions"][0]["difficulty_reason"] = "  "
    with pytest.raises(ValueError, match="must have non-empty 'difficulty_reason'"):
        validate_evaluation_dataset(bad_diff_reason, base_dir=repo_root)

    # 12. Invalid question_type
    bad_qtype = copy.deepcopy(canonical_data)
    bad_qtype["questions"][0]["question_type"] = "unsupported_custom_type"
    with pytest.raises(ValueError, match="invalid question_type"):
        validate_evaluation_dataset(bad_qtype, base_dir=repo_root)

    # 13. Employee role count exceeds 30
    bad_emp = copy.deepcopy(canonical_data)
    corpus_chunks = load_validated_corpus(repo_root)
    chunks_by_id = {c["chunk_id"]: c for c in corpus_chunks}
    switched = 0
    for q in bad_emp["questions"]:
        if q.get("answerability") and q["role"] != "employee":
            if all("employee" in chunks_by_id[cid]["allowed_roles"] for cid in q["gold_chunk_ids"]):
                q["role"] = "employee"
                switched += 1
                if switched >= 6:  # 26 + 6 = 32 > 30
                    break
    with pytest.raises(ValueError, match="exceed maximum 30"):
        validate_evaluation_dataset(bad_emp, base_dir=repo_root)

    # 14. Less than 2 non-employee restricted diagnostics
    bad_diag = copy.deepcopy(canonical_data)
    for q in bad_diag["questions"]:
        if q.get("restricted_target_chunk_ids"):
            q["role"] = "employee"
            q["restricted_target_chunk_ids"] = ["DOC-TRV-004_TRV-E01"]
    with pytest.raises(ValueError, match="Expected at least 2 non-employee restricted diagnostic"):
        validate_evaluation_dataset(bad_diag, base_dir=repo_root)

    # 15. Less than 10 historical answerable questions
    bad_hist = copy.deepcopy(canonical_data)
    for q in bad_hist["questions"]:
        if q.get("answerability") and q["query_date"] < "2026-07-01":
            q["query_date"] = "2026-08-01"
            q["gold_chunk_ids"] = ["DOC-TRV-002_TRV-C01"]
            q["gold_evidence"] = [{"chunk_id": "DOC-TRV-002_TRV-C01", "quote": "This revised policy establishes updated corporate travel guidelines"}]
            q["role"] = "employee"
    with pytest.raises(ValueError, match="historical answerable questions"):
        validate_evaluation_dataset(bad_hist, base_dir=repo_root)


def test_cli_validate_command(repo_root: Path):
    """CLI validate command must succeed with exit code 0 on valid dataset."""
    dev_path = str(repo_root / "data" / "evaluation" / "dev_questions.json")
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--questions", dev_path])
    assert exc.value.code == 0


# ==============================================================================
# 5. Gold Evidence Validation & Audit Regression Tests
# ==============================================================================


def test_gold_evidence_canonical_dataset_valid(repo_root: Path):
    """The canonical dev_questions.json must have valid gold_evidence for all 50 questions."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Must pass complete validation
    stats = validate_evaluation_dataset(data, base_dir=repo_root)
    assert stats["total_questions"] == 50
    assert stats["answerable_count"] == 40
    assert stats["restricted_count"] == 5
    assert stats["unsupported_count"] == 5

    questions = data["questions"]
    for q in questions:
        assert "gold_evidence" in q
        evidence = q["gold_evidence"]
        if q["answerability"]:
            assert len(evidence) > 0, f"Question {q['question_id']} has empty gold_evidence"
            ev_chunk_ids = {item["chunk_id"] for item in evidence}
            assert ev_chunk_ids == set(q["gold_chunk_ids"]), f"Evidence chunks mismatch for {q['question_id']}"
            for item in evidence:
                assert item["quote"].strip(), f"Empty quote in {q['question_id']}"
        else:
            assert len(evidence) == 0, f"Diagnostic question {q['question_id']} must have empty gold_evidence"


def test_gold_evidence_missing_field_raises_error(repo_root: Path):
    """validate_evaluation_dataset must fail if gold_evidence field is missing."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    del data["questions"][0]["gold_evidence"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_evaluation_dataset(data, base_dir=repo_root)


def test_gold_evidence_invented_quote_raises_error(repo_root: Path):
    """validate_evaluation_dataset must fail if a quote is not an exact verbatim substring of the chunk."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Inject an invented quote
    data["questions"][0]["gold_evidence"][0]["quote"] = "This is a completely fabricated policy sentence that never existed."
    with pytest.raises(ValueError, match="is not an exact substring"):
        validate_evaluation_dataset(data, base_dir=repo_root)


def test_gold_evidence_quote_assigned_to_wrong_chunk_raises_error(repo_root: Path):
    """validate_evaluation_dataset must fail if a quote is assigned to a chunk that does not contain it."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Swap chunk_id to another valid chunk where this quote does not appear
    q0 = data["questions"][0]
    valid_quote = q0["gold_evidence"][0]["quote"]
    q0["gold_evidence"][0]["chunk_id"] = "DOC-SEC-001_SEC-D01"
    q0["gold_chunk_ids"] = ["DOC-SEC-001_SEC-D01"]

    with pytest.raises(ValueError, match="is not an exact substring"):
        validate_evaluation_dataset(data, base_dir=repo_root)


def test_gold_evidence_missing_quote_for_chunk_raises_error(repo_root: Path):
    """validate_evaluation_dataset must fail if an answerable multi-clause question omits a quote for a gold chunk."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Find a multi-clause question with 2 gold chunks
    mc_q = next(q for q in data["questions"] if len(q.get("gold_chunk_ids", [])) >= 2)
    mc_q["gold_evidence"] = [mc_q["gold_evidence"][0]]  # Only provide quote for first chunk

    with pytest.raises(ValueError, match="have no supporting quotes in gold_evidence"):
        validate_evaluation_dataset(data, base_dir=repo_root)


def test_gold_evidence_diagnostic_with_quote_raises_error(repo_root: Path):
    """validate_evaluation_dataset must fail if a diagnostic question provides non-empty gold_evidence."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # Find a diagnostic question
    diag_q = next(q for q in data["questions"] if not q.get("answerability"))
    diag_q["gold_evidence"] = [{"chunk_id": "DOC-TRV-001_TRV-C01", "quote": "dummy"}]

    with pytest.raises(ValueError, match="must have empty gold_evidence list"):
        validate_evaluation_dataset(data, base_dir=repo_root)


def test_canonical_audit_regression_targets(repo_root: Path):
    """Verify the 5 canonical regression targets from the evidence audit."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))
    q_by_id = {q["question_id"]: q for q in data["questions"]}

    # 1. TRV-001: meal allowance cap is in TRV-C03 (not TRV-C01)
    trv001 = q_by_id["TRV-001"]
    assert trv001["gold_chunk_ids"] == ["DOC-TRV-002_TRV-C03"]
    assert "capped at INR 2,000 per diem" in trv001["gold_evidence"][0]["quote"]
    assert "TRV-C03" in trv001["gold_chunk_ids"][0]

    # 2. FIN-006: foreign currency conversion uses RBI reference rate
    fin006 = q_by_id["FIN-006"]
    assert fin006["gold_chunk_ids"] == ["DOC-FIN-002_FIN-R02"]
    assert "Reserve Bank of India reference rate" in fin006["gold_evidence"][0]["quote"]

    # 3. PPO-003: core working hours are 10:00 AM to 4:00 PM local time
    ppo003 = q_by_id["PPO-003"]
    assert ppo003["gold_chunk_ids"] == ["DOC-PPO-002_PPO-R03"]
    assert "10:00 AM and 4:00 PM local time" in ppo003["gold_evidence"][0]["quote"]

    # 4. PRC-005: inspection warranty clause is PRC-P03 (30-day inspection warranty)
    prc005 = q_by_id["PRC-005"]
    assert "DOC-PRC-001_PRC-P03" in prc005["gold_chunk_ids"]
    assert "minimum 30-day inspection warranty" in [e["quote"] for e in prc005["gold_evidence"] if e["chunk_id"] == "DOC-PRC-001_PRC-P03"][0]

    # 5. SEC-004: incident severity levels are Priority 1 through Priority 4
    sec004 = q_by_id["SEC-004"]
    assert sec004["gold_chunk_ids"] == ["DOC-SEC-003_SEC-I01"]
    assert "Priority 1 (Critical)" in sec004["gold_evidence"][0]["quote"]
    assert "Priority 4 (Minor)" in sec004["gold_evidence"][0]["quote"]


# ==============================================================================
# 11. Corpus Manifest Validation & Dynamic Cutoff Tests
# ==============================================================================


def test_manifest_validation_distinct_steps(tmp_path: Path, repo_root: Path):
    """validate_evaluation_dataset must distinctly identify manifest errors."""
    dev_path = repo_root / "data" / "evaluation" / "dev_questions.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))

    # 1. Missing manifest
    missing_dir = tmp_path / "missing_dir"
    missing_dir.mkdir()
    (missing_dir / "data" / "processed").mkdir(parents=True)
    with pytest.raises(ValueError, match="Corpus manifest file does not exist"):
        validate_evaluation_dataset(data, base_dir=missing_dir)

    # 2. Malformed JSON manifest
    bad_json_dir = tmp_path / "bad_json_dir"
    manifest_dir = bad_json_dir / "data" / "processed"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "corpus_manifest.json").write_text("{broken json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus manifest is not valid JSON"):
        validate_evaluation_dataset(data, base_dir=bad_json_dir)

    # 3. Missing processed_fingerprint field
    no_fp_dir = tmp_path / "no_fp_dir"
    no_fp_manifest_dir = no_fp_dir / "data" / "processed"
    no_fp_manifest_dir.mkdir(parents=True)
    (no_fp_manifest_dir / "corpus_manifest.json").write_text(
        json.dumps({"chunk_count": 64, "document_count": 20}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required 'processed_fingerprint' field"):
        validate_evaluation_dataset(data, base_dir=no_fp_dir)

    # 4. Missing chunk_count or document_count
    no_cnt_dir = tmp_path / "no_cnt_dir"
    no_cnt_manifest_dir = no_cnt_dir / "data" / "processed"
    no_cnt_manifest_dir.mkdir(parents=True)
    (no_cnt_manifest_dir / "corpus_manifest.json").write_text(
        json.dumps({"processed_fingerprint": data["corpus_fingerprint"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required 'chunk_count' field"):
        validate_evaluation_dataset(data, base_dir=no_cnt_dir)

    # 5. Fingerprint mismatch
    mismatch_dir = tmp_path / "mismatch_dir"
    mismatch_manifest_dir = mismatch_dir / "data" / "processed"
    mismatch_manifest_dir.mkdir(parents=True)
    (mismatch_manifest_dir / "corpus_manifest.json").write_text(
        json.dumps({
            "processed_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000",
            "chunk_count": 64,
            "document_count": 20,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match processed corpus fingerprint"):
        validate_evaluation_dataset(data, base_dir=mismatch_dir)

    # 6. Canonical validation succeeds cleanly
    validated = validate_evaluation_dataset(dev_path, base_dir=repo_root)
    assert validated["status"] == "valid"
    assert validated["total_questions"] == 50


def test_evaluate_retriever_dynamic_cutoff_k_metrics_and_markdown():
    """evaluate_retriever_method and summary markdown support arbitrary cutoffs (K=4, K=5)."""
    mock_retriever = MagicMock()

    def make_res(cid: str) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=cid,
            text="Clause text",
            doc_id="DOC-TRV-001",
            clause_id="TRV-C01",
            title="Travel Policy v1",
            department="travel",
            version="1.0",
            source_file="policies/travel.pdf",
            source_page=1,
            effective_from="2026-01-01",
            effective_to=None,
            allowed_roles=["employee"],
        )

    mock_retriever.search.return_value = [
        make_res("chunk_1"),
        make_res("chunk_2"),
        make_res("chunk_3"),
        make_res("chunk_4"),
        make_res("chunk_5"),
    ]

    questions = [
        {
            "question_id": "Q1",
            "question": "What is the policy?",
            "role": "employee",
            "query_date": "2026-11-22",
            "question_type": "direct",
            "gold_chunk_ids": ["chunk_4"],
            "answerability": True,
        }
    ]

    # Test top_k=4
    metrics_k4 = evaluate_retriever_method(mock_retriever, questions, top_k=4)
    assert metrics_k4["cutoff"] == 4
    assert metrics_k4["hit_at_k"] == 1.0
    assert metrics_k4["recall_at_k"] == 1.0
    assert metrics_k4["hit_at_4"] == 1.0
    assert metrics_k4["recall_at_4"] == 1.0

    report_k4 = generate_summary_markdown({"TestMethod": metrics_k4}, questions)
    assert "Hit@4" in report_k4
    assert "Recall@4" in report_k4
    assert "MRR@4" in report_k4

    # Test top_k=5
    metrics_k5 = evaluate_retriever_method(mock_retriever, questions, top_k=5)
    assert metrics_k5["cutoff"] == 5
    assert metrics_k5["hit_at_k"] == 1.0
    assert metrics_k5["recall_at_k"] == 1.0
    assert metrics_k5["hit_at_5"] == 1.0
    assert metrics_k5["recall_at_5"] == 1.0

    report_k5 = generate_summary_markdown({"TestMethod": metrics_k5}, questions)
    assert "Hit@5" in report_k5
    assert "Recall@5" in report_k5
    assert "Recall@5" in report_k5
    assert "MRR@5" in report_k5
