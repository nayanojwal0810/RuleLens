"""Offline retrieval metrics and permission-bound evidence checks."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock

from rulelens.retrieval import RetrievalResult
from rulelens.retrieval_eval import (compute_hit_at_k, compute_recall_at_k, compute_mrr_at_k,
                                     compute_full_evidence_at_k, evaluate_retriever_method)


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



def test_published_question_references_are_valid():
    from pathlib import Path
    from rulelens.retrieval_eval import validate_question_dataset
    root = Path(__file__).resolve().parents[1]
    assert validate_question_dataset(root / "data/evaluation/dev_questions.json", root)["total_questions"] == 81


def test_question_validation_does_not_require_a_fixed_question_count():
    import json
    from pathlib import Path
    from rulelens.retrieval_eval import validate_question_dataset
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/evaluation/dev_questions.json").read_text(encoding="utf-8"))
    data["questions"] = data["questions"][:1]
    assert validate_question_dataset(data, root)["total_questions"] == 1


def test_question_validation_rejects_invented_quotes():
    import json
    from pathlib import Path
    import pytest
    from rulelens.retrieval_eval import validate_question_dataset
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/evaluation/dev_questions.json").read_text(encoding="utf-8"))
    data["questions"][0]["gold_evidence"][0]["quote"] = "An invented policy claim."
    with pytest.raises(ValueError, match="Gold quote"):
        validate_question_dataset(data, root)


@pytest.fixture
def question_bank():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "data/evaluation/dev_questions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("path,value,message", [
    (("schema_version",), "unsupported", "schema"),
    (("corpus_fingerprint",), "changed", "fingerprint"),
    (("questions",), [], "non-empty"),
    (("questions", 0), None, "objects"),
    (("questions", 0, "question"), " ", "required field"),
    (("questions", 1, "question_id"), "TRV-001", "Duplicate"),
    (("questions", 0, "query_date"), "2026/11/22", "date"),
    (("questions", 0, "role"), "administrator", "role"),
    (("questions", 0, "difficulty"), "expert", "difficulty"),
    (("questions", 0, "department"), "unknown", "department"),
    (("questions", 0, "question_type"), "", "question type"),
    (("questions", 0, "answerability"), "true", "answerability"),
    (("questions", 0, "gold_evidence"), [], "complete evidence"),
    (("questions", 0, "gold_evidence"), [None], "Malformed gold"),
    (("questions", 0, "gold_chunk_ids"), ["missing"], "complete evidence"),
    (("questions", 0, "gold_chunk_ids"), ["DOC-TRV-002_TRV-C03"] * 2, "Malformed gold"),
    (("questions", 0, "query_date"), "2026-03-15", "authorized and active"),
    (("questions", 0, "role"), "security", "authorized and active"),
    (("questions", 0, "restricted_target_chunk_ids"), ["DOC-TRV-004_TRV-E01"], "cannot have restricted"),
    (("questions", 0, "expected"), None, "Missing expected"),
    (("questions", 0, "expected", "route"), "cached", "status or route"),
    (("questions", 0, "expected", "status"), "insufficient", "status or route"),
    (("questions", 0, "expected", "required_fact_ids"), ["invented"], "facts do not match"),
    (("questions", 0, "gold_evidence", 0, "fact_id"), [], "facts do not match"),
    (("questions", 0, "gold_evidence", 0, "content_hash"), "changed", "source hash"),
    (("questions", 0, "gold_evidence", 0, "prerequisites"), "none", "Malformed expected fact"),
])
def test_question_validation_rejects_corrupted_labels(question_bank, path, value, message):
    from rulelens.retrieval_eval import validate_question_dataset
    target = question_bank
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_question_dataset(question_bank, Path(__file__).resolve().parents[1])


def test_refusal_and_clarification_labels_fail_closed(question_bank):
    from rulelens.retrieval_eval import validate_question_dataset
    root = Path(__file__).resolve().parents[1]
    restricted = next(q for q in question_bank["questions"] if q["expected"]["status"] == "restricted")
    restricted["role"] = "manager"
    restricted["restricted_target_chunk_ids"] = ["DOC-TRV-004_TRV-E01"]
    with pytest.raises(ValueError, match="permission diagnostic"):
        validate_question_dataset(question_bank, root)
    question_bank["questions"] = [next(q for q in question_bank["questions"] if q["expected"]["route"] == "clarification")]
    question_bank["questions"][0]["expected"]["clarification"] = None
    with pytest.raises(ValueError, match="clarification needs"):
        validate_question_dataset(question_bank, root)


def test_refusal_gold_evidence_is_rejected(question_bank):
    from rulelens.retrieval_eval import validate_question_dataset
    refusal = next(q for q in question_bank["questions"] if not q["answerability"])
    refusal["gold_evidence"] = question_bank["questions"][0]["gold_evidence"]
    with pytest.raises(ValueError, match="must not include answer evidence"):
        validate_question_dataset(question_bank, Path(__file__).resolve().parents[1])


def test_legacy_retrieval_schema_remains_supported(question_bank):
    from rulelens.retrieval_eval import validate_question_dataset
    question_bank["schema_version"] = "1.0"
    question_bank["questions"] = question_bank["questions"][:1]
    question_bank["questions"][0].pop("expected")
    assert validate_question_dataset(question_bank, Path(__file__).resolve().parents[1])["status"] == "valid"
