"""Offline deterministic tests for RuleLens hybrid retrieval foundation.

Verifies:
- Tokenization (lowercase, number retention, no stopword removal, empty queries).
- BM25 indexing, exact keyword search, numerical queries, tie-breaking, safe empty corpus handling.
- BGE embedding interface contract (CPU, dim 384, normalized, finite values, blank query rejection).
- Reciprocal Rank Fusion (pure function, manual formula checks, deduplication, ties, rank/limit validation).
- Role-based permissions across all 6 roles, explicit role restrictions.
- Date boundary filtering, historical version availability, explicit empty date rejection.
- Security enforcement: exclusion of real unauthorized restricted chunk (DOC-TRV-004_TRV-E01) for employee,
  acceptance for manager, and exclusion of expired dense candidate versions.
- Full citation metadata preservation including department.
- Strict offline execution with blocked network sockets.
"""

from __future__ import annotations

import datetime
import math
import shutil
import socket
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rulelens.config import GENERATION_EVIDENCE_LIMIT
from rulelens.dataset import ROLES
from rulelens.retrieval import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_CANDIDATE_DEPTH,
    EMBEDDING_DIMENSION,
    RRF_K,
    BGEEmbedder,
    BM25Index,
    Candidate,
    HybridRetriever,
    RetrievalResult,
    filter_eligible_chunks,
    is_chunk_eligible,
    load_validated_corpus,
    normalize_dense_candidates,
    parse_query_date,
    reciprocal_rank_fusion,
    tokenize,
)


@pytest.fixture
def repo_root() -> Path:
    """Return the absolute path of the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_chunks(repo_root: Path) -> List[Dict[str, Any]]:
    """Load the validated 64 corpus chunks."""
    return load_validated_corpus(repo_root)


# Corpus Loading & Integrity Verification
def test_load_validated_corpus_loads_all_64_chunks(repo_root: Path):
    """Corpus loader must load all 64 chunks and verify fingerprint integrity."""
    chunks = load_validated_corpus(repo_root)
    assert len(chunks) == 64
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "doc_id" in chunk
        assert "clause_id" in chunk
        assert "title" in chunk
        assert "department" in chunk
        assert "version" in chunk
        assert "source_file" in chunk
        assert "source_page" in chunk
        assert "effective_from" in chunk
        assert "allowed_roles" in chunk


def test_load_validated_corpus_rejects_corrupted_fingerprint(tmp_path: Path, repo_root: Path):
    """Corpus loader must raise ValueError if corpus_manifest.json fingerprint does not match."""
    shutil.copytree(repo_root / "data" / "source", tmp_path / "data" / "source")
    shutil.copytree(repo_root / "data" / "processed", tmp_path / "data" / "processed")

    manifest_path = tmp_path / "data" / "processed" / "corpus_manifest.json"
    content = manifest_path.read_text(encoding="utf-8")
    corrupted = content.replace('"processed_fingerprint": "', '"processed_fingerprint": "corrupted_')
    manifest_path.write_text(corrupted, encoding="utf-8")

    with pytest.raises(ValueError):
        load_validated_corpus(tmp_path)


def test_load_validated_corpus_rejects_tampered_corpus_despite_recomputed_fingerprint(tmp_path: Path, repo_root: Path):
    """Corpus loader must call validate_corpus and reject tampered content even with recomputed fingerprints."""
    import json
    from rulelens.ingestion import calculate_processed_fingerprint, calculate_source_fingerprint

    shutil.copytree(repo_root / "data" / "source", tmp_path / "data" / "source")
    shutil.copytree(repo_root / "data" / "processed", tmp_path / "data" / "processed")

    chunks_path = tmp_path / "data" / "processed" / "chunks.jsonl"
    rules_path = tmp_path / "data" / "processed" / "rules.jsonl"
    manifest_path = tmp_path / "data" / "processed" / "corpus_manifest.json"

    # Tamper with chunk text
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    chunks[0]["text"] = chunks[0]["text"] + " TAMPERED POLICY CONTENT"

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, sort_keys=True) + "\n")

    rules = []
    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))

    # Recompute fingerprints so simple fingerprint checks would naively pass
    new_fp = calculate_processed_fingerprint(chunks, rules)
    source_fp = calculate_source_fingerprint(tmp_path / "data" / "source")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["processed_fingerprint"] = new_fp
    manifest["source_fingerprint"] = source_fp
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # validate_corpus() must detect mismatch against PDF ground truth and reject
    with pytest.raises(ValueError):
        load_validated_corpus(tmp_path)


# Tokenization & BM25 Lexical Search
def test_tokenize_lowercase_and_retains_numbers():
    """Tokenize must lowercase terms, retain numbers, and normalize comma-separated numbers."""
    tokens = tokenize("Asterline Travel Policy v1 limits hotel to 250 USD and flight to 40,000 USD.")
    assert "asterline" in tokens
    assert "travel" in tokens
    assert "policy" in tokens
    assert "v1" in tokens
    assert "250" in tokens
    assert "40000" in tokens
    assert "usd" in tokens


def test_tokenize_empty_and_whitespace_queries():
    """Empty or whitespace-only queries must produce empty token lists."""
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("\t\n  \r\n") == []


def test_bm25_empty_query_returns_no_results(sample_chunks: List[Dict[str, Any]]):
    """Empty or whitespace query against BM25 index must return an empty list."""
    index = BM25Index(sample_chunks)
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("\t\n") == []


def test_bm25_empty_corpus_handled_safely():
    """BM25Index over an empty corpus must handle searches safely without division by zero."""
    empty_index = BM25Index([])
    assert empty_index.search("anything") == []
    assert empty_index.search("") == []


def test_bm25_rejects_negative_top_k(sample_chunks: List[Dict[str, Any]]):
    """BM25Index must raise ValueError on negative top_k."""
    index = BM25Index(sample_chunks)
    with pytest.raises(ValueError, match="Invalid top_k"):
        index.search("flight", top_k=-1)


def test_bm25_exact_keyword_search(sample_chunks: List[Dict[str, Any]]):
    """BM25 search must find relevant chunks by exact keyword."""
    index = BM25Index(sample_chunks)
    results = index.search("domestic air travel economy", top_k=5)
    assert len(results) > 0
    top_result = results[0]
    assert top_result.score > 0.0
    assert "TRV-C02" in top_result.chunk_id
    assert "economy" in top_result.chunk["text"].lower()


def test_bm25_numerical_terms_40000_and_50000(sample_chunks: List[Dict[str, Any]]):
    """BM25 search must match numerical terms like 40000 and 50000."""
    index = BM25Index(sample_chunks)

    res_50000 = index.search("50000", top_k=5)
    assert len(res_50000) > 0
    assert any("50,000" in r.chunk["text"] or "50000" in r.chunk["text"] for r in res_50000)
    assert all(r.score > 0 for r in res_50000)

    res_40000 = index.search("40000", top_k=5)
    assert len(res_40000) > 0
    assert any("40,000" in r.chunk["text"] or "40000" in r.chunk["text"] for r in res_40000)
    assert all(r.score > 0 for r in res_40000)


def test_bm25_deterministic_tie_breaking():
    """BM25 equal-score ties must be broken deterministically by chunk_id."""
    fake_chunks = [
        {
            "chunk_id": "CHUNK-Z",
            "text": "identical target policy content",
            "doc_id": "D1",
            "clause_id": "C1",
            "title": "T1",
            "department": "Travel",
            "version": "1.0",
            "source_file": "f1.pdf",
            "source_page": 1,
            "effective_from": "2026-01-01",
            "effective_to": None,
            "allowed_roles": ["employee"],
        },
        {
            "chunk_id": "CHUNK-A",
            "text": "identical target policy content",
            "doc_id": "D2",
            "clause_id": "C2",
            "title": "T2",
            "department": "Travel",
            "version": "1.0",
            "source_file": "f2.pdf",
            "source_page": 1,
            "effective_from": "2026-01-01",
            "effective_to": None,
            "allowed_roles": ["employee"],
        },
        {
            "chunk_id": "CHUNK-M",
            "text": "identical target policy content",
            "doc_id": "D3",
            "clause_id": "C3",
            "title": "T3",
            "department": "Travel",
            "version": "1.0",
            "source_file": "f3.pdf",
            "source_page": 1,
            "effective_from": "2026-01-01",
            "effective_to": None,
            "allowed_roles": ["employee"],
        },
    ]

    index = BM25Index(fake_chunks)
    results = index.search("target policy", top_k=3)
    assert len(results) == 3
    # Equal BM25 scores; tie-break must sort alphabetically by chunk_id
    assert [r.chunk_id for r in results] == ["CHUNK-A", "CHUNK-M", "CHUNK-Z"]
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3


# Reciprocal Rank Fusion (RRF)
def test_rrf_pure_calculation_with_known_ranks():
    """Verify RRF formula calculation exactly matches 1/(k + rank) with k=60."""
    chunk_a = {
        "chunk_id": "C-A",
        "text": "Text A",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
        "department": "Travel",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    chunk_b = {
        "chunk_id": "C-B",
        "text": "Text B",
        "doc_id": "D2",
        "clause_id": "C2",
        "title": "T2",
        "department": "Travel",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    chunk_c = {
        "chunk_id": "C-C",
        "text": "Text C",
        "doc_id": "D3",
        "clause_id": "C3",
        "title": "T3",
        "department": "Travel",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }

    # C-A: BM25 rank 1, dense rank 2
    # C-B: BM25 rank 2, not in dense
    # C-C: not in BM25, dense rank 1
    bm25_cands = [
        Candidate(chunk_id="C-A", score=10.0, rank=1, chunk=chunk_a),
        Candidate(chunk_id="C-B", score=8.0, rank=2, chunk=chunk_b),
    ]
    dense_cands = [
        Candidate(chunk_id="C-C", score=0.95, rank=1, chunk=chunk_c),
        Candidate(chunk_id="C-A", score=0.90, rank=2, chunk=chunk_a),
    ]

    results = reciprocal_rank_fusion(
        bm25_candidates=bm25_cands,
        dense_candidates=dense_cands,
        chunks_lookup={"C-A": chunk_a, "C-B": chunk_b, "C-C": chunk_c},
        k=60,
        limit=4,
    )

    # Expected exact scores:
    # C-A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.03252247488...
    # C-C: 1/(60+1) = 1/61 = 0.01639344262...
    # C-B: 1/(60+2) = 1/62 = 0.01612903225...
    assert len(results) == 3
    assert results[0].chunk_id == "C-A"
    assert math.isclose(results[0].rrf_score, 1 / 61 + 1 / 62, rel_tol=1e-9)
    assert results[0].bm25_rank == 1
    assert results[0].dense_rank == 2

    assert results[1].chunk_id == "C-C"
    assert math.isclose(results[1].rrf_score, 1 / 61, rel_tol=1e-9)
    assert results[1].bm25_rank is None
    assert results[1].dense_rank == 1

    assert results[2].chunk_id == "C-B"
    assert math.isclose(results[2].rrf_score, 1 / 62, rel_tol=1e-9)
    assert results[2].bm25_rank == 2
    assert results[2].dense_rank is None


def test_rrf_same_modality_repeated_candidate_cannot_inflate_score():
    """A candidate duplicated within the same retrieval method must be counted once and not inflate score."""
    chunk = {
        "chunk_id": "C-1",
        "text": "Text 1",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
        "department": "Finance",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    # Duplicate C-1 in BM25 candidates list with different ranks
    cands_bm25 = [
        Candidate(chunk_id="C-1", score=10.0, rank=1, chunk=chunk),
        Candidate(chunk_id="C-1", score=8.0, rank=2, chunk=chunk),
    ]

    results = reciprocal_rank_fusion(
        bm25_candidates=cands_bm25,
        dense_candidates=[],
        chunks_lookup={"C-1": chunk},
        k=60,
        limit=4,
    )
    assert len(results) == 1
    # Must only receive 1/(60+1) = 1/61, NOT (1/61 + 1/62)
    assert math.isclose(results[0].rrf_score, 1 / 61, rel_tol=1e-9)
    assert results[0].bm25_rank == 1


def test_rrf_rejects_invalid_ranks_and_duplicate_ranks_for_different_chunks():
    """RRF must reject ranks < 1 and duplicate ranks assigned to different chunks within a method."""
    chunk1 = {
        "chunk_id": "C-1",
        "text": "1",
        "doc_id": "D",
        "clause_id": "C1",
        "title": "T",
        "department": "Finance",
        "version": "1",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    chunk2 = {
        "chunk_id": "C-2",
        "text": "2",
        "doc_id": "D",
        "clause_id": "C2",
        "title": "T",
        "department": "Finance",
        "version": "1",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    lookup = {"C-1": chunk1, "C-2": chunk2}

    # Rank 0 is invalid
    with pytest.raises(ValueError, match="Ranks must be >= 1"):
        reciprocal_rank_fusion(
            [Candidate(chunk_id="C-1", score=5.0, rank=0)],
            [],
            chunks_lookup=lookup,
        )

    # Rank -1 is invalid
    with pytest.raises(ValueError, match="Ranks must be >= 1"):
        reciprocal_rank_fusion(
            [Candidate(chunk_id="C-1", score=5.0, rank=-1)],
            [],
            chunks_lookup=lookup,
        )

    # Duplicate rank 1 assigned to different chunks within BM25
    with pytest.raises(ValueError, match="Duplicate rank 1 assigned to different chunks"):
        reciprocal_rank_fusion(
            [
                Candidate(chunk_id="C-1", score=5.0, rank=1),
                Candidate(chunk_id="C-2", score=4.0, rank=1),
            ],
            [],
            chunks_lookup=lookup,
        )


def test_rrf_rejects_invalid_k_limit_and_candidate_depth():
    """RRF must reject k <= 0, negative limit, and candidate_depth <= 0."""
    with pytest.raises(ValueError, match="Invalid RRF constant"):
        reciprocal_rank_fusion([], [], k=0)

    with pytest.raises(ValueError, match="Invalid limit"):
        reciprocal_rank_fusion([], [], limit=-1)

    with pytest.raises(ValueError, match="Invalid candidate_depth"):
        reciprocal_rank_fusion([], [], candidate_depth=0)


def test_rrf_never_trusts_arbitrary_candidate_metadata_over_lookup():
    """RRF must strictly use canonical chunks_lookup and ignore arbitrary candidate metadata."""
    canonical_chunk = {
        "chunk_id": "C-TRUST",
        "text": "Canonical Text",
        "doc_id": "DOC-AUTH",
        "clause_id": "AUTH-01",
        "title": "Canonical Authoritative Title",
        "department": "Security",
        "version": "1.0",
        "source_file": "auth.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["security"],
    }

    tampered_candidate_metadata = {
        "chunk_id": "C-TRUST",
        "score": 0.99,
        "rank": 1,
        "chunk": {
            "chunk_id": "C-TRUST",
            "title": "Spoofed Fake Title",
            "department": "Fake Department",
        },
    }

    results = reciprocal_rank_fusion(
        bm25_candidates=[],
        dense_candidates=[tampered_candidate_metadata],
        chunks_lookup={"C-TRUST": canonical_chunk},
    )
    assert len(results) == 1
    assert results[0].title == "Canonical Authoritative Title"
    assert results[0].department == "Security"


def test_rrf_single_modality_bm25_only_and_dense_only():
    """RRF must work seamlessly with only BM25 or only dense candidate lists."""
    chunk = {
        "chunk_id": "C-1",
        "text": "Text 1",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
        "department": "Finance",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    cand = Candidate(chunk_id="C-1", score=4.5, rank=1, chunk=chunk)

    res_bm25_only = reciprocal_rank_fusion(
        bm25_candidates=[cand], dense_candidates=[], chunks_lookup={"C-1": chunk}, limit=4
    )
    assert len(res_bm25_only) == 1
    assert res_bm25_only[0].bm25_rank == 1
    assert res_bm25_only[0].dense_rank is None
    assert math.isclose(res_bm25_only[0].rrf_score, 1 / 61, rel_tol=1e-9)

    res_dense_only = reciprocal_rank_fusion(
        bm25_candidates=[], dense_candidates=[cand], chunks_lookup={"C-1": chunk}, limit=4
    )
    assert len(res_dense_only) == 1
    assert res_dense_only[0].bm25_rank is None
    assert res_dense_only[0].dense_rank == 1
    assert math.isclose(res_dense_only[0].rrf_score, 1 / 61, rel_tol=1e-9)


def test_rrf_merged_rank_ordering_and_tie_breaking():
    """Ties in RRF scores must be broken deterministically by chunk_id."""
    chunk_x = {
        "chunk_id": "DOC-X",
        "text": "X",
        "doc_id": "D",
        "clause_id": "C",
        "title": "T",
        "department": "Travel",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    chunk_y = {
        "chunk_id": "DOC-Y",
        "text": "Y",
        "doc_id": "D",
        "clause_id": "C",
        "title": "T",
        "department": "Travel",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }

    bm25_cands = [Candidate(chunk_id="DOC-Y", score=2.0, rank=1, chunk=chunk_y)]
    dense_cands = [Candidate(chunk_id="DOC-X", score=0.9, rank=1, chunk=chunk_x)]

    results = reciprocal_rank_fusion(
        bm25_cands, dense_cands, chunks_lookup={"DOC-X": chunk_x, "DOC-Y": chunk_y}, limit=2
    )
    # Both have RRF score 1/61; DOC-X should come before DOC-Y alphabetically
    assert [r.chunk_id for r in results] == ["DOC-X", "DOC-Y"]


# Role Permissions & Date Boundary Eligibility
def test_role_authorization_all_six_roles(sample_chunks: List[Dict[str, Any]]):
    """Every approved role must have positive chunk eligibility."""
    assert len(ROLES) == 6
    for role in ROLES:
        eligible = filter_eligible_chunks(sample_chunks, role=role, query_date="2026-03-15")
        assert len(eligible) > 0, f"Role {role} should have eligible chunks"


def test_employee_restriction_from_travel_exceptions(sample_chunks: List[Dict[str, Any]]):
    """Employee role must be strictly prohibited from Travel Exceptions (DOC-TRV-004)."""
    trv_exp_chunks = [c for c in sample_chunks if c["doc_id"] == "DOC-TRV-004"]
    assert len(trv_exp_chunks) > 0

    for chunk in trv_exp_chunks:
        assert not is_chunk_eligible(chunk, role="employee", query_date="2026-03-15")


def test_auditor_restriction_from_employee_transfers(sample_chunks: List[Dict[str, Any]]):
    """Auditor role must be strictly prohibited from Employee Transfers (DOC-PPO-004)."""
    ppo_transfers = [c for c in sample_chunks if c["doc_id"] == "DOC-PPO-004"]
    assert len(ppo_transfers) > 0

    for chunk in ppo_transfers:
        assert not is_chunk_eligible(chunk, role="auditor", query_date="2026-03-15")


def test_historical_travel_policy_v1_on_2026_03_15(sample_chunks: List[Dict[str, Any]]):
    """Travel Policy v1 must be active on 2026-03-15 while Travel Policy v2 must be inactive."""
    v1_chunks = [c for c in sample_chunks if c["doc_id"] == "DOC-TRV-001"]
    v2_chunks = [c for c in sample_chunks if c["doc_id"] == "DOC-TRV-002"]

    query_date = "2026-03-15"
    for c in v1_chunks:
        assert is_chunk_eligible(c, role="employee", query_date=query_date)

    for c in v2_chunks:
        assert not is_chunk_eligible(c, role="employee", query_date=query_date)


def test_travel_policy_v2_on_2026_11_22(sample_chunks: List[Dict[str, Any]]):
    """Travel Policy v2 must be active on 2026-11-22 while Travel Policy v1 must be expired."""
    v1_chunks = [c for c in sample_chunks if c["doc_id"] == "DOC-TRV-001"]
    v2_chunks = [c for c in sample_chunks if c["doc_id"] == "DOC-TRV-002"]

    query_date = "2026-11-22"
    for c in v1_chunks:
        assert not is_chunk_eligible(c, role="employee", query_date=query_date)

    for c in v2_chunks:
        assert is_chunk_eligible(c, role="employee", query_date=query_date)


def test_correct_version_boundaries_2026_06_30_and_2026_07_01(sample_chunks: List[Dict[str, Any]]):
    """Boundary test: 2026-06-30 selects v1; 2026-07-01 selects v2."""
    v1_sample = next(c for c in sample_chunks if c["doc_id"] == "DOC-TRV-001")
    v2_sample = next(c for c in sample_chunks if c["doc_id"] == "DOC-TRV-002")

    # On 2026-06-30 (last day of v1)
    assert is_chunk_eligible(v1_sample, role="employee", query_date="2026-06-30")
    assert not is_chunk_eligible(v2_sample, role="employee", query_date="2026-06-30")

    # On 2026-07-01 (first day of v2)
    assert not is_chunk_eligible(v1_sample, role="employee", query_date="2026-07-01")
    assert is_chunk_eligible(v2_sample, role="employee", query_date="2026-07-01")


def test_invalid_role_and_date_rejection(sample_chunks: List[Dict[str, Any]]):
    """Unknown roles, malformed dates, and explicit empty strings must raise ValueError."""
    sample = sample_chunks[0]

    with pytest.raises(ValueError, match="Unknown role 'contractor'"):
        is_chunk_eligible(sample, role="contractor", query_date="2026-05-01")

    with pytest.raises(ValueError, match="Unknown role 'superadmin'"):
        filter_eligible_chunks(sample_chunks, role="superadmin", query_date="2026-05-01")

    with pytest.raises(ValueError, match="Malformed query date 'not-a-valid-date'"):
        is_chunk_eligible(sample, role="employee", query_date="not-a-valid-date")

    with pytest.raises(ValueError, match="Malformed query date '2026/05/10'"):
        parse_query_date("2026/05/10")

    with pytest.raises(ValueError, match="explicit empty string"):
        parse_query_date("")

    with pytest.raises(ValueError, match="explicit empty string"):
        parse_query_date("   ")


def test_preservation_of_citation_and_department_metadata(sample_chunks: List[Dict[str, Any]]):
    """Returned RetrievalResult must preserve all required citation fields including department."""
    retriever = HybridRetriever(chunks=sample_chunks)
    results = retriever.search(query="domestic air travel", role="employee", query_date="2026-11-22", top_k=2, mode="bm25")

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id.startswith("DOC-")
    assert top.doc_id.startswith("DOC-")
    assert top.clause_id != ""
    assert top.title != ""
    assert top.department == "Travel"
    assert top.version != ""
    assert top.source_file.endswith(".pdf")
    assert top.source_page >= 1
    assert top.effective_from != ""
    assert "employee" in top.allowed_roles
    assert top.rrf_score > 0.0

    as_dict = top.to_dict()
    assert as_dict["department"] == "Travel"
    assert as_dict["chunk_id"] == top.chunk_id


# Security Enforcements & Candidate Filtering
def test_security_real_unauthorized_dense_candidate_excluded(sample_chunks: List[Dict[str, Any]]):
    """Verify real restricted chunk DOC-TRV-004_TRV-E01 is excluded for employee, accepted for manager."""
    retriever = HybridRetriever(chunks=sample_chunks)

    # 1. Confirm DOC-TRV-004_TRV-E01 exists in the loaded corpus
    target_id = "DOC-TRV-004_TRV-E01"
    assert any(c["chunk_id"] == target_id for c in sample_chunks), f"{target_id} must exist in corpus"

    permitted_id = "DOC-TRV-002_TRV-C02"

    # Supply unauthorized chunk as top-scoring dense candidate
    dense_candidates = [
        Candidate(chunk_id=target_id, score=0.99, rank=1),
        Candidate(chunk_id=permitted_id, score=0.85, rank=2),
    ]

    # For employee on 2026-11-22:
    emp_results = retriever.search(
        query="travel exception accommodation",
        role="employee",
        query_date="2026-11-22",
        dense_candidates=dense_candidates,
        top_k=4,
    )
    # Target restricted chunk must be completely excluded
    assert all(r.chunk_id != target_id for r in emp_results)
    assert all(r.doc_id != "DOC-TRV-004" for r in emp_results)

    # Other permitted candidate must still be returned with consecutive dense rank 1
    returned_cids = [r.chunk_id for r in emp_results]
    assert permitted_id in returned_cids
    perm_result = next(r for r in emp_results if r.chunk_id == permitted_id)
    assert perm_result.dense_rank == 1
    assert perm_result.dense_score == 0.85

    # For manager on 2026-11-22 (manager has access to Travel Exceptions):
    mgr_results = retriever.search(
        query="travel exception accommodation",
        role="manager",
        query_date="2026-11-22",
        dense_candidates=dense_candidates,
        top_k=4,
    )
    mgr_cids = [r.chunk_id for r in mgr_results]
    assert target_id in mgr_cids, "Authorized manager must be able to retrieve DOC-TRV-004_TRV-E01"


def test_security_expired_dense_candidate_excluded(sample_chunks: List[Dict[str, Any]]):
    """Dense candidate belonging to an expired policy version must be excluded."""
    retriever = HybridRetriever(chunks=sample_chunks)

    # DOC-TRV-001_TRV-C02 is Travel Policy v1 (expired on 2026-06-30)
    expired_id = "DOC-TRV-001_TRV-C02"
    active_id = "DOC-TRV-002_TRV-C02"

    dense_candidates = [
        Candidate(chunk_id=expired_id, score=0.98, rank=1),
        Candidate(chunk_id=active_id, score=0.85, rank=2),
    ]

    # Query on 2026-11-22 (v1 is expired):
    results = retriever.search(
        query="domestic flight economy",
        role="employee",
        query_date="2026-11-22",
        dense_candidates=dense_candidates,
        top_k=4,
    )
    result_cids = [r.chunk_id for r in results]
    assert expired_id not in result_cids, "Expired v1 candidate must not be returned on current date"
    assert active_id in result_cids

    # Query on 2026-03-15 (v1 is active, v2 is inactive):
    hist_results = retriever.search(
        query="domestic flight economy",
        role="employee",
        query_date="2026-03-15",
        dense_candidates=dense_candidates,
        top_k=4,
    )
    hist_cids = [r.chunk_id for r in hist_results]
    assert expired_id in hist_cids, "v1 candidate must be returned for historical date"
    assert active_id not in hist_cids, "v2 candidate must not be returned for historical date"


def test_empty_or_whitespace_query_produces_no_results_even_with_dense(sample_chunks: List[Dict[str, Any]]):
    """Empty or whitespace query string must return [] when role and date are valid, even with dense candidates."""
    retriever = HybridRetriever(chunks=sample_chunks)
    dense_candidates = [
        Candidate(chunk_id="DOC-TRV-002_TRV-C02", score=0.99, rank=1),
    ]

    assert retriever.search("", role="employee", dense_candidates=dense_candidates) == []
    assert retriever.search("   ", role="employee", dense_candidates=dense_candidates) == []
    assert retriever.search("\t\n", role="employee", dense_candidates=dense_candidates) == []


def test_search_input_validation_order_and_empty_queries(sample_chunks: List[Dict[str, Any]]):
    """Verify role and date validation precede empty query check in HybridRetriever.search."""
    retriever = HybridRetriever(chunks=sample_chunks)
    dense_candidates = [
        Candidate(chunk_id="DOC-TRV-002_TRV-C02", score=0.99, rank=1),
    ]

    # 1. Empty query with unknown role 'contractor' raises ValueError
    with pytest.raises(ValueError, match="Unknown role 'contractor'"):
        retriever.search("", role="contractor", query_date="2026-11-22")

    # 2. Whitespace query with unknown role 'superadmin' raises ValueError
    with pytest.raises(ValueError, match="Unknown role 'superadmin'"):
        retriever.search("   ", role="superadmin", query_date="2026-11-22")

    # 3. Empty query with invalid date '2026/11/22' raises ValueError
    with pytest.raises(ValueError, match="Malformed query date '2026/11/22'"):
        retriever.search("", role="employee", query_date="2026/11/22")

    # 4. Empty query with explicit empty date '' raises ValueError
    with pytest.raises(ValueError, match="explicit empty string"):
        retriever.search("", role="employee", query_date="")

    # 5. Valid role and date with empty query returns []
    assert retriever.search("", role="employee", query_date="2026-11-22", mode="bm25") == []

    # 5a. Empty query with invalid mode raises ValueError before returning []
    with pytest.raises(ValueError, match="Unknown retrieval mode 'invalid'"):
        retriever.search("", role="employee", query_date="2026-11-22", mode="invalid")

    # 5b. Empty query with invalid dense_backend raises ValueError before returning []
    with pytest.raises(ValueError, match="Unknown dense backend 'invalid'"):
        retriever.search("", role="employee", query_date="2026-11-22", dense_backend="invalid")

    # 6. Valid role and date with whitespace query returns [], even when dense candidates are supplied
    assert retriever.search("   ", role="employee", query_date="2026-11-22", dense_candidates=dense_candidates) == []
    assert retriever.search("\t\n", role="employee", query_date="2026-11-22", dense_candidates=dense_candidates) == []

    # 7. A normal query still retrieves eligible chunks
    results = retriever.search("domestic flight", role="employee", query_date="2026-11-22", top_k=2, mode="bm25")
    assert len(results) > 0
    assert all("employee" in r.allowed_roles for r in results)


def test_hybrid_retriever_rejects_negative_limits_and_depth(sample_chunks: List[Dict[str, Any]]):
    """HybridRetriever must reject negative top_k and candidate_depth <= 0."""
    retriever = HybridRetriever(chunks=sample_chunks)

    with pytest.raises(ValueError, match="Invalid top_k"):
        retriever.search("flight", role="employee", top_k=-1)

    with pytest.raises(ValueError, match="Invalid candidate_depth"):
        retriever.search("flight", role="employee", candidate_depth=0)


# Offline Fake Embedding Wrapper Interface
class FakeSentenceTransformerModel:
    """Deterministic fake model for testing BGEEmbedder without network access or downloading weights."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def encode(
        self,
        sentences: List[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        import hashlib
        import numpy as np

        embeddings = []
        for s in sentences:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            vec = [(b / 255.0) - 0.5 for b in h[:16]]
            full_vec = (vec * (self.dimension // len(vec) + 1))[: self.dimension]
            arr = np.array(full_vec, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            embeddings.append(arr)
        return np.array(embeddings)


def test_bge_embedder_interface_with_fake_model():
    """Verify BGEEmbedder interface with CPU, dimension 384, and normalization."""
    fake_model = FakeSentenceTransformerModel(dimension=384)
    embedder = BGEEmbedder(model_instance=fake_model)

    query_vec = embedder.embed_query("hotel accommodation expense")
    assert len(query_vec) == 384
    norm = math.sqrt(sum(x * x for x in query_vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    batch_vecs = embedder.embed_texts(["doc1 chunk text", "doc2 chunk text"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384


def test_bge_embed_query_rejects_blank_queries():
    """BGEEmbedder.embed_query must raise ValueError on blank or whitespace queries."""
    fake_model = FakeSentenceTransformerModel(dimension=384)
    embedder = BGEEmbedder(model_instance=fake_model)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        embedder.embed_query("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        embedder.embed_query("   \t\n")


def test_bge_wrapper_validates_dimension_and_finite_values():
    """BGEEmbedder must reject invalid embedding dimensions and non-finite values."""
    # Wrong dimension
    class WrongDimModel:
        def encode(self, *args, **kwargs):
            import numpy as np

            return np.ones((1, 512), dtype=np.float32)

    embedder_wrong_dim = BGEEmbedder(model_instance=WrongDimModel())
    with pytest.raises(ValueError, match="Expected embedding dimension 384"):
        embedder_wrong_dim.embed_texts(["test text"])

    # Non-finite values
    class NonFiniteModel:
        def encode(self, *args, **kwargs):
            import numpy as np

            arr = np.ones((1, 384), dtype=np.float32)
            arr[0, 5] = float("nan")
            return arr

    embedder_non_finite = BGEEmbedder(model_instance=NonFiniteModel())
    with pytest.raises(ValueError, match="non-finite values"):
        embedder_non_finite.embed_texts(["test text"])


# Hybrid Retrieval & Caching Enhancements
def test_query_instruction_applied_to_query_only():
    """Verify BGE_QUERY_INSTRUCTION is prepended in embed_query but not in embed_texts."""
    captured_texts = []

    class RecordingModel:
        def encode(self, sentences, **kwargs):
            import numpy as np

            captured_texts.extend(sentences)
            arr = np.ones((len(sentences), 384), dtype=np.float32)
            for i in range(len(sentences)):
                arr[i] = arr[i] / np.linalg.norm(arr[i])
            return arr

    rec_model = RecordingModel()
    embedder = BGEEmbedder(model_instance=rec_model)

    embedder.embed_query("hotel accommodation expense")
    assert len(captured_texts) == 1
    assert captured_texts[0] == f"{BGE_QUERY_INSTRUCTION}hotel accommodation expense"

    captured_texts.clear()
    embedder.embed_texts(["Hotel policy text", "Flight allowance text"])
    assert len(captured_texts) == 2
    assert captured_texts[0] == "Hotel policy text"
    assert captured_texts[1] == "Flight allowance text"
    assert not captured_texts[0].startswith(BGE_QUERY_INSTRUCTION)
    assert not captured_texts[1].startswith(BGE_QUERY_INSTRUCTION)


def test_bm25_index_cache_reuse(repo_root: Path):
    """Verify retriever caches BM25Index per eligible chunk set."""
    fake_model = FakeSentenceTransformerModel(dimension=384)
    embedder = BGEEmbedder(model_instance=fake_model)
    retriever = HybridRetriever(base_dir=repo_root, embedder=embedder)

    eligible_1 = filter_eligible_chunks(retriever.chunks, "employee", "2026-11-22")
    index_1a = retriever.get_bm25_index(eligible_1)
    index_1b = retriever.get_bm25_index(eligible_1)
    assert index_1a is index_1b

    eligible_2 = filter_eligible_chunks(retriever.chunks, "manager", "2026-11-22")
    index_2 = retriever.get_bm25_index(eligible_2)
    assert index_1a is not index_2


def test_dense_embeddings_cache_reuse(repo_root: Path):
    """Verify retriever computes 64 chunk embeddings once and reuses them."""
    call_count = 0

    class CountingModel(FakeSentenceTransformerModel):
        def encode(self, sentences, **kwargs):
            nonlocal call_count
            call_count += 1
            return super().encode(sentences, **kwargs)

    counting_model = CountingModel(dimension=384)
    embedder = BGEEmbedder(model_instance=counting_model)
    retriever = HybridRetriever(base_dir=repo_root, embedder=embedder)

    assert retriever._dense_embeddings_cache is None
    emb1 = retriever.get_dense_embeddings()
    assert len(emb1) == 64
    assert call_count == 1

    emb2 = retriever.get_dense_embeddings()
    assert emb1 is emb2
    assert call_count == 1  # No re-computation


def test_retrieval_modes(repo_root: Path):
    """Verify explicit mode selection: bm25, dense, hybrid."""
    dense_called = 0

    class TrackingModel(FakeSentenceTransformerModel):
        def encode(self, sentences, **kwargs):
            nonlocal dense_called
            dense_called += 1
            return super().encode(sentences, **kwargs)

    tracking_model = TrackingModel(dimension=384)
    embedder = BGEEmbedder(model_instance=tracking_model)
    retriever = HybridRetriever(base_dir=repo_root, embedder=embedder)

    # 1. BM25 mode: should not invoke dense embedding
    dense_called = 0
    bm25_results = retriever.search(
        query="meal per diem allowance",
        role="employee",
        query_date="2026-11-22",
        mode="bm25",
        top_k=4,
    )
    assert len(bm25_results) > 0
    assert dense_called == 0
    for r in bm25_results:
        assert r.bm25_rank is not None
        assert r.dense_rank is None

    # 2. Dense mode: should invoke dense search and have dense_rank
    dense_results = retriever.search(
        query="meal per diem allowance",
        role="employee",
        query_date="2026-11-22",
        mode="dense",
        top_k=4,
    )
    assert len(dense_results) > 0
    assert dense_called > 0
    for r in dense_results:
        assert r.dense_rank is not None
        assert r.bm25_rank is None

    # 3. Hybrid mode: should run both and fuse with RRF
    hybrid_results = retriever.search(
        query="meal per diem allowance",
        role="employee",
        query_date="2026-11-22",
        mode="hybrid",
        top_k=4,
    )
    assert len(hybrid_results) > 0
    for r in hybrid_results:
        assert r.rrf_score is not None


def test_normalize_dense_candidates_sorting_and_deduplication(sample_chunks: List[Dict[str, Any]]):
    """normalize_dense_candidates must sort by score descending, deduplicate keeping highest score, and assign 1..N ranks."""
    lookup = {c["chunk_id"]: c for c in sample_chunks}
    eligible_ids = set(lookup.keys())

    raw_candidates = [
        {"chunk_id": "DOC-TRV-001_TRV-C01", "score": 0.75},
        {"chunk_id": "DOC-TRV-001_TRV-C02", "score": 0.90},
        {"chunk_id": "DOC-TRV-001_TRV-C01", "score": 0.85},  # duplicate with higher score
    ]

    normalized = normalize_dense_candidates(
        raw_candidates=raw_candidates,
        eligible_ids=eligible_ids,
        chunks_by_id=lookup,
        candidate_depth=10,
    )

    assert len(normalized) == 2
    # Highest score first: TRV-C02 (0.90), then TRV-C01 (0.85)
    assert normalized[0].chunk_id == "DOC-TRV-001_TRV-C02"
    assert math.isclose(normalized[0].score, 0.90)
    assert normalized[0].rank == 1

    assert normalized[1].chunk_id == "DOC-TRV-001_TRV-C01"
    assert math.isclose(normalized[1].score, 0.85)
    assert normalized[1].rank == 2

    # Unknown chunk ID must raise KeyError
    with pytest.raises(KeyError, match="Unknown chunk"):
        normalize_dense_candidates(
            raw_candidates=[{"chunk_id": "UNKNOWN_CHUNK_999", "score": 0.99}],
            eligible_ids=eligible_ids,
            chunks_by_id=lookup,
        )


def test_normalize_dense_candidates_rejects_non_finite_scores(sample_chunks: List[Dict[str, Any]]):
    """normalize_dense_candidates must raise ValueError on NaN or infinite scores."""
    lookup = {c["chunk_id"]: c for c in sample_chunks}
    eligible_ids = set(lookup.keys())

    with pytest.raises(ValueError, match=r"Non-finite.*score"):
        normalize_dense_candidates(
            raw_candidates=[{"chunk_id": "DOC-TRV-001_TRV-C01", "score": float("nan")}],
            eligible_ids=eligible_ids,
            chunks_by_id=lookup,
        )

    with pytest.raises(ValueError, match=r"Non-finite.*score"):
        normalize_dense_candidates(
            raw_candidates=[{"chunk_id": "DOC-TRV-001_TRV-C01", "score": float("inf")}],
            eligible_ids=eligible_ids,
            chunks_by_id=lookup,
        )


def test_dense_search_local_cosine_ranking(repo_root: Path):
    """Local dense_search correctly ranks chunks by cosine similarity."""
    fake_model = FakeSentenceTransformerModel(dimension=384)
    embedder = BGEEmbedder(model_instance=fake_model)
    retriever = HybridRetriever(base_dir=repo_root, embedder=embedder)

    eligible = filter_eligible_chunks(retriever.chunks, "employee", "2026-11-22")
    eligible_ids = {c["chunk_id"] for c in eligible}

    candidates = retriever.dense_search("hotel accommodation expense", eligible_ids=eligible_ids, top_k=5)
    assert len(candidates) == 5
    assert candidates[0].rank == 1
    assert candidates[1].rank == 2
    for i in range(len(candidates) - 1):
        assert candidates[i].score >= candidates[i + 1].score
