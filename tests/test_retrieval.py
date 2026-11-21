"""Offline deterministic tests for RuleLens Stage D1 hybrid retrieval foundation.

Verifies:
- Tokenization (lowercase, number retention, no stopword removal, empty queries).
- BM25 indexing, exact keyword search, numerical queries, tie-breaking.
- BGE embedding interface contract (CPU, dim 384, normalized, offline fake/mock).
- Reciprocal Rank Fusion (pure function, manual formula checks, deduplication, ties).
- Role-based permissions across all 6 roles, explicit role restrictions.
- Date boundary filtering and historical version availability.
- Citation metadata preservation.
- Strict offline execution with blocked network sockets.
"""

from __future__ import annotations

import datetime
import math
import socket
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rulelens.config import GENERATION_EVIDENCE_LIMIT
from rulelens.dataset import ROLES
from rulelens.retrieval import (
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
    parse_query_date,
    reciprocal_rank_fusion,
    tokenize,
)


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline unit tests")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def repo_root() -> Path:
    """Return the absolute path of the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_chunks(repo_root: Path) -> List[Dict[str, Any]]:
    """Load the validated 64 corpus chunks."""
    return load_validated_corpus(repo_root)


# ==============================================================================
# 1. Corpus Loading & Fingerprint Verification
# ==============================================================================


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
        assert "version" in chunk
        assert "source_file" in chunk
        assert "source_page" in chunk
        assert "effective_from" in chunk
        assert "allowed_roles" in chunk


def test_load_validated_corpus_rejects_corrupted_fingerprint(tmp_path: Path, repo_root: Path):
    """Corpus loader must raise ValueError if corpus_manifest.json fingerprint does not match."""
    import shutil

    processed_dir = tmp_path / "data" / "processed"
    shutil.copytree(repo_root / "data" / "processed", processed_dir)

    manifest_path = processed_dir / "corpus_manifest.json"
    content = manifest_path.read_text(encoding="utf-8")
    corrupted = content.replace('"processed_fingerprint": "', '"processed_fingerprint": "corrupted_')
    manifest_path.write_text(corrupted, encoding="utf-8")

    with pytest.raises(ValueError, match="Corpus fingerprint mismatch"):
        load_validated_corpus(tmp_path)


# ==============================================================================
# 2. Tokenization & BM25 Lexical Search
# ==============================================================================


def test_tokenize_lowercase_and_retains_numbers():
    """Tokenize must lowercase terms, retain numbers, and split on word boundaries."""
    tokens = tokenize("Asterline Travel Policy v1 limits hotel to 250 USD and flight to 40000 USD.")
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


def test_bm25_exact_keyword_search(sample_chunks: List[Dict[str, Any]]):
    """BM25 search must find relevant chunks by exact keyword."""
    index = BM25Index(sample_chunks)
    results = index.search("domestic air travel economy", top_k=5)
    assert len(results) > 0
    top_result = results[0]
    assert top_result.score > 0.0
    # Domestic air travel is in Travel Policy TRV-C02
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


# ==============================================================================
# 3. Reciprocal Rank Fusion (RRF)
# ==============================================================================


def test_rrf_pure_calculation_with_known_ranks():
    """Verify RRF formula calculation exactly matches 1/(k + rank) with k=60."""
    chunk_a = {
        "chunk_id": "C-A",
        "text": "Text A",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
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


def test_rrf_duplicate_elimination_and_limit():
    """RRF must deduplicate chunks and return at most the requested limit."""
    chunk = {
        "chunk_id": "C-1",
        "text": "Text 1",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    cands1 = [Candidate(chunk_id="C-1", score=5.0, rank=1, chunk=chunk)]
    cands2 = [Candidate(chunk_id="C-1", score=0.8, rank=1, chunk=chunk)]

    results = reciprocal_rank_fusion(cands1, cands2, limit=4)
    assert len(results) == 1
    assert results[0].chunk_id == "C-1"
    assert math.isclose(results[0].rrf_score, 2 / 61, rel_tol=1e-9)


def test_rrf_single_modality_bm25_only_and_dense_only():
    """RRF must work seamlessly with only BM25 or only dense candidate lists."""
    chunk = {
        "chunk_id": "C-1",
        "text": "Text 1",
        "doc_id": "D1",
        "clause_id": "C1",
        "title": "T1",
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }
    cand = Candidate(chunk_id="C-1", score=4.5, rank=1, chunk=chunk)

    res_bm25_only = reciprocal_rank_fusion(bm25_candidates=[cand], dense_candidates=[], limit=4)
    assert len(res_bm25_only) == 1
    assert res_bm25_only[0].bm25_rank == 1
    assert res_bm25_only[0].dense_rank is None
    assert math.isclose(res_bm25_only[0].rrf_score, 1 / 61, rel_tol=1e-9)

    res_dense_only = reciprocal_rank_fusion(bm25_candidates=[], dense_candidates=[cand], limit=4)
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
        "version": "1.0",
        "source_file": "f.pdf",
        "source_page": 1,
        "effective_from": "2026-01-01",
        "allowed_roles": ["employee"],
    }

    # Both have identical rank 1 in their respective single modality
    bm25_cands = [Candidate(chunk_id="DOC-Y", score=2.0, rank=1, chunk=chunk_y)]
    dense_cands = [Candidate(chunk_id="DOC-X", score=0.9, rank=1, chunk=chunk_x)]

    results = reciprocal_rank_fusion(bm25_cands, dense_cands, limit=2)
    # Both have RRF score 1/61; DOC-X should come before DOC-Y alphabetically
    assert [r.chunk_id for r in results] == ["DOC-X", "DOC-Y"]


# ==============================================================================
# 4. Role Permissions & Date Boundary Eligibility
# ==============================================================================


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
    """Unknown roles and malformed dates must raise ValueError."""
    sample = sample_chunks[0]

    with pytest.raises(ValueError, match="Unknown role 'contractor'"):
        is_chunk_eligible(sample, role="contractor", query_date="2026-05-01")

    with pytest.raises(ValueError, match="Unknown role 'superadmin'"):
        filter_eligible_chunks(sample_chunks, role="superadmin", query_date="2026-05-01")

    with pytest.raises(ValueError, match="Malformed query date 'not-a-valid-date'"):
        is_chunk_eligible(sample, role="employee", query_date="not-a-valid-date")

    with pytest.raises(ValueError, match="Malformed query date '2026/05/10'"):
        parse_query_date("2026/05/10")


def test_preservation_of_citation_metadata(sample_chunks: List[Dict[str, Any]]):
    """Returned RetrievalResult must preserve all required citation fields."""
    retriever = HybridRetriever(chunks=sample_chunks)
    results = retriever.search(query="domestic air travel", role="employee", query_date="2026-11-22", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id.startswith("DOC-")
    assert top.doc_id.startswith("DOC-")
    assert top.clause_id != ""
    assert top.title != ""
    assert top.version != ""
    assert top.source_file.endswith(".pdf")
    assert top.source_page >= 1
    assert top.effective_from != ""
    assert "employee" in top.allowed_roles
    assert top.rrf_score > 0.0


# ==============================================================================
# 5. Offline Fake Embedding Wrapper Interface
# ==============================================================================


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
        import numpy as np

        embeddings = []
        for s in sentences:
            # Deterministic vector based on hash of string
            import hashlib

            h = hashlib.sha256(s.encode("utf-8")).digest()
            vec = [(b / 255.0) - 0.5 for b in h[:16]]
            # Repeat to reach 384 dimensions
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
    # Check normalization: Euclidean norm should be ~1.0
    import math

    norm = math.sqrt(sum(x * x for x in query_vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    batch_vecs = embedder.embed_texts(["doc1 chunk text", "doc2 chunk text"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384


def test_hybrid_retriever_with_supplied_dense_candidates(sample_chunks: List[Dict[str, Any]]):
    """Verify HybridRetriever fuses BM25 and supplied dense candidates correctly."""
    retriever = HybridRetriever(chunks=sample_chunks)

    # Provide candidate list from external / dense source
    dense_cands = [
        Candidate(chunk_id="DOC-TRV-002_TRV-C02", score=0.92, rank=1),
        Candidate(chunk_id="DOC-TRV-002_TRV-C01", score=0.85, rank=2),
    ]

    results = retriever.search(
        query="domestic flight",
        role="employee",
        query_date="2026-11-22",
        dense_candidates=dense_cands,
        top_k=2,
    )
    assert len(results) <= 2
    # The top result should have both bm25 and dense ranks
    top = results[0]
    assert top.chunk_id == "DOC-TRV-002_TRV-C02"
    assert top.dense_rank == 1
    assert top.dense_score == 0.92
    assert top.bm25_rank is not None
    assert top.rrf_score > 0.0


def test_unauthorized_chunk_never_in_final_ranking(sample_chunks: List[Dict[str, Any]]):
    """Even if an unauthorized chunk is supplied in dense candidates, it must be filtered out."""
    retriever = HybridRetriever(chunks=sample_chunks)

    # DOC-TRV-004 is Travel Exceptions (unauthorized for employee)
    unauthorized_cands = [
        Candidate(chunk_id="DOC-TRV-004_TRV-C01", score=0.99, rank=1),
    ]

    results = retriever.search(
        query="exception approval",
        role="employee",
        query_date="2026-11-22",
        dense_candidates=unauthorized_cands,
        top_k=4,
    )

    # None of the results should be from DOC-TRV-004
    assert all(r.doc_id != "DOC-TRV-004" for r in results)
