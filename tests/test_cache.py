"""Comprehensive offline unit tests for RuleLens Stage F Guarded Semantic Caching.

Covers all 20 required safety and behavior test cases:
1. Exact repeated question, same role/date/evidence -> valid hit.
2. Meaning-preserving paraphrase -> valid hit when similarity and safeguards pass.
3. Changed amount -> miss.
4. Changed currency -> miss.
5. Changed role -> miss.
6. Changed policy date -> miss.
7. Changed corpus fingerprint -> miss.
8. Changed cited content hash -> miss.
9. Missing authorized citation -> miss.
10. Changed rule outcome -> miss.
11. Negation reversal -> miss.
12. Different exception or eligibility requirement -> miss.
13. Personal versus business travel -> never reuse incorrectly.
14. Corrupted or forged Qdrant cache payload -> reject.
15. Restricted and unsupported questions -> never cached.
16. Deterministic rule answers -> bypass cache.
17. Groq failure -> never insert a cache record.
18. Cache service failure -> safe fallback, without unauthorized output.
19. Duplicate insert -> idempotent behavior.
20. Existing retrieval, generation and security regression tests remain passing.
"""

from __future__ import annotations

import datetime
import math
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from decimal import Decimal
from qdrant_client import QdrantClient, models

from rulelens.answering import answer_question
from rulelens.cache import (
    CACHE_DEFAULT_SIMILARITY_THRESHOLD,
    CACHE_POINT_NAMESPACE,
    CACHE_SCHEMA_VERSION,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    CacheLookupResult,
    CacheRecord,
    build_canonical_citation_metadata,
    compute_chunk_content_hash,
    compute_condition_signature,
    compute_question_indicators,
    ensure_cache_collection,
    insert_cache_entry,
    lookup_cache,
    normalize_question,
    validate_cache_candidate,
)
from rulelens.config import QDRANT_CACHE_COLLECTION_NAME
from rulelens.llm import (
    AnswerResponse,
    AnswerStatus,
    GroqClientWrapper,
    ProviderTelemetrySnapshot,
    SanitizedProviderError,
)
from rulelens.retrieval import (
    ALLOWED_ROLES_SET,
    HybridRetriever,
    RetrievalResult,
    load_validated_corpus,
    parse_query_date,
)
from rulelens.rules import RuleEvaluationResult, RuleRecord


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that all cache unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline cache unit test!")

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


class DeterministicTestEmbedder:
    """Fast, deterministic 384-dimensional normalized embedder for offline tests."""

    def embed_query(self, text: str) -> List[float]:
        import hashlib
        tokens = text.lower().replace(",", " ").replace("?", " ").replace(".", " ").split()
        vec = [0.0] * 384
        for t in tokens:
            digest = hashlib.md5(t.encode("utf-8")).digest()
            for dim in range(16):
                b = digest[dim]
                idx = (b * 13 + dim * 23) % 384
                vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


@pytest.fixture
def mock_embedder() -> DeterministicTestEmbedder:
    return DeterministicTestEmbedder()


@pytest.fixture
def qdrant_cache() -> QdrantClient:
    """Isolated in-memory Qdrant client with answer cache collection initialized."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    return client


def make_mock_llm(
    answer_text: str = "Verified policy answer based on authoritative evidence.",
    citations: Optional[List[str]] = None,
    status: AnswerStatus = AnswerStatus.ANSWERED,
) -> MagicMock:
    """Factory for mocked Groq client returning valid structured responses."""
    mock = MagicMock(spec=GroqClientWrapper)
    mock.get_telemetry_snapshot.return_value = ProviderTelemetrySnapshot(
        provider_requests_attempted=1,
        provider_requests_succeeded=1,
        provider_request_latency_ms=120.0,
        known_total_tokens=250,
    )

    def mock_generate(question, role, evidence_chunks, query_date=None, applicable_rule_facts=None, max_repair_attempts=1):
        cids = list(citations) if citations is not None else None
        if cids is None and evidence_chunks:
            first = evidence_chunks[0]
            cid = first.chunk_id if hasattr(first, "chunk_id") else first["chunk_id"]
            cids = [cid]
        resp = AnswerResponse(
            answer=answer_text,
            status=status,
            citations=cids or [],
            referenced_rule_ids=[],
            reason="Supported by policy evidence.",
        )
        meta = {
            "calls_made": 1,
            "latency_ms": 120.0,
            "total_tokens": 250,
            "schema_valid": True,
        }
        return resp, meta

    mock.generate_answer.side_effect = mock_generate
    return mock


# ==============================================================================
# 20 REQUIRED TEST CASES
# ==============================================================================


def test_01_exact_repeated_question_valid_hit(repo_root, qdrant_cache, mock_embedder):
    """Case 1: Exact repeated question with same role/date/evidence produces valid hit."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    role = "employee"
    date_str = "2026-11-22"

    # Turn 1: Misses cache, generates answer, inserts into cache
    res1 = answer_question(
        question=q,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res1.status == "answered"
    assert res1.cache_hit is False
    assert res1.cache_entry_id is not None
    assert mock_llm.generate_answer.call_count == 1

    # Turn 2: Exact same question -> must hit cache with 0 LLM calls
    res2 = answer_question(
        question=q,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res2.status == "answered"
    assert res2.cache_hit is True
    assert res2.cache_similarity == 1.0
    assert res2.provider_calls_attempted == 0
    assert res2.token_usage == 0
    assert res2.schema_valid is None
    # Verify LLM was NOT called again
    assert mock_llm.generate_answer.call_count == 1


def test_02_meaning_preserving_paraphrase_valid_hit(repo_root, qdrant_cache, mock_embedder):
    """Case 2: Meaning-preserving paraphrase produces valid hit when similarity and safeguards pass."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q_src = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    q_cand = "Under Travel Policy v2, what daily meal subsistence allowance is provided for employees on domestic travel?"
    role = "employee"
    date_str = "2026-11-22"

    # Seed source question
    res1 = answer_question(
        question=q_src,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        cache_threshold=0.85,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res1.cache_entry_id is not None

    # Query candidate paraphrase
    res2 = answer_question(
        question=q_cand,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        cache_threshold=0.85,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res2.cache_hit is True
    assert res2.provider_calls_attempted == 0
    assert res2.token_usage == 0


def test_03_changed_amount_miss(canonical_chunks_by_id, mock_embedder):
    """Case 3: Numerical amount changed -> condition signature mismatch, cache miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    q_src = "Can an employee claim an expense of INR 50,000?"
    q_cand = "Can an employee claim an expense of INR 75,000?"
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question=q_src,
        answer="Claim answer for 50k.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=["DOC-TRV-002_TRV-C01"],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[{"field": "expense_amount", "value": 50000.0, "unit": "INR"}],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_cand,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[{"field": "expense_amount", "value": 75000.0, "unit": "INR"}],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.50,
    )
    assert res.hit is False
    assert res.rejection_reason in ("numerical_condition_mismatch", "condition_signature_mismatch", "low_similarity")


def test_04_changed_currency_miss(canonical_chunks_by_id, mock_embedder):
    """Case 4: Currency unit changed (e.g. INR -> USD) -> condition signature mismatch, cache miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    q_src = "Can an employee claim an expense of INR 50,000?"
    q_cand = "Can an employee claim an expense of USD 50,000?"
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question=q_src,
        answer="Claim answer for 50k.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=["DOC-TRV-002_TRV-C01"],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[{"field": "expense_amount", "value": 50000.0, "unit": "INR"}],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_cand,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[{"field": "expense_amount", "value": 50000.0, "unit": "USD"}],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.50,
    )
    assert res.hit is False
    assert res.rejection_reason in ("numerical_condition_mismatch", "condition_signature_mismatch", "low_similarity")


def test_05_changed_role_miss(repo_root, qdrant_cache, mock_embedder):
    """Case 5: Changed requester role -> filtered out / rejected, cache miss."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    date_str = "2026-11-22"

    # Seeded by employee
    answer_question(
        question=q,
        role="employee",
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )

    # Requested by manager -> server-side filter or validation rejects reuse
    res2 = answer_question(
        question=q,
        role="manager",
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res2.cache_hit is False


def test_06_changed_policy_date_miss(repo_root, qdrant_cache, mock_embedder):
    """Case 6: Changed policy effective date -> cache miss."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"

    # Seeded with v1 date
    answer_question(
        question=q,
        role="employee",
        query_date="2026-03-01",
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )

    # Queried with v2 date
    res2 = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res2.cache_hit is False
    assert res2.cache_rejection_reason in ("date_mismatch", "cache_miss")


def test_07_changed_corpus_fingerprint_miss(canonical_chunks_by_id, mock_embedder):
    """Case 7: Changed corpus fingerprint -> rejected by Qdrant filter."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    # Insert entry with old fingerprint
    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="What is the travel policy allowance?",
        answer="Allowance is INR 50,000.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fingerprint_v1",
        citations=["DOC-TRV-001_TRV-C01"],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )

    # Query with new fingerprint
    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question="What is the travel policy allowance?",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fingerprint_v2_changed",
        authorized_chunks=[canonical_chunks_by_id["DOC-TRV-001_TRV-C01"]],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
    )
    assert res.hit is False
    assert res.rejection_reason in ("cache_miss", "corpus_fingerprint_mismatch")


def test_08_changed_cited_content_hash_miss(canonical_chunks_by_id, mock_embedder):
    """Case 8: Cited chunk content modified (hash mismatch) -> candidate rejected."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    chunk_id = "DOC-TRV-001_TRV-C01"
    original_chunk = dict(canonical_chunks_by_id[chunk_id])

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="What is the daily allowance?",
        answer="INR 1,500 daily allowance.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=[chunk_id],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id={chunk_id: original_chunk},
        extracted_conditions=[],
    )

    # Modify chunk text in current canonical environment
    tampered_chunk = dict(original_chunk)
    tampered_chunk["text"] = original_chunk["text"] + " TAMPERED CONTENT"

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question="What is the daily allowance?",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[tampered_chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id={chunk_id: tampered_chunk},
    )
    assert res.hit is False
    assert res.rejection_reason in ("stale_citation_content", "cited_chunk_hash_mismatch")


def test_09_missing_authorized_citation_miss(canonical_chunks_by_id, mock_embedder):
    """Case 9: Cached entry relies on a citation not authorized in current retrieval -> miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    chunk_a = "DOC-TRV-001_TRV-C01"
    chunk_b = "DOC-FIN-001_FIN-C01"

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="What is the submission policy?",
        answer="Submit claims within 30 days.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=[chunk_a],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )

    # Retrieval only authorizes chunk_b, chunk_a is absent
    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question="What is the submission policy?",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[canonical_chunks_by_id[chunk_b]],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
    )
    assert res.hit is False
    assert res.rejection_reason in ("missing_authorized_citation", "citation_not_authorized")


def test_10_changed_rule_outcome_miss(canonical_chunks_by_id, mock_embedder):
    """Case 10: Underlying structured rule outcome changes -> cache miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    dummy_rule = RuleRecord(
        rule_id="R-TRV-003",
        doc_id="DOC-TRV-001",
        clause_id="TRV-C01",
        chunk_id="DOC-TRV-001_TRV-C01",
        condition_field="manager_approval_cap",
        operator="<=",
        threshold_value=50000.0,
        unit="INR",
        outcome="permissible",
        effective_from="2026-07-01",
        effective_to="2099-12-31",
    )
    rule_res_approved = RuleEvaluationResult(
        rule=dummy_rule,
        passed=True,
        input_value=50000.0,
        input_unit="INR",
        outcome="permissible",
        explanation="OK",
    )
    rule_res_denied = RuleEvaluationResult(
        rule=dummy_rule,
        passed=False,
        input_value=75000.0,
        input_unit="INR",
        outcome="requires_director_approval",
        explanation="Over limit",
    )

    chunk_id = "DOC-TRV-001_TRV-C01"

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="Can direct manager approve INR 50,000?",
        answer="Yes, manager may approve up to INR 50,000.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=[chunk_id],
        citation_metadata=[],
        referenced_rule_ids=["R-TRV-003"],
        rule_results=[rule_res_approved],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )

    # Current candidate evaluated with outcome 'requires_director_approval'
    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question="Can direct manager approve INR 50,000?",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[canonical_chunks_by_id[chunk_id]],
        extracted_conditions=[],
        rule_results=[rule_res_denied],
        canonical_chunks_by_id=canonical_chunks_by_id,
    )
    assert res.hit is False
    assert res.rejection_reason == "rule_outcome_mismatch"


def test_11_negation_reversal_miss(repo_root, qdrant_cache, mock_embedder):
    """Case 11: Negation indicator reversed (e.g. allowed vs not allowed) -> cache miss."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q_src = "Are flight bookings made outside the corporate portal reimbursable?"
    q_cand = "Are flight bookings made outside the corporate portal not reimbursable?"
    role = "employee"
    date_str = "2026-11-22"

    answer_question(
        question=q_src,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )

    res2 = answer_question(
        question=q_cand,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res2.cache_hit is False
    assert res2.cache_rejection_reason in ("negation_mismatch", "low_similarity", "cache_miss")


def test_12_different_exception_or_eligibility_requirement_miss(canonical_chunks_by_id, mock_embedder):
    """Case 12: Exception / emergency indicator added -> cache miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    q_src = "Can an employee travel without advance booking?"
    q_cand = "Can an employee travel under emergency exception without advance booking?"
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]

    insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question=q_src,
        answer="Standard travel rule answer.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=["DOC-TRV-002_TRV-C01"],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_cand,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.50,
    )
    assert res.hit is False
    assert res.rejection_reason in ("exception_indicator_mismatch", "comparison_mismatch", "low_similarity")


def test_13_personal_vs_business_travel_never_reused_incorrectly(repo_root, qdrant_cache, mock_embedder):
    """Case 13: Business travel approval must never be reused for personal travel."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q_biz = "Can employees spend INR 50,000 on business travel with manager approval?"
    q_pers = "Can employees spend INR 50,000 on personal travel?"
    role = "employee"
    date_str = "2026-11-22"

    answer_question(
        question=q_biz,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )

    res_pers = answer_question(
        question=q_pers,
        role=role,
        query_date=date_str,
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    # Must NOT hit the cache with an affirmative approval
    assert res_pers.cache_hit is False


def test_14_corrupted_or_forged_qdrant_payload_rejected(canonical_chunks_by_id, mock_embedder):
    """Case 14: Malformed or forged payload in Qdrant point is rejected cleanly without crash."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    chunk_id = "DOC-TRV-001_TRV-C01"
    vec = mock_embedder.embed_query("corrupted question test")

    # Insert point missing mandatory fields (e.g. no role, no citations)
    forged_point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vec,
        payload={"corrupted": True, "answer": "Forged payload"},
    )
    client.upsert(collection_name=QDRANT_CACHE_COLLECTION_NAME, points=[forged_point])

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question="corrupted question test",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[canonical_chunks_by_id[chunk_id]],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.50,
    )
    assert res.hit is False
    assert res.rejection_reason in ("malformed_cache_record", "cache_miss")


def test_15_restricted_and_unsupported_questions_never_cached(repo_root, qdrant_cache, mock_embedder):
    """Case 15: Questions resolving to restricted or insufficient are never cached."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm(
        answer_text="Insufficient evidence.",
        status=AnswerStatus.INSUFFICIENT,
        citations=[],
    )

    # 1. Unsupported question
    res_unsup = answer_question(
        question="What is the company policy on space travel and lunar hotel lodging allowances?",
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res_unsup.status == "insufficient"
    assert res_unsup.cache_entry_id is None

    # Verify cache remains completely empty
    count = qdrant_cache.count(collection_name=QDRANT_CACHE_COLLECTION_NAME).count
    assert count == 0


def test_16_deterministic_rule_answers_bypass_cache(repo_root, qdrant_cache, mock_embedder):
    """Case 16: Deterministic rule decisions bypass the cache and are never inserted."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    # Question matching deterministic rule R-TRV-003
    q = "Can an employee spend INR 50,000 on travel with manager approval?"
    res = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res.status == "answered"
    assert len(res.referenced_rule_ids) > 0
    assert res.provider_calls_attempted == 0
    assert res.cache_hit is False
    assert res.cache_entry_id is None

    # Cache collection should remain empty
    count = qdrant_cache.count(collection_name=QDRANT_CACHE_COLLECTION_NAME).count
    assert count == 0


def test_17_groq_failure_never_inserts_cache_record(repo_root, qdrant_cache, mock_embedder):
    """Case 17: Provider error during generation never inserts a cache entry."""
    retriever = HybridRetriever(base_dir=repo_root)
    failing_llm = MagicMock(spec=GroqClientWrapper)
    failing_llm.generate_answer.side_effect = SanitizedProviderError(
        category="service_unavailable",
        safe_message="Provider unavailable.",
    )

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    res = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=failing_llm,
        cache_client=qdrant_cache,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    assert res.status == "operational_error"
    assert res.cache_entry_id is None
    assert qdrant_cache.count(collection_name=QDRANT_CACHE_COLLECTION_NAME).count == 0


def test_18_cache_service_failure_safe_fallback(repo_root, mock_embedder):
    """Case 18: Cache lookup/insert failure falls back gracefully to standard generation."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    failing_cache_client = MagicMock(spec=QdrantClient)
    failing_cache_client.query_points.side_effect = RuntimeError("Qdrant cluster timeout")

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    res = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
        cache_client=failing_cache_client,
        cache_enabled=True,
        embedder=mock_embedder,
        mode="bm25",
    )
    # Generation succeeds safely via fallback
    assert res.status == "answered"
    assert res.cache_hit is False
    assert res.cache_rejection_reason == "cache_service_unavailable"
    assert mock_llm.generate_answer.call_count == 1


def test_19_duplicate_insert_idempotent(canonical_chunks_by_id, mock_embedder):
    """Case 19: Inserting the exact same record twice is idempotent without error."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    chunk_id = "DOC-TRV-001_TRV-C01"

    id1 = insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="What is the daily allowance?",
        answer="INR 1,500 daily allowance.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=[chunk_id],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )
    id2 = insert_cache_entry(
        client=client,
        embedder=mock_embedder,
        question="What is the daily allowance?",
        answer="INR 1,500 daily allowance.",
        status="answered",
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        citations=[chunk_id],
        citation_metadata=[],
        referenced_rule_ids=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[],
    )
    assert id1 == id2
    assert client.count(collection_name=QDRANT_CACHE_COLLECTION_NAME).count == 1


def test_20_existing_generation_unaffected_when_cache_disabled(repo_root):
    """Case 20: Default cache_enabled=False behavior is completely unaffected."""
    retriever = HybridRetriever(base_dir=repo_root)
    mock_llm = make_mock_llm()

    q = "What is the standard daily meal subsistence allowance for employees on domestic travel assignments under Travel Policy v2?"
    res = answer_question(
        question=q,
        role="employee",
        query_date="2026-11-22",
        retriever=retriever,
        llm_client=mock_llm,
        cache_enabled=False,
        mode="bm25",
    )
    assert res.status == "answered"
    assert res.cache_enabled is False
    assert res.cache_hit is False
    assert res.cache_entry_id is None
    assert mock_llm.generate_answer.call_count == 1


# ==============================================================================
# 21. Section 5 Mandatory High-Similarity Adversarial Cache Tests
# ==============================================================================


def test_adversarial_01_meal_vs_hotel_same_clause_high_similarity(canonical_chunks_by_id, mock_embedder):
    """Mandatory 1: Meal allowance vs hotel allowance citing same clause rejected with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C03"]

    q_meal = "What is the maximum daily meal allowance under Travel Policy v2?"
    q_hotel = "What is the maximum nightly hotel allowance under Travel Policy v2?"

    # Candidate point created with EXACT SAME vector as q_hotel query (similarity = 1.0)
    vec = mock_embedder.embed_query(q_hotel)
    meal_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q_meal,
        "normalized_question": normalize_question(q_meal),
        "answer": "The maximum daily meal allowance is INR 2,000 under Travel Policy v2.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q_meal),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C03"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C03": meal_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C03": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_hotel,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.similarity >= 0.99
    assert res.rejection_reason == "fact_target_mismatch"


def test_adversarial_02_submission_vs_review_deadline_high_similarity(canonical_chunks_by_id, mock_embedder):
    """Mandatory 2: Expense submission deadline vs manager review SLA rejected with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-FIN-001_FIN-C02"]

    q_sub = "What is the employee deadline for submitting expense reimbursement claims under Finance Policy v1?"
    q_rev = "What is the manager review turnaround time for approving expense claims under Finance Policy v1?"

    vec = mock_embedder.embed_query(q_rev)
    sub_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q_sub,
        "normalized_question": normalize_question(q_sub),
        "answer": "Employees must submit expense claims within 30 days.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q_sub),
        "cited_chunk_ids": ["DOC-FIN-001_FIN-C02"],
        "cited_chunk_hashes": {"DOC-FIN-001_FIN-C02": sub_hash},
        "cited_policy_versions": {"DOC-FIN-001_FIN-C02": chunk.get("version", "1.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_rev,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.similarity >= 0.99
    assert res.rejection_reason == "fact_target_mismatch"


def test_adversarial_03_above_vs_up_to_threshold_high_similarity(canonical_chunks_by_id, mock_embedder):
    """Mandatory 3: Amounts above threshold vs up-to threshold rejected with comparison_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]

    q_up_to = "Can a direct manager approve travel expenses up to INR 50,000 under Travel Policy v2?"
    q_above = "Can a direct manager approve travel expenses exceeding INR 50,000 under Travel Policy v2?"

    vec = mock_embedder.embed_query(q_above)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q_up_to,
        "normalized_question": normalize_question(q_up_to),
        "answer": "Standard manager approval is permitted up to INR 50,000.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "manager",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {"expense_amount": {"value": "50000", "unit": "inr", "comparison": "less_than_or_equal"}},
        "indicators": compute_question_indicators(q_up_to),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C04"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C04": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C04": chunk.get("version", "2.0")},
        "referenced_rule_ids": ["R-TRV-003"],
        "rule_outcomes": {"R-TRV-003": "standard_approval"},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    extracted_above = [{"field": "expense_amount", "value": Decimal("50000"), "unit": "INR", "comparison": "greater_than"}]
    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q_above,
        role="manager",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=extracted_above,
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.similarity >= 0.99
    assert res.rejection_reason == "comparison_mismatch"


def test_adversarial_09_missing_current_referenced_rule(canonical_chunks_by_id, mock_embedder):
    """Mandatory 9: Cached entry references a rule not active in current query -> missing_referenced_rule."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    q = "What is the approval limit under Travel Policy v2?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Manager approval limit is INR 50,000.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "manager",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C04"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C04": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C04": chunk.get("version", "2.0")},
        "referenced_rule_ids": ["R-TRV-003"],  # references R-TRV-003
        "rule_outcomes": {"R-TRV-003": "standard_approval"},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    # Empty rule_results: rule R-TRV-003 is missing in current applicable rules
    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="manager",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],  # No active rules
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "missing_referenced_rule"


def test_adversarial_11_missing_cache_schema(canonical_chunks_by_id, mock_embedder):
    """Mandatory 11: Candidate record missing schema_version is rejected with unsupported_schema_version."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]
    q = "What is the corporate travel policy purpose?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        # schema_version omitted intentionally
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel policy purpose is business travel governance.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C01"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C01": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C01": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "unsupported_schema_version"


def test_adversarial_12_incorrect_schema_version(canonical_chunks_by_id, mock_embedder):
    """Mandatory 12: Unsupported schema_version rejected with unsupported_schema_version."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]
    q = "What is the corporate travel policy purpose?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "2.0-incompatible",
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel policy purpose is business travel governance.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C01"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C01": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C01": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "unsupported_schema_version"


def test_adversarial_13_incorrect_generation_model(canonical_chunks_by_id, mock_embedder):
    """Mandatory 13: Generation model mismatch rejected with generation_model_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]
    q = "What is the corporate travel policy purpose?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel policy purpose is business travel governance.",
        "status": "answered",
        "model": "claude-3-5-sonnet-20241022",  # incompatible model
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C01"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C01": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C01": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "generation_model_mismatch"


def test_adversarial_14_incorrect_embedding_model(canonical_chunks_by_id, mock_embedder):
    """Mandatory 14: Embedding model mismatch rejected with embedding_model_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]
    q = "What is the corporate travel policy purpose?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel policy purpose is business travel governance.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": "text-embedding-3-small",  # incompatible embedding model
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C01"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C01": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C01": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "embedding_model_mismatch"


def test_adversarial_15_missing_citation_metadata(canonical_chunks_by_id, mock_embedder):
    """Mandatory 15: Missing or incomplete citation metadata rejected with incomplete_citation_metadata."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]
    q = "What is the corporate travel policy purpose?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": "1.0",
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel policy purpose is business travel governance.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C01"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C01": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C01": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [],  # Empty citation metadata
        "created_at": "2026-11-22T00:00:00Z",
    }
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "incomplete_citation_metadata"
