"""Comprehensive offline unit tests for RuleLens guarded semantic caching.

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
from rulelens.grounding import GROUNDING_VERSION, fact_fingerprint, request_meaning, request_support
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
from rulelens.rules import RuleEvaluationResult, RuleRecord, extract_conditions_from_query


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

    model_name = EMBEDDING_MODEL
    normalize_embeddings = True

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
    answer_text: Optional[str] = None,
    citations: Optional[List[str]] = None,
    status: AnswerStatus = AnswerStatus.ANSWERED,
) -> MagicMock:
    """Factory for mocked Groq client returning valid structured responses."""
    mock = MagicMock(spec=GroqClientWrapper)
    mock.model = GENERATION_MODEL
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
        generated = answer_text
        if generated is None and status == AnswerStatus.ANSWERED:
            chunks = [c.to_dict() if hasattr(c, "to_dict") else c for c in evidence_chunks]
            _, facts, _ = request_support(question, chunks)
            generated = "\n".join(dict.fromkeys(f["quote"] for f in facts))
            cids = list(dict.fromkeys(f["chunk_id"] for f in facts))
        resp = AnswerResponse(
            answer=generated or "No supported answer.",
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


def seed_guard_candidate(**kwargs):
    client = kwargs["client"]
    chunks = kwargs["canonical_chunks_by_id"]
    ids = kwargs["citations"]
    payload = dict(
        entry_id=str(uuid.uuid4()), schema_version=CACHE_SCHEMA_VERSION,
        original_question=kwargs["question"], normalized_question=normalize_question(kwargs["question"]),
        answer=kwargs["answer"], status=kwargs["status"], model=GENERATION_MODEL,
        role=kwargs["role"], query_date=kwargs["query_date"], corpus_fingerprint=kwargs["corpus_fingerprint"],
        embedding_model=EMBEDDING_MODEL, condition_signature=compute_condition_signature(kwargs["extracted_conditions"]),
        indicators=compute_question_indicators(kwargs["question"]), cited_chunk_ids=ids,
        cited_chunk_hashes={cid: compute_chunk_content_hash(chunks[cid]["text"]) for cid in ids},
        cited_policy_versions={cid: chunks[cid]["version"] for cid in ids},
        referenced_rule_ids=kwargs["referenced_rule_ids"],
        rule_outcomes={r.rule.rule_id: r.outcome for r in kwargs["rule_results"] if r.rule.rule_id in kwargs["referenced_rule_ids"]},
        citation_metadata=[build_canonical_citation_metadata(chunks[cid]) for cid in ids],
        created_at="2026-11-24T00:00:00Z",
    )
    payload.update(verified_fields(payload, chunks))
    client.upsert(collection_name=QDRANT_CACHE_COLLECTION_NAME,
                  points=[models.PointStruct(id=payload["entry_id"], vector=kwargs["embedder"].embed_query(kwargs["question"]), payload=payload)])
    return payload["entry_id"]


def verified_fields(payload, chunks):
    return {"validation_route": "generated_verified", "grounding_version": GROUNDING_VERSION,
            "meaning_signature": request_meaning(payload["original_question"]),
            "fact_fingerprint": fact_fingerprint(list(chunks.values()))}


# 20 REQUIRED TEST CASES
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

    seed_guard_candidate(
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

    seed_guard_candidate(
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
    seed_guard_candidate(
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

    seed_guard_candidate(
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

    seed_guard_candidate(
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

    seed_guard_candidate(
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
    assert res2.status == "insufficient"
    assert res2.cache_entry_id is None
    mock_llm.generate_answer.assert_not_called()


def test_12_different_exception_or_eligibility_requirement_miss(canonical_chunks_by_id, mock_embedder):
    """Case 12: Exception / emergency indicator added -> cache miss."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)

    q_src = "Can an employee travel without advance booking?"
    q_cand = "Can an employee travel under emergency exception without advance booking?"
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C01"]

    seed_guard_candidate(
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
    assert res.rejection_reason in ("exception_mismatch", "comparison_mismatch", "low_similarity")


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

    chunk_id = "DOC-TRV-002_TRV-C03"

    id1 = insert_cache_entry(
        client=client,
        validation_route="generated_verified",
        embedder=mock_embedder,
        question="What is the daily meal allowance?",
        answer="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
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
        validation_route="generated_verified",
        embedder=mock_embedder,
        question="What is the daily meal allowance?",
        answer="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
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
    assert id1 is not None
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


# Section 5 Mandatory High-Similarity Adversarial Cache Tests
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
        "schema_version": CACHE_SCHEMA_VERSION,
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
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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


# Pass 1 Final Review: Incomplete Rule Records & Fact-Equivalence Regressions
def test_adversarial_16_missing_cached_rule_outcome_rejected_incomplete_rule_record(
    canonical_chunks_by_id, mock_embedder
):
    """Cached referenced rule ID missing associated cached outcome is rejected as incomplete_rule_record."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    q = "What is the policy regarding travel reimbursement approvals?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Travel reimbursement approvals require director approval exceeding INR 50,000.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C04"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C04": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C04": chunk.get("version", "2.0")},
        "referenced_rule_ids": ["R-TRV-003"],
        "rule_outcomes": {},  # Missing cached outcome for R-TRV-003!
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
    assert res.rejection_reason == "incomplete_rule_record"


def test_adversarial_17_rule_outcome_mismatch_rejected(canonical_chunks_by_id, mock_embedder):
    """Candidate with mismatched rule outcome is rejected with rule_outcome_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    q = "Can an expense of INR 50,000 be approved under Travel Policy v2?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Expenses of INR 50,000 follow standard approval.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {"expense_amount": {"value": "50000", "unit": "inr", "comparison": "exact"}},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C04"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C04": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C04": chunk.get("version", "2.0")},
        "referenced_rule_ids": ["R-TRV-003"],
        "rule_outcomes": {"R-TRV-003": "additional_approval"},  # Cached outcome says additional_approval, but rule gives standard_approval!
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    mock_rule = RuleRecord(
        rule_id="R-TRV-003", doc_id="DOC-TRV-002", clause_id="TRV-C04", chunk_id="DOC-TRV-002_TRV-C04",
        condition_field="expense_amount", operator="<=", threshold_value=50000, unit="INR",
        outcome="standard_approval", effective_from="2026-07-01", effective_to="2099-12-31",
    )
    mock_eval = RuleEvaluationResult(
        rule=mock_rule, passed=True, input_value=Decimal("50000"), input_unit="INR",
        outcome="standard_approval", explanation="Satisfied",
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[{"field": "expense_amount", "value": Decimal("50000"), "unit": "INR", "comparison": "exact"}],
        rule_results=[mock_eval],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "rule_outcome_mismatch"


def test_adversarial_18_rule_condition_input_mismatch_rejected(canonical_chunks_by_id, mock_embedder):
    """Inconsistent rule input condition value is rejected with rule_condition_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C04"]
    q = "Can an expense of INR 50,000 be approved under Travel Policy v2?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Expenses of INR 50,000 follow standard approval.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {"expense_amount": {"value": "50000", "unit": "inr", "comparison": "exact"}},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C04"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C04": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C04": chunk.get("version", "2.0")},
        "referenced_rule_ids": ["R-TRV-003"],
        "rule_outcomes": {"R-TRV-003": "standard_approval"},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    mock_rule = RuleRecord(
        rule_id="R-TRV-003", doc_id="DOC-TRV-002", clause_id="TRV-C04", chunk_id="DOC-TRV-002_TRV-C04",
        condition_field="expense_amount", operator="<=", threshold_value=50000, unit="INR",
        outcome="standard_approval", effective_from="2026-07-01", effective_to="2099-12-31",
    )
    # Evaluated with 40,000 instead of 50,000
    mock_eval = RuleEvaluationResult(
        rule=mock_rule, passed=True, input_value=Decimal("40000"), input_unit="INR",
        outcome="standard_approval", explanation="Satisfied",
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[{"field": "expense_amount", "value": Decimal("50000"), "unit": "INR", "comparison": "exact"}],
        rule_results=[mock_eval],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is False
    assert res.rejection_reason == "rule_condition_mismatch"


def test_fact_equivalence_meal_vs_hotel_high_similarity_rejected(canonical_chunks_by_id, mock_embedder):
    """Daily meal allowance vs nightly hotel lodging with similarity >= 0.99 must reject with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C03"]

    seed_q = "What is the maximum daily meal allowance under Travel Policy v2?"
    query_q = "What is the maximum nightly hotel lodging allowance under Travel Policy v2?"

    # Force identical vector (similarity = 1.0) so vector similarity is NOT the rejection reason
    vec = mock_embedder.embed_query(query_q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": seed_q,
        "normalized_question": normalize_question(seed_q),
        "answer": "The maximum daily meal allowance is INR 2,500 under Travel Policy v2.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(seed_q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C03"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C03": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C03": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=query_q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.similarity >= 0.99
    assert res.hit is False
    assert res.rejection_reason == "fact_target_mismatch"


def test_fact_equivalence_submission_vs_review_sla_high_similarity_rejected(
    canonical_chunks_by_id, mock_embedder
):
    """Employee claim submission deadline vs manager review SLA (similarity >= 0.99) rejected with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-FIN-001_FIN-C02"]

    seed_q = "How many business days do employees have to submit expense claims after incurring an expense?"
    query_q = "How many business days do managers have to review and approve submitted expense claims?"

    vec = mock_embedder.embed_query(query_q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": seed_q,
        "normalized_question": normalize_question(seed_q),
        "answer": "Employees must submit expense claims within 30 calendar days.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(seed_q),
        "cited_chunk_ids": ["DOC-FIN-001_FIN-C02"],
        "cited_chunk_hashes": {"DOC-FIN-001_FIN-C02": chunk_hash},
        "cited_policy_versions": {"DOC-FIN-001_FIN-C02": chunk.get("version", "1.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=query_q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.similarity >= 0.99
    assert res.hit is False
    assert res.rejection_reason == "fact_target_mismatch"


def test_fact_equivalence_encryption_vs_transfer_high_similarity_rejected(
    canonical_chunks_by_id, mock_embedder
):
    """Encryption at rest vs document transfer restrictions rejected with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-SEC-001_SEC-D02"]

    seed_q = "What encryption standard is mandatory for Confidential tier data at rest?"
    query_q = "What security measures are required when transferring Confidential documents outside corporate networks?"

    vec = mock_embedder.embed_query(query_q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": seed_q,
        "normalized_question": normalize_question(seed_q),
        "answer": "AES-256 encryption is mandatory for Confidential tier data at rest.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(seed_q),
        "cited_chunk_ids": ["DOC-SEC-001_SEC-D02"],
        "cited_chunk_hashes": {"DOC-SEC-001_SEC-D02": chunk_hash},
        "cited_policy_versions": {"DOC-SEC-001_SEC-D02": chunk.get("version", "1.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=query_q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.similarity >= 0.99
    assert res.hit is False
    assert res.rejection_reason == "fact_target_mismatch"


def test_fact_equivalence_sick_leave_vs_vacation_high_similarity_rejected(
    canonical_chunks_by_id, mock_embedder
):
    """Sick leave certification vs vacation leave accrual rejected with fact_target_mismatch."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-PPO-001_PPO-L01"]

    seed_q = "How many paid annual vacation leave days do employees accrue per calendar year?"
    query_q = "How many consecutive sick leave days require a medical certificate?"

    vec = mock_embedder.embed_query(query_q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": seed_q,
        "normalized_question": normalize_question(seed_q),
        "answer": "Employees accrue 1.75 vacation days per month (21 days per year).",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(seed_q),
        "cited_chunk_ids": ["DOC-PPO-001_PPO-L01"],
        "cited_chunk_hashes": {"DOC-PPO-001_PPO-L01": chunk_hash},
        "cited_policy_versions": {"DOC-PPO-001_PPO-L01": chunk.get("version", "1.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=query_q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.similarity >= 0.99
    assert res.hit is False
    assert res.rejection_reason == "fact_target_mismatch"


def test_fact_equivalence_valid_exact_question_hit(canonical_chunks_by_id, mock_embedder):
    """Valid exact question hit produces hit=True and verified citations."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C03"]
    q = "What is the maximum daily meal allowance for domestic travel under Travel Policy v2?"

    vec = mock_embedder.embed_query(q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": q,
        "normalized_question": normalize_question(q),
        "answer": "Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C03"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C03": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C03": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
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
    assert res.hit is True
    assert res.rejection_reason is None
    assert res.record is not None
    assert "INR 2,000 per diem with itemized receipts" in res.record.answer


def test_fact_equivalence_valid_demonstrated_paraphrase_hit(canonical_chunks_by_id, mock_embedder):
    """Demonstrated meaning-preserving paraphrase of the same policy fact produces hit=True."""
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    chunk = canonical_chunks_by_id["DOC-TRV-002_TRV-C03"]

    seed_q = "What is the maximum daily meal allowance for domestic travel under Travel Policy v2?"
    para_q = "Under Travel Policy v2, what is the maximum per diem meal allowance for domestic travel?"

    vec = mock_embedder.embed_query(para_q)
    chunk_hash = compute_chunk_content_hash(chunk["text"])

    record_payload = {
        "entry_id": str(uuid.uuid4()),
        "schema_version": CACHE_SCHEMA_VERSION,
        "original_question": seed_q,
        "normalized_question": normalize_question(seed_q),
        "answer": "Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
        "status": "answered",
        "model": GENERATION_MODEL,
        "role": "employee",
        "query_date": "2026-11-22",
        "corpus_fingerprint": "fp_test",
        "embedding_model": EMBEDDING_MODEL,
        "condition_signature": {},
        "indicators": compute_question_indicators(seed_q),
        "cited_chunk_ids": ["DOC-TRV-002_TRV-C03"],
        "cited_chunk_hashes": {"DOC-TRV-002_TRV-C03": chunk_hash},
        "cited_policy_versions": {"DOC-TRV-002_TRV-C03": chunk.get("version", "2.0")},
        "referenced_rule_ids": [],
        "rule_outcomes": {},
        "citation_metadata": [build_canonical_citation_metadata(chunk)],
        "created_at": "2026-11-22T00:00:00Z",
    }
    record_payload.update(verified_fields(record_payload, canonical_chunks_by_id))
    client.upsert(
        collection_name=QDRANT_CACHE_COLLECTION_NAME,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=record_payload)],
    )

    res = lookup_cache(
        client=client,
        embedder=mock_embedder,
        question=para_q,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint="fp_test",
        authorized_chunks=[chunk],
        extracted_conditions=[],
        rule_results=[],
        canonical_chunks_by_id=canonical_chunks_by_id,
        similarity_threshold=0.85,
    )
    assert res.hit is True
    assert res.rejection_reason is None
    assert res.record is not None
    assert "INR 2,000 per diem with itemized receipts" in res.record.answer


class ConstantTestEmbedder:
    model_name = EMBEDDING_MODEL
    normalize_embeddings = True

    def embed_query(self, text):
        return [1.0] + [0.0] * 383


@pytest.mark.parametrize("source, candidate, cid", [
    ("How many weeks of paid parental leave do primary caregivers receive?", "How many weeks of paid parental leave do secondary caregivers receive?", "DOC-PPO-001_PPO-L03"),
    ("What is the hotel cap in metropolitan cities?", "What is the hotel cap in regional locations?", "DOC-TRV-002_TRV-C03"),
    ("What encryption is required for Confidential data?", "What encryption is required for Restricted data?", "DOC-SEC-001_SEC-D02"),
    ("How long are employee payroll records retained?", "How long are customer audit logs retained?", "DOC-SEC-004_SEC-R01"),
    ("How many annual vacation days do employees accrue?", "What is the monthly vacation accrual?", "DOC-PPO-001_PPO-L01"),
])
def test_different_entities_never_reuse_at_identical_similarity(canonical_chunks_by_id, source, candidate, cid):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    chunk = canonical_chunks_by_id[cid]
    _, facts, unsupported = request_support(source, [chunk])
    assert not unsupported
    record_id = insert_cache_entry(client, embedder, source, "\n".join(f["quote"] for f in facts),
                                   "answered", "auditor", "2026-11-22", "test_fp", [cid], [], [], [],
                                   canonical_chunks_by_id, extract_conditions_from_query(source), validation_route="generated_verified")
    assert record_id is not None
    result = lookup_cache(client, embedder, candidate, "auditor", "2026-11-22", "test_fp",
                          [chunk], extract_conditions_from_query(candidate), [], canonical_chunks_by_id)
    assert result.similarity == 1.0
    assert not result.hit
    assert result.rejection_reason in ("fact_target_mismatch", "numerical_condition_mismatch")


@pytest.mark.parametrize("source, candidate, cid", [
    ("What is the daily meal allowance?", "What is the per diem meal allowance?", "DOC-TRV-002_TRV-C03"),
    ("How many weeks of paid leave do primary caregivers receive?", "How many weeks of paid maternity leave are provided?", "DOC-PPO-001_PPO-L03"),
    ("What is the maximum travel expense amount my manager can approve?", "What is the travel expense limit my manager can approve without director sign-off?", "DOC-TRV-002_TRV-C04"),
    ("How many days in advance must domestic flights be booked?", "What is the minimum booking lead time for domestic business flights?", "DOC-TRV-003_TRV-B02"),
])
def test_supported_paraphrases_reuse_verified_facts(canonical_chunks_by_id, source, candidate, cid):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    chunk = canonical_chunks_by_id[cid]
    _, facts, unsupported = request_support(source, [chunk])
    assert not unsupported
    record_id = insert_cache_entry(client, embedder, source, "\n".join(f["quote"] for f in facts),
                                   "answered", "auditor", "2026-11-22", "test_fp", [cid], [], [], [],
                                   canonical_chunks_by_id, extract_conditions_from_query(source), validation_route="generated_verified")
    assert record_id is not None
    result = lookup_cache(client, embedder, candidate, "auditor", "2026-11-22", "test_fp",
                          [chunk], extract_conditions_from_query(candidate), [], canonical_chunks_by_id)
    assert result.hit, result.rejection_reason
    assert result.record.validation_route == "generated_verified"
    assert result.record.answer.startswith("> ")


@pytest.mark.parametrize("mutation, reason", [
    ({"answer": "Daily meal subsistence allowance is capped at INR 2,500 per diem with itemized receipts."}, "claim_validation_failed"),
    ({"answer": "Daily meal subsistence allowance is capped at INR 2,000 per diem."}, "claim_validation_failed"),
    ({"grounding_version": ""}, "unverified_cache_record"),
    ({"validation_route": "rule"}, "unverified_cache_record"),
    ({"meaning_signature": {}}, "unverified_cache_record"),
    ({"fact_fingerprint": "changed"}, "unverified_cache_record"),
    ({"schema_version": "1.0"}, "unsupported_schema_version"),
])
def test_forged_or_old_cache_proof_is_rejected(canonical_chunks_by_id, mutation, reason):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    question = "What is the daily meal allowance?"
    cid = "DOC-TRV-002_TRV-C03"
    record_id = seed_guard_candidate(client=client, embedder=embedder, question=question,
                                    answer="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
                                    status="answered", role="employee", query_date="2026-11-22", corpus_fingerprint="test_fp",
                                    citations=[cid], referenced_rule_ids=[], rule_results=[],
                                    canonical_chunks_by_id=canonical_chunks_by_id, extracted_conditions=[])
    record = client.retrieve(QDRANT_CACHE_COLLECTION_NAME, [record_id])[0]
    payload = dict(record.payload)
    payload.update(mutation)
    client.upsert(QDRANT_CACHE_COLLECTION_NAME,
                  [models.PointStruct(id=record_id, vector=embedder.embed_query(question), payload=payload)])
    result = lookup_cache(client, embedder, question, "employee", "2026-11-22", "test_fp",
                          [canonical_chunks_by_id[cid]], [], [], canonical_chunks_by_id)
    assert not result.hit
    assert result.rejection_reason == reason


def test_cache_identity_keeps_historical_and_current_entries(canonical_chunks_by_id):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    question = "What is the daily meal allowance?"
    ids = []
    for date, cid in [("2026-06-30", "DOC-TRV-001_TRV-C03"), ("2026-07-01", "DOC-TRV-002_TRV-C03")]:
        facts = request_support(question, [canonical_chunks_by_id[cid]])[1]
        record_id = insert_cache_entry(client, embedder, question, facts[0]["quote"], "answered", "employee", date,
                                       "test_fp", [cid], [], [], [], canonical_chunks_by_id, [],
                                       validation_route="generated_verified")
        assert record_id is not None
        ids.append(record_id)
    assert ids[0] != ids[1]
    assert client.count(QDRANT_CACHE_COLLECTION_NAME).count == 2


def test_cache_seed_requires_verified_generation_route(canonical_chunks_by_id):
    client = MagicMock()
    cid = "DOC-TRV-002_TRV-C03"
    result = insert_cache_entry(client, ConstantTestEmbedder(), "What is the daily meal allowance?",
                                "Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
                                "answered", "employee", "2026-11-22", "test_fp", [cid], [], [], [], canonical_chunks_by_id, [])
    assert result is None
    client.upsert.assert_not_called()


@pytest.mark.parametrize("field", ["doc_id", "clause_id", "title", "version", "source_file", "source_page"])
def test_every_citation_metadata_field_is_verified(canonical_chunks_by_id, field):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    question = "What is the daily meal allowance?"
    cid = "DOC-TRV-002_TRV-C03"
    record_id = seed_guard_candidate(client=client, embedder=embedder, question=question,
        answer="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
        status="answered", role="employee", query_date="2026-11-22", corpus_fingerprint="test_fp",
        citations=[cid], referenced_rule_ids=[], rule_results=[], canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[])
    payload = client.retrieve(QDRANT_CACHE_COLLECTION_NAME, [record_id])[0].payload
    payload["citation_metadata"][0][field] = "forged"
    client.upsert(QDRANT_CACHE_COLLECTION_NAME,
                  [models.PointStruct(id=record_id, vector=embedder.embed_query(question), payload=payload)])
    result = lookup_cache(client, embedder, question, "employee", "2026-11-22", "test_fp",
                          [canonical_chunks_by_id[cid]], [], [], canonical_chunks_by_id)
    assert not result.hit
    assert result.rejection_reason == "incomplete_citation_metadata"


def test_unknown_target_cannot_reuse_even_an_exact_question(canonical_chunks_by_id):
    client = QdrantClient(":memory:")
    ensure_cache_collection(client)
    embedder = ConstantTestEmbedder()
    question = "What is the sabbatical entitlement?"
    cid = "DOC-PPO-001_PPO-L03"
    seed_guard_candidate(client=client, embedder=embedder, question=question,
        answer="Secondary caregivers receive 4 weeks of fully paid paternity leave.",
        status="answered", role="employee", query_date="2026-11-22", corpus_fingerprint="test_fp",
        citations=[cid], referenced_rule_ids=[], rule_results=[], canonical_chunks_by_id=canonical_chunks_by_id,
        extracted_conditions=[])
    result = lookup_cache(client, embedder, question, "employee", "2026-11-22", "test_fp",
                          [canonical_chunks_by_id[cid]], [], [], canonical_chunks_by_id)
    assert not result.hit
    assert result.rejection_reason == "unsupported_request_meaning"


def test_cache_uses_the_active_retrievers_embedder(repo_root, qdrant_cache, monkeypatch):
    import rulelens.answering as answering
    retriever = HybridRetriever(base_dir=repo_root, embedder=ConstantTestEmbedder())
    new_embedder = MagicMock(side_effect=AssertionError("Duplicate model construction"))
    monkeypatch.setattr(answering, "BGEEmbedder", new_embedder)
    result = answer_question("What is the daily meal allowance?", "employee", "2026-11-22",
                             retriever=retriever, llm_client=make_mock_llm(), cache_enabled=True,
                             cache_client=qdrant_cache, mode="bm25")
    assert result.route == "generated_verified"
    assert result.cache_entry_id is not None
    new_embedder.assert_not_called()


def test_cache_remains_disabled_by_default(monkeypatch):
    import rulelens.config as config
    monkeypatch.setattr(config, "get_env_var", lambda name, default=None: default)
    assert config.is_cache_enabled() is False


@pytest.mark.parametrize("name, normalized", [("different/model", True), (EMBEDDING_MODEL, False), (None, True)])
def test_changed_embedding_configuration_cannot_reuse_or_seed(canonical_chunks_by_id, name, normalized):
    embedder = ConstantTestEmbedder()
    embedder.model_name = name
    embedder.normalize_embeddings = normalized
    client = MagicMock()
    cid = "DOC-TRV-002_TRV-C03"
    result = lookup_cache(client, embedder, "What is the daily meal allowance?", "employee", "2026-11-22",
                          "test_fp", [canonical_chunks_by_id[cid]], [], [], canonical_chunks_by_id)
    assert not result.hit
    assert result.rejection_reason == "embedding_configuration_mismatch"
    record_id = insert_cache_entry(client, embedder, "What is the daily meal allowance?",
        "Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
        "answered", "employee", "2026-11-22", "test_fp", [cid], [], [], [], canonical_chunks_by_id, [],
        validation_route="generated_verified")
    assert record_id is None
    client.query_points.assert_not_called()
    client.upsert.assert_not_called()


def test_custom_generation_model_bypasses_default_model_cache(repo_root):
    from types import SimpleNamespace
    response = AnswerResponse(answer="Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
                               status=AnswerStatus.ANSWERED, citations=["DOC-TRV-002_TRV-C03"],
                               referenced_rule_ids=[], reason="Source quote.")
    provider = SimpleNamespace(model="different/model", generate_answer=lambda **kwargs:
                               (response, {"calls_made": 1, "total_tokens": 9, "latency_ms": 1}))
    cache_client = MagicMock()
    result = answer_question("What is the daily meal allowance?", "employee", "2026-11-22",
                             retriever=HybridRetriever(base_dir=repo_root, embedder=ConstantTestEmbedder()),
                             llm_client=provider, cache_client=cache_client, cache_enabled=True, mode="bm25")
    assert result.route == "generated_verified"
    assert result.model == "different/model"
    assert result.cache_enabled is False
    assert result.cache_rejection_reason == "generation_configuration_mismatch"
    assert result.cache_entry_id is None
    cache_client.query_points.assert_not_called()
    cache_client.upsert.assert_not_called()
