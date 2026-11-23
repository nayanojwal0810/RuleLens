"""Guarded Semantic Caching for RuleLens (Stage F).

Provides an evidence-bound semantic cache that safely reuses previously generated
policy answers, reducing Groq requests without bypassing authorization, retrieval,
structured rules, or citation verification.

Key principles:
1. Candidate discovery only: Vector similarity is candidate discovery, never proof of equivalence.
2. Complete isolation: Operates in a dedicated Qdrant collection ('rulelens_answer_cache_v1').
3. Strict mandatory validation: Verifies role, date, corpus, numerical conditions, rule outcomes,
   negation, comparison, exception indicators, and that all cited evidence is currently authorized
   and uncorrupted before accepting any candidate.
4. Safe fallback: Cache failures safely fall back to the standard uncached generation path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from qdrant_client import QdrantClient, models

from rulelens.config import (
    CACHE_DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    QDRANT_CACHE_COLLECTION_NAME,
)
from rulelens.dataset import ROLES
from rulelens.qdrant_store import is_in_memory_or_local_client
from rulelens.retrieval import (
    EMBEDDING_DIMENSION,
    BGEEmbedder,
    RetrievalResult,
)
from rulelens.rules import (
    RuleEvaluationResult,
    _find_unsupported_query_terms,
    extract_conditions_from_query,
)

# Namespace for deterministic cache record point IDs
CACHE_POINT_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "cache.rulelens.asterline.internal")

CACHE_SCHEMA_VERSION: str = "1.0"

# Required payload indexes on non-in-memory Qdrant servers
CACHE_REQUIRED_PAYLOAD_INDEXES: List[Tuple[str, models.PayloadSchemaType]] = [
    ("role", models.PayloadSchemaType.KEYWORD),
    ("corpus_fingerprint", models.PayloadSchemaType.KEYWORD),
    ("schema_version", models.PayloadSchemaType.KEYWORD),
    ("status", models.PayloadSchemaType.KEYWORD),
]


def normalize_question(text: str) -> str:
    """Normalize question text for comparison and deterministic ID generation.

    Lowercases, strips leading/trailing whitespace and punctuation, and collapses spaces.
    """
    if not text:
        return ""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def compute_question_indicators(question: str) -> Dict[str, Any]:
    """Extract linguistic and semantic indicators from a question to guard cache reuse."""
    q_norm = normalize_question(question)
    q_lower = question.lower()

    # Negation indicators
    negation_patterns = r"\b(not|never|no|cannot|can't|without|disallow|disallowed|non-reimbursable|prohibited|neither|nor)\b"
    has_negation = bool(re.search(negation_patterns, q_lower))

    # Comparison and boundary indicators
    comparison_patterns = r"\b(exceed|exceeds|exceeding|greater|less|more|fewer|maximum|minimum|cap|at least|at most|under|over|up to|above|below|higher|lower)\b"
    has_comparison = bool(re.search(comparison_patterns, q_lower))

    # Exception and emergency indicators
    exception_patterns = r"\b(exception|exceptions|urgent|urgently|extraordinary|emergency|waiver|exempt|exemption|deviat|special)\b"
    has_exception = bool(re.search(exception_patterns, q_lower))

    # Critical approval authority entities mentioned in the query
    authorities: List[str] = []
    if "manager" in q_lower or "reporting manager" in q_lower:
        authorities.append("manager")
    if "director" in q_lower:
        authorities.append("director")
    if "vp" in q_lower or "vice president" in q_lower:
        authorities.append("vp")
    if "cfo" in q_lower or "chief financial officer" in q_lower:
        authorities.append("cfo")
    if "committee" in q_lower or "board" in q_lower:
        authorities.append("board_or_committee")
    if "lead" in q_lower or "leads" in q_lower:
        authorities.append("lead")
    if "ciso" in q_lower:
        authorities.append("ciso")
    authorities.sort()

    # Broad policy subject categorization
    subject = "general"
    if any(k in q_lower for k in ("travel", "flight", "flights", "hotel", "lodging", "meal", "subsistence", "per diem", "booking", "portal")):
        subject = "travel"
    elif any(k in q_lower for k in ("receipt", "tax invoice", "card", "credit card", "reimbursement", "claim", "submit", "submission", "settling", "bitcoin")):
        subject = "finance"
    elif any(k in q_lower for k in ("vacation", "sick", "leave", "remote", "hours", "tuition", "education", "degree", "course", "transfer", "mobility", "sabbatical")):
        subject = "people_ops"
    elif any(k in q_lower for k in ("procurement", "purchase", "order", "quote", "supplier", "vendor", "emergency purchase", "bidding", "contract", "barter")):
        subject = "procurement"
    elif any(k in q_lower for k in ("classification", "tier", "confidential", "restricted", "encryption", "aes", "privileged", "recertification", "incident", "severity", "sla", "sanitiz", "storage", "drives", "smart", "iot")):
        subject = "security"

    return {
        "has_negation": has_negation,
        "has_comparison": has_comparison,
        "has_exception": has_exception,
        "approval_authorities": authorities,
        "policy_subject": subject,
    }


def compute_condition_signature(conditions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a canonical deterministic signature of extracted condition values and units."""
    sig: Dict[str, Any] = {}
    for c in conditions:
        if not isinstance(c, dict):
            continue
        field = c.get("field")
        val = c.get("value")
        unit = (c.get("unit") or "").lower()
        if field and val is not None and not c.get("is_ambiguous", False):
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                numeric_val = str(val)
            sig[field] = {
                "value": numeric_val,
                "unit": unit,
            }
    return sig


def compute_chunk_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of a chunk's canonical text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CacheRecord:
    """Represents a validated, serializable policy answer cache entry."""

    entry_id: str
    schema_version: str
    original_question: str
    normalized_question: str
    answer: str
    status: str
    model: str
    role: str
    query_date: str
    corpus_fingerprint: str
    embedding_model: str
    condition_signature: Dict[str, Any]
    indicators: Dict[str, Any]
    cited_chunk_ids: List[str]
    cited_chunk_hashes: Dict[str, str]
    cited_policy_versions: Dict[str, str]
    referenced_rule_ids: List[str]
    rule_outcomes: Dict[str, str]
    citation_metadata: List[Dict[str, Any]]
    created_at: str

    def to_payload(self) -> Dict[str, Any]:
        """Convert record to a clean Qdrant payload dictionary."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> CacheRecord:
        """Construct record from a Qdrant payload dictionary."""
        return cls(
            entry_id=str(payload.get("entry_id", "")),
            schema_version=str(payload.get("schema_version", "1.0")),
            original_question=str(payload.get("original_question", "")),
            normalized_question=str(payload.get("normalized_question", "")),
            answer=str(payload.get("answer", "")),
            status=str(payload.get("status", "")),
            model=str(payload.get("model", "")),
            role=str(payload.get("role", "")),
            query_date=str(payload.get("query_date", "")),
            corpus_fingerprint=str(payload.get("corpus_fingerprint", "")),
            embedding_model=str(payload.get("embedding_model", "")),
            condition_signature=dict(payload.get("condition_signature") or {}),
            indicators=dict(payload.get("indicators") or {}),
            cited_chunk_ids=list(payload.get("cited_chunk_ids") or []),
            cited_chunk_hashes=dict(payload.get("cited_chunk_hashes") or {}),
            cited_policy_versions=dict(payload.get("cited_policy_versions") or {}),
            referenced_rule_ids=list(payload.get("referenced_rule_ids") or []),
            rule_outcomes=dict(payload.get("rule_outcomes") or {}),
            citation_metadata=list(payload.get("citation_metadata") or []),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass
class CacheLookupResult:
    """Outcome of attempting a semantic cache lookup."""

    hit: bool
    record: Optional[CacheRecord] = None
    similarity: Optional[float] = None
    rejection_reason: Optional[str] = None
    latency_ms: float = 0.0


def validate_cache_candidate(
    candidate: CacheRecord,
    question: str,
    role: str,
    query_date: str,
    corpus_fingerprint: str,
    authorized_chunks: Sequence[Union[RetrievalResult, Dict[str, Any]]],
    extracted_conditions: Sequence[Dict[str, Any]],
    rule_results: Sequence[RuleEvaluationResult],
    canonical_chunks_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Strictly validate whether a candidate cache record may be safely reused.

    Applies all mandatory safety checks:
    - Status must be 'answered'
    - Requesting role matches exactly
    - Query date matches exactly
    - Corpus fingerprint and embedding model match
    - Structured conditions match exactly (values and units)
    - Negation, comparison, and exception indicators match
    - Approval authority entities match
    - Policy subjects are compatible
    - All cached cited chunks are present in currently authorized evidence
    - Canonical content hashes and policy versions for all cited chunks are identical
    - Referenced rule outcomes remain consistent
    - Record is complete and well-formed
    """
    # 1. Status check
    if candidate.status != "answered":
        return False, "status_not_answered"

    # 2. Basic metadata integrity
    if not candidate.answer or not candidate.cited_chunk_ids:
        return False, "malformed_cache_record"

    # 3. Requesting role verification
    if candidate.role != role:
        return False, "role_mismatch"

    # 4. Temporal date verification
    if candidate.query_date != query_date:
        return False, "date_mismatch"

    # 5. Corpus fingerprint and model verification
    if candidate.corpus_fingerprint != corpus_fingerprint:
        return False, "corpus_fingerprint_mismatch"
    if candidate.embedding_model != EMBEDDING_MODEL:
        return False, "embedding_model_mismatch"

    # 6. Condition signature verification (numerical values and units)
    query_cond_sig = compute_condition_signature(extracted_conditions)
    candidate_cond_sig = candidate.condition_signature or {}

    # Must have identical set of condition fields
    if set(query_cond_sig.keys()) != set(candidate_cond_sig.keys()):
        return False, "numerical_condition_mismatch"

    for field, q_val_info in query_cond_sig.items():
        c_val_info = candidate_cond_sig.get(field)
        if not c_val_info:
            return False, "numerical_condition_mismatch"
        if q_val_info.get("value") != c_val_info.get("value"):
            return False, "numerical_condition_mismatch"
        if q_val_info.get("unit") != c_val_info.get("unit"):
            return False, "numerical_condition_mismatch"

    # 7. Semantic and linguistic indicator checks
    query_indicators = compute_question_indicators(question)
    cand_indicators = candidate.indicators or {}

    if query_indicators["has_negation"] != cand_indicators.get("has_negation", False):
        return False, "negation_mismatch"

    if query_indicators["has_comparison"] != cand_indicators.get("has_comparison", False):
        return False, "comparison_mismatch"

    if query_indicators["has_exception"] != cand_indicators.get("has_exception", False):
        return False, "exception_mismatch"

    # Approval authorities check
    q_auths = set(query_indicators.get("approval_authorities", []))
    c_auths = set(cand_indicators.get("approval_authorities", []))
    if q_auths != c_auths:
        return False, "critical_entity_mismatch"

    # Policy subject compatibility
    q_subj = query_indicators.get("policy_subject")
    c_subj = cand_indicators.get("policy_subject")
    if q_subj and c_subj and q_subj != "general" and c_subj != "general" and q_subj != c_subj:
        return False, "policy_subject_mismatch"

    # 8. Check rule outcomes
    active_rule_outcomes = {r.rule.rule_id: r.outcome for r in rule_results if not r.is_error}
    for rule_id, expected_outcome in (candidate.rule_outcomes or {}).items():
        if rule_id in active_rule_outcomes and active_rule_outcomes[rule_id] != expected_outcome:
            return False, "rule_outcome_mismatch"

    # 9. Verify cited evidence is authorized in current retrieval set
    authorized_cids = {
        c.chunk_id if isinstance(c, RetrievalResult) else c.get("chunk_id")
        for c in authorized_chunks
    }
    for cited_cid in candidate.cited_chunk_ids:
        if cited_cid not in authorized_cids:
            return False, "missing_authorized_citation"

    # 10. Verify canonical content hash and policy version integrity
    for cited_cid in candidate.cited_chunk_ids:
        canon_chunk = canonical_chunks_by_id.get(cited_cid)
        if not canon_chunk:
            return False, "missing_authorized_citation"

        expected_hash = candidate.cited_chunk_hashes.get(cited_cid)
        actual_hash = compute_chunk_content_hash(canon_chunk.get("text", ""))
        if expected_hash != actual_hash:
            return False, "stale_citation_content"

        expected_version = candidate.cited_policy_versions.get(cited_cid)
        actual_version = canon_chunk.get("version")
        if expected_version != actual_version:
            return False, "stale_citation_content"

    # All mandatory conditions satisfied
    return True, None


def ensure_cache_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_CACHE_COLLECTION_NAME,
) -> None:
    """Ensure that the cache collection exists with vector size 384 and Cosine distance.

    Never deletes or alters the policy collection.
    Creates payload indexes on real servers if needed.
    """
    existing_collections: List[str] = []
    try:
        cols_resp = client.get_collections()
        existing_collections = [c.name for c in cols_resp.collections]
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect Qdrant collections: {exc}") from exc

    if collection_name not in existing_collections:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create cache collection '{collection_name}': {exc}") from exc

    # Validate collection vector configuration
    try:
        col_info = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to get info for cache collection '{collection_name}': {exc}") from exc

    params = getattr(col_info.config, "params", None)
    vectors_cfg = getattr(params, "vectors", None) if params else None
    if isinstance(vectors_cfg, models.VectorParams):
        actual_size = vectors_cfg.size
        actual_dist = vectors_cfg.distance
    elif isinstance(vectors_cfg, dict):
        actual_size = vectors_cfg.get("size")
        actual_dist = vectors_cfg.get("distance")
    else:
        actual_size = None
        actual_dist = None

    if actual_size is not None and actual_size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Cache collection '{collection_name}' has vector dimension {actual_size}, "
            f"expected {EMBEDDING_DIMENSION}."
        )

    if actual_dist is not None:
        expected_dist = models.Distance.COSINE
        expected_str = expected_dist.value if hasattr(expected_dist, "value") else str(expected_dist).lower()
        actual_str = actual_dist.value if hasattr(actual_dist, "value") else str(actual_dist).lower()
        if actual_str.lower() != expected_str.lower():
            raise ValueError(
                f"Cache collection '{collection_name}' has distance metric '{actual_str}', "
                f"expected '{expected_str}'."
            )

    # Ensure payload indexes on non-in-memory clients
    if not is_in_memory_or_local_client(client):
        schema = getattr(col_info, "payload_schema", {}) or {}
        for field_name, expected_type in CACHE_REQUIRED_PAYLOAD_INDEXES:
            if field_name not in schema:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=expected_type,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to create payload index '{field_name}' in '{collection_name}': {exc}"
                    ) from exc


def lookup_cache(
    client: QdrantClient,
    embedder: BGEEmbedder,
    question: str,
    role: str,
    query_date: str,
    corpus_fingerprint: str,
    authorized_chunks: Sequence[Union[RetrievalResult, Dict[str, Any]]],
    extracted_conditions: Sequence[Dict[str, Any]],
    rule_results: Sequence[RuleEvaluationResult],
    canonical_chunks_by_id: Dict[str, Dict[str, Any]],
    similarity_threshold: float = CACHE_DEFAULT_SIMILARITY_THRESHOLD,
    collection_name: str = QDRANT_CACHE_COLLECTION_NAME,
    similarity_only: bool = False,
) -> CacheLookupResult:
    """Perform guarded semantic cache lookup for an incoming query.

    Args:
        client: QdrantClient instance (cloud or in-memory).
        embedder: BGEEmbedder instance.
        question: User query string.
        role: Requesting employee role.
        query_date: Policy effective date (ISO YYYY-MM-DD).
        corpus_fingerprint: Active corpus SHA-256 fingerprint.
        authorized_chunks: Chunks that have passed authorization and verification.
        extracted_conditions: Structured condition dicts extracted from question.
        rule_results: Structured rule evaluation results.
        canonical_chunks_by_id: Dict of canonical chunk records.
        similarity_threshold: Minimum cosine similarity required to consider a candidate.
        collection_name: Name of the cache collection.
        similarity_only: If True, bypass safety validations (used solely for evaluation benchmark).

    Returns:
        CacheLookupResult with hit status, record (if verified), similarity, and telemetry.
    """
    start_time = time.perf_counter()

    # 1. Embed query vector
    try:
        vector = embedder.embed_query(question)
    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=False,
            similarity=None,
            rejection_reason="embedding_error",
            latency_ms=round(latency, 2),
        )

    # 2. Build search filter for candidate discovery (role + corpus_fingerprint in guarded mode)
    search_filter: Optional[models.Filter] = None
    if not similarity_only:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="corpus_fingerprint",
                    match=models.MatchValue(value=corpus_fingerprint),
                ),
                models.FieldCondition(
                    key="role",
                    match=models.MatchValue(value=role),
                ),
            ]
        )

    # 3. Query Qdrant cache collection
    try:
        if hasattr(client, "query_points"):
            query_res = client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=search_filter,
                limit=5,
                with_payload=True,
            )
            scored_points = list(query_res.points) if query_res and hasattr(query_res, "points") else []
        elif hasattr(client, "search"):
            scored_points = client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=search_filter,
                limit=5,
                with_payload=True,
            )
        else:
            scored_points = []
    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=False,
            similarity=None,
            rejection_reason="cache_service_unavailable",
            latency_ms=round(latency, 2),
        )

    if not scored_points:
        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=False,
            similarity=None,
            rejection_reason="cache_miss",
            latency_ms=round(latency, 2),
        )

    top_candidate = scored_points[0]
    top_score = float(top_candidate.score)

    if top_score < similarity_threshold:
        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=False,
            similarity=round(top_score, 4),
            rejection_reason="low_similarity",
            latency_ms=round(latency, 2),
        )

    # 4. If similarity_only is requested (for evaluation mode B), reuse without safeguards
    if similarity_only:
        payload = top_candidate.payload or {}
        record = CacheRecord.from_payload(payload)
        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=True,
            record=record,
            similarity=round(top_score, 4),
            rejection_reason=None,
            latency_ms=round(latency, 2),
        )

    # 5. Guarded semantic validation of candidates
    first_rejection_reason: Optional[str] = None
    for point in scored_points:
        cand_score = float(point.score)
        if cand_score < similarity_threshold:
            break

        payload = point.payload or {}
        try:
            record = CacheRecord.from_payload(payload)
        except Exception:
            if not first_rejection_reason:
                first_rejection_reason = "malformed_cache_record"
            continue

        is_valid, reason = validate_cache_candidate(
            candidate=record,
            question=question,
            role=role,
            query_date=query_date,
            corpus_fingerprint=corpus_fingerprint,
            authorized_chunks=authorized_chunks,
            extracted_conditions=extracted_conditions,
            rule_results=rule_results,
            canonical_chunks_by_id=canonical_chunks_by_id,
        )

        if is_valid:
            latency = (time.perf_counter() - start_time) * 1000.0
            return CacheLookupResult(
                hit=True,
                record=record,
                similarity=round(cand_score, 4),
                rejection_reason=None,
                latency_ms=round(latency, 2),
            )
        elif not first_rejection_reason:
            first_rejection_reason = reason

    latency = (time.perf_counter() - start_time) * 1000.0
    return CacheLookupResult(
        hit=False,
        similarity=round(top_score, 4),
        rejection_reason=first_rejection_reason or "validation_failed",
        latency_ms=round(latency, 2),
    )


def insert_cache_entry(
    client: QdrantClient,
    embedder: BGEEmbedder,
    question: str,
    answer: str,
    status: str,
    role: str,
    query_date: str,
    corpus_fingerprint: str,
    citations: Sequence[str],
    citation_metadata: Sequence[Dict[str, Any]],
    referenced_rule_ids: Sequence[str],
    rule_results: Sequence[RuleEvaluationResult],
    canonical_chunks_by_id: Dict[str, Dict[str, Any]],
    extracted_conditions: Sequence[Dict[str, Any]],
    model: str = GENERATION_MODEL,
    collection_name: str = QDRANT_CACHE_COLLECTION_NAME,
) -> Optional[str]:
    """Insert a verified, generation-produced answer into the semantic cache.

    Only caches results with status="answered", non-empty citations, and valid evidence.
    Deterministic rule answers, refusals, and errors are NEVER cached.

    Returns:
        entry_id (str) if successfully inserted, None otherwise.
    """
    # 1. Eligibility gate
    if status != "answered":
        return None
    if not citations or not answer or not answer.strip():
        return None

    norm_q = normalize_question(question)
    if not norm_q:
        return None

    # Deterministic point ID based on corpus, role, and normalized question
    entry_id = str(uuid.uuid5(CACHE_POINT_NAMESPACE, f"{corpus_fingerprint}:{role}:{norm_q}"))

    # 2. Compute embeddings and metadata
    try:
        vector = embedder.embed_query(question)
    except Exception:
        return None

    # Compute cited chunk hashes and versions
    cited_hashes: Dict[str, str] = {}
    cited_versions: Dict[str, str] = {}
    for cid in citations:
        c = canonical_chunks_by_id.get(cid)
        if c:
            cited_hashes[cid] = compute_chunk_content_hash(c.get("text", ""))
            cited_versions[cid] = str(c.get("version", ""))

    rule_outcomes: Dict[str, str] = {}
    for r in rule_results:
        if r.rule.rule_id in referenced_rule_ids:
            rule_outcomes[r.rule.rule_id] = r.outcome

    condition_sig = compute_condition_signature(extracted_conditions)
    indicators = compute_question_indicators(question)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    record = CacheRecord(
        entry_id=entry_id,
        schema_version=CACHE_SCHEMA_VERSION,
        original_question=question.strip(),
        normalized_question=norm_q,
        answer=answer.strip(),
        status=status,
        model=model,
        role=role,
        query_date=query_date,
        corpus_fingerprint=corpus_fingerprint,
        embedding_model=EMBEDDING_MODEL,
        condition_signature=condition_sig,
        indicators=indicators,
        cited_chunk_ids=list(citations),
        cited_chunk_hashes=cited_hashes,
        cited_policy_versions=cited_versions,
        referenced_rule_ids=list(referenced_rule_ids),
        rule_outcomes=rule_outcomes,
        citation_metadata=[dict(m) for m in citation_metadata],
        created_at=now_iso,
    )

    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=entry_id,
                    vector=vector,
                    payload=record.to_payload(),
                )
            ],
        )
        return entry_id
    except Exception:
        # A cache write failure must not raise or expose secrets
        return None
