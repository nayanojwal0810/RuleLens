"""Evidence-grounded answer generation orchestration for RuleLens.

Integrates hybrid retrieval, deterministic structured-rule resolution,
defense-in-depth authorization verification, Groq LLM generation, and
strict citation validation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from rulelens.config import (
    CACHE_DEFAULT_SIMILARITY_THRESHOLD,
    GENERATION_EVIDENCE_LIMIT,
    GENERATION_MODEL,
    GROQ_API_KEY,
    QDRANT_CACHE_COLLECTION_NAME,
    get_cache_similarity_threshold,
    get_env_var,
    is_cache_enabled,
)
from rulelens.dataset import ROLES
from rulelens.llm import (
    AnswerResponse,
    AnswerStatus,
    CitationValidationError,
    GroqClientWrapper,
    ProviderTelemetrySnapshot,
    SanitizedProviderError,
    build_citation_metadata,
    build_evidence_fingerprint,
    validate_citations_and_rules,
)
from rulelens.public_policy_topics import match_public_policy_topics
from rulelens.qdrant_store import get_qdrant_client
from rulelens.retrieval import (
    ALLOWED_ROLES_SET,
    BGEEmbedder,
    HybridRetriever,
    RetrievalResult,
    is_chunk_eligible,
    parse_query_date,
)
from rulelens.rules import (
    AuthoritativeDecision,
    RuleEvaluationResult,
    _find_unsupported_query_terms,
    evaluate_authoritative_rule_decision,
    extract_conditions_from_query,
    resolve_rules_for_evidence,
)
from rulelens.cache import (
    CacheLookupResult,
    ensure_cache_collection,
    insert_cache_entry,
    lookup_cache,
)


def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return int(val)
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any) -> Optional[float]:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@dataclass
class GenerationResult:
    """Complete, structured outcome of answering a policy question."""

    question: str
    role: str
    query_date: str
    answer: str
    status: str  # "answered", "insufficient", "restricted", or "operational_error"
    reason: str
    citations: List[str]
    citation_metadata: List[Dict[str, Any]]
    referenced_rule_ids: List[str]
    model: str
    corpus_fingerprint: str
    evidence_fingerprint: Dict[str, Any]
    latency_ms: float
    provider_calls_attempted: Optional[int] = 0
    provider_latency_ms: Optional[float] = None
    token_usage: Optional[int] = None
    schema_valid: Optional[bool] = None
    has_incomplete_tokens: bool = False
    error_category: Optional[str] = None
    error: Optional[str] = None
    # Stage F: Cache telemetry
    cache_enabled: bool = False
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    cache_rejection_reason: Optional[str] = None
    cache_lookup_latency_ms: Optional[float] = None
    cache_entry_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provider_calls_attempted"] = _safe_int(self.provider_calls_attempted, default=None)
        d["provider_latency_ms"] = _safe_float(self.provider_latency_ms)
        d["token_usage"] = _safe_int(self.token_usage, default=None)
        d["schema_valid"] = self.schema_valid if isinstance(self.schema_valid, bool) else None
        d["has_incomplete_tokens"] = bool(self.has_incomplete_tokens) if isinstance(self.has_incomplete_tokens, bool) else False
        d["cache_enabled"] = bool(self.cache_enabled)
        d["cache_hit"] = bool(self.cache_hit)
        d["cache_similarity"] = _safe_float(self.cache_similarity)
        d["cache_rejection_reason"] = self.cache_rejection_reason
        d["cache_lookup_latency_ms"] = _safe_float(self.cache_lookup_latency_ms)
        d["cache_entry_id"] = self.cache_entry_id
        return d


def _stem_word(w: str) -> str:
    """Safely strip common English inflections without mangling short roots."""
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def check_restricted_policy_access(
    retriever: HybridRetriever,
    query: str,
    role: str,
    query_date: datetime.date,
) -> bool:
    """Check if query unambiguously targets a policy active on date that is restricted for the role.

    Uses only non-sensitive document metadata (doc_id, title, department, allowed_roles, effective dates).
    Never creates a text search index over chunk bodies or inspects forbidden clause text.
    If restriction cannot be established reliably, returns False so the caller falls back
    to authorized retrieval which safely returns insufficient.
    """
    date_str = query_date.isoformat()
    # Cache document-level metadata catalog on retriever to avoid redundant grouping
    doc_catalog = getattr(retriever, "_doc_metadata_catalog", None)
    if not isinstance(doc_catalog, dict):
        doc_catalog = {}
        for c in getattr(retriever, "chunks", []):
            did = c.get("doc_id")
            if not did:
                continue
            if did not in doc_catalog:
                doc_catalog[did] = {
                    "doc_id": did,
                    "title": c.get("title", ""),
                    "department": c.get("department", ""),
                    "allowed_roles": set(c.get("allowed_roles", [])),
                    "effective_from": c.get("effective_from", "0000-01-01"),
                    "effective_to": c.get("effective_to") or "9999-12-31",
                }
            else:
                doc_catalog[did]["allowed_roles"].update(c.get("allowed_roles", []))
        try:
            retriever._doc_metadata_catalog = doc_catalog
        except Exception:
            pass

    if not doc_catalog:
        return False

    active_docs = [
        d for d in doc_catalog.values()
        if d["effective_from"] <= date_str <= d["effective_to"]
    ]
    if not active_docs:
        return False

    q_norm = query.lower()
    restr_matches: List[Tuple[str, int]] = []
    auth_matches: List[Tuple[str, int]] = []

    for doc in active_docs:
        did = doc["doc_id"].lower()
        short_did = did.replace("doc-", "")
        is_allowed = role in doc["allowed_roles"]

        # Check explicit doc_id match
        if re.search(r"\b" + re.escape(did) + r"\b", q_norm) or re.search(r"\b" + re.escape(short_did) + r"\b", q_norm):
            score = 100
        else:
            # Check title match against distinctive title terms (excluding generic corporate words)
            t_clean = re.sub(r"\bpolicy\b|\bv[0-9]+\b", "", doc["title"].lower()).strip()
            words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t_clean)]
            if not words:
                score = 0
            else:
                phrase = " ".join(words)
                if phrase in q_norm:
                    score = len(words) * 10
                else:
                    # Stem matching for all title words
                    matched_words = 0
                    for w in words:
                        root = _stem_word(w)
                        if re.search(r"\b" + re.escape(root), q_norm):
                            matched_words += 1
                    if matched_words == len(words):
                        score = len(words) * 5
                    else:
                        score = 0

        if score >= 10:
            if is_allowed:
                auth_matches.append((doc["doc_id"], score))
            else:
                restr_matches.append((doc["doc_id"], score))

    # Also evaluate non-sensitive public policy topics catalog
    topic_matches = match_public_policy_topics(query)
    for t_did, t_score in topic_matches:
        doc = next((d for d in active_docs if d["doc_id"] == t_did), None)
        if doc:
            if role in doc["allowed_roles"]:
                auth_matches.append((doc["doc_id"], t_score))
            else:
                restr_matches.append((doc["doc_id"], t_score))

    if not restr_matches:
        return False

    max_restr_score = max(s[1] for s in restr_matches)
    max_auth_score = max([s[1] for s in auth_matches], default=0)

    # Return True only if the restricted policy match is unambiguous and strictly more specific
    # than any matching authorized document title or topic
    return max_restr_score > max_auth_score


def validate_candidate_evidence(
    candidate: Union[RetrievalResult, Dict[str, Any]],
    canonical_chunks_by_id: Dict[str, Dict[str, Any]],
    role: str,
    query_date: datetime.date,
) -> None:
    """Compare candidate identity and evidence fields against validated canonical corpus.

    Raises:
        SanitizedProviderError: If any mismatch, forgery, or corruption is detected.
    """
    c_dict = candidate.to_dict() if isinstance(candidate, RetrievalResult) else candidate
    cid = c_dict.get("chunk_id")
    if not cid or cid not in canonical_chunks_by_id:
        raise SanitizedProviderError(
            category="evidence_integrity_error",
            safe_message="Evidence integrity verification failed: unknown chunk ID.",
        )

    canonical = canonical_chunks_by_id[cid]

    # Verify all 12 canonical fields
    exact_fields = [
        "chunk_id", "doc_id", "clause_id", "department", "text", "title", "version",
        "effective_from", "source_file"
    ]
    for field in exact_fields:
        if c_dict.get(field) != canonical.get(field):
            raise SanitizedProviderError(
                category="evidence_integrity_error",
                safe_message=f"Evidence integrity verification failed for field '{field}'.",
            )

    # Check effective_to (normalize empty string and None)
    c_eff_to = c_dict.get("effective_to") or None
    can_eff_to = canonical.get("effective_to") or None
    if c_eff_to != can_eff_to:
        raise SanitizedProviderError(
            category="evidence_integrity_error",
            safe_message="Evidence integrity verification failed for 'effective_to'.",
        )

    # Check source_page as integer
    try:
        if int(c_dict.get("source_page", -1)) != int(canonical.get("source_page", -2)):
            raise SanitizedProviderError(
                category="evidence_integrity_error",
                safe_message="Evidence integrity verification failed for 'source_page'.",
            )
    except (ValueError, TypeError):
        raise SanitizedProviderError(
            category="evidence_integrity_error",
            safe_message="Evidence integrity verification failed for 'source_page'.",
        )

    # Check allowed_roles as sorted list
    c_roles = sorted(list(c_dict.get("allowed_roles", [])))
    can_roles = sorted(list(canonical.get("allowed_roles", [])))
    if c_roles != can_roles:
        raise SanitizedProviderError(
            category="evidence_integrity_error",
            safe_message="Evidence integrity verification failed for 'allowed_roles'.",
        )

    # Re-verify eligibility on canonical record
    if not is_chunk_eligible(canonical, role, query_date):
        raise SanitizedProviderError(
            category="evidence_integrity_error",
            safe_message="Evidence authorization verification failed.",
        )


_QUERY_STOPWORDS = {
    # Standard English function words and interrogatives
    "what", "is", "the", "how", "much", "many", "can", "an", "a", "in", "on", "at",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "will", "just", "should", "now", "does", "do", "did", "have", "has",
    "had", "be", "are", "was", "were", "been", "being", "which", "who", "whom",
    "this", "that", "these", "those", "am", "via", "per", "under", "without",
    "and", "but", "or", "as", "if", "take", "taking",
    # Currency units and transaction verbs
    "inr", "rs", "rupee", "rupees", "usd", "eur", "gbp", "dollar", "dollars", "euro", "euros", "pound", "pounds",
    "spend", "spending", "spent", "pay", "paying", "payment", "payments", "cost", "costing", "costs",
    # Enterprise policy boilerplate terms appearing across virtually all policies
    "policy", "corporate", "company", "asterline", "services", "group", "required",
    "use", "used", "employee", "employees", "rules", "rule", "guidelines", "guideline",
    "procedure", "procedures", "standard", "standards", "eligible", "eligibility",
    "approved", "approval", "approvals", "allowable", "allowed", "reimbursement",
    "reimbursements", "expense", "expenses", "claim", "claims", "travel", "request",
    "requests", "department", "applicable", "process", "submitted", "submission",
    "criteria", "amount", "limit", "limits", "within", "month", "months", "day", "days",
    "personal", "business", "domestic", "international", "official", "general",
}


def check_evidence_support(
    query: str,
    authorized_chunks: Sequence[Union[RetrievalResult, Dict[str, Any]]],
) -> bool:
    """Conservative check whether retrieved authorized evidence mentions distinctive query terms.

    Filters ubiquitous corporate policy boilerplate terms and stopwords.
    Requires distinctive term coverage before proceeding to generation.
    """
    if not authorized_chunks:
        return False

    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    salient_words = list(dict.fromkeys([w for w in words if w not in _QUERY_STOPWORDS]))
    if not salient_words:
        # If query consists entirely of generic corporate words, allow generation to assess support
        return True

    all_evidence_text = " ".join(
        (c.text if isinstance(c, RetrievalResult) else c["text"]).lower()
        for c in authorized_chunks
    )

    matched_salient = []
    for w in salient_words:
        root = _stem_word(w)
        if re.search(r"\b" + re.escape(root), all_evidence_text):
            matched_salient.append(w)

    match_count = len(matched_salient)
    total_salient = len(salient_words)

    if total_salient <= 2:
        return match_count >= 1
    else:
        # Require at least 2 distinctive words or at least 40% distinctive coverage
        return match_count >= 2 or (match_count / total_salient) >= 0.4


def answer_question(
    question: str,
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
    retriever: Optional[HybridRetriever] = None,
    llm_client: Optional[GroqClientWrapper] = None,
    qdrant_client: Optional[Any] = None,
    base_dir: Optional[Union[Path, str]] = None,
    top_k: int = GENERATION_EVIDENCE_LIMIT,
    candidate_depth: int = 20,
    mode: str = "hybrid",
    dense_backend: str = "local",
    cache_client: Optional[Any] = None,
    cache_enabled: Optional[bool] = None,
    cache_threshold: Optional[float] = None,
    embedder: Optional[BGEEmbedder] = None,
) -> GenerationResult:
    """Answer an enterprise policy question using verified retrieval, rules, and Groq.

    Args:
        question: User query string.
        role: Requesting employee role (must be one of ROLES).
        query_date: Effective date for policy retrieval.
        retriever: Injected HybridRetriever instance (optional).
        llm_client: Injected GroqClientWrapper instance (optional).
        qdrant_client: Optional Qdrant client when dense_backend="qdrant".
        base_dir: Repository root directory.
        top_k: Number of evidence chunks to retrieve for generation (default: 4, max: 4).
        candidate_depth: Retrieval candidate pool depth (default: 20).
        mode: Retrieval mode ('hybrid', 'dense', 'bm25').
        dense_backend: Dense retrieval backend ('local' or 'qdrant').

    Returns:
        GenerationResult object with grounded answer, verified citations, and metadata.
    """
    start_total_time = time.perf_counter()

    # Step A: Input validation
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Invalid question: question must be a non-empty string.")

    cleaned_question = question.strip()

    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 4:
        raise ValueError(f"Invalid top_k: top_k must be an integer between 1 and 4 inclusive, got {top_k}.")

    validated_date = parse_query_date(query_date)
    date_str = validated_date.isoformat()

    # Step B: Initialize retriever & verify corpus integrity
    try:
        active_retriever = retriever or HybridRetriever(base_dir=base_dir)
    except Exception as exc:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The policy verification system encountered an operational error.",
            status="operational_error",
            reason="Retriever initialization failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint="",
            evidence_fingerprint={},
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            error_category="service_unavailable",
            error="Retriever initialization failed.",
        )

    manifest = getattr(active_retriever, "corpus_manifest", None)
    if not manifest:
        base_path = Path(base_dir) if base_dir else Path.cwd()
        manifest_p = base_path / "data" / "processed" / "corpus_manifest.json"
        if not manifest_p.exists():
            manifest_p = base_path.parent / "data" / "processed" / "corpus_manifest.json"
        if manifest_p.exists():
            try:
                manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            except Exception:
                manifest = None

    if not manifest or not manifest.get("processed_fingerprint"):
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The policy verification system encountered an evidence integrity error.",
            status="operational_error",
            reason="Corpus fingerprint manifest is missing or inconsistent.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint="",
            evidence_fingerprint={},
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            error_category="service_unavailable",
            error="Corpus fingerprint manifest is missing or inconsistent.",
        )

    corpus_fp = manifest["processed_fingerprint"]

    # Step C: Check restricted document access via permissible policy metadata
    if check_restricted_policy_access(active_retriever, cleaned_question, role, validated_date):
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="Access restricted. Your current role does not have authorization to view this policy area.",
            status=AnswerStatus.RESTRICTED.value,
            reason="User role lacks permission for the requested policy area.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint={"corpus_fingerprint": corpus_fp},
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Step D: Retrieval
    try:
        retrieved_results = active_retriever.search(
            query=cleaned_question,
            role=role,
            query_date=validated_date,
            mode=mode,
            dense_backend=dense_backend,
            qdrant_client=qdrant_client,
            top_k=top_k,
            candidate_depth=candidate_depth,
        )
    except Exception as exc:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The policy verification system encountered an operational error during retrieval.",
            status="operational_error",
            reason="Retrieval execution failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint={},
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            error_category="service_unavailable",
            error="Retrieval execution failed.",
        )

    # Step E: Strengthened evidence validation against canonical corpus records
    canonical_chunks = getattr(active_retriever, "chunks_by_id", {})
    if not canonical_chunks and hasattr(active_retriever, "chunks"):
        canonical_chunks = {c["chunk_id"]: c for c in active_retriever.chunks}

    authorized_chunks: List[RetrievalResult] = []
    for res in retrieved_results:
        try:
            validate_candidate_evidence(res, canonical_chunks, role, validated_date)
            authorized_chunks.append(res)
        except SanitizedProviderError as integ_err:
            total_latency = (time.perf_counter() - start_total_time) * 1000.0
            return GenerationResult(
                question=cleaned_question,
                role=role,
                query_date=date_str,
                answer="The policy verification system encountered an evidence integrity error.",
                status="operational_error",
                reason=integ_err.safe_message,
                citations=[],
                citation_metadata=[],
                referenced_rule_ids=[],
                model=GENERATION_MODEL,
                corpus_fingerprint=corpus_fp,
                evidence_fingerprint={},
                latency_ms=round(total_latency, 2),
                provider_calls_attempted=0,
                provider_latency_ms=None,
                error_category=integ_err.category,
                error=integ_err.safe_message,
            )

    ev_fingerprint = build_evidence_fingerprint(authorized_chunks, corpus_fp)

    # Step F: Safe refusal if authorized evidence is empty
    if not authorized_chunks:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The provided corporate policies do not contain sufficient evidence to answer this question.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason="Retrieved evidence does not contain information supporting the requested policy facts.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Step G: Deterministic rule resolution
    try:
        extracted_conds = extract_conditions_from_query(cleaned_question)
        chunk_dicts = [c.to_dict() for c in authorized_chunks]
        rule_results = resolve_rules_for_evidence(
            authorized_chunks=chunk_dicts,
            role=role,
            query_date=validated_date,
            base_dir=base_dir,
            extracted_conditions=extracted_conds,
        )
    except Exception as exc:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The policy verification system encountered an operational error loading policy rules.",
            status="operational_error",
            reason="Structured rule loading or evaluation failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            token_usage=None,
            schema_valid=None,
            has_incomplete_tokens=False,
            error_category="rule_loading_error",
            error="Structured rule resolution failed due to an infrastructure error.",
        )

    # If any rule condition input has an invalid unit, type error, or ambiguous condition, fail safely with insufficient
    rule_errors = [r for r in rule_results if r.is_error]
    if rule_errors:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        err = rule_errors[0]
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The policy question contains rule condition inputs that could not be evaluated safely.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason=f"Rule evaluation error ({err.error_type}): {err.explanation}",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            token_usage=None,
            schema_valid=None,
            has_incomplete_tokens=False,
            error_category=err.error_type,
            error=err.explanation,
        )

    # Step G.1: Authoritative deterministic rule decision for clear single-condition decision queries
    authoritative_dec = evaluate_authoritative_rule_decision(
        question=cleaned_question,
        authorized_chunks=chunk_dicts,
        rule_results=rule_results,
        extracted_conditions=extracted_conds,
    )
    if authoritative_dec.is_authoritative:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        cited_chunk = canonical_chunks.get(authoritative_dec.citation_chunk_id)
        cited_meta = [build_citation_metadata(cited_chunk)] if cited_chunk else []
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer=authoritative_dec.answer,
            status=AnswerStatus.ANSWERED.value,
            reason=authoritative_dec.reason,
            citations=[authoritative_dec.citation_chunk_id],
            citation_metadata=cited_meta,
            referenced_rule_ids=[authoritative_dec.rule_id],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            token_usage=None,
            schema_valid=None,
            has_incomplete_tokens=False,
        )

    # Step G.1b: Boundary-crossing condition safeguard
    # For conditions crossing multiple rule outcomes (e.g. at least threshold, or range crossing threshold),
    # refuse safely with insufficient without sending ambiguous condition to Groq for an unsupported approval.
    if authoritative_dec.is_boundary_crossing or any(c.get("is_boundary_crossing") for c in extracted_conds):
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The provided corporate policies establish different approval requirements across the specified condition range; an unconditional decision cannot be determined.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason="Numerical condition crosses policy approval boundaries.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Step G.1c: Fail-safe handling of unresolved conditions for decision requests
    # If the question asks a concrete policy decision and the condition cannot be evaluated
    # or the evidence is incomplete, refuse safely with insufficient without Groq.
    if authoritative_dec.is_unresolved or any(c.get("cannot_evaluate") for c in extracted_conds):
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The provided corporate policies do not contain sufficient evidence to establish an authoritative decision for this condition.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason=authoritative_dec.reason or "Relevant policy rule condition is unresolved or cannot be evaluated safely.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Step G.2: If the query introduced unsupported terms for a numerical approval/decision rule,
    # the approval threshold cannot establish whether that expense category or purpose is eligible.
    # Refuse safely without sending unsupported evidence to Groq for an affirmative decision.
    if authoritative_dec.unsupported_terms:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The provided corporate policies do not contain sufficient evidence to answer this question.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason=f"Numerical approval threshold does not establish eligibility for unsupported query terms: {', '.join(authoritative_dec.unsupported_terms)}.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Filter applicable rules for generation: A numerical approval threshold must not establish
    # whether an expense category is eligible; reject rule satisfaction if chunk has unsupported query terms.
    chunk_map = {c.chunk_id: c for c in authorized_chunks}
    applicable_rule_results: List[RuleEvaluationResult] = []
    for r in rule_results:
        if r.passed and not r.is_error:
            chk = chunk_map.get(r.rule.chunk_id)
            if chk and _find_unsupported_query_terms(cleaned_question, chk):
                continue
        applicable_rule_results.append(r)
    applicable_rule_ids = {r.rule.rule_id for r in applicable_rule_results if r.passed and not r.is_error}

    # Step G.3: Lexical evidence-support gate for open-ended LLM generation
    if not check_evidence_support(cleaned_question, authorized_chunks):
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The provided corporate policies do not contain sufficient evidence to answer this question.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason="Retrieved evidence does not contain information supporting the requested policy facts.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
        )

    # Step G.4: Guarded semantic cache lookup
    is_cache_active = cache_enabled if cache_enabled is not None else is_cache_enabled()
    active_cache_client = cache_client or qdrant_client
    active_embedder = embedder
    if is_cache_active and active_embedder is None:
        dense_ret = getattr(active_retriever, "dense_retriever", None)
        active_embedder = getattr(dense_ret, "embedder", None)
        if active_embedder is None:
            try:
                active_embedder = BGEEmbedder()
            except Exception:
                active_embedder = None

    if is_cache_active and active_cache_client is None:
        try:
            active_cache_client = get_qdrant_client()
        except Exception:
            active_cache_client = None

    cache_lookup_res: Optional[CacheLookupResult] = None
    if is_cache_active and active_cache_client is not None and active_embedder is not None:
        eff_threshold = (
            cache_threshold
            if cache_threshold is not None
            else (get_cache_similarity_threshold() or CACHE_DEFAULT_SIMILARITY_THRESHOLD)
        )
        try:
            ensure_cache_collection(active_cache_client)
            cache_lookup_res = lookup_cache(
                client=active_cache_client,
                embedder=active_embedder,
                question=cleaned_question,
                role=role,
                query_date=date_str,
                corpus_fingerprint=corpus_fp,
                authorized_chunks=authorized_chunks,
                extracted_conditions=extracted_conds,
                rule_results=rule_results,
                canonical_chunks_by_id=canonical_chunks,
                similarity_threshold=eff_threshold,
            )
            if cache_lookup_res.hit and cache_lookup_res.record:
                total_latency = (time.perf_counter() - start_total_time) * 1000.0
                rec = cache_lookup_res.record
                verified_meta = [
                    build_citation_metadata(canonical_chunks[cid])
                    for cid in rec.cited_chunk_ids
                    if cid in canonical_chunks
                ] or rec.citation_metadata
                return GenerationResult(
                    question=cleaned_question,
                    role=role,
                    query_date=date_str,
                    answer=rec.answer,
                    status=AnswerStatus.ANSWERED.value,
                    reason=f"Reused verified answer from semantic cache (entry {rec.entry_id}, similarity {cache_lookup_res.similarity:.4f}).",
                    citations=rec.cited_chunk_ids,
                    citation_metadata=verified_meta,
                    referenced_rule_ids=rec.referenced_rule_ids,
                    model=rec.model,
                    corpus_fingerprint=corpus_fp,
                    evidence_fingerprint=ev_fingerprint,
                    latency_ms=round(total_latency, 2),
                    provider_calls_attempted=0,
                    provider_latency_ms=None,
                    token_usage=0,
                    schema_valid=None,
                    has_incomplete_tokens=False,
                    cache_enabled=True,
                    cache_hit=True,
                    cache_similarity=cache_lookup_res.similarity,
                    cache_rejection_reason=None,
                    cache_lookup_latency_ms=cache_lookup_res.latency_ms,
                    cache_entry_id=rec.entry_id,
                )
        except Exception:
            cache_lookup_res = CacheLookupResult(
                hit=False,
                rejection_reason="cache_lookup_error",
            )

    # Step H: Call Groq LLM
    try:
        active_llm = llm_client or GroqClientWrapper()
    except (ValueError, SanitizedProviderError) as client_err:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        if isinstance(client_err, SanitizedProviderError):
            safe_msg = client_err.safe_message
            err_cat = client_err.category
        else:
            safe_msg = "The generation service is unavailable due to an authentication or configuration error."
            err_cat = "authentication_error"
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The generation service is unavailable due to an authentication or configuration error.",
            status="operational_error",
            reason="LLM client initialization failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            token_usage=None,
            schema_valid=None,
            has_incomplete_tokens=False,
            error_category=err_cat,
            error=safe_msg,
        )
    except Exception:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The generation service is temporarily unavailable. Please retry your query shortly.",
            status="operational_error",
            reason="LLM client initialization failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=0,
            provider_latency_ms=None,
            token_usage=None,
            schema_valid=None,
            has_incomplete_tokens=False,
            error_category="service_unavailable",
            error="Generation service encountered an unhandled provider error.",
        )

    baseline_telemetry = None
    if hasattr(active_llm, "get_telemetry_snapshot"):
        try:
            snap = active_llm.get_telemetry_snapshot()
            if isinstance(snap, ProviderTelemetrySnapshot):
                baseline_telemetry = snap
        except Exception:
            baseline_telemetry = None

    provider_calls: Optional[int] = None
    provider_lat: Optional[float] = None
    toks_used: Optional[int] = None
    has_inc_toks: bool = False
    schema_valid_status: Optional[bool] = None

    try:
        model_resp, llm_meta = active_llm.generate_answer(
            question=cleaned_question,
            role=role,
            evidence_chunks=authorized_chunks,
            query_date=date_str,
            applicable_rule_facts=applicable_rule_results,
        )
        schema_valid_status = True
        if baseline_telemetry is not None:
            delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
            provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
            provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
            toks_used = _safe_int(getattr(delta, "known_total_tokens", None), default=None)
            has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
        elif isinstance(llm_meta, dict):
            provider_calls = _safe_int(llm_meta.get("calls_made"), default=None)
            provider_lat = _safe_float(llm_meta.get("latency_ms"))
            toks_used = _safe_int(llm_meta.get("total_tokens"), default=None)
            has_inc_toks = False
        else:
            provider_calls = None
            provider_lat = None
            toks_used = None
            has_inc_toks = False
    except SanitizedProviderError as prov_err:
        schema_valid_status = False if prov_err.category == "invalid_generation" else None
        if baseline_telemetry is not None:
            delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
            provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
            provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
            has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
        else:
            provider_calls = None
            provider_lat = None
            has_inc_toks = False
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer=prov_err.safe_message,
            status="operational_error",
            reason=f"Provider operational error: {prov_err.category}",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=provider_calls,
            provider_latency_ms=provider_lat,
            token_usage=None,
            schema_valid=schema_valid_status,
            has_incomplete_tokens=has_inc_toks,
            error_category=prov_err.category,
            error=prov_err.safe_message,
        )
    except Exception:
        schema_valid_status = None
        if baseline_telemetry is not None:
            delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
            provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
            provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
            has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
        else:
            provider_calls = None
            provider_lat = None
            has_inc_toks = False
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The generation service is temporarily unavailable. Please retry your query shortly.",
            status="operational_error",
            reason="Provider operational error: service_unavailable",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=provider_calls,
            provider_latency_ms=provider_lat,
            token_usage=None,
            schema_valid=schema_valid_status,
            has_incomplete_tokens=has_inc_toks,
            error_category="service_unavailable",
            error="Generation service encountered an unhandled provider error.",
        )

    # Step I: Validate LLM output and citations
    try:
        val_citations, val_metadata, val_rules = validate_citations_and_rules(
            response=model_resp,
            authorized_chunks=authorized_chunks,
            role=role,
            query_date=validated_date,
            applicable_rule_ids=applicable_rule_ids,
        )
    except CitationValidationError as cit_err:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="Policy evidence retrieved, but the generated response could not be verified against authorized citations.",
            status="operational_error",
            reason=cit_err.safe_message,
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=provider_calls,
            provider_latency_ms=provider_lat,
            token_usage=toks_used,
            schema_valid=True,
            has_incomplete_tokens=has_inc_toks,
            error_category=cit_err.category,
            error=cit_err.safe_message,
        )
    except Exception:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="Policy evidence retrieved, but the generated response could not be verified against authorized citations.",
            status="operational_error",
            reason="Citation validation failed.",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            provider_calls_attempted=provider_calls,
            provider_latency_ms=provider_lat,
            token_usage=toks_used,
            schema_valid=True,
            has_incomplete_tokens=has_inc_toks,
            error_category="invalid_citation",
            error="Citation validation failed.",
        )

    # Step J: Insert verified answer into semantic cache if eligible
    inserted_cache_id = None
    if is_cache_active and active_cache_client is not None and active_embedder is not None:
        if model_resp.status == AnswerStatus.ANSWERED and val_citations:
            try:
                inserted_cache_id = insert_cache_entry(
                    client=active_cache_client,
                    embedder=active_embedder,
                    question=cleaned_question,
                    answer=model_resp.answer,
                    status=model_resp.status.value,
                    role=role,
                    query_date=date_str,
                    corpus_fingerprint=corpus_fp,
                    citations=val_citations,
                    citation_metadata=val_metadata,
                    referenced_rule_ids=val_rules,
                    rule_results=rule_results,
                    canonical_chunks_by_id=canonical_chunks,
                    extracted_conditions=extracted_conds,
                    model=GENERATION_MODEL,
                )
            except Exception:
                inserted_cache_id = None

    total_latency = (time.perf_counter() - start_total_time) * 1000.0

    return GenerationResult(
        question=cleaned_question,
        role=role,
        query_date=date_str,
        answer=model_resp.answer,
        status=model_resp.status.value,
        reason=model_resp.reason,
        citations=val_citations,
        citation_metadata=val_metadata,
        referenced_rule_ids=val_rules,
        model=GENERATION_MODEL,
        corpus_fingerprint=corpus_fp,
        evidence_fingerprint=ev_fingerprint,
        latency_ms=round(total_latency, 2),
        provider_calls_attempted=provider_calls,
        provider_latency_ms=provider_lat,
        token_usage=toks_used,
        schema_valid=True,
        has_incomplete_tokens=has_inc_toks,
        cache_enabled=is_cache_active,
        cache_hit=False,
        cache_similarity=cache_lookup_res.similarity if cache_lookup_res else None,
        cache_rejection_reason=cache_lookup_res.rejection_reason if cache_lookup_res else None,
        cache_lookup_latency_ms=cache_lookup_res.latency_ms if cache_lookup_res else None,
        cache_entry_id=inserted_cache_id,
    )


def main():
    """Command-line smoke test entry point for RuleLens answer generation."""
    parser = argparse.ArgumentParser(description="RuleLens Evidence-Grounded Answer Generation CLI")
    parser.add_argument("--question", required=True, help="User policy question string")
    parser.add_argument("--role", default="employee", choices=sorted(ROLES), help="Requesting role")
    parser.add_argument("--date", default=None, help="Policy effective date (ISO YYYY-MM-DD)")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "qdrant"],
        help="Dense retrieval backend (default: local)",
    )
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=False,
        help="Enable guarded semantic cache",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=None,
        help="Semantic cache similarity threshold (default: 0.90)",
    )
    args = parser.parse_args()

    q_client = None
    if args.backend == "qdrant" or args.enable_cache:
        try:
            q_client = get_qdrant_client()
        except Exception as exc:
            print(f"Error initializing Qdrant client: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        result = answer_question(
            question=args.question,
            role=args.role,
            query_date=args.date,
            dense_backend=args.backend,
            qdrant_client=q_client,
            cache_enabled=args.enable_cache,
            cache_threshold=args.cache_threshold,
        )
    except Exception as exc:
        sanitized = re.sub(r"gsk_[a-zA-Z0-9_-]+", "[REDACTED_API_KEY]", str(exc))
        sanitized = re.sub(r"Bearer\s+[^\s]+", "Bearer [REDACTED]", sanitized)
        print(f"Operational error: {sanitized}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RuleLens Answer Generation Result")
    print("=" * 60)
    print(f"Question:   {result.question}")
    print(f"Role:       {result.role} | Effective Date: {result.query_date}")
    print(f"Status:     {result.status.upper()}")
    print(f"Answer:     {result.answer}")
    if result.citations:
        print(f"Citations:  {', '.join(result.citations)}")
    if result.referenced_rule_ids:
        print(f"Rules:      {', '.join(result.referenced_rule_ids)}")
    print(f"Reason:     {result.reason}")
    print(f"Latency:    {result.latency_ms} ms")
    if result.token_usage:
        print(f"Tokens:     {result.token_usage}")
    if result.cache_enabled:
        hit_str = "HIT" if result.cache_hit else f"MISS ({result.cache_rejection_reason or 'uncached'})"
        sim_str = f" | Similarity: {result.cache_similarity}" if result.cache_similarity is not None else ""
        print(f"Cache:      {hit_str}{sim_str}")
    if result.error:
        sanitized_err = re.sub(r"gsk_[a-zA-Z0-9_-]+", "[REDACTED_API_KEY]", result.error)
        sanitized_err = re.sub(r"Bearer\s+[^\s]+", "Bearer [REDACTED]", sanitized_err)
        print(f"Error [{result.error_category}]: {sanitized_err}")
    print("=" * 60 + "\n")

    if result.status == "operational_error":
        print(f"Exiting with status 1 due to operational error [{result.error_category}].", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
