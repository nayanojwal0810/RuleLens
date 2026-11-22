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
    GENERATION_EVIDENCE_LIMIT,
    GENERATION_MODEL,
    GROQ_API_KEY,
    get_env_var,
)
from rulelens.dataset import ROLES
from rulelens.llm import (
    AnswerResponse,
    AnswerStatus,
    CitationValidationError,
    GroqClientWrapper,
    SanitizedProviderError,
    build_evidence_fingerprint,
    validate_citations_and_rules,
)
from rulelens.qdrant_store import get_qdrant_client
from rulelens.retrieval import (
    ALLOWED_ROLES_SET,
    HybridRetriever,
    RetrievalResult,
    is_chunk_eligible,
    parse_query_date,
)
from rulelens.rules import (
    RuleEvaluationResult,
    extract_conditions_from_query,
    resolve_rules_for_evidence,
)


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
    token_usage: Optional[int] = None
    error_category: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_restricted_policy_access(
    retriever: HybridRetriever,
    query: str,
    role: str,
    query_date: datetime.date,
) -> bool:
    """Check if query targets a policy active on date that is restricted for the role.

    Uses only permissible policy metadata (allowed_roles, effective dates).
    Never inspects or forwards forbidden clause text to an LLM.
    """
    date_str = query_date.isoformat()
    active_chunks = [
        c for c in retriever.chunks
        if c["effective_from"] <= date_str <= (c.get("effective_to") or "9999-12-31")
    ]
    if not active_chunks:
        return False

    bm25 = retriever.get_bm25_index(active_chunks)
    candidates = bm25.search(query, top_k=3)
    if not candidates:
        return False

    top_candidate = candidates[0]
    try:
        cand_score = float(top_candidate.score)
    except (ValueError, TypeError, AttributeError):
        cand_score = 0.0

    # If the top matching active chunk has strong relevance and the role is not allowed
    if cand_score >= 12.0 and role not in top_candidate.chunk.get("allowed_roles", []):
        return True

    return False


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
            category="service_unavailable",
            safe_message="Evidence integrity verification failed: unknown chunk ID.",
        )

    canonical = canonical_chunks_by_id[cid]

    # Verify all 11 canonical fields
    exact_fields = [
        "chunk_id", "doc_id", "clause_id", "text", "title", "version",
        "effective_from", "source_file"
    ]
    for field in exact_fields:
        if c_dict.get(field) != canonical.get(field):
            raise SanitizedProviderError(
                category="service_unavailable",
                safe_message=f"Evidence integrity verification failed for field '{field}'.",
            )

    # Check effective_to (normalize empty string and None)
    c_eff_to = c_dict.get("effective_to") or None
    can_eff_to = canonical.get("effective_to") or None
    if c_eff_to != can_eff_to:
        raise SanitizedProviderError(
            category="service_unavailable",
            safe_message="Evidence integrity verification failed for 'effective_to'.",
        )

    # Check source_page as integer
    try:
        if int(c_dict.get("source_page", -1)) != int(canonical.get("source_page", -2)):
            raise SanitizedProviderError(
                category="service_unavailable",
                safe_message="Evidence integrity verification failed for 'source_page'.",
            )
    except (ValueError, TypeError):
        raise SanitizedProviderError(
            category="service_unavailable",
            safe_message="Evidence integrity verification failed for 'source_page'.",
        )

    # Check allowed_roles as sorted list
    c_roles = sorted(list(c_dict.get("allowed_roles", [])))
    can_roles = sorted(list(canonical.get("allowed_roles", [])))
    if c_roles != can_roles:
        raise SanitizedProviderError(
            category="service_unavailable",
            safe_message="Evidence integrity verification failed for 'allowed_roles'.",
        )

    # Re-verify eligibility on canonical record
    if not is_chunk_eligible(canonical, role, query_date):
        raise SanitizedProviderError(
            category="service_unavailable",
            safe_message="Evidence authorization verification failed.",
        )


_QUERY_STOPWORDS = {
    "what", "is", "the", "how", "much", "many", "can", "an", "a", "in", "on", "at",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "will", "just", "should", "now", "does", "do", "did", "have", "has",
    "had", "be", "are", "was", "were", "been", "being", "policy", "corporate",
    "company", "asterline", "services", "group", "required", "under", "must",
    "per", "use", "used", "employee", "rules", "rule", "guidelines",
}


def check_evidence_support(query: str, authorized_chunks: Sequence[Union[RetrievalResult, Dict[str, Any]]]) -> bool:
    """Conservative check whether retrieved authorized evidence mentions salient query terms.

    If none of the core content words appear in any retrieved chunk, the evidence cannot support an answer.
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    salient_words = [w for w in words if w not in _QUERY_STOPWORDS]
    if not salient_words:
        return True

    all_evidence_text = " ".join(
        (c.text if isinstance(c, RetrievalResult) else c["text"]).lower()
        for c in authorized_chunks
    )

    # If not a single salient term from the query is present in the evidence
    matches = sum(1 for w in salient_words if w in all_evidence_text)
    return matches > 0


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
        top_k: Number of evidence chunks to retrieve for generation (default: 4).
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

    validated_date = parse_query_date(query_date)
    date_str = validated_date.isoformat()

    # Step B: Initialize retriever & verify corpus integrity
    active_retriever = retriever or HybridRetriever(base_dir=base_dir)

    manifest = getattr(active_retriever, "corpus_manifest", None)
    if not manifest:
        base_path = Path(base_dir) if base_dir else Path.cwd()
        manifest_p = base_path / "data" / "processed" / "corpus_manifest.json"
        if not manifest_p.exists():
            manifest_p = base_path.parent / "data" / "processed" / "corpus_manifest.json"
        if manifest_p.exists():
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

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
        )

    # Step D: Retrieval
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
                error_category=integ_err.category,
                error=integ_err.safe_message,
            )

    ev_fingerprint = build_evidence_fingerprint(authorized_chunks, corpus_fp)

    # Step F: Safe refusal if authorized evidence is empty or cannot support the query
    if not authorized_chunks or not check_evidence_support(cleaned_question, authorized_chunks):
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
        )

    # Step G: Deterministic rule resolution
    extracted_conds = extract_conditions_from_query(cleaned_question)
    rule_results: List[RuleEvaluationResult] = []
    applicable_rule_ids: Set[str] = set()

    chunk_dicts = [c.to_dict() for c in authorized_chunks]
    rule_results = resolve_rules_for_evidence(
        authorized_chunks=chunk_dicts,
        role=role,
        query_date=validated_date,
        base_dir=base_dir,
        extracted_conditions=extracted_conds,
    )
    # Only rules that passed or define thresholds are applicable
    applicable_rule_ids = {r.rule.rule_id for r in rule_results if r.passed}

    # Step H: Call Groq LLM
    active_llm = llm_client or GroqClientWrapper()
    try:
        model_resp, llm_meta = active_llm.generate_answer(
            question=cleaned_question,
            role=role,
            evidence_chunks=authorized_chunks,
            query_date=date_str,
            applicable_rule_facts=rule_results,
        )
    except SanitizedProviderError as prov_err:
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
            error_category=prov_err.category,
            error=prov_err.safe_message,
        )
    except Exception:
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
            error_category="invalid_citation",
            error="Citation validation failed.",
        )

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
        token_usage=llm_meta.get("total_tokens"),
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
    args = parser.parse_args()

    q_client = None
    if args.backend == "qdrant":
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
        )
    except Exception as exc:
        print(f"Operational error: {exc}", file=sys.stderr)
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
    if result.error:
        print(f"Error [{result.error_category}]: {result.error}")
    print("=" * 60 + "\n")

    if result.status == "operational_error":
        print(f"Exiting with status 1 due to operational error [{result.error_category}].", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
