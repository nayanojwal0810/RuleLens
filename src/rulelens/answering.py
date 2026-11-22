"""Evidence-grounded answer generation orchestration for RuleLens.

Integrates hybrid retrieval, deterministic structured-rule resolution,
defense-in-depth authorization verification, Groq LLM generation, and
strict citation validation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

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
    GroqClientWrapper,
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
    status: str
    reason: str
    citations: List[str]
    citation_metadata: List[Dict[str, Any]]
    referenced_rule_ids: List[str]
    model: str
    corpus_fingerprint: str
    evidence_fingerprint: Dict[str, Any]
    latency_ms: float
    token_usage: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Mapping of sensitive terms to detect restriction when retrieval yields zero chunks
_RESTRICTED_TOPIC_KEYWORDS = {
    "employee": ["travel exception", "corporate card", "credit card", "sanitization", "vendor governance", "code of conduct"],
    "auditor": ["employee transfer", "internal transfer"],
    "finance": ["media sanitization", "storage sanitization", "decommissioned storage"],
}


def is_likely_restricted_query(query: str, role: str) -> bool:
    """Check if query targets a topic known to be restricted for the role."""
    q_lower = query.lower()
    keywords = _RESTRICTED_TOPIC_KEYWORDS.get(role, [])
    return any(k in q_lower for k in keywords)


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

    # Step B: Retrieval
    active_retriever = retriever or HybridRetriever(base_dir=base_dir)
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

    # Step C: Defense-in-depth check
    # Re-verify every retrieved chunk independently against canonical role and date constraints
    authorized_chunks: List[RetrievalResult] = []
    for res in retrieved_results:
        chunk_dict = res.to_dict()
        if is_chunk_eligible(chunk_dict, role, validated_date):
            authorized_chunks.append(res)

    corpus_fp = ""
    if hasattr(active_retriever, "corpus_manifest") and active_retriever.corpus_manifest:
        corpus_fp = active_retriever.corpus_manifest.get("processed_fingerprint", "")
    elif hasattr(active_retriever, "corpus_fingerprint"):
        corpus_fp = str(active_retriever.corpus_fingerprint)
    if not corpus_fp:
        base_path = Path(base_dir) if base_dir else Path.cwd()
        manifest_p = base_path / "data" / "processed" / "corpus_manifest.json"
        if not manifest_p.exists():
            manifest_p = base_path.parent / "data" / "processed" / "corpus_manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                corpus_fp = json.load(f).get("processed_fingerprint", "")

    ev_fingerprint = build_evidence_fingerprint(authorized_chunks, corpus_fp)

    # Step D: Deterministic rule resolution
    extracted_conds = extract_conditions_from_query(cleaned_question)
    rule_results: List[RuleEvaluationResult] = []
    applicable_rule_ids: Set[str] = set()

    if authorized_chunks:
        chunk_dicts = [c.to_dict() for c in authorized_chunks]
        rule_results = resolve_rules_for_evidence(
            authorized_chunks=chunk_dicts,
            role=role,
            query_date=validated_date,
            base_dir=base_dir,
            extracted_conditions=extracted_conds,
        )
        applicable_rule_ids = {r.rule.rule_id for r in rule_results}

    # Step E: Evidence sufficiency and safe refusal checks
    if not authorized_chunks:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        if is_likely_restricted_query(cleaned_question, role):
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
                evidence_fingerprint=ev_fingerprint,
                latency_ms=round(total_latency, 2),
            )
        else:
            return GenerationResult(
                question=cleaned_question,
                role=role,
                query_date=date_str,
                answer="The provided corporate policies do not contain sufficient evidence to answer this question.",
                status=AnswerStatus.INSUFFICIENT.value,
                reason="No matching policy clause found for query in authorized scope.",
                citations=[],
                citation_metadata=[],
                referenced_rule_ids=[],
                model=GENERATION_MODEL,
                corpus_fingerprint=corpus_fp,
                evidence_fingerprint=ev_fingerprint,
                latency_ms=round(total_latency, 2),
            )

    # Step F: Call Groq LLM
    active_llm = llm_client or GroqClientWrapper()
    try:
        model_resp, llm_meta = active_llm.generate_answer(
            question=cleaned_question,
            role=role,
            evidence_chunks=authorized_chunks,
            query_date=date_str,
            applicable_rule_facts=rule_results,
        )
    except Exception as exc:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="The generation service is temporarily unavailable. Please retry your query shortly.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason=f"LLM provider error: {exc}",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            error=str(exc),
        )

    # Step G: Validate LLM output and citations
    try:
        val_citations, val_metadata, val_rules = validate_citations_and_rules(
            response=model_resp,
            authorized_chunks=authorized_chunks,
            role=role,
            query_date=validated_date,
            applicable_rule_ids=applicable_rule_ids,
        )
    except ValueError as val_err:
        total_latency = (time.perf_counter() - start_total_time) * 1000.0
        return GenerationResult(
            question=cleaned_question,
            role=role,
            query_date=date_str,
            answer="Policy evidence retrieved, but the generated response could not be verified against authorized citations.",
            status=AnswerStatus.INSUFFICIENT.value,
            reason=f"Citation validation error: {val_err}",
            citations=[],
            citation_metadata=[],
            referenced_rule_ids=[],
            model=GENERATION_MODEL,
            corpus_fingerprint=corpus_fp,
            evidence_fingerprint=ev_fingerprint,
            latency_ms=round(total_latency, 2),
            error=str(val_err),
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
        print(f"Warning/Error: {result.error}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
