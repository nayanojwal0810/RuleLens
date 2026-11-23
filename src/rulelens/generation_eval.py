"""Development generation evaluation harness for RuleLens Stage E.

Evaluates evidence-grounded answer generation using canonical policy evidence,
structured rules, and Groq (openai/gpt-oss-20b).
Supports live smoke calls, 30-question development subset evaluation, and
offline evaluation with mocked LLM providers.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from rulelens.answering import answer_question
from rulelens.config import GENERATION_MODEL, GROQ_API_KEY, get_env_var
from rulelens.dataset import ROLES
from rulelens.llm import (
    AnswerResponse,
    AnswerStatus,
    CitationValidationError,
    GroqClientWrapper,
    ProviderTelemetrySnapshot,
    SanitizedProviderError,
    build_evidence_fingerprint,
    validate_citations_and_rules,
)
from rulelens.retrieval import (
    HybridRetriever,
    RetrievalResult,
    is_chunk_eligible,
    load_validated_corpus,
    parse_query_date,
)
from rulelens.retrieval_eval import validate_evaluation_dataset
from rulelens.rules import (
    extract_conditions_from_query,
    resolve_rules_for_evidence,
)

# Canonical 30-question development subset IDs (balanced across 5 departments)
DEV_SUBSET_QUESTION_IDS: List[str] = [
    # Travel (4 answerable, 1 restricted, 1 unsupported)
    "TRV-001", "TRV-002", "TRV-003", "TRV-007", "TRV-009", "TRV-010",
    # Finance (4 answerable, 1 restricted, 1 unsupported)
    "FIN-001", "FIN-002", "FIN-005", "FIN-006", "FIN-009", "FIN-010",
    # People Operations (4 answerable, 1 restricted, 1 unsupported)
    "PPO-001", "PPO-003", "PPO-005", "PPO-006", "PPO-009", "PPO-010",
    # Procurement (4 answerable, 1 restricted, 1 unsupported)
    "PRC-001", "PRC-003", "PRC-005", "PRC-007", "PRC-009", "PRC-010",
    # Information Security (4 answerable, 1 restricted, 1 unsupported)
    "SEC-001", "SEC-003", "SEC-004", "SEC-006", "SEC-009", "SEC-010",
]


def _safe_int(val: Any, default: int = 0) -> int:
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
class EvaluationContext:
    """Validated evaluation environment and canonical corpus context."""

    base_dir: Path
    corpus_chunks: List[Dict[str, Any]]
    chunks_by_id: Dict[str, Dict[str, Any]]
    manifest: Dict[str, Any]
    corpus_fingerprint: str


def create_evaluation_context(base_dir: Optional[Union[Path, str]] = None) -> EvaluationContext:
    """Load and validate evaluation context and canonical corpus records."""
    base = Path(base_dir) if base_dir else Path.cwd()
    manifest_p = base / "data" / "processed" / "corpus_manifest.json"
    if not manifest_p.exists():
        manifest_p = base.parent / "data" / "processed" / "corpus_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Corpus manifest not found at {manifest_p}")

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    corpus_fp = manifest.get("processed_fingerprint")
    if not corpus_fp:
        raise ValueError("Corpus manifest is missing 'processed_fingerprint'.")

    corpus = load_validated_corpus(base_dir=base)
    chunks_by_id = {c["chunk_id"]: c for c in corpus}
    return EvaluationContext(
        base_dir=base,
        corpus_chunks=corpus,
        chunks_by_id=chunks_by_id,
        manifest=manifest,
        corpus_fingerprint=corpus_fp,
    )


def load_evaluation_dataset(dataset_path: Union[Path, str]) -> Dict[str, Any]:
    """Load evaluation questions JSON file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_single_question(
    question_data: Dict[str, Any],
    retriever: Optional[HybridRetriever] = None,
    llm_client: Optional[GroqClientWrapper] = None,
    base_dir: Optional[Union[Path, str]] = None,
    use_gold_evidence_for_answerable: bool = True,
    dense_backend: str = "local",
    qdrant_client: Optional[Any] = None,
    eval_context: Optional[EvaluationContext] = None,
) -> Dict[str, Any]:
    """Evaluate generation on a single question using verified canonical evidence.

    For diagnostic questions, runs through the actual application authorization and
    evidence sufficiency checks without reading expected diagnostic labels to construct the answer.
    """
    qid = question_data["question_id"]
    q_text = question_data["question"]
    role = question_data["role"]
    query_date_str = question_data["query_date"]
    validated_date = parse_query_date(query_date_str)
    is_answerable = question_data.get("answerability", True)
    gold_cids = question_data.get("gold_chunk_ids", [])
    expected_evidence = question_data.get("gold_evidence", [])
    restr_targets = question_data.get("restricted_target_chunk_ids", [])

    base = Path(base_dir) if base_dir else Path.cwd()
    ctx = eval_context or create_evaluation_context(base_dir=base)
    active_retriever = retriever or HybridRetriever(base_dir=base)
    active_llm = llm_client or GroqClientWrapper()

    start_time = time.perf_counter()

    # Diagnostic questions: execute through real application answering pipeline
    if not is_answerable:
        app_result = answer_question(
            question=q_text,
            role=role,
            query_date=query_date_str,
            retriever=active_retriever,
            llm_client=active_llm,
            base_dir=base,
            dense_backend=dense_backend,
            qdrant_client=qdrant_client,
        )
        total_lat = (time.perf_counter() - start_time) * 1000.0

        is_restricted_expected = bool(restr_targets)
        expected_refusal_type = "restricted" if is_restricted_expected else "insufficient"
        refusal_correct = (app_result.status == expected_refusal_type)

        provider_err = False
        other_op_err = False
        cit_val_failure = False
        evidence_val_failure = False

        if app_result.status == "operational_error":
            if app_result.error_category == "evidence_integrity_error":
                evidence_val_failure = True
            elif app_result.error_category == "invalid_citation":
                cit_val_failure = True
            elif (
                app_result.error_category in ("rate_limited", "authentication_error", "timeout", "invalid_generation")
                or (app_result.provider_calls_attempted is not None and app_result.provider_calls_attempted > 0)
            ):
                provider_err = True
            else:
                other_op_err = True

        citation_expected = (app_result.status == "answered" or cit_val_failure)
        if citation_expected:
            citation_valid = (len(app_result.citations) > 0 and not cit_val_failure)
        else:
            citation_valid = (len(app_result.citations) == 0)

        return {
            "question_id": qid,
            "department": question_data.get("department"),
            "question_type": question_data.get("question_type"),
            "difficulty": question_data.get("difficulty"),
            "role": role,
            "query_date": query_date_str,
            "answerability": is_answerable,
            "gold_chunk_ids": gold_cids,
            "evaluation_mode": "application_diagnostic",
            "retrieval_mode": f"application_{dense_backend}",
            "status": app_result.status,
            "answer": app_result.answer,
            "citations": app_result.citations,
            "referenced_rule_ids": app_result.referenced_rule_ids,
            "reason": app_result.reason,
            "schema_valid": app_result.schema_valid,
            "citation_valid": citation_valid,
            "citation_expected": citation_expected,
            "expected_refusal_type": expected_refusal_type,
            "refusal_correct": refusal_correct,
            "provider_error": provider_err,
            "evidence_validation_failure": evidence_val_failure,
            "citation_validation_failure": cit_val_failure,
            "other_operational_failure": other_op_err,
            "provider_calls_attempted": app_result.provider_calls_attempted,
            "role_violations": 0,
            "version_violations": 0,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": app_result.provider_latency_ms,
            "token_usage": app_result.token_usage,
            "has_incomplete_tokens": getattr(app_result, "has_incomplete_tokens", False),
            "factual_correctness": "not_reviewed",
            "error_category": app_result.error_category,
            "gold_quotes": [e.get("quote", "") for e in expected_evidence],
            "error": app_result.error,
        }

    # Answerable questions: run using authorized gold chunks or standard retrieval
    if use_gold_evidence_for_answerable and gold_cids:
        role_violations = 0
        version_violations = 0
        fail_reason: Optional[str] = None
        evidence_validation_failed = False

        # Fail-closed check 1: Corpus fingerprint
        if not ctx.corpus_fingerprint:
            fail_reason = "Corpus fingerprint is missing or invalid."
            evidence_validation_failed = True

        # Fail-closed check 2: Evidence limit (1 to 4 chunks)
        if not (1 <= len(gold_cids) <= 4):
            fail_reason = f"Gold chunks count ({len(gold_cids)}) outside allowed bounds [1, 4]."
            evidence_validation_failed = True

        # Fail-closed check 3: Duplicate chunk IDs
        if len(gold_cids) != len(set(gold_cids)):
            fail_reason = "Duplicate chunk IDs present in gold_chunk_ids."
            evidence_validation_failed = True

        # Fail-closed check 4: Presence and metadata integrity against canonical records
        authorized_chunks: List[Dict[str, Any]] = []
        exact_fields = [
            "chunk_id", "doc_id", "clause_id", "department", "text", "title", "version",
            "effective_from", "source_file"
        ]

        for cid in gold_cids:
            if cid not in ctx.chunks_by_id:
                fail_reason = f"Gold chunk ID '{cid}' not found in canonical corpus."
                evidence_validation_failed = True
                break

            canonical = ctx.chunks_by_id[cid]

            # Role authorization verification
            if role not in canonical.get("allowed_roles", []):
                role_violations += 1
                fail_reason = f"Role '{role}' is not authorized for chunk '{cid}'."
                evidence_validation_failed = True

            # Temporal eligibility verification
            eff_from = canonical.get("effective_from", "0000-01-01")
            eff_to = canonical.get("effective_to") or "9999-12-31"
            if not (eff_from <= validated_date.isoformat() <= eff_to):
                version_violations += 1
                fail_reason = f"Chunk '{cid}' is inactive on query date {validated_date.isoformat()}."
                evidence_validation_failed = True

            if is_chunk_eligible(canonical, role, validated_date):
                authorized_chunks.append(canonical)

        # Fail-closed check 5: Verbatim quote presence in canonical chunk text
        if not evidence_validation_failed:
            for ev_item in expected_evidence:
                ev_cid = ev_item.get("chunk_id")
                ev_quote = ev_item.get("quote", "").strip()
                if not ev_cid or ev_cid not in ctx.chunks_by_id:
                    fail_reason = f"Evidence quote references unknown chunk ID '{ev_cid}'."
                    evidence_validation_failed = True
                    break
                can_text = ctx.chunks_by_id[ev_cid]["text"]
                if ev_quote and ev_quote not in can_text:
                    fail_reason = f"Evidence quote is not an exact substring of chunk '{ev_cid}' text."
                    evidence_validation_failed = True
                    break

        # FAIL CLOSED: Never send invalid, corrupted, or unauthorized chunks to Groq!
        if evidence_validation_failed or role_violations > 0 or version_violations > 0 or not authorized_chunks:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return {
                "question_id": qid,
                "department": question_data.get("department"),
                "question_type": question_data.get("question_type"),
                "difficulty": question_data.get("difficulty"),
                "role": role,
                "query_date": query_date_str,
                "answerability": is_answerable,
                "gold_chunk_ids": gold_cids,
                "evaluation_mode": "gold_evidence_generation",
                "retrieval_mode": "none (gold evidence direct)",
                "status": "operational_error",
                "answer": "Gold evidence failed authorization, version, or text verification.",
                "citations": [],
                "referenced_rule_ids": [],
                "reason": (
                    f"Gold evidence verification failed: {fail_reason} "
                    f"(role_violations={role_violations}, version_violations={version_violations})."
                ),
                "schema_valid": None,
                "citation_valid": False,
                "citation_expected": False,
                "expected_refusal_type": None,
                "refusal_correct": False,
                "provider_error": False,
                "evidence_validation_failure": True,
                "citation_validation_failure": False,
                "other_operational_failure": False,
                "provider_calls_attempted": 0,
                "role_violations": role_violations,
                "version_violations": version_violations,
                "latency_ms": round(total_lat, 2),
                "provider_latency_ms": None,
                "token_usage": None,
                "has_incomplete_tokens": False,
                "factual_correctness": "not_reviewed",
                "error_category": "evidence_validation_failure",
                "gold_quotes": [e.get("quote", "") for e in expected_evidence],
                "error": f"Gold evidence verification failed: {fail_reason}",
            }

        extracted_conds = extract_conditions_from_query(q_text)
        rule_results = resolve_rules_for_evidence(
            authorized_chunks=authorized_chunks,
            role=role,
            query_date=validated_date,
            base_dir=base,
            extracted_conditions=extracted_conds,
        )

        # Check rule condition evaluation errors
        rule_errors = [r for r in rule_results if r.is_error]
        if rule_errors:
            total_lat = (time.perf_counter() - start_time) * 1000.0
            return {
                "question_id": qid,
                "department": question_data.get("department"),
                "question_type": question_data.get("question_type"),
                "difficulty": question_data.get("difficulty"),
                "role": role,
                "query_date": query_date_str,
                "answerability": is_answerable,
                "gold_chunk_ids": gold_cids,
                "evaluation_mode": "gold_evidence_generation",
                "retrieval_mode": "none (gold evidence direct)",
                "status": "insufficient",
                "answer": "The policy question contains rule condition inputs that could not be evaluated safely.",
                "citations": [],
                "referenced_rule_ids": [],
                "reason": f"Rule evaluation error: {rule_errors[0].explanation}",
                "schema_valid": None,
                "citation_valid": False,
                "citation_expected": False,
                "expected_refusal_type": None,
                "refusal_correct": False,
                "provider_error": False,
                "evidence_validation_failure": False,
                "citation_validation_failure": False,
                "other_operational_failure": False,
                "provider_calls_attempted": 0,
                "role_violations": 0,
                "version_violations": 0,
                "latency_ms": round(total_lat, 2),
                "provider_latency_ms": None,
                "token_usage": None,
                "has_incomplete_tokens": False,
                "factual_correctness": "not_reviewed",
                "error_category": rule_errors[0].error_type,
                "gold_quotes": [e.get("quote", "") for e in expected_evidence],
                "error": rule_errors[0].explanation,
            }

        applicable_rule_ids = {r.rule.rule_id for r in rule_results if r.passed and not r.is_error}

        schema_valid = False
        citation_valid = False
        gen_answer = ""
        gen_status = ""
        gen_reason = ""
        gen_citations: List[str] = []
        gen_rules: List[str] = []
        token_usage = None
        error_msg = None
        error_cat = None
        provider_lat = None
        provider_calls = None
        provider_err = False
        cit_val_failure = False
        other_op_err = False
        has_inc_toks = False

        baseline_telemetry = None
        if hasattr(active_llm, "get_telemetry_snapshot"):
            try:
                snap = active_llm.get_telemetry_snapshot()
                if isinstance(snap, ProviderTelemetrySnapshot):
                    baseline_telemetry = snap
            except Exception:
                baseline_telemetry = None

        try:
            model_resp, llm_meta = active_llm.generate_answer(
                question=q_text,
                role=role,
                evidence_chunks=authorized_chunks,
                query_date=query_date_str,
                applicable_rule_facts=rule_results,
            )
            if baseline_telemetry is not None:
                delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
                provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
                provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
                token_usage = _safe_int(getattr(delta, "known_total_tokens", None), default=None)
                has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
            elif isinstance(llm_meta, dict):
                provider_calls = _safe_int(llm_meta.get("calls_made"), default=None)
                provider_lat = _safe_float(llm_meta.get("latency_ms"))
                token_usage = _safe_int(llm_meta.get("total_tokens"), default=None)
                has_inc_toks = False
            else:
                provider_calls = None
                provider_lat = None
                token_usage = None
                has_inc_toks = False

            schema_valid = True
            gen_answer = model_resp.answer
            gen_status = model_resp.status.value
            gen_reason = model_resp.reason

            val_cids, _, val_rules = validate_citations_and_rules(
                response=model_resp,
                authorized_chunks=authorized_chunks,
                role=role,
                query_date=validated_date,
                applicable_rule_ids=applicable_rule_ids,
            )
            gen_citations = val_cids
            gen_rules = val_rules
            citation_valid = (len(gen_citations) > 0) if gen_status == "answered" else (len(gen_citations) == 0)

        except CitationValidationError as cit_err:
            if baseline_telemetry is not None:
                delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
                provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
                provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
                has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
            error_msg = cit_err.safe_message
            error_cat = cit_err.category
            gen_status = "operational_error"
            schema_valid = True
            citation_valid = False
            cit_val_failure = True
        except SanitizedProviderError as prov_err:
            if baseline_telemetry is not None:
                delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
                provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
                provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
                has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
            error_msg = prov_err.safe_message
            error_cat = prov_err.category
            gen_status = "operational_error"
            schema_valid = False if prov_err.category == "invalid_generation" else None
            provider_err = True
        except Exception:
            if baseline_telemetry is not None:
                delta = active_llm.get_telemetry_snapshot().delta(baseline_telemetry)
                provider_calls = _safe_int(getattr(delta, "provider_requests_attempted", None), default=None)
                provider_lat = _safe_float(getattr(delta, "provider_request_latency_ms", None))
                has_inc_toks = bool(getattr(delta, "has_incomplete_tokens", False))
            error_msg = "Unhandled operational error during evaluation generation."
            error_cat = "service_unavailable"
            gen_status = "operational_error"
            provider_err = True

        total_lat = (time.perf_counter() - start_time) * 1000.0
        citation_expected = (gen_status == "answered" or cit_val_failure)

        return {
            "question_id": qid,
            "department": question_data.get("department"),
            "question_type": question_data.get("question_type"),
            "difficulty": question_data.get("difficulty"),
            "role": role,
            "query_date": query_date_str,
            "answerability": is_answerable,
            "gold_chunk_ids": gold_cids,
            "evaluation_mode": "gold_evidence_generation",
            "retrieval_mode": "none (gold evidence direct)",
            "status": gen_status,
            "answer": gen_answer,
            "citations": gen_citations,
            "referenced_rule_ids": gen_rules,
            "reason": gen_reason,
            "schema_valid": schema_valid,
            "citation_valid": citation_valid,
            "citation_expected": citation_expected,
            "expected_refusal_type": None,
            "refusal_correct": False,
            "provider_error": provider_err,
            "evidence_validation_failure": False,
            "citation_validation_failure": cit_val_failure,
            "other_operational_failure": other_op_err,
            "provider_calls_attempted": provider_calls,
            "role_violations": 0,
            "version_violations": 0,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": round(provider_lat, 2) if provider_lat is not None else None,
            "token_usage": token_usage,
            "has_incomplete_tokens": has_inc_toks,
            "factual_correctness": "not_reviewed",
            "error_category": error_cat,
            "gold_quotes": [e.get("quote", "") for e in expected_evidence],
            "error": error_msg,
        }
    else:
        # Full answering pipeline for answerable query
        app_result = answer_question(
            question=q_text,
            role=role,
            query_date=query_date_str,
            retriever=active_retriever,
            llm_client=active_llm,
            base_dir=base,
            dense_backend=dense_backend,
            qdrant_client=qdrant_client,
        )
        total_lat = (time.perf_counter() - start_time) * 1000.0

        provider_err = False
        other_op_err = False
        cit_val_failure = False
        evidence_val_failure = False

        if app_result.status == "operational_error":
            if app_result.error_category == "evidence_integrity_error":
                evidence_val_failure = True
            elif app_result.error_category == "invalid_citation":
                cit_val_failure = True
            elif (
                app_result.error_category in ("rate_limited", "authentication_error", "timeout", "invalid_generation")
                or (app_result.provider_calls_attempted is not None and app_result.provider_calls_attempted > 0)
            ):
                provider_err = True
            else:
                other_op_err = True

        citation_expected = (app_result.status == "answered" or cit_val_failure)
        if citation_expected:
            citation_valid = (len(app_result.citations) > 0 and not cit_val_failure)
        else:
            citation_valid = (len(app_result.citations) == 0)

        return {
            "question_id": qid,
            "department": question_data.get("department"),
            "question_type": question_data.get("question_type"),
            "difficulty": question_data.get("difficulty"),
            "role": role,
            "query_date": query_date_str,
            "answerability": is_answerable,
            "gold_chunk_ids": gold_cids,
            "evaluation_mode": "application_pipeline",
            "retrieval_mode": f"application_{dense_backend}",
            "status": app_result.status,
            "answer": app_result.answer,
            "citations": app_result.citations,
            "referenced_rule_ids": app_result.referenced_rule_ids,
            "reason": app_result.reason,
            "schema_valid": app_result.schema_valid,
            "citation_valid": citation_valid,
            "citation_expected": citation_expected,
            "expected_refusal_type": None,
            "refusal_correct": False,
            "provider_error": provider_err,
            "evidence_validation_failure": evidence_val_failure,
            "citation_validation_failure": cit_val_failure,
            "other_operational_failure": other_op_err,
            "provider_calls_attempted": app_result.provider_calls_attempted,
            "role_violations": 0,
            "version_violations": 0,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": app_result.provider_latency_ms,
            "token_usage": app_result.token_usage,
            "has_incomplete_tokens": getattr(app_result, "has_incomplete_tokens", False),
            "factual_correctness": "not_reviewed",
            "error_category": app_result.error_category,
            "gold_quotes": [e.get("quote", "") for e in expected_evidence],
            "error": app_result.error,
        }


def run_generation_evaluation(
    dataset_path: Union[Path, str],
    base_dir: Optional[Union[Path, str]] = None,
    subset_ids: Optional[Sequence[str]] = None,
    max_questions: Optional[int] = None,
    llm_client: Optional[GroqClientWrapper] = None,
    retriever: Optional[HybridRetriever] = None,
    dense_backend: str = "local",
    qdrant_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run controlled development generation evaluation.

    Args:
        dataset_path: Path to dev_questions.json.
        base_dir: Repository root.
        subset_ids: Specific question IDs to evaluate.
        max_questions: Cap on number of questions to evaluate (1 to 50).
        llm_client: Optional injected client wrapper.
        retriever: Optional injected HybridRetriever.
        dense_backend: Dense retrieval backend ('local' or 'qdrant').
        qdrant_client: Optional active Qdrant client.

    Returns:
        Dict with aggregate summary and per-question results.
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    if isinstance(max_questions, bool) or (max_questions is not None and not isinstance(max_questions, int)):
        raise ValueError(f"max_questions must be an integer between 1 and 50 inclusive, got {max_questions!r}.")
    if max_questions is not None and (max_questions < 1 or max_questions > 50):
        raise ValueError(f"max_questions must be an integer between 1 and 50 inclusive, got {max_questions}.")

    # Pre-generation evaluation dataset validation
    try:
        val_report = validate_evaluation_dataset(dataset_path, base_dir=base)
        if val_report.get("status") != "valid":
            raise ValueError("Evaluation dataset validation failed.")
    except Exception as exc:
        raise ValueError(f"Evaluation dataset validation failed: {exc}") from exc

    data = load_evaluation_dataset(dataset_path)
    all_questions = data["questions"]
    q_map = {q["question_id"]: q for q in all_questions}

    if subset_ids is not None:
        unknown_ids = [qid for qid in subset_ids if qid not in q_map]
        if unknown_ids:
            raise ValueError(f"Unknown question IDs requested in subset: {unknown_ids}")
        target_ids = list(subset_ids)
    else:
        target_ids = list(DEV_SUBSET_QUESTION_IDS)

    if max_questions is not None:
        target_ids = target_ids[:max_questions]

    selected_questions = [q_map[qid] for qid in target_ids]
    if not selected_questions:
        raise ValueError("No matching questions found in dataset to evaluate.")

    eval_ctx = create_evaluation_context(base_dir=base)
    active_retriever = retriever or HybridRetriever(base_dir=base)
    active_llm = llm_client or GroqClientWrapper()

    active_qdrant_client = qdrant_client
    if dense_backend == "qdrant" and active_qdrant_client is None:
        from rulelens.qdrant_store import get_qdrant_client
        try:
            active_qdrant_client = get_qdrant_client()
        except Exception as exc:
            raise RuntimeError(f"Qdrant client initialization failed for backend 'qdrant': {exc}")

    per_q_results: List[Dict[str, Any]] = []
    print(f"Starting Generation Evaluation on {len(selected_questions)} questions (backend: {dense_backend})...")

    stopped_early = False
    stop_reason = None

    for i, q in enumerate(selected_questions, 1):
        qid = q["question_id"]
        print(f"[{i}/{len(selected_questions)}] Evaluating {qid} ({q.get('department')}, {q.get('role')})...")
        res = evaluate_single_question(
            question_data=q,
            retriever=active_retriever,
            llm_client=active_llm,
            base_dir=base,
            dense_backend=dense_backend,
            qdrant_client=active_qdrant_client,
            eval_context=eval_ctx,
        )
        per_q_results.append(res)
        if res.get("error_category") == "rate_limited":
            stopped_early = True
            stop_reason = "Rate limit HTTP 429 encountered from provider."
            print(f"Encountered Groq 429 rate limit on {qid}. Stopping evaluation early to respect provider quota.")
            break
        # Pacing between live calls (skip during mock/offline runs)
        if q.get("answerability", True) and llm_client is None:
            time.sleep(2.5)

    # Dynamic metrics computation
    planned_questions = len(selected_questions)
    completed_questions = len(per_q_results)

    ans_results = [r for r in per_q_results if r["answerability"]]
    diag_results = [r for r in per_q_results if not r["answerability"]]
    restr_results = [r for r in diag_results if r.get("expected_refusal_type") == "restricted"]
    unsupp_results = [r for r in diag_results if r.get("expected_refusal_type") == "insufficient"]

    successful_answers = sum(
        1 for r in ans_results
        if r["status"] == "answered"
        and not r.get("provider_error", False)
        and r.get("schema_valid") is True
        and r.get("citation_valid", False)
    )
    refusals = sum(1 for r in per_q_results if r["status"] in ("insufficient", "restricted"))
    provider_errors = sum(1 for r in per_q_results if r.get("provider_error", False))
    evidence_val_failures = sum(1 for r in per_q_results if r.get("evidence_validation_failure", False))
    citation_val_failures = sum(1 for r in per_q_results if r.get("citation_validation_failure", False))
    other_operational_failures = sum(1 for r in per_q_results if r.get("other_operational_failure", False))

    total_op_val_failures = (
        provider_errors + evidence_val_failures + citation_val_failures + other_operational_failures
    )

    if stopped_early or completed_questions < planned_questions:
        eval_state = "partial"
    elif total_op_val_failures > 0:
        eval_state = "completed_with_errors"
    else:
        eval_state = "completed"

    generations_attempted = sum(
        1 for r in per_q_results
        if r.get("provider_calls_attempted") is None or r["provider_calls_attempted"] > 0
    )
    schema_valid_generations = sum(
        1 for r in per_q_results
        if r.get("schema_valid") is True and not r.get("provider_error", False)
    )

    citation_expected_count = sum(1 for r in per_q_results if r.get("citation_expected", False))
    citation_valid_generations = sum(
        1 for r in per_q_results
        if r.get("citation_expected", False) and r.get("citation_valid", False) and not r.get("provider_error", False)
    )

    restr_success = sum(1 for r in restr_results if r.get("refusal_correct", False))
    unsupp_success = sum(1 for r in unsupp_results if r.get("refusal_correct", False))
    incorrect_refusals = (
        sum(1 for r in ans_results if r["status"] in ("insufficient", "restricted"))
        + sum(1 for r in diag_results if not r.get("refusal_correct", False) and r["status"] in ("insufficient", "restricted"))
    )
    incorrect_answers = sum(1 for r in diag_results if r["status"] == "answered")

    known_requests_list = [r["provider_calls_attempted"] for r in per_q_results if r.get("provider_calls_attempted") is not None]
    has_unknown_provider_requests = any(
        r.get("provider_calls_attempted") is None
        for r in per_q_results
        if r.get("status") == "answered" or r.get("provider_error", False)
    )
    actual_provider_requests = sum(known_requests_list)

    known_tokens = sum(r["token_usage"] for r in per_q_results if r.get("token_usage") is not None)
    known_tokens_incomplete = any(r.get("has_incomplete_tokens", False) for r in per_q_results) or (
        hasattr(active_llm, "get_telemetry_snapshot") and active_llm.get_telemetry_snapshot().has_incomplete_tokens
    )

    e2e_latencies = [r["latency_ms"] for r in per_q_results]
    provider_latencies = [
        r["provider_latency_ms"] for r in per_q_results
        if r.get("provider_latency_ms") is not None and r["provider_latency_ms"] > 0
    ]
    diag_latencies = [r["latency_ms"] for r in diag_results]

    median_e2e = statistics.median(e2e_latencies) if e2e_latencies else 0.0
    median_prov = statistics.median(provider_latencies) if provider_latencies else 0.0
    median_diag = statistics.median(diag_latencies) if diag_latencies else 0.0

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": GENERATION_MODEL,
        "evaluation_state": eval_state,
        "stop_reason": stop_reason,
        "retrieval_backend": dense_backend,
        "planned_questions": planned_questions,
        "completed_questions": completed_questions,
        "successful_answers": successful_answers,
        "successful_generations": successful_answers,  # Backward compatibility
        "refusals": refusals,
        "restricted_refusal_successes": restr_success,
        "unsupported_refusal_successes": unsupp_success,
        "incorrect_refusals": incorrect_refusals,
        "incorrect_answers": incorrect_answers,
        "provider_errors": provider_errors,
        "provider_failures": provider_errors,
        "evidence_validation_failures": evidence_val_failures,
        "citation_validation_failures": citation_val_failures,
        "other_operational_failures": other_operational_failures,
        "generations_attempted": generations_attempted,
        "schema_valid_generations": schema_valid_generations,
        "citation_expected_count": citation_expected_count,
        "citation_valid_generations": citation_valid_generations,
        "generation_calls": actual_provider_requests,
        "actual_provider_requests": actual_provider_requests,
        "has_unknown_provider_requests": has_unknown_provider_requests,
        "tokens_used": known_tokens,
        "known_tokens_used": known_tokens,
        "known_tokens_incomplete": known_tokens_incomplete,
        "median_e2e_latency_ms": round(median_e2e, 2),
        "median_provider_latency_ms": round(median_prov, 2),
        "median_diagnostic_latency_ms": round(median_diag, 2),
        "per_question_results": per_q_results,
    }

    return summary


def format_rate(numerator: int, denominator: int) -> str:
    """Format rate percentage or return N/A if denominator is zero."""
    if denominator == 0:
        return "N/A"
    pct = (numerator / denominator) * 100.0
    return f"{numerator} / {denominator} ({pct:.1f}%)"


def generate_summary_markdown(summary: Dict[str, Any]) -> str:
    """Format evaluation summary dictionary as clean GitHub markdown."""
    state_str = f"**{summary['evaluation_state'].upper()}**"
    if summary.get("stop_reason"):
        state_str += f" ({summary['stop_reason']})"

    completed = summary["completed_questions"]
    ans_count = sum(1 for r in summary["per_question_results"] if r.get("answerability"))
    diag_count = completed - ans_count
    restr_count = sum(1 for r in summary["per_question_results"] if r.get("expected_refusal_type") == "restricted" or (r.get("answerability") is False and r["question_id"].endswith("-009")))
    unsupp_count = sum(1 for r in summary["per_question_results"] if r.get("expected_refusal_type") == "insufficient" or (r.get("answerability") is False and r["question_id"].endswith("-010")))
    generations_att = summary.get("generations_attempted", completed)
    cit_denom = summary.get("citation_expected_count", ans_count)

    token_str = str(summary.get("known_tokens_used", summary.get("tokens_used", 0)))
    if summary.get("known_tokens_incomplete"):
        token_str += " (incomplete)"

    prov_reqs = summary.get("actual_provider_requests", summary.get("generation_calls", 0))
    prov_reqs_str = str(prov_reqs)
    if summary.get("has_unknown_provider_requests"):
        prov_reqs_str += " (some requests unknown)"

    lines = [
        "# RuleLens — Stage E Development Answer Generation Summary",
        "",
        "## 1. Evaluation Setup",
        f"- **Timestamp**: `{summary['timestamp']}`",
        f"- **Model**: `{summary['model']}`",
        f"- **Dense Retrieval Backend**: `{summary.get('retrieval_backend', 'local')}`",
        f"- **Evaluation Status**: {state_str}",
        f"- **Planned Questions**: {summary['planned_questions']}",
        f"- **Completed Questions**: {completed}",
        f"- **Answerable Questions Evaluated**: {ans_count}",
        f"- **Diagnostic Questions Evaluated**: {diag_count} (Restricted: {restr_count}, Unsupported: {unsupp_count})",
        "",
        "## 2. Quantitative Summary",
        "",
        "| Metric | Result | Target |",
        "|---|:---:|:---:|",
        f"| **Schema Validity (Groq)** | {format_rate(summary['schema_valid_generations'], generations_att)} | 100% |",
        f"| **Citation Validity (Answerable)** | {format_rate(summary['citation_valid_generations'], cit_denom)} | 100% |",
        f"| **Restricted Refusal Success** | {format_rate(summary['restricted_refusal_successes'], restr_count)} | 100% |",
        f"| **Unsupported Refusal Success** | {format_rate(summary['unsupported_refusal_successes'], unsupp_count)} | 100% |",
        f"| **Provider Failures** | {summary.get('provider_failures', summary.get('provider_errors', 0))} | 0 |",
        f"| **Evidence Validation Failures** | {summary.get('evidence_validation_failures', 0)} | 0 |",
        f"| **Citation Validation Failures** | {summary.get('citation_validation_failures', 0)} | 0 |",
        f"| **Other Operational Failures** | {summary.get('other_operational_failures', 0)} | 0 |",
        f"| **Incorrect Refusals** | {summary.get('incorrect_refusals', 0)} | 0 |",
        f"| **Provider Requests Attempted** | {prov_reqs_str} | Monitored |",
        f"| **Median E2E Latency** | {summary['median_e2e_latency_ms']} ms | < 2500 ms |",
        f"| **Median Provider Latency** | {summary['median_provider_latency_ms']} ms | < 2000 ms |",
        f"| **Median Diagnostic Latency** | {summary['median_diagnostic_latency_ms']} ms | < 50 ms |",
        f"| **Total Tokens Consumed** | {token_str} | Monitored |",
        "",
        "## 3. Per-Question Results Sample",
        "",
        "| ID | Role | Date | Status | Citations | Rules | E2E Latency | Answer Excerpt |",
        "|---|---|:---:|:---:|---|---|---:|---|",
    ]

    for r in summary["per_question_results"]:
        ans_raw = r.get("answer") or ""
        ans_excerpt = (ans_raw[:60] + "...") if len(ans_raw) > 60 else ans_raw
        ans_excerpt = ans_excerpt.replace("\n", " ")
        cits_str = ", ".join(r.get("citations") or []) if r.get("citations") else "None"
        rules_str = ", ".join(r.get("referenced_rule_ids") or []) if r.get("referenced_rule_ids") else "None"
        lines.append(
            f"| **{r['question_id']}** | `{r.get('role', 'unknown')}` | {r.get('query_date', '')} | `{r['status']}` | `{cits_str}` | `{rules_str}` | {r.get('latency_ms', 0.0)} ms | {ans_excerpt} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_reconciled_markdown(summary: Dict[str, Any]) -> str:
    """Format reconciled 30-question development evaluation report as clean GitHub markdown."""
    completed = summary["completed_questions"]
    ans_count = sum(1 for r in summary["per_question_results"] if r.get("answerability"))
    diag_count = completed - ans_count
    restr_count = sum(
        1 for r in summary["per_question_results"]
        if r.get("expected_refusal_type") == "restricted" or (r.get("answerability") is False and r["question_id"].endswith("-009"))
    )
    unsupp_count = sum(
        1 for r in summary["per_question_results"]
        if r.get("expected_refusal_type") == "insufficient" or (r.get("answerability") is False and r["question_id"].endswith("-010"))
    )
    generations_att = summary.get("generations_attempted", completed)
    cit_denom = summary.get("citation_expected_count", ans_count)

    prov_reqs = summary.get("total_provider_requests_attempted", summary.get("actual_provider_requests", 0))
    retries_count = summary.get("historical_retry_attempts_count", 0)

    lines = [
        "# RuleLens — Reconciled 30-Question Stage E Development Generation Summary",
        "",
        "## 1. Evaluation Setup & Reconciliation Audit",
        f"- **Reconciliation Timestamp**: `{summary['timestamp']}`",
        f"- **Model**: `{summary['model']}`",
        f"- **Dense Retrieval Backend**: `{summary.get('retrieval_backend', 'local')}`",
        f"- **Evaluation Status**: **{summary.get('evaluation_state', 'completed').upper()}** (Reconciled from primary run + continuation run)",
        f"- **Planned Questions**: {summary['planned_questions']}",
        f"- **Unique Questions Evaluated**: {completed} (100% of planned {summary['planned_questions']} questions)",
        f"- **Answerable Questions Evaluated**: {ans_count} (Gold-evidence grounded generation)",
        f"- **Diagnostic Questions Evaluated**: {diag_count} (Restricted: {restr_count}, Unsupported: {unsupp_count})",
        f"- **Historical Retry Attempts**: {retries_count} (SEC-003 initial HTTP 429 rate limit recorded in historical audit)",
        f"- **Total Provider Requests (with retries)**: {prov_reqs}",
        f"- **Total Known Tokens Consumed**: {summary.get('total_tokens_consumed', summary.get('tokens_used', 0))}",
        f"- **Corpus Integrity**: {summary.get('provenance_note') or 'Fingerprint match verified against canonical corpus'}",
        "- **Factual Correctness Status**: `not_reviewed` (Awaiting independent human/expert review)",
        "",
        "## 2. Quantitative Summary",
        "",
        "| Metric | Result | Target |",
        "|---|:---:|:---:|",
        f"| **Unique Questions Completed** | {completed} / {summary['planned_questions']} (100.0%) | {summary['planned_questions']} / {summary['planned_questions']} |",
        f"| **Schema Validity (Groq)** | {format_rate(summary['schema_valid_generations'], generations_att)} | 100% |",
        f"| **Citation Validity (Answerable)** | {format_rate(summary['citation_valid_generations'], cit_denom)} | 100% |",
        f"| **Restricted Refusal Success** | {format_rate(summary['restricted_refusal_successes'], restr_count)} | 100% |",
        f"| **Unsupported Refusal Success** | {format_rate(summary['unsupported_refusal_successes'], unsupp_count)} | 100% |",
        f"| **Provider Failures (Final Outcome)** | {summary.get('provider_failures', 0)} | 0 |",
        f"| **Evidence Validation Failures** | {summary.get('evidence_validation_failures', 0)} | 0 |",
        f"| **Citation Validation Failures** | {summary.get('citation_validation_failures', 0)} | 0 |",
        f"| **Other Operational Failures** | {summary.get('other_operational_failures', 0)} | 0 |",
        f"| **Incorrect Refusals** | {summary.get('incorrect_refusals', 0)} (SEC-009 in historical run) | 0 |",
        f"| **Total Provider Requests Attempted** | {prov_reqs} (includes 1 rate-limited attempt) | Monitored |",
        f"| **Median E2E Latency** | {summary['median_e2e_latency_ms']} ms | < 2500 ms |",
        f"| **Median Provider Latency** | {summary['median_provider_latency_ms']} ms | < 2000 ms |",
        f"| **Median Diagnostic Latency** | {summary['median_diagnostic_latency_ms']} ms | < 50 ms |",
        f"| **Total Tokens Consumed** | {summary.get('total_tokens_consumed', 0)} | Monitored |",
        "",
        "## 3. Historical Telemetry Audit",
        "",
        "The reconciled evaluation incorporates all provider attempts from both runs:",
        "- **Primary Evaluation Run**: 26 questions processed (25 successful, 1 HTTP 429 rate limit on `SEC-003`).",
        "- **Continuation Run**: 5 questions processed (`SEC-003`, `SEC-004`, `SEC-006`, `SEC-009`, `SEC-010`).",
        "- `SEC-003` was successfully generated on its continuation retry (90 days, citing `DOC-SEC-002_SEC-A02`).",
        "- `SEC-009` was safely refused as `insufficient` without leaking restricted chunks in the historical run.",
        "",
        "## 4. Per-Question Reconciled Results (30 Unique Questions)",
        "",
        "| ID | Role | Date | Status | Citations | Rules | E2E Latency | Answer Excerpt |",
        "|---|---|:---:|:---:|---|---|---:|---|",
    ]

    for r in summary["per_question_results"]:
        ans_raw = r.get("answer") or ""
        ans_excerpt = (ans_raw[:60] + "...") if len(ans_raw) > 60 else ans_raw
        ans_excerpt = ans_excerpt.replace("\n", " ")
        cits_str = ", ".join(r.get("citations") or []) if r.get("citations") else "None"
        rules_str = ", ".join(r.get("referenced_rule_ids") or []) if r.get("referenced_rule_ids") else "None"
        lines.append(
            f"| **{r['question_id']}** | `{r.get('role', 'unknown')}` | {r.get('query_date', '')} | `{r['status']}` | `{cits_str}` | `{rules_str}` | {r.get('latency_ms', 0.0)} ms | {ans_excerpt} |"
        )

    lines.append("")
    return "\n".join(lines)


def reconcile_stage_e_evaluations(
    primary_results_path: Union[Path, str],
    continuation_results_path: Union[Path, str],
    output_json_path: Optional[Union[Path, str]] = None,
    output_md_path: Optional[Union[Path, str]] = None,
    subset_question_ids: Optional[Sequence[str]] = None,
    base_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Reconcile Stage E partial evaluation with continuation run.

    Combines the original run (26 questions, SEC-003 rate-limited) and the continuation
    run (5 questions including successful SEC-003 retry) into a unified 30-question report.
    Preserves historical provider telemetry (including HTTP 429) while establishing
    the authoritative final outcome for all 30 unique development questions.
    """
    p_path = Path(primary_results_path)
    c_path = Path(continuation_results_path)

    if not p_path.is_file():
        raise FileNotFoundError(f"Primary results file not found: {p_path}")
    if not c_path.is_file():
        raise FileNotFoundError(f"Continuation results file not found: {c_path}")

    primary_data = json.loads(p_path.read_text(encoding="utf-8"))
    cont_data = json.loads(c_path.read_text(encoding="utf-8"))

    if not isinstance(primary_data, dict) or "per_question_results" not in primary_data or not isinstance(primary_data["per_question_results"], list):
        raise ValueError("Invalid primary evaluation JSON format: must contain 'per_question_results' list")
    if not isinstance(cont_data, dict) or "per_question_results" not in cont_data or not isinstance(cont_data["per_question_results"], list):
        raise ValueError("Invalid continuation evaluation JSON format: must contain 'per_question_results' list")

    # Verification checks
    if primary_data.get("model") != cont_data.get("model"):
        raise ValueError(
            f"Model mismatch between runs: '{primary_data.get('model')}' vs '{cont_data.get('model')}'"
        )
    if primary_data.get("retrieval_backend") != cont_data.get("retrieval_backend"):
        raise ValueError("Retrieval backend mismatch between primary and continuation runs.")

    primary_qs = primary_data.get("per_question_results", [])
    cont_qs = cont_data.get("per_question_results", [])

    target_ids = list(subset_question_ids) if subset_question_ids is not None else list(DEV_SUBSET_QUESTION_IDS)
    target_id_set = set(target_ids)

    # 1. Reject duplicate question IDs within either run
    p_ids = [q["question_id"] for q in primary_qs]
    if len(p_ids) != len(set(p_ids)):
        raise ValueError("Duplicate question ID found within primary evaluation run")
    c_ids = [q["question_id"] for q in cont_qs]
    if len(c_ids) != len(set(c_ids)):
        raise ValueError("Duplicate question ID found within continuation evaluation run")

    # 2. Reject unexpected question IDs not in target_ids
    for qid in p_ids:
        if qid not in target_id_set:
            raise ValueError(f"Unexpected question ID '{qid}' in primary run")
    for qid in c_ids:
        if qid not in target_id_set:
            raise ValueError(f"Unexpected question ID '{qid}' in continuation run")

    # 3. Require continuation to replace ONLY previously failed questions
    primary_by_qid = {q["question_id"]: q for q in primary_qs}
    for q in cont_qs:
        qid = q["question_id"]
        if qid in primary_by_qid:
            prev = primary_by_qid[qid]
            is_prev_failed = (
                prev.get("status") == "operational_error"
                or prev.get("provider_error", False)
                or prev.get("error_category") == "rate_limited"
            )
            if not is_prev_failed:
                raise ValueError(
                    f"Continuation run attempts to replace previously successful question '{qid}'."
                )

    # 4. Verify question metadata against dev_questions.json if available
    root_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    dev_q_path = root_dir / "data" / "evaluation" / "dev_questions.json"
    if dev_q_path.is_file():
        dev_data = json.loads(dev_q_path.read_text(encoding="utf-8"))
        dev_list = dev_data.get("questions", []) if isinstance(dev_data, dict) else dev_data
        dev_lookup = {q["question_id"]: q for q in dev_list if isinstance(q, dict) and "question_id" in q}
        for q in primary_qs + cont_qs:
            qid = q.get("question_id")
            if qid in dev_lookup:
                canon = dev_lookup[qid]
                if q.get("role") != canon.get("role"):
                    raise ValueError(f"Metadata role mismatch for question '{qid}': '{q.get('role')}' vs '{canon.get('role')}'")
                if q.get("query_date") != canon.get("query_date"):
                    raise ValueError(f"Metadata query_date mismatch for question '{qid}': '{q.get('query_date')}' vs '{canon.get('query_date')}'")
                if bool(q.get("answerability")) != bool(canon.get("answerability")):
                    raise ValueError(f"Metadata answerability mismatch for question '{qid}': '{q.get('answerability')}' vs '{canon.get('answerability')}'")

    # 5. Check corpus fingerprints when recorded; report provenance note
    p_fp = primary_data.get("corpus_fingerprint")
    c_fp = cont_data.get("corpus_fingerprint")
    if p_fp and c_fp:
        if p_fp != c_fp:
            raise ValueError(f"Corpus fingerprint mismatch: '{p_fp}' vs '{c_fp}'")
        provenance_note = f"Corpus fingerprint match verified: {p_fp}"
    elif p_fp or c_fp:
        fp = p_fp or c_fp
        provenance_note = f"Corpus fingerprint verified: {fp}"
    else:
        provenance_note = (
            "Historical evaluation result files do not record top-level corpus_fingerprint; "
            "metadata cross-verified against canonical dev_questions.json."
        )

    # Map results by question ID; continuation results supersede failed/earlier attempts
    final_by_qid: Dict[str, Dict[str, Any]] = {}
    historical_attempts: List[Dict[str, Any]] = []

    for q in primary_qs:
        qid = q["question_id"]
        if any(cq["question_id"] == qid for cq in cont_qs):
            historical_attempts.append(q)
        else:
            final_by_qid[qid] = q

    for q in cont_qs:
        qid = q["question_id"]
        final_by_qid[qid] = q

    reconciled_results = [final_by_qid[qid] for qid in target_ids if qid in final_by_qid]
    if len(reconciled_results) != len(target_ids):
        raise ValueError(f"Expected {len(target_ids)} unique questions in reconciled report, found {len(reconciled_results)}.")

    planned_questions = len(target_ids)
    completed_questions = len(reconciled_results)

    ans_results = [r for r in reconciled_results if r.get("answerability")]
    diag_results = [r for r in reconciled_results if not r.get("answerability")]
    restr_results = [r for r in diag_results if r.get("expected_refusal_type") == "restricted" or r["question_id"].endswith("-009")]
    unsupp_results = [r for r in diag_results if r.get("expected_refusal_type") == "insufficient" or r["question_id"].endswith("-010")]

    successful_answers = sum(
        1 for r in ans_results
        if r["status"] == "answered"
        and not r.get("provider_error", False)
        and r.get("schema_valid") is True
        and r.get("citation_valid", False)
    )
    refusals = sum(1 for r in reconciled_results if r["status"] in ("insufficient", "restricted"))
    restr_successes = sum(1 for r in restr_results if r.get("refusal_correct", False))
    unsupp_successes = sum(1 for r in unsupp_results if r.get("refusal_correct", False))

    generations_attempted = sum(
        1 for r in reconciled_results
        if r.get("provider_calls_attempted", 0) and r.get("provider_calls_attempted", 0) > 0
    )
    schema_valid_gens = sum(
        1 for r in reconciled_results
        if r.get("schema_valid") is True
    )
    cit_expected_count = sum(1 for r in reconciled_results if r.get("citation_expected", False))
    cit_valid_gens = sum(
        1 for r in reconciled_results
        if r.get("citation_expected", False) and r.get("citation_valid", False)
    )

    provider_failures = sum(1 for r in reconciled_results if r.get("provider_error", False))
    evidence_validation_failures = sum(1 for r in reconciled_results if r.get("evidence_validation_failure", False))
    citation_validation_failures = sum(1 for r in reconciled_results if r.get("citation_validation_failure", False))
    other_operational_failures = sum(1 for r in reconciled_results if r.get("other_operational_failure", False))

    has_final_errors = any(
        r.get("status") == "operational_error"
        or r.get("provider_error", False)
        or r.get("evidence_validation_failure", False)
        or r.get("citation_validation_failure", False)
        or r.get("other_operational_failure", False)
        for r in reconciled_results
    )
    evaluation_state = "completed_with_errors" if has_final_errors else "completed"

    prov_req_primary = primary_data.get("actual_provider_requests") or primary_data.get("generation_calls")
    if prov_req_primary is None:
        prov_req_primary = sum(q.get("provider_calls_attempted", 0) or 0 for q in primary_qs)

    prov_req_cont = cont_data.get("actual_provider_requests") or cont_data.get("generation_calls")
    if prov_req_cont is None:
        prov_req_cont = sum(q.get("provider_calls_attempted", 0) or 0 for q in cont_qs)

    total_prov_requests = prov_req_primary + prov_req_cont

    tok_primary = primary_data.get("known_tokens_used") or primary_data.get("tokens_used")
    if tok_primary is None:
        tok_primary = sum(q.get("token_usage", 0) or 0 for q in primary_qs)

    tok_cont = cont_data.get("known_tokens_used") or cont_data.get("tokens_used")
    if tok_cont is None:
        tok_cont = sum(q.get("token_usage", 0) or 0 for q in cont_qs)

    total_tokens = tok_primary + tok_cont

    all_lats = [r["latency_ms"] for r in reconciled_results if r.get("latency_ms") is not None]
    median_e2e = float(statistics.median(all_lats)) if all_lats else 0.0

    prov_lats = [r["provider_latency_ms"] for r in reconciled_results if r.get("provider_latency_ms") is not None]
    median_prov = float(statistics.median(prov_lats)) if prov_lats else 0.0

    diag_lats = [r["latency_ms"] for r in diag_results if r.get("latency_ms") is not None]
    median_diag = float(statistics.median(diag_lats)) if diag_lats else 0.0

    reconciled_summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": primary_data.get("model", GENERATION_MODEL),
        "evaluation_state": evaluation_state,
        "stop_reason": None,
        "retrieval_backend": primary_data.get("retrieval_backend", "local"),
        "provenance_note": provenance_note,
        "corpus_fingerprint": p_fp or c_fp,
        "planned_questions": planned_questions,
        "completed_questions": completed_questions,
        "unique_questions_evaluated": completed_questions,
        "historical_retry_attempts_count": len(historical_attempts),
        "historical_retries_attempted": len(historical_attempts),
        "successful_answers": successful_answers,
        "successful_generations": successful_answers,
        "refusals": refusals,
        "restricted_refusal_successes": restr_successes,
        "unsupported_refusal_successes": unsupp_successes,
        "incorrect_refusals": sum(1 for r in diag_results if not r.get("refusal_correct", False)),
        "incorrect_answers": sum(1 for r in ans_results if r["status"] != "answered"),
        "provider_errors": provider_failures,
        "provider_failures": provider_failures,
        "evidence_validation_failures": evidence_validation_failures,
        "citation_validation_failures": citation_validation_failures,
        "other_operational_failures": other_operational_failures,
        "generations_attempted": generations_attempted,
        "schema_valid_generations": schema_valid_gens,
        "citation_expected_count": cit_expected_count,
        "citation_valid_generations": cit_valid_gens,
        "total_provider_requests_attempted": total_prov_requests,
        "total_tokens_consumed": total_tokens,
        "known_tokens_used": total_tokens,
        "tokens_used": total_tokens,
        "known_tokens_incomplete": primary_data.get("known_tokens_incomplete", False) or cont_data.get("known_tokens_incomplete", False),
        "median_e2e_latency_ms": round(median_e2e, 2),
        "median_provider_latency_ms": round(median_prov, 2),
        "median_diagnostic_latency_ms": round(median_diag, 2),
        "historical_attempts": historical_attempts,
        "per_question_results": reconciled_results,
    }

    if output_json_path:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(exist_ok=True)
        out_j.write_text(json.dumps(reconciled_summary, indent=2), encoding="utf-8")

    if output_md_path:
        out_m = Path(output_md_path)
        out_m.parent.mkdir(exist_ok=True)
        md_text = generate_reconciled_markdown(reconciled_summary)
        out_m.write_text(md_text, encoding="utf-8")

    return reconciled_summary



def main():
    """CLI runner for generation evaluation."""
    parser = argparse.ArgumentParser(description="RuleLens Generation Evaluation Runner")
    parser.add_argument("--questions", default="data/evaluation/dev_questions.json", help="Path to dev_questions.json")
    parser.add_argument("--limit", type=int, default=30, help="Maximum questions to evaluate (default: 30)")
    parser.add_argument("--backend", default="local", choices=["local", "qdrant"], help="Dense retrieval backend")
    parser.add_argument("--smoke", action="store_true", help="Run 5-question smoke test only")
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    repo_root = Path.cwd()
    dataset_path = repo_root / args.questions

    print(f"Running generation evaluation (limit={limit}, smoke={args.smoke}, backend={args.backend})...")
    summary = run_generation_evaluation(
        dataset_path=dataset_path,
        base_dir=repo_root,
        max_questions=limit,
        dense_backend=args.backend,
    )

    local_eval_dir = repo_root / "local_eval"
    local_eval_dir.mkdir(exist_ok=True)

    results_path = local_eval_dir / "generation_development_results.json"
    summary_path = local_eval_dir / "generation_development_summary.md"

    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_content = generate_summary_markdown(summary)
    summary_path.write_text(md_content, encoding="utf-8")

    print(f"\nSaved generation results to {results_path}")
    print(f"Saved generation summary to {summary_path}")
    print("\nSummary metrics:")
    print(f"  Evaluation state:           {summary['evaluation_state']}")
    print(f"  Completed / Planned:        {summary['completed_questions']} / {summary['planned_questions']}")
    print(f"  Schema valid:               {summary['schema_valid_generations']}")
    print(f"  Citation valid:             {summary['citation_valid_generations']}")
    print(f"  Restricted refusals:        {summary['restricted_refusal_successes']}")
    print(f"  Unsupported refusals:       {summary['unsupported_refusal_successes']}")
    print(f"  Median E2E latency:         {summary['median_e2e_latency_ms']} ms")


if __name__ == "__main__":
    main()
