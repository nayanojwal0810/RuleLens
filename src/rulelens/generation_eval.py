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


def load_evaluation_dataset(dataset_path: Union[Path, str]) -> Dict[str, Any]:
    """Load evaluation questions JSON file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_single_question(
    question_data: Dict[str, Any],
    retriever: HybridRetriever,
    llm_client: GroqClientWrapper,
    base_dir: Optional[Union[Path, str]] = None,
    use_gold_evidence_for_answerable: bool = True,
) -> Dict[str, Any]:
    """Evaluate generation on a single question using the real answering pipeline.

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

    start_time = time.perf_counter()

    # Diagnostic questions: execute through real application answering pipeline
    if not is_answerable:
        app_result = answer_question(
            question=q_text,
            role=role,
            query_date=query_date_str,
            retriever=retriever,
            llm_client=llm_client,
            base_dir=base_dir,
        )
        total_lat = (time.perf_counter() - start_time) * 1000.0

        is_restricted_expected = bool(restr_targets)
        # Score the refusal based on whether it correctly refused and matched expected refusal type
        refusal_correct = False
        if is_restricted_expected:
            refusal_correct = (app_result.status == AnswerStatus.RESTRICTED.value)
        else:
            refusal_correct = (app_result.status == AnswerStatus.INSUFFICIENT.value)

        return {
            "question_id": qid,
            "department": question_data.get("department"),
            "question_type": question_data.get("question_type"),
            "difficulty": question_data.get("difficulty"),
            "role": role,
            "query_date": query_date_str,
            "answerability": is_answerable,
            "gold_chunk_ids": gold_cids,
            "status": app_result.status,
            "answer": app_result.answer,
            "citations": app_result.citations,
            "referenced_rule_ids": app_result.referenced_rule_ids,
            "reason": app_result.reason,
            "schema_valid": app_result.status in ("answered", "insufficient", "restricted"),
            "citation_valid": len(app_result.citations) == 0,
            "refusal_correct": refusal_correct,
            "provider_error": app_result.status == "operational_error",
            "role_violations": 0,
            "version_violations": 0,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": 0.0,
            "token_usage": 0,
            "factual_correctness": "not_reviewed",
            "gold_quotes": [e.get("quote", "") for e in expected_evidence],
            "error": app_result.error,
        }

    # Answerable questions: run using authorized gold chunks or standard retrieval
    if use_gold_evidence_for_answerable and gold_cids:
        # Load and verify gold evidence through canonical corpus
        corpus_chunks = getattr(retriever, "chunks_by_id", {})
        if not corpus_chunks and hasattr(retriever, "chunks"):
            corpus_chunks = {c["chunk_id"]: c for c in retriever.chunks}

        authorized_chunks = []
        role_violations = 0
        version_violations = 0
        for cid in gold_cids:
            if cid in corpus_chunks:
                c = corpus_chunks[cid]
                if not is_chunk_eligible(c, role, validated_date):
                    role_violations += 1
                authorized_chunks.append(c)

        extracted_conds = extract_conditions_from_query(q_text)
        rule_results = resolve_rules_for_evidence(
            authorized_chunks=authorized_chunks,
            role=role,
            query_date=validated_date,
            base_dir=base_dir,
            extracted_conditions=extracted_conds,
        )
        applicable_rule_ids = {r.rule.rule_id for r in rule_results if r.passed}

        schema_valid = False
        citation_valid = False
        gen_answer = ""
        gen_status = ""
        gen_reason = ""
        gen_citations: List[str] = []
        gen_rules: List[str] = []
        token_usage = None
        error_msg = None
        provider_lat = 0.0

        try:
            model_resp, llm_meta = llm_client.generate_answer(
                question=q_text,
                role=role,
                evidence_chunks=authorized_chunks,
                query_date=query_date_str,
                applicable_rule_facts=rule_results,
            )
            schema_valid = True
            gen_answer = model_resp.answer
            gen_status = model_resp.status.value
            gen_reason = model_resp.reason
            token_usage = llm_meta.get("total_tokens")
            provider_lat = llm_meta.get("latency_ms", 0.0)

            val_cids, _, val_rules = validate_citations_and_rules(
                response=model_resp,
                authorized_chunks=authorized_chunks,
                role=role,
                query_date=validated_date,
                applicable_rule_ids=applicable_rule_ids,
            )
            gen_citations = val_cids
            gen_rules = val_rules
            citation_valid = True

        except (SanitizedProviderError, CitationValidationError) as prov_err:
            error_msg = prov_err.safe_message
            gen_status = "operational_error"
        except Exception:
            error_msg = "Unhandled operational error during evaluation generation."
            gen_status = "operational_error"

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
            "status": gen_status,
            "answer": gen_answer,
            "citations": gen_citations,
            "referenced_rule_ids": gen_rules,
            "reason": gen_reason,
            "schema_valid": schema_valid,
            "citation_valid": citation_valid,
            "refusal_correct": False,
            "provider_error": gen_status == "operational_error",
            "role_violations": role_violations,
            "version_violations": version_violations,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": round(provider_lat, 2),
            "token_usage": token_usage,
            "factual_correctness": "not_reviewed",
            "gold_quotes": [e.get("quote", "") for e in expected_evidence],
            "error": error_msg,
        }
    else:
        # Full answering pipeline for answerable query
        app_result = answer_question(
            question=q_text,
            role=role,
            query_date=query_date_str,
            retriever=retriever,
            llm_client=llm_client,
            base_dir=base_dir,
        )
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
            "status": app_result.status,
            "answer": app_result.answer,
            "citations": app_result.citations,
            "referenced_rule_ids": app_result.referenced_rule_ids,
            "reason": app_result.reason,
            "schema_valid": app_result.status in ("answered", "insufficient", "restricted"),
            "citation_valid": len(app_result.citations) > 0,
            "refusal_correct": False,
            "provider_error": app_result.status == "operational_error",
            "role_violations": 0,
            "version_violations": 0,
            "latency_ms": round(total_lat, 2),
            "provider_latency_ms": round(app_result.latency_ms, 2),
            "token_usage": app_result.token_usage,
            "factual_correctness": "not_reviewed",
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
) -> Dict[str, Any]:
    """Run controlled development generation evaluation.

    Args:
        dataset_path: Path to dev_questions.json.
        base_dir: Repository root.
        subset_ids: Specific question IDs to evaluate.
        max_questions: Cap on number of questions to evaluate.
        llm_client: Optional injected client wrapper.
        retriever: Optional injected HybridRetriever.

    Returns:
        Dict with aggregate summary and per-question results.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    data = load_evaluation_dataset(dataset_path)
    all_questions = data["questions"]
    q_map = {q["question_id"]: q for q in all_questions}

    target_ids = list(subset_ids) if subset_ids is not None else DEV_SUBSET_QUESTION_IDS
    if max_questions is not None:
        target_ids = target_ids[:max_questions]

    selected_questions = [q_map[qid] for qid in target_ids if qid in q_map]

    active_retriever = retriever or HybridRetriever(base_dir=base)
    active_llm = llm_client or GroqClientWrapper()

    per_q_results: List[Dict[str, Any]] = []
    print(f"Starting Generation Evaluation on {len(selected_questions)} questions...")

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
        )
        per_q_results.append(res)
        if res.get("provider_error") and res.get("error") and "rate limit" in res["error"].lower():
            stopped_early = True
            stop_reason = "Rate limit HTTP 429 encountered from provider."
            print(f"Encountered Groq 429 rate limit on {qid}. Stopping evaluation early to respect provider quota.")
            break
        # Pacing between live calls
        if q.get("answerability", True) and llm_client is None:
            time.sleep(2.5)

    # Dynamic metrics computation
    planned_questions = len(selected_questions)
    completed_questions = len(per_q_results)

    ans_results = [r for r in per_q_results if r["answerability"]]
    diag_results = [r for r in per_q_results if not r["answerability"]]
    restr_results = [r for r in diag_results if r["question_id"].endswith("-009")]
    unsupp_results = [r for r in diag_results if r["question_id"].endswith("-010")]

    successful_generations = sum(1 for r in ans_results if r["status"] == "answered" and not r["provider_error"])
    refusals = sum(1 for r in per_q_results if r["status"] in ("insufficient", "restricted"))
    provider_errors = sum(1 for r in per_q_results if r["provider_error"])

    schema_valid_generations = sum(1 for r in per_q_results if r["schema_valid"] and not r["provider_error"])
    citation_valid_generations = sum(1 for r in ans_results if r["citation_valid"] and not r["provider_error"])

    restr_success = sum(1 for r in restr_results if r["refusal_correct"])
    unsupp_success = sum(1 for r in unsupp_results if r["refusal_correct"])

    generation_calls = sum(1 for r in per_q_results if r["answerability"] and not r["provider_error"])
    total_tokens = sum(r.get("token_usage") or 0 for r in per_q_results)

    e2e_latencies = [r["latency_ms"] for r in per_q_results]
    provider_latencies = [r["provider_latency_ms"] for r in ans_results if r["provider_latency_ms"] > 0]
    diag_latencies = [r["latency_ms"] for r in diag_results]

    median_e2e = statistics.median(e2e_latencies) if e2e_latencies else 0.0
    median_prov = statistics.median(provider_latencies) if provider_latencies else 0.0
    median_diag = statistics.median(diag_latencies) if diag_latencies else 0.0

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": GENERATION_MODEL,
        "evaluation_state": "partial" if stopped_early else "completed",
        "stop_reason": stop_reason,
        "planned_questions": planned_questions,
        "completed_questions": completed_questions,
        "successful_generations": successful_generations,
        "refusals": refusals,
        "provider_errors": provider_errors,
        "schema_valid_generations": schema_valid_generations,
        "citation_valid_generations": citation_valid_generations,
        "restricted_refusal_successes": restr_success,
        "unsupported_refusal_successes": unsupp_success,
        "generation_calls": generation_calls,
        "tokens_used": total_tokens,
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
    restr_count = sum(1 for r in summary["per_question_results"] if r["question_id"].endswith("-009"))
    unsupp_count = sum(1 for r in summary["per_question_results"] if r["question_id"].endswith("-010"))

    lines = [
        "# RuleLens — Stage E Development Answer Generation Summary",
        "",
        "## 1. Evaluation Setup",
        f"- **Timestamp**: `{summary['timestamp']}`",
        f"- **Model**: `{summary['model']}`",
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
        f"| **Schema Validity** | {format_rate(summary['schema_valid_generations'], completed)} | 100% |",
        f"| **Citation Validity (Answerable)** | {format_rate(summary['citation_valid_generations'], ans_count)} | 100% |",
        f"| **Restricted Refusal Success** | {format_rate(summary['restricted_refusal_successes'], restr_count)} | 100% |",
        f"| **Unsupported Refusal Success** | {format_rate(summary['unsupported_refusal_successes'], unsupp_count)} | 100% |",
        f"| **Provider Errors** | {summary['provider_errors']} | 0 |",
        f"| **Median E2E Latency** | {summary['median_e2e_latency_ms']} ms | < 2500 ms |",
        f"| **Median Provider Latency** | {summary['median_provider_latency_ms']} ms | < 2000 ms |",
        f"| **Median Diagnostic Latency** | {summary['median_diagnostic_latency_ms']} ms | < 50 ms |",
        f"| **Total Tokens Consumed** | {summary['tokens_used']} | Monitored |",
        "",
        "## 3. Per-Question Results Sample",
        "",
        "| ID | Role | Date | Status | Citations | Rules | E2E Latency | Answer Excerpt |",
        "|---|---|:---:|:---:|---|---|---:|---|",
    ]

    for r in summary["per_question_results"]:
        ans_excerpt = (r["answer"][:60] + "...") if len(r["answer"]) > 60 else r["answer"]
        ans_excerpt = ans_excerpt.replace("\n", " ")
        cits_str = ", ".join(r["citations"]) if r["citations"] else "None"
        rules_str = ", ".join(r["referenced_rule_ids"]) if r["referenced_rule_ids"] else "None"
        lines.append(
            f"| **{r['question_id']}** | `{r['role']}` | {r['query_date']} | `{r['status']}` | `{cits_str}` | `{rules_str}` | {r['latency_ms']} ms | {ans_excerpt} |"
        )

    lines.append("")
    return "\n".join(lines)


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
