"""Retrieval metrics and evidence coverage for explicit question workloads."""

from __future__ import annotations

import math
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from rulelens.dataset import DEPARTMENTS
from rulelens.retrieval import HybridRetriever, ALLOWED_ROLES_SET, is_chunk_eligible, parse_query_date
from rulelens.ingestion import get_corpus_snapshot


def compute_hit_at_k(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Set[str], k: int = 5) -> float:
    """Return 1.0 if any gold chunk ID is present in top K retrieved chunks, else 0.0."""
    if not gold_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in gold_chunk_ids for cid in top_k) else 0.0

def compute_recall_at_k(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Set[str], k: int = 5) -> float:
    """Return fraction of gold chunk IDs retrieved in top K (multi-gold Recall@K)."""
    if not gold_chunk_ids:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    found = len(top_k.intersection(gold_chunk_ids))
    return float(found) / float(len(gold_chunk_ids))

def compute_full_evidence_at_k(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Set[str], k: int = 5) -> float:
    """Return 1.0 if ALL gold chunk IDs are present in top K retrieved chunks, else 0.0."""
    if not gold_chunk_ids:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    return 1.0 if gold_chunk_ids.issubset(top_k) else 0.0

def compute_mrr_at_k(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Set[str], k: int = 5) -> float:
    """Return reciprocal rank of first gold chunk in top K (1-based), or 0.0 if not found."""
    if not gold_chunk_ids:
        return 0.0
    for rank_idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        if cid in gold_chunk_ids:
            return 1.0 / float(rank_idx)
    return 0.0

def evaluate_retriever_method(
    retriever: HybridRetriever,
    questions: Sequence[Dict[str, Any]],
    mode: str = "hybrid",
    top_k: int = 5,
    dense_backend: str = "local",
    qdrant_client: Optional[Any] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """Evaluate a retriever method over development questions.

    Distinguishes answerable questions from diagnostic (forbidden / unanswerable) questions.
    Measures warmed-up latency, permission violations, version violations, and miss analysis.
    """
    if warmup and questions:
        # Warm up caches and models before recording latency (propagate any unexpected failures)
        first_q = questions[0]
        retriever.search(
            query=first_q["question"],
            role=first_q["role"],
            query_date=first_q["query_date"],
            mode=mode,
            dense_backend=dense_backend,
            qdrant_client=qdrant_client,
            top_k=top_k,
        )

    latencies_ms: List[float] = []
    hits: List[float] = []
    recalls: List[float] = []
    mrrs: List[float] = []
    full_evidences: List[float] = []

    # Breakdown tracking
    by_diff: Dict[str, Dict[str, List[float]]] = {
        "easy": {"hits": [], "recalls": [], "mrrs": [], "full_evidences": []},
        "medium": {"hits": [], "recalls": [], "mrrs": [], "full_evidences": []},
        "hard": {"hits": [], "recalls": [], "mrrs": [], "full_evidences": []},
    }
    by_dept: Dict[str, Dict[str, List[float]]] = {
        d: {"hits": [], "recalls": [], "mrrs": [], "full_evidences": []} for d in DEPARTMENTS
    }
    by_qtype: Dict[str, Dict[str, List[float]]] = {}

    answerable_count = 0
    permission_violations = 0
    version_violations = 0

    misses: List[Dict[str, Any]] = []
    partial_evidence_misses: List[Dict[str, Any]] = []
    suboptimal_rankings: List[Dict[str, Any]] = []
    diagnostic_results: List[Dict[str, Any]] = []
    per_question_details: List[Dict[str, Any]] = []

    for q in questions:
        qid = q["question_id"]
        query_text = q["question"]
        role = q["role"]
        q_date = q["query_date"]
        dept = q.get("department", "Unknown")
        diff = q.get("difficulty", "medium")
        q_type = q.get("question_type", "direct")
        gold_cids = set(q.get("gold_chunk_ids", []))
        restr_targets = set(q.get("restricted_target_chunk_ids", []))
        is_answerable = q.get("answerability", True)

        start_time = time.perf_counter()
        results = retriever.search(
            query=query_text,
            role=role,
            query_date=q_date,
            mode=mode,
            dense_backend=dense_backend,
            qdrant_client=qdrant_client,
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_cids = [r.chunk_id for r in results]

        # Check permission violations: does any result violate role permissions?
        for r in results:
            if role not in r.allowed_roles:
                permission_violations += 1

            # Check version violations: does any result fall outside date boundaries?
            eff_from = r.effective_from
            eff_to = r.effective_to or "2099-12-31"
            q_date_str = str(q_date)
            if not (eff_from <= q_date_str <= eff_to):
                version_violations += 1

        # Check restricted target leakage for diagnostic queries
        if restr_targets:
            for cid in retrieved_cids:
                if cid in restr_targets:
                    permission_violations += 1

        hit: Optional[float] = None
        recall: Optional[float] = None
        mrr: Optional[float] = None
        full_ev: Optional[float] = None

        if is_answerable and gold_cids:
            answerable_count += 1
            hit = compute_hit_at_k(retrieved_cids, gold_cids, k=top_k)
            recall = compute_recall_at_k(retrieved_cids, gold_cids, k=top_k)
            mrr = compute_mrr_at_k(retrieved_cids, gold_cids, k=top_k)
            full_ev = compute_full_evidence_at_k(retrieved_cids, gold_cids, k=top_k)

            hits.append(hit)
            recalls.append(recall)
            mrrs.append(mrr)
            full_evidences.append(full_ev)

            if diff in by_diff:
                by_diff[diff]["hits"].append(hit)
                by_diff[diff]["recalls"].append(recall)
                by_diff[diff]["mrrs"].append(mrr)
                by_diff[diff]["full_evidences"].append(full_ev)

            if dept in by_dept:
                by_dept[dept]["hits"].append(hit)
                by_dept[dept]["recalls"].append(recall)
                by_dept[dept]["mrrs"].append(mrr)
                by_dept[dept]["full_evidences"].append(full_ev)

            if q_type not in by_qtype:
                by_qtype[q_type] = {"hits": [], "recalls": [], "mrrs": [], "full_evidences": []}
            by_qtype[q_type]["hits"].append(hit)
            by_qtype[q_type]["recalls"].append(recall)
            by_qtype[q_type]["mrrs"].append(mrr)
            by_qtype[q_type]["full_evidences"].append(full_ev)

            if hit == 0.0:
                misses.append(
                    {
                        "question_id": qid,
                        "question": query_text,
                        "department": dept,
                        "difficulty": diff,
                        "role": role,
                        "query_date": q_date,
                        "question_type": q_type,
                        "expected_gold_chunk_ids": sorted(gold_cids),
                        "actual_top_k_chunk_ids": retrieved_cids,
                    }
                )
            elif 0.0 < recall < 1.0:
                partial_evidence_misses.append(
                    {
                        "question_id": qid,
                        "question": query_text,
                        "department": dept,
                        "difficulty": diff,
                        "role": role,
                        "query_date": q_date,
                        "question_type": q_type,
                        "recall": round(recall, 4),
                        "expected_gold_chunk_ids": sorted(gold_cids),
                        "actual_top_k_chunk_ids": retrieved_cids,
                        "retrieved_gold_ids": sorted(set(retrieved_cids).intersection(gold_cids)),
                        "missing_gold_ids": sorted(gold_cids - set(retrieved_cids)),
                    }
                )

            if mrr < 1.0:
                # Retrieved in top K, but first gold not at rank 1
                first_rank = next((idx for idx, c in enumerate(retrieved_cids, start=1) if c in gold_cids), None)
                suboptimal_rankings.append(
                    {
                        "question_id": qid,
                        "question": query_text,
                        "department": dept,
                        "difficulty": diff,
                        "role": role,
                        "query_date": q_date,
                        "question_type": q_type,
                        "first_gold_rank": first_rank,
                        "expected_gold_chunk_ids": sorted(gold_cids),
                        "actual_top_k_chunk_ids": retrieved_cids,
                    }
                )
        else:
            # Diagnostic question (forbidden permission test or unsupported answer)
            diagnostic_results.append(
                {
                    "question_id": qid,
                    "question": query_text,
                    "department": dept,
                    "role": role,
                    "query_date": q_date,
                    "question_type": q_type,
                    "retrieved_chunk_ids": retrieved_cids,
                    "restricted_targets": sorted(restr_targets),
                    "is_answerable": is_answerable,
                    "leakage_detected": any(c in restr_targets for c in retrieved_cids),
                }
            )

        per_question_details.append(
            {
                "question_id": qid,
                "question": query_text,
                "department": dept,
                "difficulty": diff,
                "role": role,
                "query_date": q_date,
                "question_type": q_type,
                "answerability": is_answerable,
                "retrieved_chunk_ids": retrieved_cids,
                "gold_chunk_ids": sorted(gold_cids),
                "restricted_target_chunk_ids": sorted(restr_targets),
                "hit_at_k": hit,
                "recall_at_k": recall,
                "mrr_at_k": mrr,
                "full_evidence_at_k": full_ev,
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    median_lat = float(statistics.median(latencies_ms)) if latencies_ms else 0.0
    p95_lat = sorted(latencies_ms)[math.ceil(0.95 * len(latencies_ms)) - 1] if latencies_ms else 0.0

    avg_hit = float(sum(hits) / len(hits)) if hits else 0.0
    avg_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    avg_mrr = float(sum(mrrs) / len(mrrs)) if mrrs else 0.0
    avg_full_ev = float(sum(full_evidences) / len(full_evidences)) if full_evidences else 0.0

    # Aggregate breakdown by difficulty
    diff_summary = {}
    for d_name, d_data in by_diff.items():
        cnt = len(d_data["hits"])
        h_val = round(float(sum(d_data["hits"]) / cnt), 4) if cnt else 0.0
        r_val = round(float(sum(d_data["recalls"]) / cnt), 4) if cnt else 0.0
        m_val = round(float(sum(d_data["mrrs"]) / cnt), 4) if cnt else 0.0
        fe_val = round(float(sum(d_data["full_evidences"]) / cnt), 4) if cnt else 0.0
        d_entry = {
            "count": cnt,
            "hit_at_k": h_val,
            "recall_at_k": r_val,
            "mrr_at_k": m_val,
            "full_evidence_at_k": fe_val,
            f"hit_at_{top_k}": h_val,
            f"recall_at_{top_k}": r_val,
            f"mrr_at_{top_k}": m_val,
            f"full_evidence_at_{top_k}": fe_val,
        }
        if top_k == 5:
            d_entry["hit_at_5"] = h_val
            d_entry["recall_at_5"] = r_val
            d_entry["mrr_at_5"] = m_val
            d_entry["full_evidence_at_5"] = fe_val
        diff_summary[d_name] = d_entry

    # Aggregate breakdown by department
    dept_summary = {}
    for dep_name, dep_data in by_dept.items():
        cnt = len(dep_data["hits"])
        h_val = round(float(sum(dep_data["hits"]) / cnt), 4) if cnt else 0.0
        r_val = round(float(sum(dep_data["recalls"]) / cnt), 4) if cnt else 0.0
        m_val = round(float(sum(dep_data["mrrs"]) / cnt), 4) if cnt else 0.0
        fe_val = round(float(sum(dep_data["full_evidences"]) / cnt), 4) if cnt else 0.0
        dep_entry = {
            "count": cnt,
            "hit_at_k": h_val,
            "recall_at_k": r_val,
            "mrr_at_k": m_val,
            "full_evidence_at_k": fe_val,
            f"hit_at_{top_k}": h_val,
            f"recall_at_{top_k}": r_val,
            f"mrr_at_{top_k}": m_val,
            f"full_evidence_at_{top_k}": fe_val,
        }
        if top_k == 5:
            dep_entry["hit_at_5"] = h_val
            dep_entry["recall_at_5"] = r_val
            dep_entry["mrr_at_5"] = m_val
            dep_entry["full_evidence_at_5"] = fe_val
        dept_summary[dep_name] = dep_entry

    # Aggregate breakdown by question type
    qtype_summary = {}
    for qt_name, qt_data in sorted(by_qtype.items()):
        cnt = len(qt_data["hits"])
        h_val = round(float(sum(qt_data["hits"]) / cnt), 4) if cnt else 0.0
        r_val = round(float(sum(qt_data["recalls"]) / cnt), 4) if cnt else 0.0
        m_val = round(float(sum(qt_data["mrrs"]) / cnt), 4) if cnt else 0.0
        fe_val = round(float(sum(qt_data["full_evidences"]) / cnt), 4) if cnt else 0.0
        qt_entry = {
            "count": cnt,
            "hit_at_k": h_val,
            "recall_at_k": r_val,
            "mrr_at_k": m_val,
            "full_evidence_at_k": fe_val,
            f"hit_at_{top_k}": h_val,
            f"recall_at_{top_k}": r_val,
            f"mrr_at_{top_k}": m_val,
            f"full_evidence_at_{top_k}": fe_val,
        }
        if top_k == 5:
            qt_entry["hit_at_5"] = h_val
            qt_entry["recall_at_5"] = r_val
            qt_entry["mrr_at_5"] = m_val
            qt_entry["full_evidence_at_5"] = fe_val
        qtype_summary[qt_name] = qt_entry

    res_dict = {
        "mode": mode,
        "dense_backend": dense_backend,
        "cutoff": top_k,
        "total_questions_evaluated": len(questions),
        "total_answerable_questions": answerable_count,
        "total_diagnostic_questions": len(diagnostic_results),
        "hit_at_k": round(avg_hit, 4),
        "recall_at_k": round(avg_recall, 4),
        "mrr_at_k": round(avg_mrr, 4),
        "full_evidence_at_k": round(avg_full_ev, 4),
        f"hit_at_{top_k}": round(avg_hit, 4),
        f"recall_at_{top_k}": round(avg_recall, 4),
        f"mrr_at_{top_k}": round(avg_mrr, 4),
        f"full_evidence_at_{top_k}": round(avg_full_ev, 4),
        "complete_evidence_count": sum(1 for fe in full_evidences if fe == 1.0),
        "rank_one_hits_count": sum(1 for m in mrrs if m == 1.0),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "permission_violations": permission_violations,
        "version_violations": version_violations,
        "by_difficulty": diff_summary,
        "by_department": dept_summary,
        "by_question_type": qtype_summary,
        "missed_questions_count": len(misses),
        "misses": misses,
        "partial_evidence_misses_count": len(partial_evidence_misses),
        "partial_evidence_misses": partial_evidence_misses,
        "suboptimal_rankings_count": len(suboptimal_rankings),
        "suboptimal_rankings": suboptimal_rankings,
        "diagnostic_results": diagnostic_results,
        "per_question_details": per_question_details,
    }
    if top_k == 5:
        res_dict["hit_at_5"] = round(avg_hit, 4)
        res_dict["recall_at_5"] = round(avg_recall, 4)
        res_dict["mrr_at_5"] = round(avg_mrr, 4)
        res_dict["full_evidence_at_5"] = round(avg_full_ev, 4)

    return res_dict


def validate_question_dataset(dataset, base_dir=None):
    """Check question references against the corpus without prescribing a distribution."""
    data = json.loads(Path(dataset).read_text(encoding="utf-8")) if isinstance(dataset, (str, Path)) else dataset
    if not isinstance(data, dict) or data.get("schema_version") not in ("1.0", "2.0"):
        raise ValueError("Expected question schema 1.0 or 2.0.")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a non-empty questions list.")
    snapshot = get_corpus_snapshot(base_dir)
    if data.get("corpus_fingerprint") != snapshot.fingerprint:
        raise ValueError("Question-set corpus fingerprint mismatch.")
    chunks = {c["chunk_id"]: c for c in snapshot.chunks}
    seen = set()
    for q in questions:
        if not isinstance(q, dict):
            raise ValueError("Questions must be objects.")
        for field in ("question_id", "question", "role", "query_date"):
            if not isinstance(q.get(field), str) or not q[field].strip():
                raise ValueError(f"Missing required field: {field}.")
        if q["question_id"] in seen:
            raise ValueError("Duplicate question ID.")
        seen.add(q["question_id"])
        if q["role"] not in ALLOWED_ROLES_SET or not isinstance(q.get("answerability"), bool):
            raise ValueError("Invalid role or answerability.")
        if q.get("difficulty") not in ("easy", "medium", "hard"):
            raise ValueError("Invalid difficulty.")
        if q.get("department") not in (*DEPARTMENTS, "Cross-policy"):
            raise ValueError("Invalid department.")
        if not isinstance(q.get("question_type"), str) or not q["question_type"].strip():
            raise ValueError("Missing question type.")
        date = parse_query_date(q["query_date"])
        gold = q.get("gold_chunk_ids", [])
        evidence = q.get("gold_evidence", [])
        if (not isinstance(gold, list) or not all(isinstance(cid, str) for cid in gold)
                or not isinstance(evidence, list) or not all(isinstance(e, dict) for e in evidence)
                or len(gold) != len(set(gold))):
            raise ValueError("Malformed gold evidence.")
        if q["answerability"]:
            if not 1 <= len(gold) <= 4 or {e.get("chunk_id") for e in evidence} != set(gold):
                raise ValueError("Answerable questions need complete evidence within four chunks.")
            for cid in gold:
                if cid not in chunks or not is_chunk_eligible(chunks[cid], q["role"], date):
                    raise ValueError("Gold evidence is not authorized and active.")
            for e in evidence:
                quote = e.get("quote", "")
                if not isinstance(quote, str) or not quote.strip() or " ".join(quote.split()) not in " ".join(chunks[e["chunk_id"]]["text"].split()):
                    raise ValueError("Gold quote does not match its source.")
        elif gold or evidence:
            raise ValueError("Refusal questions must not include answer evidence.")
        restricted = q.get("restricted_target_chunk_ids", [])
        if not isinstance(restricted, list) or not all(isinstance(cid, str) for cid in restricted):
            raise ValueError("Malformed restricted targets.")
        if q["answerability"] and restricted:
            raise ValueError("Answerable questions cannot have restricted targets.")
        for cid in restricted:
            if cid not in chunks or q["role"] in chunks[cid]["allowed_roles"]:
                raise ValueError("Restricted target is not a valid permission diagnostic.")
        if data["schema_version"] == "2.0":
            validate_expected_result(q, chunks)
    return {"status": "valid", "total_questions": len(questions)}


def validate_expected_result(question, chunks):
    """Check structured gold references; semantic entailment still needs source review."""
    expected = question.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Missing expected result.")
    status, route = expected.get("status"), expected.get("route")
    routes = {"answered": {"rule", "generated_verified"},
              "restricted": {"refusal"}, "insufficient": {"refusal", "clarification"}}
    if status not in routes or route not in routes[status]:
        raise ValueError("Invalid expected status or route.")
    if question["answerability"] != (status == "answered"):
        raise ValueError("Expected status conflicts with answerability.")
    if (status == "restricted") != bool(question.get("restricted_target_chunk_ids")):
        raise ValueError("Expected restriction conflicts with target evidence.")
    if route == "clarification" and not expected.get("clarification"):
        raise ValueError("Expected clarification needs a question.")
    if not isinstance(expected.get("reason_code"), str) or not expected["reason_code"].strip():
        raise ValueError("Missing expected reason code.")
    evidence = question["gold_evidence"]
    fact_ids = [e.get("fact_id") for e in evidence]
    required = expected.get("required_fact_ids")
    if (not all(isinstance(fid, str) and fid for fid in fact_ids)
            or not isinstance(required, list) or not all(isinstance(fid, str) and fid for fid in required)
            or len(required) != len(set(required)) or len(fact_ids) != len(set(fact_ids))
            or set(required) != set(fact_ids)):
        raise ValueError("Expected facts do not match gold evidence.")
    for fact in evidence:
        if (not isinstance(fact.get("target"), str) or not fact["target"].strip()
                or not isinstance(fact.get("slots"), dict) or "value" not in fact
                or not isinstance(fact.get("unit"), str)
                or not isinstance(fact.get("prerequisites"), list)
                or not all(isinstance(p, str) and p for p in fact["prerequisites"])
                or fact.get("content_hash") != chunks[fact["chunk_id"]]["content_hash"]):
            raise ValueError("Malformed expected fact or source hash mismatch.")
