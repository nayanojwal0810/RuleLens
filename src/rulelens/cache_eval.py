"""Comparable cache workloads, explicit outcomes and request accounting."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Sequence, Union

from qdrant_client import QdrantClient, models

from rulelens.answering import GenerationResult, answer_question
from rulelens.cache import (
    CACHE_SCHEMA_VERSION, CacheRecord, ensure_cache_collection,
    lookup_cache, validate_cache_candidate,
)
from rulelens.config import CACHE_DEFAULT_SIMILARITY_THRESHOLD, QDRANT_CACHE_COLLECTION_NAME
from rulelens.grounding import GROUNDING_VERSION, validate_grounded_answer
from rulelens.ingestion import get_corpus_snapshot
from rulelens.llm import GroqClientWrapper
from rulelens.retrieval import ALLOWED_ROLES_SET, BGEEmbedder, HybridRetriever, parse_query_date
from rulelens.rules import extract_conditions_from_query, resolve_rules_for_evidence


DEFAULT_CACHE_DEV_PAIRS_PATH = Path("data/evaluation/cache_dev_pairs.json")
MODES = ("no_cache", "similarity_only", "guarded")
CACHE_ERRORS = {
    "embedding_error", "cache_service_unavailable", "cache_lookup_error", "cache_insert_error",
    "generation_configuration_mismatch", "embedding_configuration_mismatch", "invalid_similarity",
}


def validate_cache_dev_dataset(dataset_path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(dataset_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "2.0":
        raise ValueError("Expected cache question schema 2.0.")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Expected a non-empty pairs list.")
    seen = set()
    for row in pairs:
        if not isinstance(row, dict):
            raise ValueError("Each pair must be an object.")
        for field in ("pair_id", "scenario", "source_question", "candidate_question", "rationale"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing required field: {field}.")
        if row["pair_id"] in seen:
            raise ValueError("Duplicate pair_id.")
        seen.add(row["pair_id"])
        for side in ("source", "candidate"):
            if row.get(side + "_role") not in ALLOWED_ROLES_SET:
                raise ValueError(f"Invalid {side}_role.")
            date = row.get(side + "_date")
            if not isinstance(date, str) or len(date) != 10:
                raise ValueError(f"Invalid {side}_date.")
            parse_query_date(date)
        if row.get("reuse_label") not in ("allow", "deny"):
            raise ValueError("reuse_label must be allow or deny.")
    return {"total_pairs": len(pairs), "scenarios": dict(Counter(p["scenario"] for p in pairs)),
            "labels": dict(Counter(p["reuse_label"] for p in pairs))}


def request_usage(result: GenerationResult, execution_mode: str) -> Dict[str, Any]:
    calls = result.provider_calls_attempted
    tokens = result.token_usage
    complete = not result.has_incomplete_tokens and (calls == 0 or tokens is not None)
    if execution_mode == "offline_mock":
        return {"provider_calls": 0, "known_total_tokens": None,
                "has_incomplete_tokens": False, "mock_provider_calls": calls,
                "simulated_tokens_used": tokens, "simulated_tokens_complete": complete}
    if execution_mode != "live":
        raise ValueError("execution_mode must be offline_mock or live.")
    return {"provider_calls": calls, "known_total_tokens": 0 if calls == 0 else tokens,
            "has_incomplete_tokens": not complete, "mock_provider_calls": None,
            "simulated_tokens_used": None, "simulated_tokens_complete": None}


def cache_outcome(result: GenerationResult, mode: str) -> str:
    if mode == "no_cache":
        return "disabled"
    if result.cache_hit:
        return "hit"
    if result.cache_rejection_reason in CACHE_ERRORS or result.cache_insert_outcome == "error":
        return "error"
    if result.route != "generated_verified":
        return "bypassed"
    if result.cache_rejection_reason in ("cache_miss", "low_similarity"):
        return "miss"
    return "rejected" if result.cache_rejection_reason else "bypassed"


def prepare_cache_seed(pair: Dict[str, Any], *, retriever: HybridRetriever, embedder: BGEEmbedder,
                       llm_client: Any, execution_mode: str, answer_options: Dict[str, Any],
                       client_factory: Callable[[], QdrantClient]) -> Dict[str, Any]:
    client = None
    result = None
    started = time.perf_counter()
    try:
        client = client_factory()
        result = answer_question(pair["source_question"], pair["source_role"], pair["source_date"],
                                 retriever=retriever, llm_client=llm_client, embedder=embedder,
                                 cache_client=client, cache_enabled=True, **answer_options)
        elapsed = (time.perf_counter() - started) * 1000
        eligible = result.status == "answered" and result.route == "generated_verified"
        error = result.status == "operational_error" or cache_outcome(result, "guarded") == "error"
        point = None
        if eligible and result.cache_insert_outcome == "stored" and result.cache_entry_id:
            records = client.retrieve(QDRANT_CACHE_COLLECTION_NAME, [result.cache_entry_id], with_vectors=True)
            if len(records) == 1:
                saved = records[0]
                point = models.PointStruct(id=saved.id, vector=saved.vector, payload=saved.payload)
            else:
                error = True
        elif eligible and result.cache_insert_outcome != "ineligible":
            error = True
        reason = ("seed_error" if error else "eligible" if point else
                  "ineligible_meaning" if eligible else result.route)
        return {"pair_id": pair["pair_id"], "eligible": bool(point) and not error,
                "error": error, "reason": reason, "route": result.route, "status": result.status,
                "error_category": result.error_category, "cache_rejection_reason": result.cache_rejection_reason,
                "cache_insert_outcome": result.cache_insert_outcome, "cache_insert_reason": result.cache_insert_reason,
                "answer": result.answer, "citations": result.citations, "e2e_latency_ms": round(elapsed, 3),
                "usage": request_usage(result, execution_mode), "point": point}
    except Exception:
        usage = request_usage(result, execution_mode) if result else {
            "provider_calls": 0, "known_total_tokens": 0 if execution_mode == "live" else None,
            "mock_provider_calls": 0 if execution_mode == "offline_mock" else None,
            "simulated_tokens_used": None, "simulated_tokens_complete": False,
            "has_incomplete_tokens": False}
        return {"pair_id": pair["pair_id"], "eligible": False, "error": True, "reason": "seed_setup_error",
                "route": "error", "status": "operational_error", "usage": usage, "point": None,
                "e2e_latency_ms": round((time.perf_counter() - started) * 1000, 3)}
    finally:
        if client is not None:
            client.close()


def audit_cache_hit(pair: Dict[str, Any], result: GenerationResult, seed: Dict[str, Any],
                    retriever: HybridRetriever, canonical: Dict[str, Any]) -> Dict[str, Any]:
    ids = result.evidence_fingerprint.get("chunk_ids", [])
    chunks = [canonical[cid] for cid in ids if cid in canonical]
    conditions = extract_conditions_from_query(pair["candidate_question"])
    rules = resolve_rules_for_evidence(chunks, pair["candidate_role"], pair["candidate_date"],
                                      base_dir=retriever.base_dir, extracted_conditions=conditions)
    record = CacheRecord.from_payload(deepcopy(seed["point"].payload))
    valid, reason = validate_cache_candidate(record, pair["candidate_question"], pair["candidate_role"],
                                             pair["candidate_date"], result.corpus_fingerprint,
                                             chunks, conditions, rules, canonical)
    try:
        validate_grounded_answer(pair["candidate_question"], result.answer, result.citations, chunks)
        supported = True
    except ValueError:
        supported = False
    return {"mandatory_guards_passed": valid, "guard_violation": reason, "claims_supported": supported,
            "correct_reuse": pair["reuse_label"] == "allow" and valid and supported}


def evaluate_cache_pair(pair: Dict[str, Any], *, seed: Dict[str, Any], retriever: HybridRetriever,
                        embedder: BGEEmbedder, llm_client: Any, canonical_chunks_by_id: Dict[str, Any],
                        mode: str, similarity_threshold: float, execution_mode: str,
                        answer_options: Dict[str, Any],
                        client_factory: Callable[[], QdrantClient] = lambda: QdrantClient(":memory:")) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError("Unknown cache mode.")
    row = {"pair_id": pair["pair_id"], "scenario": pair["scenario"], "reuse_label": pair["reuse_label"],
           "mode": mode, "seed_eligible": seed["eligible"], "seed_reason": seed["reason"],
           "cache_hit": False, "cache_similarity": None, "cache_outcome": "bypassed",
           "provider_calls": 0, "known_total_tokens": 0 if execution_mode == "live" else None,
           "has_incomplete_tokens": False, "mock_provider_calls": 0 if execution_mode == "offline_mock" else None,
           "simulated_tokens_used": 0 if execution_mode == "offline_mock" else None,
           "simulated_tokens_complete": True if execution_mode == "offline_mock" else None,
           "e2e_latency_ms": None, "candidate_executed": False,
           "decision": "seed_error" if seed["error"] else "seed_ineligible"}
    if not seed["eligible"]:
        return row
    client = None
    try:
        client = client_factory()
        if mode != "no_cache":
            ensure_cache_collection(client)
            client.upsert(QDRANT_CACHE_COLLECTION_NAME, [deepcopy(seed["point"])])
        lookup = partial(lookup_cache, similarity_only=True) if mode == "similarity_only" else None
        started = time.perf_counter()
        result = answer_question(pair["candidate_question"], pair["candidate_role"], pair["candidate_date"],
                                 retriever=retriever, llm_client=llm_client, embedder=embedder,
                                 cache_client=client, cache_enabled=mode != "no_cache",
                                 cache_threshold=similarity_threshold, cache_lookup_fn=lookup, **answer_options)
        elapsed = (time.perf_counter() - started) * 1000
        outcome = cache_outcome(result, mode)
        row.update(request_usage(result, execution_mode))
        row.update({"candidate_executed": True, "e2e_latency_ms": round(elapsed, 3), "status": result.status,
                    "route": result.route, "answer": result.answer, "citations": result.citations,
                    "error_category": result.error_category, "cache_hit": result.cache_hit,
                    "cache_similarity": result.cache_similarity, "cache_outcome": outcome,
                    "rejection_reason": result.cache_rejection_reason,
                    "cache_insert_outcome": result.cache_insert_outcome, "cache_insert_reason": result.cache_insert_reason})
        if result.status == "operational_error" or outcome == "error":
            row["decision"] = "error"
        elif result.cache_hit:
            try:
                row.update(audit_cache_hit(pair, result, seed, retriever, canonical_chunks_by_id))
                row["decision"] = "correct_hit" if row["correct_reuse"] else "unsafe_hit"
            except (ValueError, KeyError, OSError):
                row["decision"] = "audit_error"
        elif mode == "no_cache":
            row["decision"] = "baseline"
        elif outcome == "bypassed":
            row["decision"] = "bypassed"
        else:
            row["decision"] = "safe_nonreuse" if pair["reuse_label"] == "deny" else "missed_allow"
        return row
    except Exception:
        row.update({"decision": "error", "cache_outcome": "error", "error_category": "evaluation_service_error"})
        return row
    finally:
        if client is not None:
            client.close()


def complete_total(rows: Sequence[Dict[str, Any]], field: str) -> Optional[int]:
    values = [r.get(field) for r in rows]
    return sum(values) if all(v is not None for v in values) else None


def percentile95(values: Sequence[float]) -> Optional[float]:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def summarize_cache_records(rows: Sequence[Dict[str, Any]], execution_mode: str) -> Dict[str, Any]:
    eligible = [r for r in rows if r["seed_eligible"]]
    opportunities = sum(r["reuse_label"] == "allow" for r in eligible)
    hits = [r for r in rows if r["cache_hit"]]
    correct = sum(r["decision"] == "correct_hit" for r in rows)
    unsafe = sum(r["decision"] == "unsafe_hit" for r in rows)
    audits_unknown = sum(r["decision"] not in ("correct_hit", "unsafe_hit") for r in hits)
    latencies = [r["e2e_latency_ms"] for r in eligible if r["candidate_executed"]]
    token_complete = (all(not r["has_incomplete_tokens"] and (r["provider_calls"] == 0 or r["known_total_tokens"] is not None)
                          for r in rows) if execution_mode == "live" else
                      all(r["simulated_tokens_complete"] and (r["mock_provider_calls"] == 0 or r["simulated_tokens_used"] is not None)
                          for r in rows))
    result = {"total_pairs": len(rows), "eligible_pairs": len(eligible), "eligible_allow_opportunities": opportunities,
              "eligible_coverage": len(eligible) / len(rows) if rows else None,
              "total_hits": len(hits), "correct_hits": correct, "unsafe_hits": unsafe,
              "unresolved_hit_audits": audits_unknown,
              "safe_reuse_precision": correct / len(hits) if hits and not audits_unknown else None,
              "valid_reuse_rate": correct / opportunities if opportunities else None,
              "cache_hit_rate": len(hits) / len(eligible) if eligible else None,
              "all_input_hit_rate": len(hits) / len(rows) if rows else None,
              "decisions": dict(Counter(r["decision"] for r in rows)),
              "cache_outcomes": dict(Counter(r["cache_outcome"] for r in rows)),
              "scenario_breakdown": {}, "token_totals_complete": token_complete,
              "latency_sample_count": len(latencies),
              "latency_median_ms": statistics.median(latencies) if latencies else None,
              "latency_p95_ms": percentile95(latencies),
              "factual_error_hits": sum(r.get("claims_supported") is False for r in hits),
              "mandatory_guard_violation_hits": sum(r.get("mandatory_guards_passed") is False for r in hits),
              "pair_records": list(rows)}
    for field in ("provider_calls", "known_total_tokens", "mock_provider_calls", "simulated_tokens_used"):
        result[field] = complete_total(rows, field)
        result["known_partial_" + field] = sum(r[field] for r in rows if r.get(field) is not None)
        result["unknown_" + field + "_rows"] = sum(r.get(field) is None for r in rows)
    if execution_mode == "live" and not token_complete:
        result["known_total_tokens"] = None
    if execution_mode == "offline_mock" and not token_complete:
        result["simulated_tokens_used"] = None
    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        result["scenario_breakdown"][scenario] = {
            "total": len(subset), "eligible": sum(r["seed_eligible"] for r in subset),
            "decisions": dict(Counter(r["decision"] for r in subset)),
            "cache_outcomes": dict(Counter(r["cache_outcome"] for r in subset))}
    return result


def add_paired_savings(baseline: Dict[str, Any], compared: Dict[str, Any], execution_mode: str) -> None:
    left = baseline["pair_records"]
    right = compared["pair_records"]
    if [(r["pair_id"], r["seed_eligible"]) for r in left] != [(r["pair_id"], r["seed_eligible"]) for r in right]:
        raise ValueError("Savings require identical ordered cohorts.")
    paired_complete = any(r["seed_eligible"] for r in left) and all(
        not r["seed_eligible"] or r["candidate_executed"] for r in left + right)
    compared["paired_execution_complete"] = paired_complete
    compared["comparison_has_errors"] = any(r["decision"] in ("error", "audit_error", "seed_error") for r in left + right)
    for field, output in (("provider_calls", "provider_requests_saved"),
                          ("known_total_tokens", "known_tokens_saved"),
                          ("mock_provider_calls", "mock_requests_saved"),
                          ("simulated_tokens_used", "simulated_tokens_saved")):
        a, b = baseline.get(field), compared.get(field)
        compared[output] = a - b if paired_complete and a is not None and b is not None else None


def evaluation_identity(base: Path, dataset_path: Path, embedder: BGEEmbedder, llm_client: Any,
                        execution_mode: str, retrieval_mode: str, dense_backend: str) -> Dict[str, Any]:
    snapshot = get_corpus_snapshot(base)
    commit = None
    try:
        repository = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=base,
                                    capture_output=True, text=True, check=True).stdout.strip()
        if Path(repository).resolve() != base.resolve():
            raise ValueError("The data directory is not a repository root.")
        files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                               cwd=base, capture_output=True, text=True, check=True).stdout.splitlines()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        paths = list((base / "src").rglob("*.py")) + list((base / "tests").rglob("*.py"))
        paths += [p for p in (base / "data").rglob("*") if p.is_file()]
        paths += list((base / ".github/workflows").glob("*.yml"))
        paths += [base / name for name in ("README.md", "pyproject.toml", "requirements_test.txt", ".gitignore", ".env.example")]
        files = [p.relative_to(base).as_posix() for p in paths if p.is_file()]
    identities = []
    for name in sorted(set(files)):
        path = base / name
        if path.is_file() and (not path.name.startswith(".env") or path.name == ".env.example"):
            identities.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    dependencies = {}
    for package in ("qdrant-client", "groq", "sentence-transformers", "pydantic", "rank-bm25", "pymupdf"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {"execution_mode": execution_mode, "cache_backend": "isolated_qdrant_memory",
            "retrieval_mode": retrieval_mode, "dense_backend": dense_backend,
            "embedding_implementation": type(embedder).__module__ + "." + type(embedder).__qualname__,
            "embedding_model": getattr(embedder, "model_name", None),
            "real_embedding_model": isinstance(embedder, BGEEmbedder),
            "generation_model": getattr(llm_client, "model", None),
            "corpus_fingerprint": snapshot.fingerprint, "cache_schema": CACHE_SCHEMA_VERSION,
            "grounding_schema": GROUNDING_VERSION,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "git_commit": commit,
            "worktree_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
            "python": platform.python_version(), "platform": platform.system(), "dependencies": dependencies,
            "latency_scope": "complete candidate request, including retrieval, guards and miss regeneration",
            "seed_cost_included_in_candidate_totals": False,
            "warmup": "shared retriever and embedder; no synthetic latency subtraction"}


def run_cache_evaluation(dataset_path: Union[Path, str] = DEFAULT_CACHE_DEV_PAIRS_PATH, *,
                         llm_client: Any, embedder: BGEEmbedder, execution_mode: str,
                         modes: Sequence[str] = MODES, similarity_threshold: float = CACHE_DEFAULT_SIMILARITY_THRESHOLD,
                         thresholds: Optional[Sequence[float]] = None, base_dir: Optional[Union[Path, str]] = None,
                         retriever: Optional[HybridRetriever] = None, retrieval_mode: str = "bm25",
                         dense_backend: str = "local", policy_client: Any = None,
                         pair_limit: Optional[int] = None,
                         client_factory: Callable[[], QdrantClient] = lambda: QdrantClient(":memory:")) -> Dict[str, Any]:
    if execution_mode not in ("offline_mock", "live") or not modes or any(m not in MODES for m in modes):
        raise ValueError("Invalid execution mode or cache modes.")
    if execution_mode == "offline_mock" and isinstance(llm_client, GroqClientWrapper):
        raise ValueError("A live provider cannot be labelled offline_mock.")
    values = list(thresholds) if thresholds is not None else [similarity_threshold]
    if not values or len(set(values)) != len(values) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in values):
        raise ValueError("Thresholds must be unique finite values between zero and one.")
    if pair_limit is not None and (isinstance(pair_limit, bool) or not isinstance(pair_limit, int) or pair_limit < 1):
        raise ValueError("pair_limit must be positive.")
    validation = validate_cache_dev_dataset(dataset_path)
    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    dataset_path = Path(dataset_path).resolve()
    pairs = json.loads(dataset_path.read_text(encoding="utf-8"))["pairs"][:pair_limit]
    snapshot = get_corpus_snapshot(base)
    declared = json.loads(dataset_path.read_text(encoding="utf-8")).get("corpus_fingerprint")
    if declared and declared != snapshot.fingerprint:
        raise ValueError("Question-set corpus fingerprint mismatch.")
    canonical = {c["chunk_id"]: c for c in snapshot.chunks}
    active_retriever = retriever or HybridRetriever(base_dir=base, embedder=embedder)
    options = {"base_dir": base, "mode": retrieval_mode, "dense_backend": dense_backend,
               "qdrant_client": policy_client}
    identity = evaluation_identity(base, dataset_path, embedder, llm_client, execution_mode, retrieval_mode, dense_backend)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    seeds = [prepare_cache_seed(p, retriever=active_retriever, embedder=embedder, llm_client=llm_client,
                                execution_mode=execution_mode, answer_options=options,
                                client_factory=client_factory) for p in pairs]
    common = dict(retriever=active_retriever, embedder=embedder, llm_client=llm_client,
                  canonical_chunks_by_id=canonical, execution_mode=execution_mode,
                  answer_options=options, client_factory=client_factory)
    baseline_rows = [evaluate_cache_pair(p, seed=s, mode="no_cache", similarity_threshold=values[0], **common)
                     for p, s in zip(pairs, seeds)]
    baseline = summarize_cache_records(baseline_rows, execution_mode)
    comparisons = []
    for threshold in values:
        measured = {}
        for mode in dict.fromkeys(modes):
            if mode == "no_cache":
                continue
            rows = [evaluate_cache_pair(p, seed=s, mode=mode, similarity_threshold=threshold, **common)
                    for p, s in zip(pairs, seeds)]
            measured[mode] = summarize_cache_records(rows, execution_mode)
            add_paired_savings(baseline, measured[mode], execution_mode)
        comparisons.append({"threshold": threshold, "modes": measured})
    final_identity = evaluation_identity(base, dataset_path, embedder, llm_client, execution_mode, retrieval_mode, dense_backend)
    identity["input_identity_unchanged"] = all(identity[k] == final_identity[k] for k in
        ("git_commit", "worktree_sha256", "dataset_sha256", "corpus_fingerprint"))
    identity["claim_audit"] = "mandatory guards and exact source spans; independent human correctness review is separate"
    usages = [s["usage"] for s in seeds] + baseline_rows
    for comparison in comparisons:
        for measured in comparison["modes"].values():
            usages.extend(measured["pair_records"])
    request_totals = {}
    for field in ("provider_calls", "known_total_tokens", "mock_provider_calls", "simulated_tokens_used"):
        request_totals[field] = complete_total(usages, field)
        request_totals["known_partial_" + field] = sum(r[field] for r in usages if r.get(field) is not None)
    request_totals["has_incomplete_tokens"] = any(r["has_incomplete_tokens"] for r in usages)
    if request_totals["has_incomplete_tokens"]:
        request_totals["known_total_tokens"] = None
    if execution_mode == "offline_mock" and not all(r["simulated_tokens_complete"] for r in usages):
        request_totals["simulated_tokens_used"] = None
    has_errors = any(s["error"] for s in seeds) or any(
        r["decision"] in ("error", "audit_error") for r in usages if "decision" in r)
    state = "invalidated" if not identity["input_identity_unchanged"] else "completed_with_errors" if has_errors else "completed"
    return {"schema_version": "2.0", "timestamp": started_at, "execution_state": state, "identity": identity,
            "dataset_validation": validation, "evaluated_pairs": len(pairs), "pair_limit": pair_limit,
            "request_totals_including_seeding": request_totals,
            "seeds": [{k: v for k, v in s.items() if k != "point"} for s in seeds],
            "baseline": baseline, "comparisons": comparisons}


def format_cache_eval_summary(data: Dict[str, Any]) -> str:
    identity = data["identity"]
    lines = ["# Cache evaluation", "", f"Execution: {identity['execution_mode']}. Cache backend: {identity['cache_backend']}.",
             "Seeding is reported separately. Latency covers complete candidate requests.",
             "Offline mock timings and simulated tokens are not live API performance.", ""]
    if not identity.get("input_identity_unchanged", True):
        lines.append("Input identity changed during execution; this run is not comparable evidence.")
    def display(value):
        return "undefined" if value is None else str(round(value, 4)) if isinstance(value, float) else str(value)
    for comparison in data["comparisons"]:
        lines.extend([f"Threshold: {comparison['threshold']}", "", "| Mode | Eligible / input | Correct / all hits | Unsafe hits | Precision | Valid reuse | Errors |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for mode, result in [("no_cache", data["baseline"]), *comparison["modes"].items()]:
            errors = sum(result["decisions"].get(k, 0) for k in ("error", "audit_error", "seed_error"))
            lines.append(f"| {mode} | {result['eligible_pairs']} / {result['total_pairs']} | {result['correct_hits']} / {result['total_hits']} | {result['unsafe_hits']} | {display(result['safe_reuse_precision'])} | {display(result['valid_reuse_rate'])} | {errors} |")
        lines.append("")
    return "\n".join(lines)
