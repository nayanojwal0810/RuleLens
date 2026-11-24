"""Offline checks for comparison cohorts, measured requests and nullable metrics."""

import json
from pathlib import Path
import socket

import pytest

from rulelens.cache_eval import (
    DEFAULT_CACHE_DEV_PAIRS_PATH, add_paired_savings, format_cache_eval_summary,
    percentile95, request_usage, run_cache_evaluation, summarize_cache_records,
    validate_cache_dev_dataset,
)
from rulelens.config import EMBEDDING_MODEL, GENERATION_MODEL
from rulelens.grounding import request_support
from rulelens.ingestion import get_corpus_snapshot
from rulelens.llm import AnswerResponse, AnswerStatus, ProviderTelemetrySnapshot, SanitizedProviderError
from rulelens.answering import GenerationResult


class ConstantEmbedder:
    model_name = EMBEDDING_MODEL
    normalize_embeddings = True

    def embed_query(self, text):
        return [1.0] + [0.0] * 383


class CountingLLM:
    model = GENERATION_MODEL

    def __init__(self, fail_on=None, tokens=13):
        self.invocations = 0
        self.fail_on = fail_on
        self.tokens = tokens

    def generate_answer(self, question, role, evidence_chunks, **kwargs):
        self.invocations += 1
        if self.invocations == self.fail_on:
            exc = SanitizedProviderError("timeout", "Provider timed out.")
            exc.request_telemetry = ProviderTelemetrySnapshot(provider_requests_attempted=1, has_incomplete_tokens=True)
            raise exc
        _, facts, missing = request_support(question, [c.to_dict() for c in evidence_chunks])
        assert not missing
        response = AnswerResponse(answer="\n".join(dict.fromkeys(f["quote"] for f in facts)),
            status=AnswerStatus.ANSWERED, citations=list(dict.fromkeys(f["chunk_id"] for f in facts)),
            referenced_rule_ids=[], reason="Source text.")
        return response, {"calls_made": 1, "total_tokens": self.tokens, "has_incomplete_tokens": self.tokens is None,
                          "latency_ms": 0.1}


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[1]


def pair(pair_id="one", source="What is the daily meal allowance?", candidate="What is the per diem meal allowance?", label="allow"):
    return {"pair_id": pair_id, "scenario": "paraphrase" if label == "allow" else "different_fact",
            "source_question": source, "candidate_question": candidate,
            "source_role": "employee", "candidate_role": "employee",
            "source_date": "2026-11-22", "candidate_date": "2026-11-22",
            "reuse_label": label, "rationale": "Controlled comparison fixture."}


def dataset(tmp_path, repo_root, pairs):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"schema_version": "2.0", "corpus_fingerprint": get_corpus_snapshot(repo_root).fingerprint,
                                "pairs": pairs}), encoding="utf-8")
    return path


def run(path, repo_root, llm=None, **kwargs):
    return run_cache_evaluation(path, llm_client=llm or CountingLLM(), embedder=ConstantEmbedder(),
                                execution_mode="offline_mock", base_dir=repo_root, **kwargs)


def test_development_labels_are_dynamic_and_separate_from_outcomes():
    info = validate_cache_dev_dataset(DEFAULT_CACHE_DEV_PAIRS_PATH)
    assert info["total_pairs"] == 52
    assert info["labels"] == {"allow": 16, "deny": 36}
    assert sum(info["scenarios"].values()) == 52


@pytest.mark.parametrize("field,value", [("candidate_role", "admin"), ("source_date", "2026-02-30"),
    ("candidate_question", ""), ("reuse_label", "hit")])
def test_invalid_dataset_fields_are_rejected(tmp_path, repo_root, field, value):
    row = pair()
    row[field] = value
    with pytest.raises(ValueError):
        validate_cache_dev_dataset(dataset(tmp_path, repo_root, [row]))


def test_duplicate_pairs_are_rejected(tmp_path, repo_root):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cache_dev_dataset(dataset(tmp_path, repo_root, [pair(), pair()]))


def test_same_eligible_cohort_and_seed_cost_separate(tmp_path, repo_root):
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm)
    baseline = result["baseline"]
    sim = result["comparisons"][0]["modes"]["similarity_only"]
    guarded = result["comparisons"][0]["modes"]["guarded"]
    assert llm.invocations == 2
    assert result["request_totals_including_seeding"]["mock_provider_calls"] == 2
    assert result["request_totals_including_seeding"]["simulated_tokens_used"] == 26
    assert result["seeds"][0]["usage"]["mock_provider_calls"] == 1
    assert baseline["mock_provider_calls"] == 1
    assert guarded["mock_provider_calls"] == sim["mock_provider_calls"] == 0
    assert baseline["eligible_pairs"] == guarded["eligible_pairs"] == sim["eligible_pairs"] == 1
    assert guarded["correct_hits"] == sim["correct_hits"] == 1
    assert guarded["mock_requests_saved"] == 1
    assert guarded["provider_requests_saved"] == 0
    assert guarded["known_tokens_saved"] is None
    assert guarded["simulated_tokens_saved"] == 13


def test_same_clause_wrong_caregiver_is_a_real_unsafe_hit(tmp_path, repo_root):
    row = pair(source="How many weeks of paid parental leave do primary caregivers receive?",
               candidate="How many weeks of paid parental leave do secondary caregivers receive?", label="deny")
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [row]), repo_root, llm)
    compared = result["comparisons"][0]["modes"]
    assert compared["similarity_only"]["unsafe_hits"] == 1
    assert compared["similarity_only"]["factual_error_hits"] == 1
    assert compared["guarded"]["unsafe_hits"] == 0
    assert compared["guarded"]["safe_reuse_precision"] is None
    assert compared["guarded"]["pair_records"][0]["cache_outcome"] == "rejected"
    assert compared["guarded"]["mock_provider_calls"] == 1
    assert llm.invocations == 3


def test_similarity_miss_runs_and_times_the_full_fallback(tmp_path, repo_root):
    class DistinctEmbedder(ConstantEmbedder):
        def embed_query(self, text):
            return [1.0, 0.0] + [0.0] * 382 if "daily" in text else [0.0, 1.0] + [0.0] * 382
    llm = CountingLLM()
    result = run_cache_evaluation(dataset(tmp_path, repo_root, [pair()]), llm_client=llm, embedder=DistinctEmbedder(),
                                  execution_mode="offline_mock", base_dir=repo_root)
    row = result["comparisons"][0]["modes"]["similarity_only"]["pair_records"][0]
    assert not row["cache_hit"] and row["route"] == "generated_verified"
    assert row["cache_outcome"] == "miss" and row["mock_provider_calls"] == 1
    assert row["simulated_tokens_used"] == 13 and row["e2e_latency_ms"] > 0
    assert llm.invocations == 4


def test_rule_source_is_excluded_in_every_mode(tmp_path, repo_root):
    row = pair(source="Can my manager approve a travel expense of INR 50,000?",
               candidate="Can my manager approve a travel expense of INR 75,000?", label="deny")
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [row]), repo_root, llm)
    assert llm.invocations == 0 and result["seeds"][0]["route"] == "rule"
    for mode in [result["baseline"], *result["comparisons"][0]["modes"].values()]:
        assert mode["eligible_pairs"] == 0
        assert mode["decisions"] == {"seed_ineligible": 1}
        assert mode["safe_reuse_precision"] is None and mode["valid_reuse_rate"] is None


def test_source_failure_is_an_error_not_a_safe_rejection(tmp_path, repo_root):
    llm = CountingLLM(fail_on=1)
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm)
    assert result["seeds"][0]["error"]
    assert result["seeds"][0]["usage"]["mock_provider_calls"] == 1
    for mode in [result["baseline"], *result["comparisons"][0]["modes"].values()]:
        assert mode["decisions"] == {"seed_error": 1} and mode["eligible_pairs"] == 0


def test_provider_error_stays_in_the_comparison_rows(tmp_path, repo_root):
    llm = CountingLLM(fail_on=2)
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm)
    assert result["baseline"]["decisions"] == {"error": 1}
    assert result["baseline"]["mock_provider_calls"] == 1
    guarded = result["comparisons"][0]["modes"]["guarded"]
    assert guarded["eligible_pairs"] == result["baseline"]["eligible_pairs"] == 1
    assert guarded["comparison_has_errors"] is True


def test_threshold_comparisons_reuse_one_measured_baseline(tmp_path, repo_root):
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm,
                 modes=("similarity_only", "guarded"), thresholds=(0.8, 0.95))
    assert llm.invocations == 2 and result["baseline"]["mock_provider_calls"] == 1
    assert len(result["comparisons"]) == 2
    for comparison in result["comparisons"]:
        assert comparison["modes"]["guarded"]["mock_requests_saved"] == 1


def metric_row(pid, label="allow", decision="missed_allow", hit=False, eligible=True, calls=1, tokens=10):
    return {"pair_id": pid, "scenario": "controlled", "reuse_label": label, "seed_eligible": eligible,
            "cache_hit": hit, "cache_outcome": "hit" if hit else "miss", "decision": decision,
            "candidate_executed": eligible, "e2e_latency_ms": 10 if eligible else None,
            "provider_calls": calls, "known_total_tokens": tokens, "has_incomplete_tokens": tokens is None,
            "mock_provider_calls": None, "simulated_tokens_used": None, "simulated_tokens_complete": None}


def test_denominators_and_scenario_totals_reconcile():
    rows = [metric_row("a", decision="correct_hit", hit=True), metric_row("b"),
            metric_row("c", label="deny", decision="unsafe_hit", hit=True),
            metric_row("d", eligible=False, decision="seed_ineligible")]
    summary = summarize_cache_records(rows, "live")
    assert summary["safe_reuse_precision"] == 0.5 and summary["valid_reuse_rate"] == 0.5
    assert summary["cache_hit_rate"] == 2 / 3 and summary["all_input_hit_rate"] == 0.5
    assert summary["eligible_coverage"] == 0.75
    assert sum(summary["decisions"].values()) == 4
    assert sum(summary["scenario_breakdown"]["controlled"]["decisions"].values()) == 4


def test_empty_hit_and_latency_denominators_are_undefined():
    summary = summarize_cache_records([], "live")
    assert summary["safe_reuse_precision"] is None and summary["valid_reuse_rate"] is None
    assert summary["latency_median_ms"] is None and percentile95([]) is None
    assert percentile95(list(range(1, 21))) == 19


def test_unknown_or_partial_usage_cannot_become_complete_totals():
    rows = [metric_row("a", calls=None, tokens=None), metric_row("b", calls=2, tokens=17)]
    rows[1]["has_incomplete_tokens"] = True
    summary = summarize_cache_records(rows, "live")
    assert summary["provider_calls"] is None and summary["known_partial_provider_calls"] == 2
    assert summary["known_total_tokens"] is None and summary["known_partial_known_total_tokens"] == 17
    assert summary["token_totals_complete"] is False


def test_paired_cost_differences_allow_negative_savings():
    baseline = summarize_cache_records([metric_row("a", calls=1, tokens=10)], "live")
    compared = summarize_cache_records([metric_row("a", calls=2, tokens=30)], "live")
    add_paired_savings(baseline, compared, "live")
    assert compared["provider_requests_saved"] == -1 and compared["known_tokens_saved"] == -20
    compared["pair_records"][0]["pair_id"] = "other"
    with pytest.raises(ValueError, match="identical"):
        add_paired_savings(baseline, compared, "live")


def test_mock_calls_and_simulated_tokens_never_become_live_usage():
    result = GenerationResult("question", "employee", "2026-11-22", "answer", "answered", "verified", [], [], [],
                              GENERATION_MODEL, "fingerprint", {}, 5, provider_calls_attempted=2, token_usage=27)
    usage = request_usage(result, "offline_mock")
    assert usage["provider_calls"] == 0 and usage["known_total_tokens"] is None
    assert usage["mock_provider_calls"] == 2 and usage["simulated_tokens_used"] == 27


def test_report_does_not_invent_perfect_precision(tmp_path, repo_root):
    row = pair(source="Can my manager approve a travel expense of INR 50,000?", label="deny")
    result = run(dataset(tmp_path, repo_root, [row]), repo_root)
    report = format_cache_eval_summary(result)
    assert "undefined" in report and "100%" not in report and "Zero Unsafe" not in report
    assert "offline_mock" in report and result["identity"]["real_embedding_model"] is False
    assert result["identity"]["dataset_sha256"] and result["identity"]["worktree_sha256"]


@pytest.mark.parametrize("thresholds", [(float("nan"),), (float("inf"),), (-0.1,), (0.5, 0.5)])
def test_invalid_thresholds_fail_before_requests(tmp_path, repo_root, thresholds):
    llm = CountingLLM()
    with pytest.raises(ValueError):
        run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm, thresholds=thresholds)
    assert llm.invocations == 0


def test_seed_write_failure_is_not_labelled_ineligible(tmp_path, repo_root):
    from qdrant_client import QdrantClient
    class FailingWriteClient(QdrantClient):
        def upsert(self, *args, **kwargs):
            raise RuntimeError("Do not expose this service message.")
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm,
                 client_factory=lambda: FailingWriteClient(":memory:"))
    assert llm.invocations == 1
    assert result["seeds"][0]["error"]
    assert result["seeds"][0]["reason"] == "seed_error"
    assert result["seeds"][0]["usage"]["mock_provider_calls"] == 1
    assert "Do not expose" not in json.dumps(result)
    assert result["baseline"]["decisions"] == {"seed_error": 1}


def test_candidate_cache_service_error_is_not_a_safe_rejection(tmp_path, repo_root):
    from qdrant_client import QdrantClient
    created = 0
    class FailingQueryClient(QdrantClient):
        def query_points(self, *args, **kwargs):
            raise RuntimeError("Private service details")
    def factory():
        nonlocal created
        created += 1
        return FailingQueryClient(":memory:") if created == 3 else QdrantClient(":memory:")
    llm = CountingLLM()
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, llm, modes=("guarded",), client_factory=factory)
    guarded = result["comparisons"][0]["modes"]["guarded"]
    assert guarded["eligible_pairs"] == result["baseline"]["eligible_pairs"] == 1
    assert guarded["decisions"] == {"error": 1}
    assert guarded["cache_outcomes"] == {"error": 1}
    assert guarded["mock_provider_calls"] == 1 and llm.invocations == 3
    assert "Private service details" not in json.dumps(result)


def test_candidate_setup_failure_keeps_the_cohort_and_has_no_savings(tmp_path, repo_root):
    from qdrant_client import QdrantClient
    created = 0
    def factory():
        nonlocal created
        created += 1
        if created == 3:
            raise RuntimeError("Unavailable")
        return QdrantClient(":memory:")
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root, modes=("guarded",), client_factory=factory)
    guarded = result["comparisons"][0]["modes"]["guarded"]
    assert guarded["eligible_pairs"] == 1 and guarded["decisions"] == {"error": 1}
    assert guarded["pair_records"][0]["candidate_executed"] is False
    assert guarded["paired_execution_complete"] is False
    assert guarded["mock_requests_saved"] is None


def test_changing_inputs_invalidates_comparison_provenance(tmp_path, repo_root, monkeypatch):
    import rulelens.cache_eval as evaluator
    original = evaluator.evaluation_identity
    calls = 0
    def changing_identity(*args, **kwargs):
        nonlocal calls
        calls += 1
        identity = original(*args, **kwargs)
        if calls == 2:
            identity["worktree_sha256"] = "changed"
        return identity
    monkeypatch.setattr(evaluator, "evaluation_identity", changing_identity)
    result = run(dataset(tmp_path, repo_root, [pair()]), repo_root)
    assert result["execution_state"] == "invalidated"
    assert result["identity"]["input_identity_unchanged"] is False
    assert "not comparable evidence" in format_cache_eval_summary(result)


def test_completed_month_is_different_from_annual_accrual(tmp_path, repo_root):
    row = pair(source="How many annual vacation days does a full-time employee accrue per year?",
               candidate="How many annual vacation days does a full-time employee accrue per completed month?", label="deny")
    result = run(dataset(tmp_path, repo_root, [row]), repo_root)
    compared = result["comparisons"][0]["modes"]
    assert compared["similarity_only"]["unsafe_hits"] == 1
    assert compared["guarded"]["unsafe_hits"] == 0
    assert compared["guarded"]["pair_records"][0]["cache_outcome"] == "rejected"
    assert compared["guarded"]["mock_provider_calls"] == 1


def test_context_partition_violation_is_distinct_from_wrong_facts(tmp_path, repo_root):
    row = pair(source="How many weeks of paid parental leave do primary caregivers receive?",
               candidate="How many weeks of paid parental leave do primary caregivers receive?", label="deny")
    row["candidate_date"] = "2026-11-23"
    result = run(dataset(tmp_path, repo_root, [row]), repo_root)
    sim = result["comparisons"][0]["modes"]["similarity_only"]
    assert sim["unsafe_hits"] == 1 and sim["mandatory_guard_violation_hits"] == 1
    assert sim["factual_error_hits"] == 0
    assert result["comparisons"][0]["modes"]["guarded"]["unsafe_hits"] == 0



def test_unversioned_source_has_explicit_provenance(repo_root, monkeypatch):
    import rulelens.cache_eval as evaluator
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")
    monkeypatch.setattr(evaluator.subprocess, "run", missing_git)
    identity = evaluator.evaluation_identity(repo_root, DEFAULT_CACHE_DEV_PAIRS_PATH, ConstantEmbedder(),
                                           CountingLLM(), "offline_mock", "bm25", "local")
    assert identity["git_commit"] is None
    assert identity["worktree_sha256"] and identity["dataset_sha256"]
