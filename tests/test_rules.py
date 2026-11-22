"""Offline deterministic tests for RuleLens Stage E structured rule resolution.

Verifies:
- Loading all 32 canonical rules and linking to policy chunk IDs.
- Boundary conditions for all approved thresholds (historical and current).
- Exact operator semantics (<, <=, >, >=, ==).
- Type safety, unit matching, and fail-closed error handling.
- Eligibility filtering by role and query date.
- Natural language condition extraction.
"""

from __future__ import annotations

import datetime
import socket
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rulelens.rules import (
    RuleEvaluationResult,
    RuleRecord,
    evaluate_rule_condition,
    extract_conditions_from_query,
    is_rule_active,
    load_rules,
    resolve_rules_for_evidence,
)


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that unit tests make zero network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline unit tests")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def all_rules(repo_root: Path) -> List[RuleRecord]:
    return load_rules(base_dir=repo_root, reload=True)


# ==============================================================================
# 1. Rule Loading and Structure
# ==============================================================================


def test_load_all_32_canonical_rules(all_rules: List[RuleRecord]):
    """load_rules must return exactly 32 canonical rules with valid chunk_ids."""
    assert len(all_rules) == 32
    for r in all_rules:
        assert r.rule_id.startswith("R-")
        assert r.doc_id.startswith("DOC-")
        assert r.chunk_id == f"{r.doc_id}_{r.clause_id}"
        assert r.operator in ("<", "<=", ">", ">=", "==")
        assert r.unit
        assert r.outcome
        assert r.effective_from <= r.effective_to


# ==============================================================================
# 2. Mandatory Boundary Tests (Section 10)
# ==============================================================================


def test_travel_approval_historical_boundary(all_rules: List[RuleRecord]):
    """Travel approval v1 (on 2026-06-30): INR 40,000 standard vs 40,001 additional."""
    r1 = next(r for r in all_rules if r.rule_id == "R-TRV-001")  # <= 40000
    r2 = next(r for r in all_rules if r.rule_id == "R-TRV-002")  # > 40000

    q_date = datetime.date(2026, 6, 30)
    assert is_rule_active(r1, q_date)
    assert is_rule_active(r2, q_date)

    # 40,000
    assert evaluate_rule_condition(r1, 40000, "INR") is True
    assert evaluate_rule_condition(r2, 40000, "INR") is False

    # 40,001
    assert evaluate_rule_condition(r1, 40001, "INR") is False
    assert evaluate_rule_condition(r2, 40001, "INR") is True


def test_travel_approval_revised_boundary(all_rules: List[RuleRecord]):
    """Travel approval v2 (on 2026-07-01): INR 50,000 standard vs 50,001 additional."""
    r3 = next(r for r in all_rules if r.rule_id == "R-TRV-003")  # <= 50000
    r4 = next(r for r in all_rules if r.rule_id == "R-TRV-004")  # > 50000

    q_date = datetime.date(2026, 7, 1)
    assert is_rule_active(r3, q_date)
    assert is_rule_active(r4, q_date)

    # Historical rules must be inactive
    r1 = next(r for r in all_rules if r.rule_id == "R-TRV-001")
    assert is_rule_active(r1, q_date) is False

    # 50,000
    assert evaluate_rule_condition(r3, 50000, "INR") is True
    assert evaluate_rule_condition(r4, 50000, "INR") is False

    # 50,001
    assert evaluate_rule_condition(r3, 50001, "INR") is False
    assert evaluate_rule_condition(r4, 50001, "INR") is True


def test_finance_receipt_boundary(all_rules: List[RuleRecord]):
    """Finance incidental receipts: INR 1,000 no receipt vs 1,001 itemized receipt."""
    r_no_rcpt = next(r for r in all_rules if r.rule_id == "R-FIN-003")  # <= 1000
    r_itemized = next(r for r in all_rules if r.rule_id == "R-FIN-004")  # > 1000

    # 1,000
    assert evaluate_rule_condition(r_no_rcpt, 1000, "INR") is True
    assert evaluate_rule_condition(r_itemized, 1000, "INR") is False

    # 1,001
    assert evaluate_rule_condition(r_no_rcpt, 1001, "INR") is False
    assert evaluate_rule_condition(r_itemized, 1001, "INR") is True


def test_travel_advance_booking_boundary(all_rules: List[RuleRecord]):
    """Travel booking advance: 14 days standard booking vs 13 days expedited approval."""
    r_std = next(r for r in all_rules if r.rule_id == "R-TRV-005")  # >= 14
    r_exp = next(r for r in all_rules if r.rule_id == "R-TRV-006")  # < 14

    # 14 days
    assert evaluate_rule_condition(r_std, 14, "days") is True
    assert evaluate_rule_condition(r_exp, 14, "days") is False

    # 13 days
    assert evaluate_rule_condition(r_std, 13, "days") is False
    assert evaluate_rule_condition(r_exp, 13, "days") is True


def test_security_classification_tier_equality(all_rules: List[RuleRecord]):
    """Security classification tier == comparison for Confidential and Restricted."""
    r_conf = next(r for r in all_rules if r.rule_id == "R-SEC-001")  # == Confidential
    r_restr = next(r for r in all_rules if r.rule_id == "R-SEC-002")  # == Restricted

    assert evaluate_rule_condition(r_conf, "Confidential", "tier") is True
    assert evaluate_rule_condition(r_conf, "confidential", "tier") is True
    assert evaluate_rule_condition(r_conf, "Restricted", "tier") is False

    assert evaluate_rule_condition(r_restr, "Restricted", "tier") is True
    assert evaluate_rule_condition(r_restr, "Public", "tier") is False


# ==============================================================================
# 3. Unit and Type Safety
# ==============================================================================


def test_rule_unit_mismatch_raises_error(all_rules: List[RuleRecord]):
    """Evaluating rule with incompatible unit must raise ValueError."""
    r_inr = next(r for r in all_rules if r.rule_id == "R-TRV-001")  # INR
    with pytest.raises(ValueError, match="Incompatible unit 'days'"):
        evaluate_rule_condition(r_inr, 40000, "days")


def test_rule_invalid_numeric_type_raises_error(all_rules: List[RuleRecord]):
    """Evaluating numeric rule with unconvertible string must raise TypeError."""
    r_num = next(r for r in all_rules if r.rule_id == "R-TRV-001")
    with pytest.raises(TypeError, match="Cannot evaluate operator"):
        evaluate_rule_condition(r_num, "not_a_number", "INR")


# ==============================================================================
# 4. Evidence Rule Resolution
# ==============================================================================


def test_resolve_rules_for_evidence_filters_by_chunk_and_date(repo_root: Path):
    """resolve_rules_for_evidence must only link rules for authorized chunks active on date."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-001_TRV-C04", "doc_id": "DOC-TRV-001", "clause_id": "TRV-C04"},
        {"chunk_id": "DOC-FIN-002_FIN-R03", "doc_id": "DOC-FIN-002", "clause_id": "FIN-R03"},
    ]

    # Date when TRV-001 is active (2026-06-30)
    res_hist = resolve_rules_for_evidence(
        authorized_chunks=mock_chunks,
        role="employee",
        query_date="2026-06-30",
        base_dir=repo_root,
        extracted_conditions=[{"field": "expense_amount", "value": 35000, "unit": "INR"}],
    )
    rule_ids = {r.rule.rule_id for r in res_hist}
    assert "R-TRV-001" in rule_ids
    assert "R-TRV-002" in rule_ids
    # R-TRV-003 is v2 and not active on 2026-06-30
    assert "R-TRV-003" not in rule_ids

    # Check evaluation result
    eval_r1 = next(r for r in res_hist if r.rule.rule_id == "R-TRV-001")
    assert eval_r1.passed is True
    assert eval_r1.outcome == "standard_approval"


# ==============================================================================
# 5. Natural Language Condition Extraction
# ==============================================================================


def test_extract_conditions_from_query():
    """Conservative condition extraction from queries with currency, days, tiers."""
    # Currency
    c1 = extract_conditions_from_query("Can I get reimbursed for a flight ticket costing INR 45,000?")
    assert any(c["field"] == "expense_amount" and c["value"] == 45000.0 and c["unit"] == "INR" for c in c1)

    # Days
    c2 = extract_conditions_from_query("Is standard booking permitted if I book 14 days in advance?")
    assert any(c["field"] == "advance_booking_days" and c["value"] == 14 and c["unit"] == "days" for c in c2)

    # Tier
    c3 = extract_conditions_from_query("What encryption standard is required for Confidential data?")
    assert any(c["field"] == "data_classification_tier" and c["value"] == "Confidential" for c in c3)

    # Ambiguous / query without conditions
    c4 = extract_conditions_from_query("What is the general vacation policy?")
    assert len(c4) == 0


def test_unrelated_numeric_condition_not_cross_assigned():
    """Bug regression: One currency amount must NOT be assigned to all unrelated condition fields."""
    query = "Is an incidental receipt required for a dinner expense of INR 800?"
    conds = extract_conditions_from_query(query)
    fields = [c["field"] for c in conds]
    assert "receipt_amount" in fields
    # Must NOT assign to unrelated fields
    assert "monthly_spend" not in fields
    assert "exception_amount" not in fields
    assert "annual_tuition_claim" not in fields
    assert "order_amount" not in fields
    assert "emergency_purchase_amount" not in fields


def test_failed_rule_not_treated_as_applicable(repo_root: Path):
    """Bug regression: Failed rule conditions must NOT be reported as satisfied outcomes."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    # TRV-C04 has R-TRV-003 (<= 50,000 standard_approval) and R-TRV-004 (> 50,000 additional_approval)
    # Query with 60,000 INR
    res = resolve_rules_for_evidence(
        authorized_chunks=mock_chunks,
        role="employee",
        query_date="2026-11-22",
        base_dir=repo_root,
        extracted_conditions=[{"field": "expense_amount", "value": 60000, "unit": "INR"}],
    )
    r_pass = next(r for r in res if r.rule.rule_id == "R-TRV-004")
    r_fail = next(r for r in res if r.rule.rule_id == "R-TRV-003")

    assert r_pass.passed is True
    assert r_pass.outcome == "additional_approval"
    assert r_pass.is_error is False
    assert r_pass.error_type is None

    # Failed rule must NOT report standard_approval as satisfied!
    assert r_fail.passed is False
    assert r_fail.outcome != "standard_approval"
    assert r_fail.is_error is False
    assert r_fail.error_type is None
    assert "NOT satisfied" in r_fail.explanation


def test_rule_unit_mismatch_marks_is_error(repo_root: Path):
    """When condition input has incompatible unit, resolve_rules_for_evidence marks is_error=True."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    # R-TRV-003 requires unit INR, input supplies days
    res = resolve_rules_for_evidence(
        authorized_chunks=mock_chunks,
        role="employee",
        query_date="2026-11-22",
        base_dir=repo_root,
        extracted_conditions=[{"field": "expense_amount", "value": 50000, "unit": "days"}],
    )
    r_err = next(r for r in res if r.rule.rule_id == "R-TRV-003")
    assert r_err.passed is False
    assert r_err.is_error is True
    assert r_err.error_type == "invalid_unit"
    assert r_err.outcome == "invalid_unit"
    assert "unit error" in r_err.explanation.lower()


def test_rule_type_mismatch_marks_is_error(repo_root: Path):
    """When condition input has unconvertible type, resolve_rules_for_evidence marks is_error=True."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    # R-TRV-003 requires numeric value, input supplies unconvertible string
    res = resolve_rules_for_evidence(
        authorized_chunks=mock_chunks,
        role="employee",
        query_date="2026-11-22",
        base_dir=repo_root,
        extracted_conditions=[{"field": "expense_amount", "value": "unconvertible_str", "unit": "INR"}],
    )
    r_err = next(r for r in res if r.rule.rule_id == "R-TRV-003")
    assert r_err.passed is False
    assert r_err.is_error is True
    assert r_err.error_type == "invalid_type"
    assert r_err.outcome == "invalid_type"
    assert "type error" in r_err.explanation.lower()


def test_rules_cache_isolation_by_base_dir(repo_root: Path, tmp_path: Path):
    """Bug regression: _RULES_CACHE must be isolated by base_dir to avoid cross-corpus pollution."""
    rules_orig = load_rules(base_dir=repo_root, reload=True)
    assert len(rules_orig) == 32

    # A different directory without data/processed/rules.jsonl must fail and not reuse cache
    with pytest.raises(FileNotFoundError):
        load_rules(base_dir=tmp_path, reload=False)
