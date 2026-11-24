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
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rulelens.rules import (
    ConditionExtractionResult,
    RuleEvaluationResult,
    RuleRecord,
    clear_rules_cache,
    evaluate_authoritative_rule_decision,
    evaluate_rule_condition,
    extract_condition_result,
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
    assert r_err.error_type in ("invalid_unit", "incompatible_unit")
    assert r_err.outcome in ("invalid_unit", "incompatible_unit")
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


# ==============================================================================
# 5. Decimal Precision and Ambiguous Condition Tests
# ==============================================================================


def test_decimal_currency_precision_preserved(all_rules: List[RuleRecord]):
    """Currency conditions preserve exact Decimal precision without float conversion loss."""
    r3 = next(r for r in all_rules if r.rule_id == "R-TRV-003")  # <= 50000 INR

    # Exact decimal threshold check
    val_below = Decimal("50000.00")
    val_above = Decimal("50000.01")

    assert evaluate_rule_condition(r3, val_below, "INR") is True
    assert evaluate_rule_condition(r3, val_above, "INR") is False


def test_extract_condition_result_detects_ambiguous_amounts():
    """Queries mentioning multiple conflicting amounts or 'or' conjunctions emit ambiguous status."""
    ambig_query = "What happens if travel expenses are INR 40,000 or INR 50,000 under the policy?"
    cond_res = extract_condition_result(ambig_query)

    assert cond_res.status == "ambiguous"
    assert any(c.get("field") == "expense_amount" for c in cond_res.conditions)


def test_ambiguous_condition_marks_rule_evaluation_error(repo_root: Path):
    """When condition extraction detects ambiguity, resolve_rules_for_evidence marks is_error=True."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    query = "What approval is required if expenses are INR 40,000 or INR 60,000?"
    conds = extract_conditions_from_query(query)

    res = resolve_rules_for_evidence(
        authorized_chunks=mock_chunks,
        role="employee",
        query_date="2026-11-22",
        base_dir=repo_root,
        extracted_conditions=conds,
    )

    r_err = next(r for r in res if r.rule.rule_id in ("R-TRV-003", "R-TRV-004"))
    assert r_err.passed is False
    assert r_err.is_error is True
    assert r_err.error_type == "ambiguous_condition"
    assert "ambiguous" in r_err.explanation.lower()


# ==============================================================================
# 6. Comprehensive Numerical Unit Ambiguity Tests (Correction C)
# ==============================================================================


@pytest.mark.parametrize(
    "query,expected_field",
    [
        ("Can we book flights with 14 days or 10 days advance notice?", "advance_booking_days"),
        ("If reimbursement submission is delayed by 30 days and 45 days, what applies?", "submission_delay_days"),
        ("Can a contract be for 12 months versus 18 months without legal signoff?", "contract_duration_months"),
        ("Is a P1 response time of 1 hour or 2 hours compliant with SLA?", "incident_severity_p1_response_hours"),
        ("Does home internet of 50 Mbps or 30 Mbps qualify for remote work?", "internet_speed_mbps"),
        ("Is the project data classified as Confidential versus Restricted tier?", "data_classification_tier"),
    ],
)
def test_ambiguity_detection_across_all_units(query: str, expected_field: str):
    """Multiple distinct values for any unit (days, months, hours, mbps, tiers) trigger ambiguous status."""
    cond_res = extract_condition_result(query)
    assert cond_res.status == "ambiguous"
    ambig_cond = next((c for c in cond_res.conditions if c.get("field") == expected_field), None)
    assert ambig_cond is not None
    assert ambig_cond.get("is_ambiguous") is True


def test_repeated_identical_values_deduplicated():
    """Repeated identical values for amounts or days are deduplicated and valid."""
    # Repeated amount
    q_amount = "Can a manager approve an expense of INR 50,000 or INR 50,000?"
    res_amt = extract_condition_result(q_amount)
    assert res_amt.status == "valid"
    assert len([c for c in res_amt.conditions if c["field"] == "expense_amount"]) == 1
    assert res_amt.conditions[0]["value"] == Decimal("50000")

    # Repeated days
    q_days = "If booked with 14 days or 14 days advance notice, is approval required?"
    res_days = extract_condition_result(q_days)
    assert res_days.status == "valid"
    assert len([c for c in res_days.conditions if c["field"] == "advance_booking_days"]) == 1
    assert res_days.conditions[0]["value"] == 14


def test_valid_cross_document_distinct_fields_not_ambiguous():
    """Valid cross-document questions referring to different fields must not trigger ambiguity."""
    cross_q = "If a flight is booked 14 days in advance for an expense of INR 50,000, what approval is needed?"
    res = extract_condition_result(cross_q)
    assert res.status == "valid"
    fields = {c["field"] for c in res.conditions}
    assert "advance_booking_days" in fields
    assert "expense_amount" in fields
    assert not any(c.get("is_ambiguous") for c in res.conditions)


# ==============================================================================
# 7. Authoritative Rule Decision Tests (Correction B)
# ==============================================================================


from rulelens.rules import evaluate_authoritative_rule_decision


def test_authoritative_decision_travel_v1_boundaries(repo_root: Path):
    """INR 40,000 standard vs 40,001 additional under Travel Policy v1."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-001_TRV-C04", "doc_id": "DOC-TRV-001", "clause_id": "TRV-C04"},
    ]
    # 40,000 on 2026-03-15
    q_40k = "Can a direct manager approve INR 40,000 under Travel Policy v1?"
    conds_40k = extract_conditions_from_query(q_40k)
    rules_40k = resolve_rules_for_evidence(mock_chunks, "manager", "2026-03-15", repo_root, conds_40k)
    dec_40k = evaluate_authoritative_rule_decision(q_40k, mock_chunks, rules_40k, conds_40k)
    assert dec_40k.is_authoritative is True
    assert dec_40k.rule_id == "R-TRV-001"
    assert dec_40k.outcome == "standard_approval"
    assert "Standard manager approval is permitted" in dec_40k.answer

    # 40,001 on 2026-03-15
    q_40k1 = "Can a direct manager approve INR 40,001 under Travel Policy v1?"
    conds_40k1 = extract_conditions_from_query(q_40k1)
    rules_40k1 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-03-15", repo_root, conds_40k1)
    dec_40k1 = evaluate_authoritative_rule_decision(q_40k1, mock_chunks, rules_40k1, conds_40k1)
    assert dec_40k1.is_authoritative is True
    assert dec_40k1.rule_id == "R-TRV-002"
    assert dec_40k1.outcome == "additional_approval"
    assert "Additional approval is required" in dec_40k1.answer


def test_authoritative_decision_travel_v2_boundaries(repo_root: Path):
    """INR 50,000 standard vs 50,001 additional under Travel Policy v2."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    # 50,000 on 2026-08-15
    q_50k = "Can a direct manager approve an expense of INR 50,000 under Travel Policy v2?"
    conds_50k = extract_conditions_from_query(q_50k)
    rules_50k = resolve_rules_for_evidence(mock_chunks, "manager", "2026-08-15", repo_root, conds_50k)
    dec_50k = evaluate_authoritative_rule_decision(q_50k, mock_chunks, rules_50k, conds_50k)
    assert dec_50k.is_authoritative is True
    assert dec_50k.rule_id == "R-TRV-003"
    assert dec_50k.outcome == "standard_approval"

    # 50,001 on 2026-08-15
    q_50k1 = "Can a direct manager approve an expense of INR 50,001 under Travel Policy v2?"
    conds_50k1 = extract_conditions_from_query(q_50k1)
    rules_50k1 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-08-15", repo_root, conds_50k1)
    dec_50k1 = evaluate_authoritative_rule_decision(q_50k1, mock_chunks, rules_50k1, conds_50k1)
    assert dec_50k1.is_authoritative is True
    assert dec_50k1.rule_id == "R-TRV-004"
    assert dec_50k1.outcome == "additional_approval"
    assert "Additional approval is required" in dec_50k1.answer


def test_authoritative_decision_receipt_boundaries(repo_root: Path):
    """INR 1,000 no receipt vs 1,001 itemized receipt required."""
    mock_chunks = [
        {"chunk_id": "DOC-FIN-002_FIN-R03", "doc_id": "DOC-FIN-002", "clause_id": "FIN-R03"},
    ]
    # 1,000
    q_1k = "Does an incidental expense of INR 1,000 require an itemized receipt?"
    conds_1k = extract_conditions_from_query(q_1k)
    rules_1k = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_1k)
    dec_1k = evaluate_authoritative_rule_decision(q_1k, mock_chunks, rules_1k, conds_1k)
    assert dec_1k.is_authoritative is True
    assert dec_1k.rule_id == "R-FIN-003"
    assert dec_1k.outcome == "no_receipt_required"
    assert "do not require an itemized tax receipt, provided digital payment confirmation is attached" in dec_1k.answer
    assert "No receipt is required" not in dec_1k.answer

    # 1,001
    q_1k1 = "Does an incidental expense of INR 1,001 require an itemized receipt?"
    conds_1k1 = extract_conditions_from_query(q_1k1)
    rules_1k1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_1k1)
    dec_1k1 = evaluate_authoritative_rule_decision(q_1k1, mock_chunks, rules_1k1, conds_1k1)
    assert dec_1k1.is_authoritative is True
    assert dec_1k1.rule_id == "R-FIN-004"
    assert dec_1k1.outcome == "itemized_receipt_required"
    assert "itemized tax invoice is strictly required" in dec_1k1.answer


def test_authoritative_decision_procurement_boundaries(repo_root: Path):
    """INR 150,000 single quote vs 150,001 three competitive quotes."""
    mock_chunks = [
        {"chunk_id": "DOC-PRC-001_PRC-P02", "doc_id": "DOC-PRC-001", "clause_id": "PRC-P02"},
    ]
    # 150,000
    q_150k = "Is a single quote permitted for a purchase order of INR 150,000?"
    conds_150k = extract_conditions_from_query(q_150k)
    rules_150k = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_150k)
    dec_150k = evaluate_authoritative_rule_decision(q_150k, mock_chunks, rules_150k, conds_150k)
    assert dec_150k.is_authoritative is True
    assert dec_150k.rule_id == "R-PRC-001"
    assert dec_150k.outcome == "single_quote_permitted"

    # 150,001
    q_150k1 = "Is a single quote permitted for a purchase order of INR 150,001?"
    conds_150k1 = extract_conditions_from_query(q_150k1)
    rules_150k1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_150k1)
    dec_150k1 = evaluate_authoritative_rule_decision(q_150k1, mock_chunks, rules_150k1, conds_150k1)
    assert dec_150k1.is_authoritative is True
    assert dec_150k1.rule_id == "R-PRC-002"
    assert dec_150k1.outcome == "three_competitive_quotes_required"


def test_authoritative_decision_emergency_purchases(repo_root: Path):
    """INR 300,000 director emergency approval vs 300,001 board procurement committee."""
    mock_chunks = [
        {"chunk_id": "DOC-PRC-004_PRC-E01", "doc_id": "DOC-PRC-004", "clause_id": "PRC-E01"},
    ]
    # 300,000
    q_300k = "Can a director approve an emergency purchase of INR 300,000?"
    conds_300k = extract_conditions_from_query(q_300k)
    rules_300k = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds_300k)
    dec_300k = evaluate_authoritative_rule_decision(q_300k, mock_chunks, rules_300k, conds_300k)
    assert dec_300k.is_authoritative is True
    assert dec_300k.rule_id == "R-PRC-005"
    assert dec_300k.outcome == "director_emergency_approval"

    # 300,001
    q_300k1 = "Can a director approve an emergency purchase of INR 300,001?"
    conds_300k1 = extract_conditions_from_query(q_300k1)
    rules_300k1 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds_300k1)
    dec_300k1 = evaluate_authoritative_rule_decision(q_300k1, mock_chunks, rules_300k1, conds_300k1)
    assert dec_300k1.is_authoritative is True
    assert dec_300k1.rule_id == "R-PRC-006"
    assert dec_300k1.outcome == "board_procurement_committee"


def test_authoritative_decision_non_currency_boundaries(repo_root: Path):
    """Non-currency boundaries: 14 days booking, 90 days review, 1 hour incident response."""
    # 14 days advance booking
    mock_trv_b02 = [{"chunk_id": "DOC-TRV-003_TRV-B02", "doc_id": "DOC-TRV-003", "clause_id": "TRV-B02"}]
    q_14d = "Is 14 days advance notice sufficient for standard flight booking?"
    conds_14d = extract_conditions_from_query(q_14d)
    rules_14d = resolve_rules_for_evidence(mock_trv_b02, "employee", "2026-11-22", repo_root, conds_14d)
    dec_14d = evaluate_authoritative_rule_decision(q_14d, mock_trv_b02, rules_14d, conds_14d)
    assert dec_14d.is_authoritative is True
    assert dec_14d.rule_id == "R-TRV-005"
    assert dec_14d.outcome == "standard_booking"

    # 90 days recertification
    mock_sec_a02 = [{"chunk_id": "DOC-SEC-002_SEC-A02", "doc_id": "DOC-SEC-002", "clause_id": "SEC-A02"}]
    q_90d = "Is an access review interval of 90 days compliant before automatic suspension?"
    conds_90d = extract_conditions_from_query(q_90d)
    rules_90d = resolve_rules_for_evidence(mock_sec_a02, "manager", "2026-11-22", repo_root, conds_90d)
    dec_90d = evaluate_authoritative_rule_decision(q_90d, mock_sec_a02, rules_90d, conds_90d)
    assert dec_90d.is_authoritative is True
    assert dec_90d.rule_id == "R-SEC-003"
    assert dec_90d.outcome == "access_recertification_compliant"

    # 1 hour incident response
    mock_sec_i02 = [{"chunk_id": "DOC-SEC-003_SEC-I02", "doc_id": "DOC-SEC-003", "clause_id": "SEC-I02"}]
    q_1h = "Does a 1 hour response time meet the SLA for a Priority 1 incident?"
    conds_1h = extract_conditions_from_query(q_1h)
    rules_1h = resolve_rules_for_evidence(mock_sec_i02, "employee", "2026-11-22", repo_root, conds_1h)
    dec_1h = evaluate_authoritative_rule_decision(q_1h, mock_sec_i02, rules_1h, conds_1h)
    assert dec_1h.is_authoritative is True
    assert dec_1h.rule_id == "R-SEC-005"
    assert dec_1h.outcome == "sla_met"


def test_threshold_definition_and_multidoc_questions_bypass_authoritative_decision(repo_root: Path):
    """Threshold definition and multi-document questions must NOT trigger authoritative decision."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
    ]
    # Threshold definition (no condition value)
    thresh_q = "What is the maximum reimbursement threshold under Travel Policy v2?"
    conds_th = extract_conditions_from_query(thresh_q)
    rules_th = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_th)
    dec_th = evaluate_authoritative_rule_decision(thresh_q, mock_chunks, rules_th, conds_th)
    assert dec_th.is_authoritative is False

    # Multi-document question
    multi_q = "What deployment restriction applies to developers, and within what timeframe must a report be published?"
    conds_m = extract_conditions_from_query(multi_q)
    rules_m = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds_m)
    dec_m = evaluate_authoritative_rule_decision(multi_q, mock_chunks, rules_m, conds_m)
    assert dec_m.is_authoritative is False


def test_foreign_currency_handling_rejected(repo_root: Path):
    """Foreign currencies (USD, EUR, GBP, $, €, £) must produce incompatible_unit and cannot produce an INR approval decision."""
    mock_chunks = [
        {"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04"},
        {"chunk_id": "DOC-PRC-001_PRC-P02", "doc_id": "DOC-PRC-001", "clause_id": "PRC-P02"},
        {"chunk_id": "DOC-FIN-002_FIN-R03", "doc_id": "DOC-FIN-002", "clause_id": "FIN-R03"},
    ]
    test_queries = [
        "Is a single quote permitted for a purchase order of USD 150,001?",
        "Can an employee submit an expense of EUR 50,000?",
        "Can an employee submit an expense of $50,000?",
        "Can an employee submit an expense of €50,000?",
        "Can an employee submit an expense of £1,000?",
    ]
    for q in test_queries:
        conds = extract_conditions_from_query(q)
        assert len(conds) >= 1
        unit = conds[0]["unit"]
        assert unit in ("USD", "EUR", "GBP")
        rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
        # Check rule results: all matching rules must fail with incompatible_unit
        incomp_rules = [r for r in rules if r.is_error and r.error_type == "incompatible_unit"]
        assert len(incomp_rules) > 0
        # Authoritative decision must reject and not return an authoritative approval
        dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
        assert dec.is_authoritative is False


def test_adversarial_spa_treatment_not_authoritative(repo_root: Path):
    """An adversarial question about personal spa treatments costing INR 50,000 must NOT produce an authoritative approval."""
    mock_chunks = [
        {
            "chunk_id": "DOC-TRV-002_TRV-C04",
            "doc_id": "DOC-TRV-002",
            "clause_id": "TRV-C04",
            "title": "Travel Policy v2",
            "department": "Travel",
            "text": (
                "[TRV-C04] Travel Expense Approval Thresholds\n"
                "Travel expenses up to and including INR 50,000 follow standard approval by the direct reporting manager. "
                "Any travel expenses exceeding INR 50,000 require additional approval from the department director prior to reimbursement."
            ),
        },
    ]
    q = "Can employees expense personal spa treatments costing INR 50,000 during business trips?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False


def test_personal_travel_not_authoritative(repo_root: Path):
    """An expense query about personal travel (INR 50,000) must NOT produce an authoritative approval."""
    mock_chunks = [
        {
            "chunk_id": "DOC-TRV-002_TRV-C04",
            "doc_id": "DOC-TRV-002",
            "clause_id": "TRV-C04",
            "title": "Travel Policy v2",
            "department": "Travel",
            "text": (
                "[TRV-C04] Travel Expense Approval Thresholds\n"
                "Travel expenses up to and including INR 50,000 follow standard approval by the direct reporting manager. "
                "Any travel expenses exceeding INR 50,000 require additional approval from the department director prior to reimbursement."
            ),
        },
    ]
    q = "Can employees spend INR 50,000 on personal travel?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.unsupported_terms == ["personal"]


def test_business_travel_authoritative_approval(repo_root: Path):
    """Standard business travel expense of INR 50,000 MUST produce authoritative approval."""
    mock_chunks = [
        {
            "chunk_id": "DOC-TRV-002_TRV-C04",
            "doc_id": "DOC-TRV-002",
            "clause_id": "TRV-C04",
            "title": "Travel Policy v2",
            "department": "Travel",
            "text": (
                "[TRV-C04] Travel Expense Approval Thresholds\n"
                "Travel expenses up to and including INR 50,000 follow standard approval by the direct reporting manager. "
                "Any travel expenses exceeding INR 50,000 require additional approval from the department director prior to reimbursement."
            ),
        },
    ]
    q = "Can a direct manager approve an expense of INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert "Standard manager approval is permitted" in dec.answer


# ==============================================================================
# 9. Focused Numerical Comparison Directional Regression Tests (Section 2.5)
# ==============================================================================


def test_section_2_5_01_exact_threshold(repo_root: Path):
    """1. Exact threshold: INR 50,000 -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve an expense of INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert dec.outcome == "standard_approval"


def test_section_2_5_02_immediately_above_threshold(repo_root: Path):
    """2. Immediately above threshold: INR 50,001 -> additional approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve an expense of INR 50,001 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-004"
    assert dec.outcome == "additional_approval"


def test_section_2_5_03_immediately_below_threshold(repo_root: Path):
    """3. Immediately below threshold: INR 49,999 -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve an expense of INR 49,999 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert dec.outcome == "standard_approval"


def test_section_2_5_04_exceeding_threshold(repo_root: Path):
    """4. 'Exceeding' the threshold: exceeding INR 50,000 -> additional approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses exceeding INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "greater_than"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-004"
    assert dec.outcome == "additional_approval"
    assert "exceeding" in dec.answer.lower()


def test_section_2_5_05_at_least_threshold_boundary_crossing(repo_root: Path):
    """5. 'At least' the threshold: at least INR 50,000 -> boundary crossing, cannot give unconditional decision."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses of at least INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "greater_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is True
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_boundary_crossing is True


def test_section_2_5_06_up_to_threshold(repo_root: Path):
    """6. 'Up to' the threshold: up to INR 50,000 -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses up to INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "less_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert dec.outcome == "standard_approval"
    assert "up to" in dec.answer.lower()


def test_section_2_5_07_range_crossing_threshold(repo_root: Path):
    """7. Range crossing the threshold: between INR 40,000 and 60,000 -> boundary crossing."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses between INR 40,000 and 60,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "range"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is True
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_boundary_crossing is True


def test_section_2_5_08_opposite_comparison_wording(repo_root: Path):
    """8. Opposite comparison wording: below INR 50,000 -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses below INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "less_than"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert dec.outcome == "standard_approval"


def test_section_2_5_09_negation_with_comparison(repo_root: Path):
    """9. Negation with a comparison: not exceeding INR 50,000 -> less_than_or_equal -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses not exceeding INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "less_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.rule_id == "R-TRV-003"
    assert dec.outcome == "standard_approval"


def test_section_2_5_10_invalid_currency(repo_root: Path):
    """10. Foreign currency: USD 50,000 against INR rule produces incompatible_unit error and no authoritative decision."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses of USD 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["unit"] == "USD"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert any(r.is_error and r.error_type == "incompatible_unit" for r in rules)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False


def test_section_2_5_11_historical_current_boundaries(repo_root: Path):
    """11. Historical vs current boundaries: 40,000 on 2026-06-30 vs 50,000 on 2026-07-01."""
    mock_chunks_v1 = [{"chunk_id": "DOC-TRV-001_TRV-C04", "doc_id": "DOC-TRV-001", "clause_id": "TRV-C04", "text": "Travel expenses up to INR 40,000 follow standard approval."}]
    mock_chunks_v2 = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to INR 50,000 follow standard approval."}]

    # Historical
    q_hist = "Can a direct manager approve travel expenses of INR 40,000 under Travel Policy v1?"
    c_hist = extract_conditions_from_query(q_hist)
    r_hist = resolve_rules_for_evidence(mock_chunks_v1, "manager", "2026-06-30", repo_root, c_hist)
    d_hist = evaluate_authoritative_rule_decision(q_hist, mock_chunks_v1, r_hist, c_hist)
    assert d_hist.is_authoritative is True
    assert d_hist.rule_id == "R-TRV-001"

    # Current
    q_curr = "Can a direct manager approve travel expenses of INR 50,000 under Travel Policy v2?"
    c_curr = extract_conditions_from_query(q_curr)
    r_curr = resolve_rules_for_evidence(mock_chunks_v2, "manager", "2026-07-01", repo_root, c_curr)
    d_curr = evaluate_authoritative_rule_decision(q_curr, mock_chunks_v2, r_curr, c_curr)
    assert d_curr.is_authoritative is True
    assert d_curr.rule_id == "R-TRV-003"


def test_section_2_5_12_unsupported_expense_eligibility(repo_root: Path):
    """12. Unsupported expense category: personal travel rejected from deterministic approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can employees spend INR 50,000 on personal travel?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.unsupported_terms == ["personal"]


# ==============================================================================
# 10. Pass 1 Final Review: Numerical Boundary Regression Tests
# ==============================================================================


def test_boundary_travel_v2_more_than_40k_crosses_50k_boundary(repo_root: Path):
    """Travel Policy v2: More than INR 40,000 crosses the 50,000 boundary."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses of more than INR 40,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "greater_than"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is True
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_boundary_crossing is True


def test_boundary_travel_v2_up_to_60k_crosses_50k_boundary(repo_root: Path):
    """Travel Policy v2: Up to INR 60,000 crosses the 50,000 boundary."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses up to INR 60,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "less_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is True
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_boundary_crossing is True


def test_boundary_travel_v2_at_least_45k_crosses_50k_boundary(repo_root: Path):
    """Travel Policy v2: At least INR 45,000 crosses the 50,000 boundary."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses of at least INR 45,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "greater_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is True
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_boundary_crossing is True


def test_boundary_travel_v2_between_40k_and_50k_stays_within_standard(repo_root: Path):
    """Travel Policy v2: Between INR 40,000 and 50,000 stays within standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses between INR 40,000 and 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "range"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is False
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.outcome == "standard_approval"
    assert dec.rule_id == "R-TRV-003"


def test_boundary_travel_v2_exceeding_50k_additional_approval(repo_root: Path):
    """Travel Policy v2: Exceeding INR 50,000 -> additional approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses exceeding INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "greater_than"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is False
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.outcome == "additional_approval"
    assert dec.rule_id == "R-TRV-004"


def test_boundary_travel_v2_up_to_50k_standard_approval(repo_root: Path):
    """Travel Policy v2: Up to INR 50,000 -> standard approval."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses up to INR 50,000 under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    assert conds[0]["comparison"] == "less_than_or_equal"
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    assert conds[0]["is_boundary_crossing"] is False
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is True
    assert dec.outcome == "standard_approval"
    assert dec.rule_id == "R-TRV-003"


def test_boundary_travel_v1_historical_evaluations(repo_root: Path):
    """Historical Travel Policy v1: boundary is dynamically 40,000 from active rules."""
    mock_chunks = [{"chunk_id": "DOC-TRV-001_TRV-C04", "doc_id": "DOC-TRV-001", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 40,000 follow standard approval by direct reporting manager."}]

    # Up to 40k -> standard approval
    q1 = "Can a direct manager approve travel expenses up to INR 40,000 under Travel Policy v1?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-06-01", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "standard_approval"
    assert d1.rule_id == "R-TRV-001"

    # Exceeding 40k -> additional approval
    q2 = "Can a direct manager approve travel expenses exceeding INR 40,000 under Travel Policy v1?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-06-01", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "additional_approval"
    assert d2.rule_id == "R-TRV-002"

    # More than 35k -> crosses 40k boundary
    q3 = "Can a direct manager approve travel expenses of more than INR 35,000 under Travel Policy v1?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-06-01", repo_root, c3)
    assert c3[0]["is_boundary_crossing"] is True
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is False
    assert d3.is_boundary_crossing is True

    # Up to 45k -> crosses 40k boundary
    q4 = "Can a direct manager approve travel expenses up to INR 45,000 under Travel Policy v1?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-06-01", repo_root, c4)
    assert c4[0]["is_boundary_crossing"] is True
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is False
    assert d4.is_boundary_crossing is True


def test_boundary_corporate_card_monthly_spend_at_200k(repo_root: Path):
    """Corporate card monthly spend: boundary is dynamically 200,000 from DOC-FIN-003."""
    mock_chunks = [{"chunk_id": "DOC-FIN-003_FIN-K02", "doc_id": "DOC-FIN-003", "clause_id": "FIN-K02", "text": "Monthly corporate credit card spend up to INR 200,000 is within standard card limit."}]

    # Exact 200,000 -> within_standard_card_limit
    q1 = "Can an employee spend INR 200,000 on corporate credit card?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "within_standard_card_limit"
    assert d1.rule_id == "R-FIN-005"

    # Exact 200,001 -> card_limit_escalation_required
    q2 = "Can an employee spend INR 200,001 on corporate credit card?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "card_limit_escalation_required"
    assert d2.rule_id == "R-FIN-006"

    # Up to 200,000 -> within_standard_card_limit
    q3 = "Can an employee have monthly card spending up to INR 200,000?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "within_standard_card_limit"
    assert d3.rule_id == "R-FIN-005"

    # Exceeding 200,000 -> card_limit_escalation_required
    q4 = "Can an employee have monthly card spending exceeding INR 200,000?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "card_limit_escalation_required"
    assert d4.rule_id == "R-FIN-006"

    # More than 150,000 -> crosses 200k boundary
    q5 = "Can an employee have monthly card spending of more than INR 150,000?"
    c5 = extract_conditions_from_query(q5)
    r5 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c5)
    assert c5[0]["is_boundary_crossing"] is True
    d5 = evaluate_authoritative_rule_decision(q5, mock_chunks, r5, c5)
    assert d5.is_authoritative is False
    assert d5.is_boundary_crossing is True

    # Up to 250,000 -> crosses 200k boundary
    q6 = "Can an employee have monthly card spending up to INR 250,000?"
    c6 = extract_conditions_from_query(q6)
    r6 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c6)
    assert c6[0]["is_boundary_crossing"] is True
    d6 = evaluate_authoritative_rule_decision(q6, mock_chunks, r6, c6)
    assert d6.is_authoritative is False
    assert d6.is_boundary_crossing is True


def test_boundary_tuition_claim_at_75k(repo_root: Path):
    """Tuition claims: boundary is dynamically 75,000 from DOC-PPO-003."""
    mock_chunks = [{"chunk_id": "DOC-PPO-003_PPO-B02", "doc_id": "DOC-PPO-003", "clause_id": "PPO-B02", "text": "Annual tuition reimbursement up to INR 75,000 follows standard grant."}]

    # Exact 75,000 -> standard_tuition_grant
    q1 = "Can an employee claim tuition assistance of INR 75,000?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "standard_tuition_grant"
    assert d1.rule_id == "R-PPO-005"

    # Exact 75,001 -> committee_review_required
    q2 = "Can an employee claim tuition assistance of INR 75,001?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "committee_review_required"
    assert d2.rule_id == "R-PPO-006"

    # Up to 75,000 -> standard_tuition_grant
    q3 = "Can an employee claim tuition assistance up to INR 75,000?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "standard_tuition_grant"
    assert d3.rule_id == "R-PPO-005"

    # Exceeding 75,000 -> committee_review_required
    q4 = "Can an employee claim tuition assistance exceeding INR 75,000?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "committee_review_required"
    assert d4.rule_id == "R-PPO-006"

    # More than 50,000 -> crosses 75k boundary
    q5 = "Can an employee claim tuition assistance of more than INR 50,000?"
    c5 = extract_conditions_from_query(q5)
    r5 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c5)
    assert c5[0]["is_boundary_crossing"] is True
    d5 = evaluate_authoritative_rule_decision(q5, mock_chunks, r5, c5)
    assert d5.is_authoritative is False
    assert d5.is_boundary_crossing is True

    # Up to 100,000 -> crosses 75k boundary
    q6 = "Can an employee claim tuition assistance up to INR 100,000?"
    c6 = extract_conditions_from_query(q6)
    r6 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c6)
    assert c6[0]["is_boundary_crossing"] is True
    d6 = evaluate_authoritative_rule_decision(q6, mock_chunks, r6, c6)
    assert d6.is_authoritative is False
    assert d6.is_boundary_crossing is True


# ==============================================================================
# 9. Stale-Rule Cache Invalidation, Reload, and Repository Isolation Tests
# ==============================================================================


def test_rules_cache_invalidation_on_content_change(repo_root: Path, tmp_path: Path):
    """Rules cache is invalidated when rules.jsonl content changes on disk."""
    import shutil
    import json

    tmp_processed = tmp_path / "data" / "processed"
    tmp_processed.mkdir(parents=True, exist_ok=True)
    orig_rules = repo_root / "data" / "processed" / "rules.jsonl"
    target_rules = tmp_processed / "rules.jsonl"
    shutil.copy2(orig_rules, target_rules)

    # Initial load
    rules1 = load_rules(base_dir=tmp_path)
    assert len(rules1) == 32
    r_trv3 = next(r for r in rules1 if r.rule_id == "R-TRV-003")
    assert r_trv3.threshold_value == 50000

    # Modify rules.jsonl on disk (change threshold to 55000 for R-TRV-003)
    lines = target_rules.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        if data["rule_id"] == "R-TRV-003":
            data["threshold_value"] = 55000
        new_lines.append(json.dumps(data))
    target_rules.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # Second load WITHOUT reload=True: must detect content fingerprint change and reload
    rules2 = load_rules(base_dir=tmp_path, reload=False)
    assert len(rules2) == 32
    r_trv3_mod = next(r for r in rules2 if r.rule_id == "R-TRV-003")
    assert r_trv3_mod.threshold_value == 55000


def test_rules_cache_validation_failure_on_invalid_file(repo_root: Path, tmp_path: Path):
    """Corrupted or incomplete rules.jsonl fails validation rather than reusing old cache."""
    import shutil

    tmp_processed = tmp_path / "data" / "processed"
    tmp_processed.mkdir(parents=True, exist_ok=True)
    orig_rules = repo_root / "data" / "processed" / "rules.jsonl"
    target_rules = tmp_processed / "rules.jsonl"
    shutil.copy2(orig_rules, target_rules)

    # Initial valid load
    rules1 = load_rules(base_dir=tmp_path)
    assert len(rules1) == 32

    # Truncate file so it has only 31 rules
    lines = target_rules.read_text(encoding="utf-8").splitlines()
    target_rules.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    # Next load must raise ValueError and NOT reuse old 32-rule cache
    with pytest.raises(ValueError, match="Expected 32 canonical rules"):
        load_rules(base_dir=tmp_path, reload=False)


def test_explicit_reload_flag_reloads_from_disk(repo_root: Path):
    """Explicit reload=True reloads rules successfully and resets cache."""
    clear_rules_cache()
    rules = load_rules(base_dir=repo_root, reload=True)
    assert len(rules) == 32


# ==============================================================================
# 10. Mandatory 10 Travel Policy v2 Checks (Prompt Specification Table)
# ==============================================================================


def test_mandatory_travel_v2_all_10_checks(repo_root: Path):
    """Test all 10 conditions specified in the Travel Policy v2 mandatory table."""
    mock_chunks = [{
        "chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04",
        "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager.",
    }]

    # 1. Exactly INR 50,000 -> Standard approval
    q1 = "Can a direct manager approve travel expenses of exactly INR 50,000 under Travel Policy v2?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "standard_approval"
    assert d1.rule_id == "R-TRV-003"

    # 2. INR 50,001 -> Additional approval
    q2 = "Can a direct manager approve travel expenses of INR 50,001 under Travel Policy v2?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "additional_approval"
    assert d2.rule_id == "R-TRV-004"

    # 3. Exceeding INR 50,000 -> Additional approval
    q3 = "Can a direct manager approve travel expenses exceeding INR 50,000 under Travel Policy v2?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "additional_approval"
    assert d3.rule_id == "R-TRV-004"

    # 4. Below INR 50,000 -> Standard approval
    q4 = "Can a direct manager approve travel expenses below INR 50,000 under Travel Policy v2?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "standard_approval"
    assert d4.rule_id == "R-TRV-003"

    # 5. Up to INR 50,000 -> Standard approval
    q5 = "Can a direct manager approve travel expenses up to INR 50,000 under Travel Policy v2?"
    c5 = extract_conditions_from_query(q5)
    r5 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c5)
    d5 = evaluate_authoritative_rule_decision(q5, mock_chunks, r5, c5)
    assert d5.is_authoritative is True
    assert d5.outcome == "standard_approval"
    assert d5.rule_id == "R-TRV-003"

    # 6. More than INR 40,000 -> Mixed outcomes (safe refusal)
    q6 = "Can a direct manager approve travel expenses of more than INR 40,000 under Travel Policy v2?"
    c6 = extract_conditions_from_query(q6)
    r6 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c6)
    assert c6[0]["is_boundary_crossing"] is True
    d6 = evaluate_authoritative_rule_decision(q6, mock_chunks, r6, c6)
    assert d6.is_authoritative is False
    assert d6.is_boundary_crossing is True

    # 7. Up to INR 60,000 -> Mixed outcomes (safe refusal)
    q7 = "Can a direct manager approve travel expenses up to INR 60,000 under Travel Policy v2?"
    c7 = extract_conditions_from_query(q7)
    r7 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c7)
    assert c7[0]["is_boundary_crossing"] is True
    d7 = evaluate_authoritative_rule_decision(q7, mock_chunks, r7, c7)
    assert d7.is_authoritative is False
    assert d7.is_boundary_crossing is True

    # 8. At least INR 45,000 -> Mixed outcomes (safe refusal)
    q8 = "Can a direct manager approve travel expenses of at least INR 45,000 under Travel Policy v2?"
    c8 = extract_conditions_from_query(q8)
    r8 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c8)
    assert c8[0]["is_boundary_crossing"] is True
    d8 = evaluate_authoritative_rule_decision(q8, mock_chunks, r8, c8)
    assert d8.is_authoritative is False
    assert d8.is_boundary_crossing is True

    # 9. Between INR 40,000 and 50,000 -> Standard approval
    q9 = "Can a direct manager approve travel expenses between INR 40,000 and 50,000 under Travel Policy v2?"
    c9 = extract_conditions_from_query(q9)
    r9 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c9)
    d9 = evaluate_authoritative_rule_decision(q9, mock_chunks, r9, c9)
    assert d9.is_authoritative is True
    assert d9.outcome == "standard_approval"
    assert d9.rule_id == "R-TRV-003"

    # 10. Between INR 40,000 and 60,000 -> Mixed outcomes (safe refusal)
    q10 = "Can a direct manager approve travel expenses between INR 40,000 and 60,000 under Travel Policy v2?"
    c10 = extract_conditions_from_query(q10)
    r10 = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, c10)
    assert c10[0]["is_boundary_crossing"] is True
    d10 = evaluate_authoritative_rule_decision(q10, mock_chunks, r10, c10)
    assert d10.is_authoritative is False
    assert d10.is_boundary_crossing is True


# ==============================================================================
# 11. Multi-Department Boundary Decisions Across All Five Departments
# ==============================================================================


def test_procurement_order_amount_boundaries(repo_root: Path):
    """Procurement order amount (150,000 INR boundary from DOC-PRC-001)."""
    mock_chunks = [{"chunk_id": "DOC-PRC-001_PRC-P02", "doc_id": "DOC-PRC-001", "clause_id": "PRC-P02", "text": "Purchase orders up to INR 150,000 require a single supplier quote."}]

    # Exactly 150,000 -> single_quote_permitted
    q1 = "Can procurement issue a purchase order of INR 150,000 with a single quote?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "single_quote_permitted"
    assert d1.rule_id == "R-PRC-001"

    # 150,001 -> three_competitive_quotes_required
    q2 = "Can procurement issue a purchase order of INR 150,001 with a single quote?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "three_competitive_quotes_required"
    assert d2.rule_id == "R-PRC-002"

    # Up to 150,000 -> single_quote_permitted
    q3 = "Can procurement issue a purchase order up to INR 150,000 with a single quote?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "single_quote_permitted"
    assert d3.rule_id == "R-PRC-001"

    # Exceeding 150,000 -> three_competitive_quotes_required
    q4 = "Are three quotes required for purchase orders exceeding INR 150,000?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "three_competitive_quotes_required"
    assert d4.rule_id == "R-PRC-002"

    # More than 100,000 -> crosses 150k boundary
    q5 = "Are three quotes required for purchase orders of more than INR 100,000?"
    c5 = extract_conditions_from_query(q5)
    r5 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c5)
    assert c5[0]["is_boundary_crossing"] is True
    d5 = evaluate_authoritative_rule_decision(q5, mock_chunks, r5, c5)
    assert d5.is_authoritative is False
    assert d5.is_boundary_crossing is True

    # Between 100,000 and 150,000 -> single_quote_permitted
    q6 = "Is a single quote permitted for purchase orders between INR 100,000 and 150,000?"
    c6 = extract_conditions_from_query(q6)
    r6 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c6)
    d6 = evaluate_authoritative_rule_decision(q6, mock_chunks, r6, c6)
    assert d6.is_authoritative is True
    assert d6.outcome == "single_quote_permitted"

    # Between 100,000 and 200,000 -> crosses boundary
    q7 = "Is a single quote permitted for purchase orders between INR 100,000 and 200,000?"
    c7 = extract_conditions_from_query(q7)
    r7 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c7)
    assert c7[0]["is_boundary_crossing"] is True
    d7 = evaluate_authoritative_rule_decision(q7, mock_chunks, r7, c7)
    assert d7.is_authoritative is False
    assert d7.is_boundary_crossing is True


def test_procurement_contract_duration_boundaries(repo_root: Path):
    """Procurement vendor contract duration (12 months boundary from DOC-PRC-003)."""
    mock_chunks = [{"chunk_id": "DOC-PRC-003_PRC-C02", "doc_id": "DOC-PRC-003", "clause_id": "PRC-C02", "text": "Vendor contracts up to 12 months duration require department lead signoff."}]

    # Exactly 12 months -> department_lead_signoff
    q1 = "Can a department lead sign off on a vendor contract duration of 12 months?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "department_lead_signoff"
    assert d1.rule_id == "R-PRC-003"

    # 13 months -> legal_and_procurement_signoff
    q2 = "Is legal signoff required for a vendor contract duration of 13 months?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "legal_and_procurement_signoff"
    assert d2.rule_id == "R-PRC-004"

    # Exceeding 12 months -> legal_and_procurement_signoff
    q3 = "Is legal signoff required for vendor contracts exceeding 12 months?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "legal_and_procurement_signoff"


def test_people_ops_sick_leave_boundaries(repo_root: Path):
    """People Operations consecutive sick days (3 days boundary from DOC-PPO-001)."""
    mock_chunks = [{"chunk_id": "DOC-PPO-001_PPO-L02", "doc_id": "DOC-PPO-001", "clause_id": "PPO-L02", "text": "Sick leave up to 3 consecutive days permits self-certification."}]

    # 3 days -> self_certification
    q1 = "Can an employee use self-certification for sick leave of 3 consecutive days?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "self_certification"
    assert d1.rule_id == "R-PPO-001"

    # 4 days -> medical_certificate_required
    q2 = "Is a medical certificate required for sick leave of 4 consecutive days?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "medical_certificate_required"
    assert d2.rule_id == "R-PPO-002"

    # More than 2 days -> crosses 3 days boundary
    q3 = "Is a medical certificate required for sick leave of more than 2 days?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    assert c3[0]["is_boundary_crossing"] is True
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is False
    assert d3.is_boundary_crossing is True


def test_people_ops_internet_speed_boundaries(repo_root: Path):
    """People Operations remote setup internet speed (50 Mbps boundary from DOC-PPO-002)."""
    mock_chunks = [{"chunk_id": "DOC-PPO-002_PPO-R02", "doc_id": "DOC-PPO-002", "clause_id": "PPO-R02", "text": "Remote work internet speed minimum requirement is 50 mbps."}]

    # Exactly 50 Mbps -> remote_setup_compliant
    q1 = "Does an internet speed of 50 Mbps comply with remote work setup policy?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_chunks, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "remote_setup_compliant"
    assert d1.rule_id == "R-PPO-003"

    # 49 Mbps -> remediation_required
    q2 = "Does an internet speed of 49 Mbps require remediation under remote policy?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_chunks, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "remediation_required"
    assert d2.rule_id == "R-PPO-004"

    # At least 50 Mbps -> remote_setup_compliant
    q3 = "Is an internet speed of at least 50 Mbps compliant with remote work policy?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_chunks, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "remote_setup_compliant"

    # Below 50 Mbps -> remediation_required
    q4 = "Does an internet speed below 50 Mbps require remediation?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_chunks, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "remediation_required"

    # More than 40 Mbps -> crosses 50 Mbps boundary
    q5 = "Is an internet speed of more than 40 Mbps compliant?"
    c5 = extract_conditions_from_query(q5)
    r5 = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, c5)
    assert c5[0]["is_boundary_crossing"] is True
    d5 = evaluate_authoritative_rule_decision(q5, mock_chunks, r5, c5)
    assert d5.is_authoritative is False
    assert d5.is_boundary_crossing is True


def test_infosec_privileged_review_and_p1_sla_boundaries(repo_root: Path):
    """InfoSec privileged role review (90 days) and P1 SLA (1 hour)."""
    mock_sec2 = [{"chunk_id": "DOC-SEC-002_SEC-A02", "doc_id": "DOC-SEC-002", "clause_id": "SEC-A02", "text": "Privileged access recertification interval up to 90 days is compliant."}]
    mock_sec3 = [{"chunk_id": "DOC-SEC-003_SEC-I02", "doc_id": "DOC-SEC-003", "clause_id": "SEC-I02", "text": "Priority 1 incident response time SLA is 1 hour."}]

    # 90 days -> access_recertification_compliant
    q1 = "Is privileged access review compliant within 90 days interval?"
    c1 = extract_conditions_from_query(q1)
    r1 = resolve_rules_for_evidence(mock_sec2, "security", "2026-11-22", repo_root, c1)
    d1 = evaluate_authoritative_rule_decision(q1, mock_sec2, r1, c1)
    assert d1.is_authoritative is True
    assert d1.outcome == "access_recertification_compliant"
    assert d1.rule_id == "R-SEC-003"

    # 91 days -> automatic_account_suspension
    q2 = "Does a privileged access review interval of 91 days trigger automatic suspension?"
    c2 = extract_conditions_from_query(q2)
    r2 = resolve_rules_for_evidence(mock_sec2, "security", "2026-11-22", repo_root, c2)
    d2 = evaluate_authoritative_rule_decision(q2, mock_sec2, r2, c2)
    assert d2.is_authoritative is True
    assert d2.outcome == "automatic_account_suspension"
    assert d2.rule_id == "R-SEC-004"

    # P1 1 hour -> sla_met
    q3 = "Is a Priority 1 incident response time of 1 hour compliant with SLA?"
    c3 = extract_conditions_from_query(q3)
    r3 = resolve_rules_for_evidence(mock_sec3, "security", "2026-11-22", repo_root, c3)
    d3 = evaluate_authoritative_rule_decision(q3, mock_sec3, r3, c3)
    assert d3.is_authoritative is True
    assert d3.outcome == "sla_met"
    assert d3.rule_id == "R-SEC-005"

    # P1 2 hours -> sla_breach_escalation
    q4 = "Does a Priority 1 incident response time of 2 hours require escalation?"
    c4 = extract_conditions_from_query(q4)
    r4 = resolve_rules_for_evidence(mock_sec3, "security", "2026-11-22", repo_root, c4)
    d4 = evaluate_authoritative_rule_decision(q4, mock_sec3, r4, c4)
    assert d4.is_authoritative is True
    assert d4.outcome == "sla_breach_escalation"
    assert d4.rule_id == "R-SEC-006"


# ==============================================================================
# 12. Fail-Safe Handling of Unresolved Conditions
# ==============================================================================


def test_failsafe_negative_values_cannot_evaluate(repo_root: Path):
    """Negative values cannot be evaluated and trigger is_unresolved=True."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "Can a direct manager approve travel expenses of -50,000 INR?"
    conds = extract_conditions_from_query(q)
    assert len(conds) == 1
    assert conds[0]["value"] < 0
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    # resolve_rules_for_evidence catches ValueError and marks is_error
    assert any(r.is_error for r in rules)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_unresolved is True


def test_failsafe_incomplete_retrieval_evidence_marks_unresolved(repo_root: Path):
    """When decision question's rule chunk is missing from authorized evidence, mark is_unresolved=True."""
    # Chunk provided is from Information Security, but question asks about travel expenses
    mock_chunks = [{"chunk_id": "DOC-SEC-001_SEC-D02", "doc_id": "DOC-SEC-001", "clause_id": "SEC-D02", "text": "Security data classification tier."}]
    q = "Can a direct manager approve travel expenses of INR 50,000?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "manager", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_unresolved is True
    assert "Incomplete retrieval evidence" in dec.reason


def test_explanatory_questions_preserve_non_decision_status(repo_root: Path):
    """Explanatory questions must not be classified as decision requests or marked is_unresolved."""
    mock_chunks = [{"chunk_id": "DOC-TRV-002_TRV-C04", "doc_id": "DOC-TRV-002", "clause_id": "TRV-C04", "text": "Travel expenses up to and including INR 50,000 follow standard approval by direct reporting manager."}]
    q = "What is the maximum travel expense threshold that a direct manager can approve under Travel Policy v2?"
    conds = extract_conditions_from_query(q)
    rules = resolve_rules_for_evidence(mock_chunks, "employee", "2026-11-22", repo_root, conds)
    dec = evaluate_authoritative_rule_decision(q, mock_chunks, rules, conds)
    assert dec.is_authoritative is False
    assert dec.is_decision_request is False
    assert dec.is_unresolved is False

