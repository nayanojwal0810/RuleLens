"""Deterministic structured rule resolution for RuleLens.

Loads, links, filters, and evaluates deterministic policy rules from
data/processed/rules.jsonl against authorized chunks, user roles, query dates,
and condition values.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from rulelens.dataset import ROLES
from rulelens.retrieval import ALLOWED_ROLES_SET, parse_query_date


@dataclass(frozen=True)
class RuleRecord:
    """Canonical representation of a structured policy rule."""

    rule_id: str
    doc_id: str
    clause_id: str
    chunk_id: str
    condition_field: str
    operator: str
    threshold_value: Union[int, float, str]
    unit: str
    outcome: str
    effective_from: str
    effective_to: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "doc_id": self.doc_id,
            "clause_id": self.clause_id,
            "chunk_id": self.chunk_id,
            "condition_field": self.condition_field,
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "unit": self.unit,
            "outcome": self.outcome,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
        }


@dataclass(frozen=True)
class RuleEvaluationResult:
    """Outcome of evaluating a structured rule against an input value."""

    rule: RuleRecord
    passed: bool
    input_value: Any
    input_unit: Optional[str]
    outcome: str
    explanation: str
    is_error: bool = False
    error_type: Optional[str] = None


@dataclass(frozen=True)
class ConditionExtractionResult:
    """Structured outcome of natural language condition extraction."""

    status: str  # "valid", "missing", "ambiguous", "invalid"
    conditions: List[Dict[str, Any]]
    reason: Optional[str] = None


# Cache loaded rules keyed by base_dir to avoid cross-corpus pollution
_RULES_CACHE: Dict[Path, List[RuleRecord]] = {}


def load_rules(base_dir: Optional[Union[Path, str]] = None, reload: bool = False) -> List[RuleRecord]:
    """Load and validate all 32 canonical rules from rules.jsonl.

    Args:
        base_dir: Repository root directory. Defaults to cwd.
        reload: If True, bypass cache and reload from disk.

    Returns:
        List of RuleRecord objects.
    """
    global _RULES_CACHE
    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    if not reload and base in _RULES_CACHE:
        return _RULES_CACHE[base]

    rules_path = base / "data" / "processed" / "rules.jsonl"
    if not rules_path.is_file():
        # Check parent in case cwd is inside tests or a subpackage
        alt_path = base.parent / "data" / "processed" / "rules.jsonl"
        if alt_path.is_file():
            rules_path = alt_path
        else:
            raise FileNotFoundError(f"Canonical rules file not found: {rules_path}")

    rules = []
    with open(rules_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunk_id = f"{data['doc_id']}_{data['clause_id']}"
            rule = RuleRecord(
                rule_id=data["rule_id"],
                doc_id=data["doc_id"],
                clause_id=data["clause_id"],
                chunk_id=chunk_id,
                condition_field=data["condition_field"],
                operator=data["operator"],
                threshold_value=data["threshold_value"],
                unit=data["unit"],
                outcome=data["outcome"],
                effective_from=data["effective_from"],
                effective_to=data["effective_to"],
            )
            rules.append(rule)

    if len(rules) != 32:
        raise ValueError(f"Expected 32 canonical rules in {rules_path}, found {len(rules)}.")

    _RULES_CACHE[base] = rules
    return rules


def is_rule_active(rule: Union[RuleRecord, Dict[str, Any]], query_date: datetime.date) -> bool:
    """Check if a rule is effective on query_date."""
    eff_from_str = rule.effective_from if isinstance(rule, RuleRecord) else rule["effective_from"]
    eff_to_str = rule.effective_to if isinstance(rule, RuleRecord) else rule["effective_to"]

    eff_from = datetime.date.fromisoformat(eff_from_str)
    eff_to = datetime.date.fromisoformat(eff_to_str)
    return eff_from <= query_date <= eff_to


def evaluate_rule_condition(
    rule: Union[RuleRecord, Dict[str, Any]],
    value: Any,
    unit: Optional[str] = None,
) -> bool:
    """Evaluate whether an input value satisfies a rule's condition using exact numeric comparisons.

    Args:
        rule: RuleRecord or dict.
        value: Input value to test.
        unit: Optional unit for comparison.

    Returns:
        True if the condition is satisfied, False otherwise.

    Raises:
        TypeError: If value type is incompatible with operator or threshold.
        ValueError: If units conflict.
    """
    rule_unit = rule.unit if isinstance(rule, RuleRecord) else rule["unit"]
    rule_op = rule.operator if isinstance(rule, RuleRecord) else rule["operator"]
    threshold = rule.threshold_value if isinstance(rule, RuleRecord) else rule["threshold_value"]

    # Unit validation
    if unit is not None:
        if unit.strip().lower() != rule_unit.strip().lower():
            raise ValueError(
                f"Incompatible unit '{unit}' for rule '{getattr(rule, 'rule_id', '')}'. Expected '{rule_unit}'."
            )

    # Operator evaluation
    if rule_op == "==":
        if isinstance(threshold, str):
            return str(value).strip().lower() == threshold.strip().lower()
        return value == threshold

    # Exact decimal numeric comparisons: <, <=, >, >=
    try:
        clean_val_str = str(value).strip().replace(",", "")
        clean_thresh_str = str(threshold).strip().replace(",", "")
        numeric_val = Decimal(clean_val_str)
        numeric_threshold = Decimal(clean_thresh_str)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise TypeError(
            f"Cannot evaluate operator '{rule_op}' with non-numeric value '{value}' and threshold '{threshold}'."
        ) from exc

    if rule_op == "<":
        return numeric_val < numeric_threshold
    elif rule_op == "<=":
        return numeric_val <= numeric_threshold
    elif rule_op == ">":
        return numeric_val > numeric_threshold
    elif rule_op == ">=":
        return numeric_val >= numeric_threshold
    else:
        raise ValueError(f"Unsupported rule operator: '{rule_op}'.")


def resolve_rules_for_evidence(
    authorized_chunks: Sequence[Dict[str, Any]],
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
    base_dir: Optional[Union[Path, str]] = None,
    extracted_conditions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[RuleEvaluationResult]:
    """Resolve applicable structured rules for authorized evidence chunks.

    Only rules whose linked chunk is in `authorized_chunks` and whose date is active
    on `query_date` are considered. If `extracted_conditions` are provided,
    applicable conditions are evaluated against these rules.

    Args:
        authorized_chunks: Chunks that have passed role and date authorization.
        role: Requesting user role.
        query_date: Effective date.
        base_dir: Repository root directory.
        extracted_conditions: Optional list of dicts with keys:
            'field', 'value', 'unit' (optional).

    Returns:
        List of RuleEvaluationResult objects for applicable rules.
    """
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    q_date = parse_query_date(query_date)
    all_rules = load_rules(base_dir=base_dir)

    # Set of chunk IDs in authorized evidence
    authorized_cids = {c["chunk_id"] for c in authorized_chunks}

    # Map conditions by field for evaluation
    cond_map: Dict[str, Tuple[Any, Optional[str], bool, Optional[str]]] = {}
    has_ambiguity = False
    ambiguity_reason = None
    if extracted_conditions:
        for cond in extracted_conditions:
            field = cond.get("field")
            is_ambig = bool(cond.get("is_ambiguous", False))
            if is_ambig:
                has_ambiguity = True
                ambiguity_reason = cond.get("error", "Ambiguous or conflicting condition inputs detected")
            if field:
                cond_map[field] = (cond.get("value"), cond.get("unit"), is_ambig, cond.get("error"))

    results: List[RuleEvaluationResult] = []

    for rule in all_rules:
        # 1. Chunk must be present in authorized evidence
        if rule.chunk_id not in authorized_cids:
            continue

        # 2. Rule must be active on query date
        if not is_rule_active(rule, q_date):
            continue

        # 3. Check for condition evaluation or ambiguity
        if rule.condition_field in cond_map or (has_ambiguity and rule.condition_field in ("expense_amount", "order_amount", "monthly_spend", "receipt_amount", "exception_amount")):
            cond_entry = cond_map.get(rule.condition_field)
            if (cond_entry and cond_entry[2]) or has_ambiguity:
                # Ambiguous input condition
                explanation = (
                    f"Rule {rule.rule_id} condition ambiguous: "
                    f"{ambiguity_reason or 'Multiple conflicting condition inputs detected in query.'}"
                )
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=None,
                        input_unit=rule.unit,
                        outcome="ambiguous_condition",
                        explanation=explanation,
                        is_error=True,
                        error_type="ambiguous_condition",
                    )
                )
                continue

            val, unit, _, _ = cond_entry
            try:
                passed = evaluate_rule_condition(rule, val, unit)
                if passed:
                    explanation = (
                        f"Rule {rule.rule_id} condition satisfied: {rule.condition_field} {rule.operator} "
                        f"{rule.threshold_value} {rule.unit} for input {val} {unit or rule.unit} -> outcome: {rule.outcome}"
                    )
                    results.append(
                        RuleEvaluationResult(
                            rule=rule,
                            passed=True,
                            input_value=val,
                            input_unit=unit or rule.unit,
                            outcome=rule.outcome,
                            explanation=explanation,
                            is_error=False,
                            error_type=None,
                        )
                    )
                else:
                    # Valid comparison evaluating to False (normal unsatisfied rule)
                    explanation = (
                        f"Rule {rule.rule_id} condition NOT satisfied: {rule.condition_field} {rule.operator} "
                        f"{rule.threshold_value} {rule.unit} for input {val} {unit or rule.unit}."
                    )
                    results.append(
                        RuleEvaluationResult(
                            rule=rule,
                            passed=False,
                            input_value=val,
                            input_unit=unit or rule.unit,
                            outcome="condition_not_met",
                            explanation=explanation,
                            is_error=False,
                            error_type=None,
                        )
                    )
            except ValueError as val_err:
                # Incompatible unit: invalid rule evaluation error
                explanation = f"Rule {rule.rule_id} unit error: {val_err}"
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=val,
                        input_unit=unit or rule.unit,
                        outcome="invalid_unit",
                        explanation=explanation,
                        is_error=True,
                        error_type="invalid_unit",
                    )
                )
            except TypeError as type_err:
                # Incompatible type: invalid rule evaluation error
                explanation = f"Rule {rule.rule_id} type error: {type_err}"
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=val,
                        input_unit=unit or rule.unit,
                        outcome="invalid_type",
                        explanation=explanation,
                        is_error=True,
                        error_type="invalid_type",
                    )
                )
        else:
            # Rule applies structurally to the clause, but no transaction amount was provided (threshold definition query)
            explanation = (
                f"Rule {rule.rule_id} defines: {rule.condition_field} {rule.operator} {rule.threshold_value} {rule.unit} "
                f"for {rule.outcome}."
            )
            results.append(
                RuleEvaluationResult(
                    rule=rule,
                    passed=True,
                    input_value=None,
                    input_unit=rule.unit,
                    outcome=rule.outcome,
                    explanation=explanation,
                    is_error=False,
                    error_type=None,
                )
            )

    return results


# Regular expressions for conservative condition extraction from natural language questions
_CURRENCY_PATTERN = re.compile(
    r"(?:inr|rs\.?|₹)?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?:inr|rupees)?",
    re.IGNORECASE,
)
_DAYS_PATTERN = re.compile(
    r"\b([0-9]+)\s*(?:consecutive\s*)?(?:business\s*|working\s*|calendar\s*)?days?\b",
    re.IGNORECASE,
)
_MONTHS_PATTERN = re.compile(
    r"\b([0-9]+)\s*months?\b",
    re.IGNORECASE,
)
_MBPS_PATTERN = re.compile(
    r"\b([0-9]+)\s*mbps\b",
    re.IGNORECASE,
)
_HOURS_PATTERN = re.compile(
    r"\b([0-9]+)\s*hours?\b",
    re.IGNORECASE,
)


def _determine_currency_field(query_lower: str, span_lower: str) -> Optional[str]:
    """Identify the specific currency condition field from query context."""
    if any(k in query_lower for k in ("receipt", "incidental", "tax invoice")):
        return "receipt_amount"
    if "exception" in query_lower:
        return "exception_amount"
    if any(k in query_lower for k in ("card", "monthly spend", "credit card")):
        return "monthly_spend"
    if any(k in query_lower for k in ("tuition", "education", "course", "degree")):
        return "annual_tuition_claim"
    if "emergency" in query_lower and any(k in query_lower for k in ("purchase", "order", "procurement")):
        return "emergency_purchase_amount"
    if any(k in query_lower for k in ("purchase order", "po ", "contract", "quote", "single quote", "three competitive")):
        return "order_amount"
    if any(k in query_lower for k in ("travel", "flight", "hotel", "per diem", "manager", "director", "approval threshold")):
        return "expense_amount"
    if "expense" in query_lower:
        return "expense_amount"
    return None


def extract_conditions_from_query(
    query: str,
    relevant_fields: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Conservatively extract typed condition values from a natural language query.

    Only extracts when unambiguous. Rejects ambiguous or conflicting field/value matches.

    Args:
        query: User question string.
        relevant_fields: Optional set of condition_field names to restrict extraction to.

    Returns:
        List of dicts: [{"field": condition_field, "value": value, "unit": unit}]
    """
    if not query or not query.strip():
        return []

    q_lower = query.lower()
    conditions: List[Dict[str, Any]] = []

    # Check for tier values (Information Security)
    if "confidential" in q_lower:
        conditions.append({"field": "data_classification_tier", "value": "Confidential", "unit": "tier"})
    elif "restricted" in q_lower and any(k in q_lower for k in ("restricted data", "restricted tier", "data classification")):
        conditions.append({"field": "data_classification_tier", "value": "Restricted", "unit": "tier"})

    # Check for INR currency amounts using Decimal precision directly from string
    found_currency_amounts: List[Tuple[Decimal, str]] = []
    has_or_connector = bool(re.search(r"\b(?:or|and/or)\b|/", q_lower))
    for match in _CURRENCY_PATTERN.finditer(query):
        raw_num = match.group(1).replace(",", "")
        try:
            val = Decimal(raw_num)
        except (InvalidOperation, ValueError):
            continue
        # Exclude years (e.g. 2026) unless explicitly preceded/followed by currency mark
        span = query[max(0, match.start() - 10) : min(len(query), match.end() + 10)].lower()
        has_curr_mark = any(k in span for k in ("inr", "rs", "₹", "rupee", "cost", "amount", "spend", "claim", "price"))
        if 2000 <= val <= 2099 and not any(k in span for k in ("inr", "rs", "₹")):
            continue
        if has_curr_mark:
            field = _determine_currency_field(q_lower, span)
            if field:
                found_currency_amounts.append((val, field))

    # Reject if multiple distinct amounts claim the same field or if query presents ambiguous alternatives
    seen_amounts_by_field: Dict[str, List[Decimal]] = {}
    for val, field in found_currency_amounts:
        seen_amounts_by_field.setdefault(field, []).append(val)

    ambiguous_fields = {f for f, vals in seen_amounts_by_field.items() if len(set(vals)) > 1}
    distinct_vals = {v for v, _ in found_currency_amounts}
    if len(distinct_vals) > 1 and has_or_connector:
        for f in seen_amounts_by_field:
            ambiguous_fields.add(f)

    if ambiguous_fields:
        for f in ambiguous_fields:
            vals_str = ", ".join(str(v) for v in sorted(set(seen_amounts_by_field[f])))
            conditions.append({
                "field": f,
                "value": None,
                "unit": "INR",
                "is_ambiguous": True,
                "error": f"Multiple conflicting amounts [{vals_str}] detected for condition '{f}'.",
            })
    else:
        for val, field in found_currency_amounts:
            conditions.append({"field": field, "value": val, "unit": "INR", "is_ambiguous": False})

    # Check for days
    for match in _DAYS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("advance", "booking", "departure", "flight")):
            conditions.append({"field": "advance_booking_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("submit", "submission", "delay", "expense claim", "reimbursement")):
            conditions.append({"field": "submission_delay_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("sick", "illness", "medical")):
            conditions.append({"field": "consecutive_sick_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("recertification", "access review", "privileged")):
            conditions.append({"field": "privileged_role_review_interval_days", "value": val, "unit": "days"})

    # Check for months
    for match in _MONTHS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("contract", "duration", "agreement")):
            conditions.append({"field": "contract_duration_months", "value": val, "unit": "months"})

    # Check for mbps
    for match in _MBPS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        conditions.append({"field": "internet_speed_mbps", "value": val, "unit": "mbps"})

    # Check for hours
    for match in _HOURS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("incident", "p1", "severity", "sla")):
            conditions.append({"field": "incident_severity_p1_response_hours", "value": val, "unit": "hours"})

    # Filter to relevant fields if specified
    if relevant_fields is not None:
        conditions = [c for c in conditions if c["field"] in relevant_fields]

    return conditions


def extract_condition_result(
    query: str,
    relevant_fields: Optional[Set[str]] = None,
) -> ConditionExtractionResult:
    """Return structured ConditionExtractionResult distinguishing valid, missing, and ambiguous conditions."""
    if not query or not query.strip():
        return ConditionExtractionResult(status="missing", conditions=[])

    conds = extract_conditions_from_query(query, relevant_fields=relevant_fields)
    if any(c.get("is_ambiguous") for c in conds):
        ambig_reasons = [c.get("error") for c in conds if c.get("is_ambiguous")]
        return ConditionExtractionResult(
            status="ambiguous",
            conditions=conds,
            reason="; ".join(filter(None, ambig_reasons)) or "Ambiguous condition values in query",
        )
    if conds:
        return ConditionExtractionResult(status="valid", conditions=conds)
    return ConditionExtractionResult(status="missing", conditions=[])
