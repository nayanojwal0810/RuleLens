"""Deterministic structured rule resolution for RuleLens.

Loads, links, filters, and evaluates deterministic policy rules from
data/processed/rules.jsonl against authorized chunks, user roles, query dates,
and condition values.
"""

from __future__ import annotations

import datetime
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


# Cache loaded rules
_RULES_CACHE: Optional[List[RuleRecord]] = None


def load_rules(base_dir: Optional[Union[Path, str]] = None, reload: bool = False) -> List[RuleRecord]:
    """Load and validate all 32 canonical rules from rules.jsonl.

    Args:
        base_dir: Repository root directory. Defaults to cwd.
        reload: If True, bypass cache and reload from disk.

    Returns:
        List of RuleRecord objects.
    """
    global _RULES_CACHE
    if _RULES_CACHE is not None and not reload:
        return _RULES_CACHE

    base = Path(base_dir) if base_dir else Path.cwd()
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

    _RULES_CACHE = rules
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
    """Evaluate whether an input value satisfies a rule's condition.

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

    # Numeric comparisons: <, <=, >, >=
    try:
        numeric_val = float(value)
        numeric_threshold = float(threshold)
    except (ValueError, TypeError) as exc:
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
    cond_map: Dict[str, Tuple[Any, Optional[str]]] = {}
    if extracted_conditions:
        for cond in extracted_conditions:
            field = cond.get("field")
            if field and "value" in cond:
                cond_map[field] = (cond["value"], cond.get("unit"))

    results: List[RuleEvaluationResult] = []

    for rule in all_rules:
        # 1. Chunk must be present in authorized evidence
        if rule.chunk_id not in authorized_cids:
            continue

        # 2. Rule must be active on query date
        if not is_rule_active(rule, q_date):
            continue

        # 3. If condition value provided for this field, evaluate it
        if rule.condition_field in cond_map:
            val, unit = cond_map[rule.condition_field]
            try:
                passed = evaluate_rule_condition(rule, val, unit)
            except (ValueError, TypeError) as exc:
                # Incompatible unit or type: fails closed
                passed = False

            explanation = (
                f"Rule {rule.rule_id} ({rule.condition_field} {rule.operator} {rule.threshold_value} {rule.unit}): "
                f"Input {val} {unit or rule.unit} -> {'MET' if passed else 'NOT MET'} -> outcome: {rule.outcome}"
            )
            results.append(
                RuleEvaluationResult(
                    rule=rule,
                    passed=passed,
                    input_value=val,
                    input_unit=unit or rule.unit,
                    outcome=rule.outcome,
                    explanation=explanation,
                )
            )
        else:
            # Rule applies structurally to the clause, but no input value was given
            explanation = (
                f"Rule {rule.rule_id} ({rule.condition_field} {rule.operator} {rule.threshold_value} {rule.unit}): "
                f"Threshold definition for {rule.outcome}."
            )
            results.append(
                RuleEvaluationResult(
                    rule=rule,
                    passed=True,
                    input_value=None,
                    input_unit=rule.unit,
                    outcome=rule.outcome,
                    explanation=explanation,
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


def extract_conditions_from_query(
    query: str,
    relevant_fields: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Conservatively extract typed condition values from a natural language query.

    Only extracts when unambiguous. Returns empty list if no clear condition is found.

    Args:
        query: User question string.
        relevant_fields: Optional set of condition_field names to restrict extraction to.

    Returns:
        List of dicts: [{"field": condition_field, "value": value, "unit": unit}]
    """
    if not query or not query.strip():
        return []

    conditions: List[Dict[str, Any]] = []

    # Check for tier values (Information Security)
    if "confidential" in query.lower():
        if relevant_fields is None or "data_classification_tier" in relevant_fields:
            conditions.append({"field": "data_classification_tier", "value": "Confidential", "unit": "tier"})
    elif "restricted" in query.lower() and "restricted data" in query.lower():
        if relevant_fields is None or "data_classification_tier" in relevant_fields:
            conditions.append({"field": "data_classification_tier", "value": "Restricted", "unit": "tier"})

    # Check for INR currency amounts
    # Avoid picking up years like 2026 as amounts: check for explicit currency symbol, keyword, or large values
    for match in _CURRENCY_PATTERN.finditer(query):
        raw_num = match.group(1).replace(",", "")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        # Check surrounding text for currency markers or context
        span = query[max(0, match.start() - 10) : min(len(query), match.end() + 10)].lower()
        has_currency_marker = any(k in span for k in ("inr", "rs", "₹", "rupee", "cost", "amount", "spend", "claim", "price"))
        # Exclude years (e.g. 2026) unless explicitly marked with INR
        if 2000 <= val <= 2099 and not any(k in span for k in ("inr", "rs", "₹")):
            continue

        if has_currency_marker:
            # Map to potential currency fields if relevant_fields specifies them
            cand_fields = [
                "expense_amount",
                "exception_amount",
                "receipt_amount",
                "monthly_spend",
                "annual_tuition_claim",
                "order_amount",
                "emergency_purchase_amount",
            ]
            for f in cand_fields:
                if relevant_fields is None or f in relevant_fields:
                    conditions.append({"field": f, "value": val, "unit": "INR"})

    # Check for days
    for match in _DAYS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        cand_fields = [
            "advance_booking_days",
            "submission_delay_days",
            "consecutive_sick_days",
            "privileged_role_review_interval_days",
        ]
        for f in cand_fields:
            if relevant_fields is None or f in relevant_fields:
                conditions.append({"field": f, "value": val, "unit": "days"})

    # Check for months
    for match in _MONTHS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        cand_fields = ["contract_duration_months"]
        for f in cand_fields:
            if relevant_fields is None or f in relevant_fields:
                conditions.append({"field": f, "value": val, "unit": "months"})

    # Check for mbps
    for match in _MBPS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        cand_fields = ["internet_speed_mbps"]
        for f in cand_fields:
            if relevant_fields is None or f in relevant_fields:
                conditions.append({"field": f, "value": val, "unit": "mbps"})

    # Check for hours
    for match in _HOURS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        cand_fields = ["incident_severity_p1_response_hours"]
        for f in cand_fields:
            if relevant_fields is None or f in relevant_fields:
                conditions.append({"field": f, "value": val, "unit": "hours"})

    return conditions
