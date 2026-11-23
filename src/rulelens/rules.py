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


@dataclass(frozen=True)
class AuthoritativeDecision:
    """Represents a deterministic, authoritative rule decision."""

    is_authoritative: bool
    rule: Optional[RuleRecord] = None
    answer: Optional[str] = None
    citation_chunk_id: Optional[str] = None
    rule_id: Optional[str] = None
    outcome: Optional[str] = None
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
        if rule.condition_field in cond_map:
            cond_entry = cond_map.get(rule.condition_field)
            val, unit, is_ambig, ambig_err = cond_entry
            if is_ambig:
                # Ambiguous input condition
                explanation = (
                    f"Rule {rule.rule_id} condition ambiguous: "
                    f"{ambig_err or 'Multiple conflicting condition inputs detected in query.'}"
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
                err_type = "incompatible_unit" if "incompatible unit" in str(val_err).lower() else "invalid_unit"
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=val,
                        input_unit=unit or rule.unit,
                        outcome=err_type,
                        explanation=explanation,
                        is_error=True,
                        error_type=err_type,
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
    r"(?P<prefix>inr|rs\.?|₹|\$|usd|€|eur|£|gbp)?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?P<suffix>inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?)?",
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

    Only extracts when unambiguous. Rejects ambiguous or conflicting field/value matches
    across all numerical units (currency, days, months, mbps, hours) and categorical tiers.

    Args:
        query: User question string.
        relevant_fields: Optional set of condition_field names to restrict extraction to.

    Returns:
        List of dicts: [{"field": condition_field, "value": value, "unit": unit, "is_ambiguous": bool}]
    """
    if not query or not query.strip():
        return []

    q_lower = query.lower()
    raw_candidates: List[Dict[str, Any]] = []

    # Check for tier values (Information Security)
    has_confidential = "confidential" in q_lower
    has_restricted = "restricted" in q_lower and any(
        k in q_lower for k in ("restricted data", "restricted tier", "data classification", "restricted")
    )
    if has_confidential:
        raw_candidates.append({"field": "data_classification_tier", "value": "Confidential", "unit": "tier"})
    if has_restricted:
        raw_candidates.append({"field": "data_classification_tier", "value": "Restricted", "unit": "tier"})

    # Check for currency amounts using Decimal precision directly from string
    for match in _CURRENCY_PATTERN.finditer(query):
        raw_num = match.group(2).replace(",", "")
        try:
            val = Decimal(raw_num)
        except (InvalidOperation, ValueError):
            continue
        prefix = (match.group(1) or "").lower()
        suffix = (match.group(3) or "").lower()
        span = query[max(0, match.start() - 10) : min(len(query), match.end() + 10)].lower()

        # Determine explicit currency unit
        if prefix in ("$", "usd") or suffix in ("usd", "dollar", "dollars") or any(k in span for k in ("usd", "$", "dollar")):
            curr_unit = "USD"
        elif prefix in ("€", "eur") or suffix in ("eur", "euro", "euros") or any(k in span for k in ("eur", "€", "euro")):
            curr_unit = "EUR"
        elif prefix in ("£", "gbp") or suffix in ("gbp", "pound", "pounds") or any(k in span for k in ("gbp", "£", "pound")):
            curr_unit = "GBP"
        elif prefix in ("₹", "inr", "rs", "rs.") or suffix in ("inr", "rupee", "rupees") or any(k in span for k in ("inr", "rs", "₹", "rupee")):
            curr_unit = "INR"
        else:
            # Check global query currency markers
            if any(k in q_lower for k in ("usd", "$", "dollar")):
                curr_unit = "USD"
            elif any(k in q_lower for k in ("eur", "€", "euro")):
                curr_unit = "EUR"
            elif any(k in q_lower for k in ("gbp", "£", "pound")):
                curr_unit = "GBP"
            else:
                curr_unit = "INR"

        has_curr_mark = (
            prefix != ""
            or suffix != ""
            or any(k in span for k in ("inr", "rs", "₹", "rupee", "$", "usd", "dollar", "€", "eur", "euro", "£", "gbp", "pound", "cost", "amount", "spend", "claim", "price"))
        )
        if 2000 <= val <= 2099 and not (prefix or suffix or any(k in span for k in ("inr", "rs", "₹", "$", "usd", "€", "eur", "£", "gbp"))):
            continue
        if has_curr_mark:
            field = _determine_currency_field(q_lower, span)
            if field:
                raw_candidates.append({"field": field, "value": val, "unit": curr_unit})

    # Check for days
    for match in _DAYS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("advance", "booking", "departure", "flight")):
            raw_candidates.append({"field": "advance_booking_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("submit", "submission", "delay", "expense claim", "reimbursement")):
            raw_candidates.append({"field": "submission_delay_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("sick", "illness", "medical")):
            raw_candidates.append({"field": "consecutive_sick_days", "value": val, "unit": "days"})
        elif any(k in q_lower for k in ("recertification", "access review", "privileged")):
            raw_candidates.append({"field": "privileged_role_review_interval_days", "value": val, "unit": "days"})

    # Check for months
    for match in _MONTHS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("contract", "duration", "agreement")):
            raw_candidates.append({"field": "contract_duration_months", "value": val, "unit": "months"})

    # Check for mbps
    for match in _MBPS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        raw_candidates.append({"field": "internet_speed_mbps", "value": val, "unit": "mbps"})

    # Check for hours
    for match in _HOURS_PATTERN.finditer(query):
        try:
            val = int(match.group(1))
        except ValueError:
            continue
        if any(k in q_lower for k in ("incident", "p1", "severity", "sla")):
            raw_candidates.append({"field": "incident_severity_p1_response_hours", "value": val, "unit": "hours"})

    # Filter to relevant fields if specified
    if relevant_fields is not None:
        raw_candidates = [c for c in raw_candidates if c["field"] in relevant_fields]

    # Group candidate conditions by condition_field
    candidates_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for cand in raw_candidates:
        field = cand["field"]
        candidates_by_field.setdefault(field, []).append(cand)

    has_alternative_connector = bool(re.search(r"\b(?:or|and/or|versus|vs\.?)\b|/", q_lower))
    conditions: List[Dict[str, Any]] = []

    for field, c_list in candidates_by_field.items():
        # Check for conflicting units
        units = list(dict.fromkeys(c["unit"] for c in c_list if c.get("unit")))
        if len(units) > 1:
            conditions.append({
                "field": field,
                "value": None,
                "unit": units[0],
                "is_ambiguous": True,
                "error": f"Conflicting units [{', '.join(sorted(units))}] detected for condition '{field}'.",
            })
            continue

        primary_unit = units[0] if units else ""

        # Check for multiple distinct values
        vals = [c["value"] for c in c_list]
        unique_vals = list(dict.fromkeys(vals))

        if len(unique_vals) > 1 or (len(unique_vals) > 1 and has_alternative_connector):
            vals_str = ", ".join(str(v) for v in sorted(unique_vals, key=lambda x: str(x)))
            conditions.append({
                "field": field,
                "value": None,
                "unit": primary_unit,
                "is_ambiguous": True,
                "error": f"Multiple conflicting values [{vals_str}] detected for condition '{field}'.",
            })
        else:
            # Exactly one unique value: valid (deduplicated)
            conditions.append({
                "field": field,
                "value": unique_vals[0],
                "unit": primary_unit,
                "is_ambiguous": False,
            })

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


def _format_canonical_decision_answer(
    rule: RuleRecord,
    cond_val: Any,
    cond_unit: str,
) -> str:
    """Format an authoritative, policy-grounded natural language decision answer from a canonical rule outcome."""
    outcome = rule.outcome
    field = rule.condition_field
    thresh = rule.threshold_value
    unit = cond_unit or rule.unit

    if outcome == "standard_approval":
        return f"Standard manager approval is permitted. The expense amount of {cond_val} {unit} is within the policy approval limit of {thresh} {rule.unit}."
    elif outcome == "additional_approval":
        return f"Additional approval is required. Standard manager approval is not sufficient for an expense of {cond_val} {unit} as it exceeds the standard limit of {thresh} {rule.unit}."
    elif outcome == "standard_booking":
        return f"Standard flight booking is permitted as the travel is booked with {cond_val} days advance notice, meeting the {thresh}-day policy requirement."
    elif outcome == "expedited_approval_required":
        return f"Expedited approval is required because the flight is booked with {cond_val} days advance notice, which is less than the required {thresh} days."
    elif outcome == "vp_approval":
        return f"Vice President (VP) approval is required for travel exception expenses of {cond_val} {unit} (up to {thresh} {rule.unit})."
    elif outcome == "cfo_approval":
        return f"Chief Financial Officer (CFO) approval is required for travel exception expenses of {cond_val} {unit} (exceeding {thresh} {rule.unit})."
    elif outcome == "eligible_reimbursement":
        return f"The expense claim submitted within {cond_val} days is eligible for standard reimbursement (within the {thresh}-day limit)."
    elif outcome == "requires_late_exception":
        return f"A late submission exception approval is required for expense claims submitted after {cond_val} days (exceeding the {thresh}-day deadline)."
    elif outcome == "no_receipt_required":
        return f"Incidental business expenses of {cond_val} {unit} (up to {thresh} {rule.unit}) do not require an itemized tax receipt, provided digital payment confirmation is attached."
    elif outcome == "itemized_receipt_required":
        return f"An itemized tax invoice is strictly required for incidental expenses of {cond_val} {unit} (exceeding {thresh} {rule.unit})."
    elif outcome == "within_standard_card_limit":
        return f"The monthly spend of {cond_val} {unit} is within the standard corporate credit card limit of {thresh} {rule.unit}."
    elif outcome == "card_limit_escalation_required":
        return f"Corporate credit card limit escalation approval is required for monthly spend of {cond_val} {unit} exceeding the standard {thresh} {rule.unit} limit."
    elif outcome == "self_certification":
        return f"Self-certification is permitted for sick leave of {cond_val} consecutive days (up to {thresh} days)."
    elif outcome == "medical_certificate_required":
        return f"A formal medical certificate is required for sick leave of {cond_val} consecutive days (exceeding {thresh} days)."
    elif outcome == "remote_setup_compliant":
        return f"The internet connection speed of {cond_val} Mbps meets the remote work setup policy minimum of {thresh} {rule.unit}."
    elif outcome == "remediation_required":
        return f"Internet speed remediation is required because {cond_val} Mbps does not meet the minimum requirement of {thresh} {rule.unit}."
    elif outcome == "standard_tuition_grant":
        return f"The tuition claim of {cond_val} {unit} is eligible for a standard tuition grant (within {thresh} {rule.unit})."
    elif outcome == "committee_review_required":
        return f"Tuition reimbursement committee review is required for claims of {cond_val} {unit} exceeding the standard {thresh} {rule.unit} grant."
    elif outcome == "single_quote_permitted":
        return f"A single supplier quote is permitted for purchase orders of {cond_val} {unit} (up to {thresh} {rule.unit})."
    elif outcome == "three_competitive_quotes_required":
        return f"Three competitive supplier quotes are required for purchase orders of {cond_val} {unit} (exceeding {thresh} {rule.unit})."
    elif outcome == "department_lead_signoff":
        return f"Department lead signoff is sufficient for vendor contracts of {cond_val} months duration (up to {thresh} months)."
    elif outcome == "legal_and_procurement_signoff":
        return f"Both Legal and Procurement signoff are required for vendor contracts of {cond_val} months duration (exceeding {thresh} months)."
    elif outcome == "director_emergency_approval":
        return f"Director emergency approval is permitted for emergency purchases of {cond_val} {unit} (up to {thresh} {rule.unit})."
    elif outcome == "board_procurement_committee":
        return f"Board Procurement Committee approval is required for emergency purchases of {cond_val} {unit} (exceeding {thresh} {rule.unit})."
    elif outcome == "aes_256_mandatory":
        return "AES-256 encryption is mandatory for Confidential tier data at rest."
    elif outcome == "hsm_tokenized_encryption":
        return "Hardware Security Module (HSM) tokenized encryption is mandatory for Restricted tier data at rest."
    elif outcome == "access_recertification_compliant":
        return f"Privileged access review interval of {cond_val} days is compliant within the permitted {thresh}-day recertification cycle."
    elif outcome == "automatic_account_suspension":
        return f"Automatic account suspension applies to privileged accounts exceeding the {thresh}-day recertification interval ({cond_val} days)."
    elif outcome == "sla_met":
        return f"Priority 1 incident response within {cond_val} hour(s) meets the SLA requirement of {thresh} hour(s)."
    elif outcome == "sla_breach_escalation":
        return f"Priority 1 incident response time of {cond_val} hour(s) exceeds the {thresh}-hour SLA window, requiring immediate breach escalation."
    else:
        return f"The policy rule ({rule.rule_id}) establishes the outcome '{outcome}' for {field} {rule.operator} {thresh} {rule.unit}."


_DECISION_BENIGN_TERMS: Set[str] = {
    # Functional / grammatical terms
    "what", "is", "are", "was", "were", "can", "could", "does", "do", "did", "may", "might",
    "a", "an", "the", "of", "for", "under", "in", "on", "at", "by", "with", "about", "within",
    "after", "before", "to", "from", "than", "into", "through", "during", "and", "or", "as", "if",
    "not", "no", "per", "via", "without", "this", "that", "these", "those", "who", "whom", "which",
    # Currency and units
    "inr", "rs", "rupee", "rupees", "usd", "eur", "gbp", "dollar", "dollars", "euro", "euros", "pound", "pounds",
    "day", "days", "month", "months", "hour", "hours", "mbps",
    # Decision / permission framing
    "approve", "approval", "approved", "permits", "permit", "permitted", "permission", "permissible",
    "allows", "allow", "allowed", "allowable", "requires", "require", "required", "requirement",
    "standard", "additional", "compliant", "compliance", "meet", "meets", "meeting", "suffice",
    "sufficient", "enough", "exceed", "exceeds", "exceeding", "eligible", "eligibility", "status",
    # Corporate roles and entities
    "director", "manager", "lead", "employee", "employees", "role", "roles", "policy", "corporate",
    "company", "asterline", "group", "threshold", "limit", "limits", "amount", "expense", "expenses",
    "spend", "spending", "claim", "claims", "reimbursement", "reimburse", "cost", "costing", "price", "direct",
    "functional", "department", "reporting",
    # Canonical condition field domain vocabulary
    "booking", "notice", "advance", "departure", "flight", "receipt", "tax", "invoice", "itemized",
    "incidental", "payment", "confirmation", "attached", "quote", "quotes", "single", "competitive",
    "three", "two", "multiple", "supplier", "purchase", "order", "po", "emergency", "card", "credit",
    "monthly", "sick", "illness", "leave", "medical", "certificate", "self", "remote", "setup", "work",
    "internet", "speed", "tuition", "annual", "grant", "committee", "contract", "vendor", "agreement",
    "duration", "signoff", "legal", "procurement", "board", "data", "classification", "tier",
    "confidential", "restricted", "encryption", "tokenized", "hsm", "aes", "privileged", "access",
    "review", "recertification", "suspension", "automatic", "cycle", "interval", "incident", "severity",
    "priority", "response", "time", "window", "sla", "breach", "escalation", "late", "exception", "exceptions",
    # Document tags
    "doc", "trv", "fin", "ppo", "prc", "sec",
    "c04", "b02", "r03", "p02", "e01", "a02", "i02", "v1", "v2", "v3",
    # Broad travel qualifiers
    "travel", "trip", "trips", "business", "personal",
}

_RULE_FIELD_SUBJECT_TERMS: Dict[str, Set[str]] = {
    "expense_amount": {"travel", "trip", "trv", "lodging", "hotel", "meal", "per diem", "policy v1", "policy v2", "travel policy"},
    "advance_booking_days": {"booking", "book", "flight", "departure", "advance", "notice"},
    "receipt_amount": {"receipt", "invoice", "incidental", "itemized"},
    "order_amount": {"quote", "supplier", "purchase", "order", "po", "procurement"},
    "emergency_purchase_amount": {"emergency", "purchase", "procurement"},
    "monthly_spend": {"card", "corporate card", "credit card", "monthly spend"},
    "consecutive_sick_days": {"sick", "illness", "leave", "medical", "certificate"},
    "internet_speed_mbps": {"internet", "speed", "remote", "connection", "mbps"},
    "annual_tuition_claim": {"tuition", "course", "education", "degree", "grant"},
    "contract_duration_months": {"contract", "vendor", "agreement", "duration"},
    "privileged_role_review_interval_days": {"privileged", "access", "review", "recertification", "suspension"},
    "incident_severity_p1_response_hours": {"incident", "severity", "p1", "priority 1", "response", "sla"},
    "data_classification_tier": {"classification", "tier", "confidential", "restricted", "encryption"},
    "submission_delay_days": {"submit", "submission", "delay", "deadline"},
}


def _find_unsupported_query_terms(query: str, chunk_dict: Union[Dict[str, Any], Any]) -> List[str]:
    """Find query terms that introduce unsupported requirements not covered by chunk text, title, or benign framing."""
    if isinstance(chunk_dict, dict):
        c_text = (
            (chunk_dict.get("text") or "") + " " +
            (chunk_dict.get("title") or "") + " " +
            (chunk_dict.get("department") or "")
        ).lower()
    else:
        c_text = (
            (getattr(chunk_dict, "text", "") or "") + " " +
            (getattr(chunk_dict, "title", "") or "") + " " +
            (getattr(chunk_dict, "department", "") or "")
        ).lower()

    words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
    unsupported = []
    for w in words:
        if w in _DECISION_BENIGN_TERMS:
            continue
        stem = w.rstrip("s")
        if not re.search(r"\b" + re.escape(stem), c_text):
            unsupported.append(w)
    return unsupported


def evaluate_authoritative_rule_decision(
    question: str,
    authorized_chunks: Sequence[Dict[str, Any]],
    rule_results: Sequence[RuleEvaluationResult],
    extracted_conditions: Sequence[Dict[str, Any]],
) -> AuthoritativeDecision:
    """Evaluate whether a single-condition policy query can be answered deterministically from canonical rules.

    Requirements:
    1. Query asks for a concrete policy decision (e.g., approval, permission, compliance).
    2. Exactly one unambiguous condition is extracted.
    3. Exactly one active, authorized rule condition passes for this condition.
    4. Query aligns with the matched rule's policy subject.
    5. The passing rule's chunk is present in authorized evidence.
    6. Query contains no additional unsupported requirements not present in evidence.
    7. Query is not an open-ended definition question or multi-clause question.
    """
    if not question or not question.strip():
        return AuthoritativeDecision(is_authoritative=False)

    q_lower = question.lower()

    # Rule out open-ended threshold definitions or cross-clause questions
    explanatory_markers = (
        "what is the maximum", "what is the threshold", "what is the limit",
        "what are the tiers", "into what four", "what four severity",
        "and what timeframe", "and who", "and when", "and within what"
    )
    if any(em in q_lower for em in explanatory_markers):
        return AuthoritativeDecision(is_authoritative=False)

    # Must contain a decision indicator
    decision_markers = (
        "can ", "could ", "is ", "are ", "does ", "do ", "may ", "require", "required",
        "permit", "permitted", "allow", "allowed", "eligible", "approval", "approve",
        "compliant", "exceed", "escalat"
    )
    if not any(m in q_lower for m in decision_markers):
        return AuthoritativeDecision(is_authoritative=False)

    # Must have exactly one valid condition, and no ambiguous conditions
    if any(c.get("is_ambiguous") for c in extracted_conditions):
        return AuthoritativeDecision(is_authoritative=False)

    valid_conds = [c for c in extracted_conditions if c.get("value") is not None and not c.get("is_ambiguous")]
    if len(valid_conds) != 1:
        return AuthoritativeDecision(is_authoritative=False)

    target_cond = valid_conds[0]
    cond_field = target_cond["field"]
    cond_val = target_cond["value"]
    cond_unit = target_cond.get("unit", "")

    # Find rules evaluated against this condition field with input_value
    evaluated_for_field = [
        r for r in rule_results
        if not r.is_error and r.input_value is not None and r.rule.condition_field == cond_field
    ]
    passing_rules = [r for r in evaluated_for_field if r.passed]
    if len(passing_rules) != 1:
        return AuthoritativeDecision(is_authoritative=False)

    passing_result = passing_rules[0]
    rule = passing_result.rule

    # 1. Subject match verification: Question must be about the matched rule's policy subject
    subject_terms = _RULE_FIELD_SUBJECT_TERMS.get(cond_field)
    if subject_terms and not any(st in q_lower for st in subject_terms):
        return AuthoritativeDecision(is_authoritative=False)

    # 2. Verify rule's linked chunk is present in authorized chunks
    rule_chunk = None
    for c in authorized_chunks:
        cid = c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", None)
        if cid == rule.chunk_id:
            rule_chunk = c
            break

    if not rule_chunk:
        return AuthoritativeDecision(is_authoritative=False)

    # 3. Unsupported requirements check: Question must not contain additional unsupported entities
    unsupported_terms = _find_unsupported_query_terms(question, rule_chunk)
    if unsupported_terms:
        return AuthoritativeDecision(is_authoritative=False)

    answer_text = _format_canonical_decision_answer(rule, cond_val, cond_unit)

    return AuthoritativeDecision(
        is_authoritative=True,
        rule=rule,
        answer=answer_text,
        citation_chunk_id=rule.chunk_id,
        rule_id=rule.rule_id,
        outcome=rule.outcome,
        reason=f"Authoritative deterministic decision based on canonical rule {rule.rule_id} ({rule.outcome}).",
    )
