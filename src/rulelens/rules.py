"""Deterministic structured rule resolution for RuleLens.

Loads, links, filters, and evaluates deterministic policy rules from
data/processed/rules.jsonl against authorized chunks, user roles, query dates,
and condition values.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import hashlib
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
class NumericalBoundaryAnalysis:
    """Outcome of analyzing a numerical query condition against active structured rules."""

    status: str  # "single_outcome", "crossing_outcomes", "cannot_evaluate"
    satisfying_rule: Optional[RuleRecord] = None
    outcome: Optional[str] = None
    intersecting_rules: List[RuleRecord] = None
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
    unsupported_terms: Optional[List[str]] = None
    is_boundary_crossing: bool = False
    is_unresolved: bool = False
    is_decision_request: bool = False


# Cache loaded rules keyed by base_dir to avoid cross-corpus pollution
_RULES_CACHE: Dict[Path, List[RuleRecord]] = {}
@dataclass(frozen=True)
class _RulesCacheEntry:
    fingerprint: str
    rules: List[RuleRecord]


# Content-fingerprinted cache of loaded rules keyed by base_dir to avoid cross-corpus pollution
# and stale in-memory reuse when the underlying rules file changes.
_RULES_CACHE: Dict[Path, _RulesCacheEntry] = {}


def clear_rules_cache() -> None:
    """Clear the in-memory rules cache."""
    global _RULES_CACHE
    _RULES_CACHE.clear()


def get_rules_fingerprint(base_dir: Optional[Union[Path, str]] = None) -> str:
    """Compute content fingerprint (SHA-256) of canonical rules file."""
    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    rules_path = base / "data" / "processed" / "rules.jsonl"
    if not rules_path.is_file():
        alt_path = base.parent / "data" / "processed" / "rules.jsonl"
        if alt_path.is_file():
            rules_path = alt_path
        else:
            raise FileNotFoundError(f"Canonical rules file not found: {rules_path}")
    return hashlib.sha256(rules_path.read_bytes()).hexdigest()


def load_rules(base_dir: Optional[Union[Path, str]] = None, reload: bool = False) -> List[RuleRecord]:
    """Load and validate all 32 canonical rules from rules.jsonl.

    Validates content fingerprint on every call to prevent stale rule reuse
    when the rules file or validated corpus changes on disk.

    Args:
        base_dir: Repository root directory. Defaults to cwd.
        reload: If True, bypass cache and reload from disk.

    Returns:
        List of RuleRecord objects.

    Raises:
        FileNotFoundError: If rules.jsonl cannot be found.
        ValueError: If file is malformed or rule count != 32.
    """
    global _RULES_CACHE
    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    rules_path = base / "data" / "processed" / "rules.jsonl"
    if not rules_path.is_file():
        # Check parent in case cwd is inside tests or a subpackage
        alt_path = base.parent / "data" / "processed" / "rules.jsonl"
        if alt_path.is_file():
            rules_path = alt_path
        else:
            raise FileNotFoundError(f"Canonical rules file not found: {rules_path}")

    # Read raw content to compute content fingerprint
    raw_bytes = rules_path.read_bytes()
    content_fingerprint = hashlib.sha256(raw_bytes).hexdigest()

    # Reuse cache only if not reload and content fingerprint matches
    if not reload and base in _RULES_CACHE:
        entry = _RULES_CACHE[base]
        if entry.fingerprint == content_fingerprint:
            return entry.rules

    # Parse and validate the new content
    rules = []
    lines = raw_bytes.decode("utf-8").splitlines()
    for line_num, line in enumerate(lines, 1):
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

    # Only update cache if validation succeeds completely
    _RULES_CACHE[base] = _RulesCacheEntry(fingerprint=content_fingerprint, rules=rules)
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
    comparison: str = "exact",
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> bool:
    """Evaluate whether an input condition satisfies a rule using exact Decimal comparisons.

    Supports directional comparisons and boundary checks:
    - exact: value == / <= / >= / < / > threshold
    - greater_than: values > value uniformly satisfy rule
    - greater_than_or_equal: values >= value uniformly satisfy rule
    - less_than: values < value uniformly satisfy rule
    - less_than_or_equal: values <= value uniformly satisfy rule
    - range: all values in [min_value, max_value] uniformly satisfy rule

    Args:
        rule: RuleRecord or dict.
        value: Input value to test.
        unit: Optional unit for comparison.
        comparison: Comparison modifier string.
        min_value: Minimum value for range comparison.
        max_value: Maximum value for range comparison.

    Returns:
        True if the condition uniformly satisfies the rule, False otherwise.

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

    comp = (comparison or "exact").strip().lower()

    # Operator evaluation for categorical equality
    if rule_op == "==":
        if comp != "exact":
            return False
        if isinstance(threshold, str):
            return str(value).strip().lower() == threshold.strip().lower()
        return value == threshold

    # Exact decimal numeric comparisons: <, <=, >, >=
    try:
        clean_thresh_str = str(threshold).strip().replace(",", "")
        numeric_threshold = Decimal(clean_thresh_str)

        if comp == "range":
            c_min = min_value if min_value is not None else (value[0] if isinstance(value, (list, tuple)) else value)
            c_max = max_value if max_value is not None else (value[1] if isinstance(value, (list, tuple)) else value)
            numeric_min = Decimal(str(c_min).strip().replace(",", ""))
            numeric_max = Decimal(str(c_max).strip().replace(",", ""))
            if numeric_min > numeric_max:
                numeric_min, numeric_max = numeric_max, numeric_min
            if numeric_min < 0:
                raise ValueError(f"Negative values ({numeric_min}) are unsupported for policy rules.")
        else:
            clean_val_str = str(value).strip().replace(",", "")
            numeric_val = Decimal(clean_val_str)
            if numeric_val < 0:
                raise ValueError(f"Negative values ({numeric_val}) are unsupported for policy rules.")
    except (InvalidOperation, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and "Negative values" in str(exc):
            raise
        raise TypeError(
            f"Cannot evaluate operator '{rule_op}' with non-numeric value '{value}' and threshold '{threshold}'."
        ) from exc

    # Range condition evaluation
    if comp == "range":
        if rule_op == "<=":
            return numeric_max <= numeric_threshold
        elif rule_op == "<":
            return numeric_max < numeric_threshold
        elif rule_op == ">=":
            return numeric_min >= numeric_threshold
        elif rule_op == ">":
            return numeric_min > numeric_threshold
        return False

    # Exact value comparison
    if comp == "exact":
        if rule_op == "<":
            return numeric_val < numeric_threshold
        elif rule_op == "<=":
            return numeric_val <= numeric_threshold
        elif rule_op == ">":
            return numeric_val > numeric_threshold
        elif rule_op == ">=":
            return numeric_val >= numeric_threshold
        return False

    # Directional comparison: greater_than (values strictly > numeric_val)
    if comp == "greater_than":
        if rule_op in (">", ">="):
            return numeric_val >= numeric_threshold
        return False

    # Directional comparison: greater_than_or_equal (values >= numeric_val)
    if comp == "greater_than_or_equal":
        if rule_op == ">=":
            return numeric_val >= numeric_threshold
        elif rule_op == ">":
            return numeric_val > numeric_threshold
        return False

    # Directional comparison: less_than (values strictly < numeric_val)
    if comp == "less_than":
        if rule_op in ("<=", "<"):
            return numeric_val <= numeric_threshold
        return False

    # Directional comparison: less_than_or_equal (values <= numeric_val)
    if comp == "less_than_or_equal":
        if rule_op == "<=":
            return numeric_val <= numeric_threshold
        elif rule_op == "<":
            return numeric_val < numeric_threshold
        return False

    raise ValueError(f"Unsupported rule comparison modifier: '{comp}'.")


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
    cond_map: Dict[str, Dict[str, Any]] = {}
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
                cond_map[field] = cond

    # Dynamically analyze numerical boundaries against active, authorized rules for this query date
    active_authorized_rules = [
        r for r in all_rules
        if r.chunk_id in authorized_cids and is_rule_active(r, q_date)
    ]
    for field, cond_entry in cond_map.items():
        if not cond_entry.get("is_ambiguous", False) and cond_entry.get("value") is not None:
            field_rules = [r for r in active_authorized_rules if r.condition_field == field]
            if field_rules:
                b_res = analyze_numerical_boundary(
                    active_rules=field_rules,
                    comparison=cond_entry.get("comparison", "exact"),
                    value=cond_entry.get("value"),
                    unit=cond_entry.get("unit"),
                    min_value=cond_entry.get("min_value"),
                    max_value=cond_entry.get("max_value"),
                )
                if b_res.status == "crossing_outcomes":
                    cond_entry["is_boundary_crossing"] = True
                elif b_res.status == "cannot_evaluate":
                    cond_entry["cannot_evaluate"] = True
                    cond_entry["cannot_evaluate_reason"] = b_res.reason

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
            cond_entry = cond_map[rule.condition_field]
            is_ambig = bool(cond_entry.get("is_ambiguous", False))
            ambig_err = cond_entry.get("error")
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

            if cond_entry.get("cannot_evaluate"):
                reason_str = str(cond_entry.get("cannot_evaluate_reason") or "")
                if "incompatible unit" in reason_str.lower() or "missing unit" in reason_str.lower():
                    err_type = "incompatible_unit"
                    explanation = f"Rule {rule.rule_id} unit error: {reason_str}"
                elif "unsupported numerical expression" in reason_str.lower() or "non-numeric" in reason_str.lower():
                    err_type = "invalid_type"
                    explanation = f"Rule {rule.rule_id} type error: {reason_str}"
                elif "negative" in reason_str.lower():
                    err_type = "cannot_evaluate"
                    explanation = f"Rule {rule.rule_id} condition cannot be evaluated: {reason_str}"
                else:
                    err_type = "cannot_evaluate"
                    explanation = f"Rule {rule.rule_id} condition cannot be evaluated: {reason_str}"
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=cond_entry.get("value"),
                        input_unit=rule.unit,
                        outcome=err_type,
                        explanation=explanation,
                        is_error=True,
                        error_type=err_type,
                    )
                )
                continue

            val = cond_entry.get("value")
            unit = cond_entry.get("unit")
            comp = cond_entry.get("comparison", "exact")
            min_val = cond_entry.get("min_value")
            max_val = cond_entry.get("max_value")

            try:
                passed = evaluate_rule_condition(
                    rule, val, unit, comparison=comp, min_value=min_val, max_value=max_val
                )
                comp_desc = f" ({comp})" if comp != "exact" else ""
                if passed:
                    explanation = (
                        f"Rule {rule.rule_id} condition satisfied: {rule.condition_field} {rule.operator} "
                        f"{rule.threshold_value} {rule.unit} for input {val}{comp_desc} {unit or rule.unit} -> outcome: {rule.outcome}"
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
                        f"{rule.threshold_value} {rule.unit} for input {val}{comp_desc} {unit or rule.unit}."
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
                # Incompatible unit or negative value: invalid rule evaluation error
                explanation = f"Rule {rule.rule_id} error: {val_err}"
                if "incompatible unit" in str(val_err).lower():
                    err_type = "incompatible_unit"
                elif "negative value" in str(val_err).lower():
                    err_type = "unsupported_negative_value"
                else:
                    err_type = "invalid_unit"
                results.append(
                    RuleEvaluationResult(
                        rule=rule,
                        passed=False,
                        input_value=val,
                        input_unit=unit or rule.unit,
                        outcome="rule_error",
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
    r"(?P<neg1>-)?\s*(?P<prefix>inr|rs\.?|₹|\$|usd|€|eur|£|gbp)?\s*(?P<neg2>-)?\s*(?P<num>[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?P<suffix>inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?)?",
    re.IGNORECASE,
)
_RANGE_CURRENCY_PATTERN = re.compile(
    r"\b(?:between|from)\s+(?:(?P<neg1>-)?\s*(?P<pre1>inr|rs\.?|₹|\$|usd|€|eur|£|gbp)?\s*(?P<neg2>-)?\s*(?P<num1>[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?))\s*(?:(?P<suf1>inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?)\s*)?(?:and|to)\s+(?:(?P<neg3>-)?\s*(?P<pre2>inr|rs\.?|₹|\$|usd|€|eur|£|gbp)?\s*(?P<neg4>-)?\s*(?P<num2>[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?))\s*(?:(?P<suf2>inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?))?\b",
    re.IGNORECASE,
)
_DAYS_PATTERN = re.compile(
    r"(?P<neg>-)?\s*\b(?P<num>[0-9]+)\s*(?:consecutive\s*)?(?:business\s*|working\s*|calendar\s*)?days?\b",
    re.IGNORECASE,
)
_RANGE_DAYS_PATTERN = re.compile(
    r"\b(?:between|from)\s+(?:(?P<neg1>-)?\s*(?P<num1>[0-9]+))\s+(?:and|to)\s+(?:(?P<neg2>-)?\s*(?P<num2>[0-9]+))\s*(?:consecutive\s*)?(?:business\s*|working\s*|calendar\s*)?days?\b",
    re.IGNORECASE,
)
_MONTHS_PATTERN = re.compile(
    r"(?P<neg>-)?\s*\b(?P<num>[0-9]+)\s*months?\b",
    re.IGNORECASE,
)
_MBPS_PATTERN = re.compile(
    r"(?P<neg>-)?\s*\b(?P<num>[0-9]+)\s*mbps\b",
    re.IGNORECASE,
)
_HOURS_PATTERN = re.compile(
    r"(?P<neg>-)?\s*\b(?P<num>[0-9]+)\s*hours?\b",
    re.IGNORECASE,
)

def _detect_comparison_operator(query: str, match_start: int, match_end: int) -> str:
    """Detect directional comparison modifiers in the vicinity of a numerical match."""
    q_lower = query.lower()
    pre_window = q_lower[max(0, match_start - 35) : match_start]
    post_window = q_lower[match_end : min(len(q_lower), match_end + 25)]

    # Check for negated comparisons first in pre_window
    if re.search(r"\b(?:not\s+exceed(?:ing|s)?|does\s+not\s+exceed|not\s+more\s+than|not\s+above|not\s+over)\b", pre_window):
        return "less_than_or_equal"
    if re.search(r"\b(?:not\s+less\s+than|not\s+below|not\s+under)\b", pre_window):
        return "greater_than_or_equal"

    # Post-window modifiers: "50,000 or more", "50,000 and above", "50,000 or less"
    if re.search(r"\b(?:or\s+more|or\s+above|and\s+above|or\s+higher|and\s+higher)\b", post_window):
        return "greater_than_or_equal"
    if re.search(r"\b(?:or\s+less|or\s+below|and\s+below|or\s+fewer)\b", post_window):
        return "less_than_or_equal"

    # Pre-window modifiers
    if re.search(r"\b(?:at\s+least|minimum(?:\s+of)?|no\s+less\s+than)\b|>=|\bmin\b", pre_window):
        return "greater_than_or_equal"
    if re.search(r"\b(?:up\s+to|at\s+most|no\s+more\s+than|maximum(?:\s+of)?)\b|<=|\bmax\b", pre_window):
        return "less_than_or_equal"
    if re.search(r"\b(?:exceeding|exceeds|exceed|more\s+than|greater\s+than|above|over|strictly\s+more\s+than|strictly\s+greater\s+than)\b|>", pre_window):
        return "greater_than"
    if re.search(r"\b(?:below|less\s+than|fewer\s+than|strictly\s+less\s+than)\b|<", pre_window):
        return "less_than"
    if re.search(r"\bunder\b", pre_window) and not re.search(r"\bunder\s+(?:the\s+)?(?:corporate\s+|company\s+|travel\s+|finance\s+|procurement\s+|security\s+)?policy\b", pre_window):
        return "less_than"
    if re.search(r"\b(?:exactly|equal\s+to)\b|==", pre_window):
        return "exact"

    return "exact"


def does_condition_intersect_rule(
    rule: Union[RuleRecord, Dict[str, Any]],
    comparison: str,
    value: Any,
    unit: Optional[str] = None,
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> bool:
    """Determine whether a query condition range/value intersects a rule's condition set."""
    rule_unit = rule.unit if isinstance(rule, RuleRecord) else rule["unit"]
    rule_op = rule.operator if isinstance(rule, RuleRecord) else rule["operator"]
    threshold = rule.threshold_value if isinstance(rule, RuleRecord) else rule["threshold_value"]

    if unit is not None and rule_unit is not None:
        if unit.strip().lower() != rule_unit.strip().lower():
            return False

    comp = (comparison or "exact").strip().lower()

    if rule_op == "==":
        if comp == "exact":
            if isinstance(threshold, str):
                return str(value).strip().lower() == threshold.strip().lower()
            return value == threshold
        elif comp == "range":
            c_min = min_value if min_value is not None else (value[0] if isinstance(value, (list, tuple)) else value)
            c_max = max_value if max_value is not None else (value[1] if isinstance(value, (list, tuple)) else value)
            try:
                d_min = Decimal(str(c_min).strip().replace(",", ""))
                d_max = Decimal(str(c_max).strip().replace(",", ""))
                t_dec = Decimal(str(threshold).strip().replace(",", ""))
                if d_min > d_max:
                    d_min, d_max = d_max, d_min
                return d_min <= t_dec <= d_max
            except (InvalidOperation, ValueError, TypeError):
                return False
        return False

    try:
        clean_thresh = str(threshold).strip().replace(",", "")
        t_dec = Decimal(clean_thresh)
        if comp == "range":
            c_min = min_value if min_value is not None else (value[0] if isinstance(value, (list, tuple)) else value)
            c_max = max_value if max_value is not None else (value[1] if isinstance(value, (list, tuple)) else value)
            d_min = Decimal(str(c_min).strip().replace(",", ""))
            d_max = Decimal(str(c_max).strip().replace(",", ""))
            if d_min > d_max:
                d_min, d_max = d_max, d_min
            if d_min < 0 or d_max < 0:
                return False
        else:
            clean_val = str(value).strip().replace(",", "")
            d_val = Decimal(clean_val)
            if d_val < 0:
                return False
    except (InvalidOperation, ValueError, TypeError):
        return False

    # Check intersection based on rule operator
    if rule_op == "<=":
        if comp == "exact":
            return d_val <= t_dec
        elif comp in ("less_than", "less_than_or_equal"):
            return True
        elif comp == "greater_than":
            return d_val < t_dec
        elif comp == "greater_than_or_equal":
            return d_val <= t_dec
        elif comp == "range":
            return d_min <= t_dec

    elif rule_op == "<":
        if comp == "exact":
            return d_val < t_dec
        elif comp in ("less_than", "less_than_or_equal"):
            return True
        elif comp in ("greater_than", "greater_than_or_equal"):
            return d_val < t_dec
        elif comp == "range":
            return d_min < t_dec

    elif rule_op == ">":
        if comp == "exact":
            return d_val > t_dec
        elif comp in ("greater_than", "greater_than_or_equal"):
            return True
        elif comp in ("less_than", "less_than_or_equal"):
            return d_val > t_dec
        elif comp == "range":
            return d_max > t_dec

    elif rule_op == ">=":
        if comp == "exact":
            return d_val >= t_dec
        elif comp in ("greater_than", "greater_than_or_equal"):
            return True
        elif comp == "less_than":
            return d_val > t_dec
        elif comp == "less_than_or_equal":
            return d_val >= t_dec
        elif comp == "range":
            return d_max >= t_dec

    return False


def analyze_numerical_boundary(
    active_rules: Sequence[Union[RuleRecord, Dict[str, Any]]],
    comparison: str,
    value: Any,
    unit: Optional[str] = None,
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> NumericalBoundaryAnalysis:
    """Determine whether the requested set of values fits completely within one rule outcome,
    crosses two or more outcomes, or cannot be evaluated safely.

    Uses active, authorized structured rules dynamically for the requested policy date.
    Uses exact Decimal arithmetic.
    """
    if not active_rules:
        return NumericalBoundaryAnalysis(
            status="cannot_evaluate",
            reason="No active rules provided for numerical boundary evaluation.",
        )

    comp = (comparison or "exact").strip().lower()

    # 1. Check unit compatibility across active rules
    all_rules_have_unit = all(bool(getattr(r, "unit", None) if isinstance(r, RuleRecord) else r.get("unit")) for r in active_rules)
    if unit is not None and str(unit).strip():
        for r in active_rules:
            r_unit = r.unit if isinstance(r, RuleRecord) else r["unit"]
            if r_unit and unit.strip().lower() != r_unit.strip().lower():
                return NumericalBoundaryAnalysis(
                    status="cannot_evaluate",
                    reason=f"Incompatible unit '{unit}' for rule. Expected '{r_unit}'.",
                )
    elif all_rules_have_unit and (unit is None or not str(unit).strip()):
        return NumericalBoundaryAnalysis(
            status="cannot_evaluate",
            reason="Missing unit for numerical condition.",
        )

    # 2. Check rule intersection (which outcomes does the condition intersect?)
    # 2. Check for negative or unsupported numerical expressions
    try:
        rule_op_0 = active_rules[0].operator if isinstance(active_rules[0], RuleRecord) else active_rules[0]["operator"]
        if rule_op_0 != "==":
            if comp == "range":
                c_min = min_value if min_value is not None else (value[0] if isinstance(value, (list, tuple)) else value)
                c_max = max_value if max_value is not None else (value[1] if isinstance(value, (list, tuple)) else value)
                d_min = Decimal(str(c_min).strip().replace(",", ""))
                d_max = Decimal(str(c_max).strip().replace(",", ""))
                if d_min < 0 or d_max < 0:
                    return NumericalBoundaryAnalysis(
                        status="cannot_evaluate",
                        reason=f"Negative values [{d_min}, {d_max}] are unsupported for policy rules.",
                    )
            elif value is not None:
                d_val = Decimal(str(value).strip().replace(",", ""))
                if d_val < 0:
                    return NumericalBoundaryAnalysis(
                        status="cannot_evaluate",
                        reason=f"Negative value '{d_val}' is unsupported for policy rules.",
                    )
    except (InvalidOperation, ValueError, TypeError):
        return NumericalBoundaryAnalysis(
            status="cannot_evaluate",
            reason=f"Unsupported numerical expression: '{value}'.",
        )

    # 3. Check rule intersection (which outcomes does the condition intersect?)
    intersecting = []
    for r in active_rules:
        if does_condition_intersect_rule(r, comparison, value, unit, min_value, max_value):
            intersecting.append(r)

    intersecting_outcomes = {r.outcome if isinstance(r, RuleRecord) else r["outcome"] for r in intersecting}

    # 3. Check complete satisfaction (which rule outcome completely contains all query values?)
    satisfying = []
    for r in active_rules:
        try:
            if evaluate_rule_condition(r, value, unit, comparison, min_value, max_value):
                satisfying.append(r)
        except (ValueError, TypeError):
            pass

    # If condition intersects rules with multiple different outcomes:
    if len(intersecting_outcomes) > 1:
        return NumericalBoundaryAnalysis(
            status="crossing_outcomes",
            intersecting_rules=intersecting,
            reason=f"Condition crosses multiple policy rule outcomes: {sorted(intersecting_outcomes)}.",
        )

    # If condition fits completely within one rule outcome:
    if len(satisfying) == 1 and len(intersecting_outcomes) <= 1:
        chosen_rule = satisfying[0]
        chosen_outcome = chosen_rule.outcome if isinstance(chosen_rule, RuleRecord) else chosen_rule["outcome"]
        return NumericalBoundaryAnalysis(
            status="single_outcome",
            satisfying_rule=chosen_rule if isinstance(chosen_rule, RuleRecord) else None,
            outcome=chosen_outcome,
            intersecting_rules=intersecting,
            reason=f"Condition fits completely within rule outcome '{chosen_outcome}'.",
        )

    if len(satisfying) > 1:
        # Multiple satisfying rules with the same outcome is fine if unanimous
        outcomes = {r.outcome if isinstance(r, RuleRecord) else r["outcome"] for r in satisfying}
        if len(outcomes) == 1:
            chosen_rule = satisfying[0]
            chosen_outcome = list(outcomes)[0]
            return NumericalBoundaryAnalysis(
                status="single_outcome",
                satisfying_rule=chosen_rule if isinstance(chosen_rule, RuleRecord) else None,
                outcome=chosen_outcome,
                intersecting_rules=intersecting,
                reason=f"Condition uniformly satisfies outcome '{chosen_outcome}'.",
            )
        return NumericalBoundaryAnalysis(
            status="crossing_outcomes",
            intersecting_rules=satisfying,
            reason=f"Condition crosses multiple rule outcomes: {sorted(outcomes)}.",
        )

    # If no rule completely satisfies the condition:
    return NumericalBoundaryAnalysis(
        status="cannot_evaluate",
        intersecting_rules=intersecting,
        reason="Condition does not fit completely within any single active rule outcome.",
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

    Distinguishes exact values, directional comparisons (greater_than, greater_than_or_equal,
    less_than, less_than_or_equal), ranges with two boundaries, and ambiguous/conflicting conditions.

    Args:
        query: User question string.
        relevant_fields: Optional set of condition_field names to restrict extraction to.

    Returns:
        List of dicts: [
            {
                "field": condition_field,
                "value": value,
                "unit": unit,
                "comparison": comp,
                "min_value": min_val,
                "max_value": max_val,
                "is_ambiguous": bool,
                "is_boundary_crossing": bool,
            }
        ]
    """
    if not query or not query.strip():
        return []

    q_lower = query.lower()
    raw_candidates: List[Dict[str, Any]] = []
    covered_spans: List[Tuple[int, int]] = []

    # Check for tier values (Information Security)
    has_confidential = "confidential" in q_lower
    has_restricted = "restricted" in q_lower and any(
        k in q_lower for k in ("restricted data", "restricted tier", "data classification", "restricted")
    )
    if has_confidential:
        raw_candidates.append({
            "field": "data_classification_tier",
            "value": "Confidential",
            "unit": "tier",
            "comparison": "exact",
            "min_value": None,
            "max_value": None,
            "is_boundary_crossing": False,
        })
    if has_restricted:
        raw_candidates.append({
            "field": "data_classification_tier",
            "value": "Restricted",
            "unit": "tier",
            "comparison": "exact",
            "min_value": None,
            "max_value": None,
            "is_boundary_crossing": False,
        })

    # Check for currency ranges: "between INR 40,000 and 60,000" / "from 40000 to 60000 INR"
    for match in _RANGE_CURRENCY_PATTERN.finditer(query):
        raw_num1 = match.group("num1").replace(",", "")
        raw_num2 = match.group("num2").replace(",", "")
        try:
            val1 = Decimal(raw_num1)
            if match.group("neg1") or match.group("neg2"):
                val1 = -val1
            val2 = Decimal(raw_num2)
            if match.group("neg3") or match.group("neg4"):
                val2 = -val2
        except (InvalidOperation, ValueError):
            continue

        min_val = min(val1, val2)
        max_val = max(val1, val2)
        cannot_eval = (min_val < 0 or max_val < 0)
        span = query[max(0, match.start() - 10) : min(len(query), match.end() + 10)].lower()

        # Determine currency unit
        p1 = (match.group("pre1") or "").lower()
        s1 = (match.group("suf1") or "").lower()
        p2 = (match.group("pre2") or "").lower()
        s2 = (match.group("suf2") or "").lower()

        if any(p in ("$", "usd") or s in ("usd", "dollar", "dollars") for p, s in ((p1, s1), (p2, s2))) or any(k in span for k in ("usd", "$", "dollar")):
            curr_unit = "USD"
        elif any(p in ("€", "eur") or s in ("eur", "euro", "euros") for p, s in ((p1, s1), (p2, s2))) or any(k in span for k in ("eur", "€", "euro")):
            curr_unit = "EUR"
        elif any(p in ("£", "gbp") or s in ("gbp", "pound", "pounds") for p, s in ((p1, s1), (p2, s2))) or any(k in span for k in ("gbp", "£", "pound")):
            curr_unit = "GBP"
        else:
            curr_unit = "INR"

        field = _determine_currency_field(q_lower, span)
        if field:
            raw_candidates.append({
                "field": field,
                "value": (min_val, max_val),
                "unit": curr_unit,
                "comparison": "range",
                "min_value": min_val,
                "max_value": max_val,
                "is_boundary_crossing": False,
                "cannot_evaluate": cannot_eval,
                "cannot_evaluate_reason": "Negative values are unsupported for policy rules." if cannot_eval else None,
            })
            covered_spans.append((match.start(), match.end()))

    # Check for days ranges: "between 10 and 20 days"
    for match in _RANGE_DAYS_PATTERN.finditer(query):
        try:
            d1 = int(match.group("num1"))
            if match.group("neg1"):
                d1 = -d1
            d2 = int(match.group("num2"))
            if match.group("neg2"):
                d2 = -d2
        except (ValueError, IndexError):
            continue
        min_d = min(d1, d2)
        max_d = max(d1, d2)
        cannot_eval = (min_d < 0 or max_d < 0)
        field = None
        if any(k in q_lower for k in ("advance", "booking", "departure", "flight")):
            field = "advance_booking_days"
        elif any(k in q_lower for k in ("submit", "submission", "delay", "expense claim", "reimbursement")):
            field = "submission_delay_days"
        elif any(k in q_lower for k in ("sick", "illness", "medical")):
            field = "consecutive_sick_days"
        elif any(k in q_lower for k in ("recertification", "access review", "privileged")):
            field = "privileged_role_review_interval_days"

        if field:
            raw_candidates.append({
                "field": field,
                "value": (min_d, max_d),
                "unit": "days",
                "comparison": "range",
                "min_value": min_d,
                "max_value": max_d,
                "is_boundary_crossing": False,
                "cannot_evaluate": cannot_eval,
                "cannot_evaluate_reason": "Negative values are unsupported for policy rules." if cannot_eval else None,
            })
            covered_spans.append((match.start(), match.end()))

    # Check for single currency amounts using Decimal precision directly from string
    for match in _CURRENCY_PATTERN.finditer(query):
        # Skip if within an already-extracted range
        if any(start <= match.start() < end for start, end in covered_spans):
            continue

        raw_num = match.group("num").replace(",", "")
        try:
            val = Decimal(raw_num)
            if match.group("neg1") or match.group("neg2"):
                val = -val
        except (InvalidOperation, ValueError):
            continue
        prefix = (match.group("prefix") or "").lower()
        suffix = (match.group("suffix") or "").lower()
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
                comp = _detect_comparison_operator(query, match.start(), match.end())
                cannot_eval = (val < 0)
                raw_candidates.append({
                    "field": field,
                    "value": val,
                    "unit": curr_unit,
                    "comparison": comp,
                    "min_value": None,
                    "max_value": None,
                    "is_boundary_crossing": False,
                    "cannot_evaluate": cannot_eval,
                    "cannot_evaluate_reason": f"Negative value '{val}' is unsupported for policy rules." if cannot_eval else None,
                })

    # Check for days
    for match in _DAYS_PATTERN.finditer(query):
        if any(start <= match.start() < end for start, end in covered_spans):
            continue
        try:
            val = int(match.group("num"))
            if match.group("neg"):
                val = -val
        except (ValueError, IndexError):
            continue
        field = None
        if any(k in q_lower for k in ("advance", "booking", "departure", "flight")):
            field = "advance_booking_days"
        elif any(k in q_lower for k in ("submit", "submission", "delay", "expense claim", "reimbursement")):
            field = "submission_delay_days"
        elif any(k in q_lower for k in ("sick", "illness", "medical")):
            field = "consecutive_sick_days"
        elif any(k in q_lower for k in ("recertification", "access review", "privileged")):
            field = "privileged_role_review_interval_days"

        if field:
            comp = _detect_comparison_operator(query, match.start(), match.end())
            cannot_eval = (val < 0)
            raw_candidates.append({
                "field": field,
                "value": val,
                "unit": "days",
                "comparison": comp,
                "min_value": None,
                "max_value": None,
                "is_boundary_crossing": False,
                "cannot_evaluate": cannot_eval,
                "cannot_evaluate_reason": f"Negative value '{val}' is unsupported for policy rules." if cannot_eval else None,
            })

    # Check for months
    for match in _MONTHS_PATTERN.finditer(query):
        try:
            val = int(match.group("num"))
            if match.group("neg"):
                val = -val
        except (ValueError, IndexError):
            continue
        if any(k in q_lower for k in ("contract", "duration", "agreement")):
            comp = _detect_comparison_operator(query, match.start(), match.end())
            cannot_eval = (val < 0)
            raw_candidates.append({
                "field": "contract_duration_months",
                "value": val,
                "unit": "months",
                "comparison": comp,
                "min_value": None,
                "max_value": None,
                "is_boundary_crossing": False,
                "cannot_evaluate": cannot_eval,
                "cannot_evaluate_reason": f"Negative value '{val}' is unsupported for policy rules." if cannot_eval else None,
            })

    # Check for mbps
    for match in _MBPS_PATTERN.finditer(query):
        try:
            val = int(match.group("num"))
            if match.group("neg"):
                val = -val
        except (ValueError, IndexError):
            continue
        comp = _detect_comparison_operator(query, match.start(), match.end())
        cannot_eval = (val < 0)
        raw_candidates.append({
            "field": "internet_speed_mbps",
            "value": val,
            "unit": "mbps",
            "comparison": comp,
            "min_value": None,
            "max_value": None,
            "is_boundary_crossing": False,
            "cannot_evaluate": cannot_eval,
            "cannot_evaluate_reason": f"Negative value '{val}' is unsupported for policy rules." if cannot_eval else None,
        })

    # Check for hours
    for match in _HOURS_PATTERN.finditer(query):
        try:
            val = int(match.group("num"))
            if match.group("neg"):
                val = -val
        except (ValueError, IndexError):
            continue
        if any(k in q_lower for k in ("incident", "p1", "severity", "sla")):
            comp = _detect_comparison_operator(query, match.start(), match.end())
            cannot_eval = (val < 0)
            raw_candidates.append({
                "field": "incident_severity_p1_response_hours",
                "value": val,
                "unit": "hours",
                "comparison": comp,
                "min_value": None,
                "max_value": None,
                "is_boundary_crossing": False,
                "cannot_evaluate": cannot_eval,
                "cannot_evaluate_reason": f"Negative value '{val}' is unsupported for policy rules." if cannot_eval else None,
            })

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
                "comparison": "exact",
                "min_value": None,
                "max_value": None,
                "is_ambiguous": True,
                "is_boundary_crossing": False,
                "error": f"Conflicting units [{', '.join(sorted(units))}] detected for condition '{field}'.",
            })
            continue

        primary_unit = units[0] if units else ""

        # Check for multiple distinct values (unless it's an exact duplicate match)
        if len(c_list) > 1:
            vals = [c["value"] for c in c_list]
            unique_vals = list(dict.fromkeys(vals))
            comparisons = list(dict.fromkeys(c.get("comparison", "exact") for c in c_list))
            if len(unique_vals) > 1 or len(comparisons) > 1:
                vals_str = ", ".join(str(v) for v in sorted(unique_vals, key=lambda x: str(x)))
                conditions.append({
                    "field": field,
                    "value": None,
                    "unit": primary_unit,
                    "comparison": "ambiguous",
                    "min_value": None,
                    "max_value": None,
                    "is_ambiguous": True,
                    "is_boundary_crossing": False,
                    "error": f"Multiple conflicting values [{vals_str}] detected for condition '{field}'.",
                })
                continue

        chosen = c_list[0]
        conditions.append({
            "field": field,
            "value": chosen["value"],
            "unit": primary_unit,
            "comparison": chosen.get("comparison", "exact"),
            "min_value": chosen.get("min_value"),
            "max_value": chosen.get("max_value"),
            "is_ambiguous": False,
            "is_boundary_crossing": chosen.get("is_boundary_crossing", False),
            "cannot_evaluate": chosen.get("cannot_evaluate", False),
            "cannot_evaluate_reason": chosen.get("cannot_evaluate_reason"),
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
    comparison: str = "exact",
) -> str:
    """Format an authoritative, policy-grounded natural language decision answer from a canonical rule outcome."""
    outcome = rule.outcome
    field = rule.condition_field
    thresh = rule.threshold_value
    unit = cond_unit or rule.unit

    comp = (comparison or "exact").lower()
    if comp == "greater_than":
        val_phrase = f"exceeding {cond_val} {unit}"
        days_phrase = f"more than {cond_val} days"
        months_phrase = f"exceeding {cond_val} months"
        hours_phrase = f"exceeding {cond_val} hour(s)"
    elif comp == "less_than":
        val_phrase = f"below {cond_val} {unit}"
        days_phrase = f"less than {cond_val} days"
        months_phrase = f"under {cond_val} months"
        hours_phrase = f"under {cond_val} hour(s)"
    elif comp == "less_than_or_equal":
        val_phrase = f"up to {cond_val} {unit}"
        days_phrase = f"up to {cond_val} days"
        months_phrase = f"up to {cond_val} months"
        hours_phrase = f"within {cond_val} hour(s)"
    elif comp == "range" and isinstance(cond_val, (list, tuple)):
        val_phrase = f"between {cond_val[0]} and {cond_val[1]} {unit}"
        days_phrase = f"between {cond_val[0]} and {cond_val[1]} days"
        months_phrase = f"between {cond_val[0]} and {cond_val[1]} months"
        hours_phrase = f"between {cond_val[0]} and {cond_val[1]} hour(s)"
    else:
        val_phrase = f"of {cond_val} {unit}"
        days_phrase = f"with {cond_val} days"
        months_phrase = f"of {cond_val} months"
        hours_phrase = f"within {cond_val} hour(s)"

    if outcome == "standard_approval":
        return f"Standard manager approval is permitted. The expense amount {val_phrase} is within the policy approval limit of {thresh} {rule.unit}."
    elif outcome == "additional_approval":
        return f"Additional approval is required. Standard manager approval is not sufficient for an expense {val_phrase} as it exceeds the standard limit of {thresh} {rule.unit}."
    elif outcome == "standard_booking":
        return f"Standard flight booking is permitted as the travel is booked {days_phrase} advance notice, meeting the {thresh}-day policy requirement."
    elif outcome == "expedited_approval_required":
        return f"Expedited approval is required because the flight is booked {days_phrase} advance notice, which is less than the required {thresh} days."
    elif outcome == "vp_approval":
        return f"Vice President (VP) approval is required for travel exception expenses {val_phrase} (up to {thresh} {rule.unit})."
    elif outcome == "cfo_approval":
        return f"Chief Financial Officer (CFO) approval is required for travel exception expenses {val_phrase} (exceeding {thresh} {rule.unit})."
    elif outcome == "eligible_reimbursement":
        return f"The expense claim submitted {days_phrase} is eligible for standard reimbursement (within the {thresh}-day limit)."
    elif outcome == "requires_late_exception":
        return f"A late submission exception approval is required for expense claims submitted after {cond_val} days (exceeding the {thresh}-day deadline)."
    elif outcome == "no_receipt_required":
        return f"Incidental business expenses {val_phrase} (up to {thresh} {rule.unit}) do not require an itemized tax receipt, provided digital payment confirmation is attached."
    elif outcome == "itemized_receipt_required":
        return f"An itemized tax invoice is strictly required for incidental expenses {val_phrase} (exceeding {thresh} {rule.unit})."
    elif outcome == "within_standard_card_limit":
        return f"The monthly spend {val_phrase} is within the standard corporate credit card limit of {thresh} {rule.unit}."
    elif outcome == "card_limit_escalation_required":
        return f"Corporate credit card limit escalation approval is required for monthly spend {val_phrase} exceeding the standard {thresh} {rule.unit} limit."
    elif outcome == "self_certification":
        return f"Self-certification is permitted for sick leave {days_phrase} consecutive days (up to {thresh} days)."
    elif outcome == "medical_certificate_required":
        return f"A formal medical certificate is required for sick leave {days_phrase} consecutive days (exceeding {thresh} days)."
    elif outcome == "remote_setup_compliant":
        return f"The internet connection speed {val_phrase} meets the remote work setup policy minimum of {thresh} {rule.unit}."
    elif outcome == "remediation_required":
        return f"Internet speed remediation is required because {cond_val} Mbps does not meet the minimum requirement of {thresh} {rule.unit}."
    elif outcome == "standard_tuition_grant":
        return f"The tuition claim {val_phrase} is eligible for a standard tuition grant (within {thresh} {rule.unit})."
    elif outcome == "committee_review_required":
        return f"Tuition reimbursement committee review is required for claims {val_phrase} exceeding the standard {thresh} {rule.unit} grant."
    elif outcome == "single_quote_permitted":
        return f"A single supplier quote is permitted for purchase orders {val_phrase} (up to {thresh} {rule.unit})."
    elif outcome == "three_competitive_quotes_required":
        return f"Three competitive supplier quotes are required for purchase orders {val_phrase} (exceeding {thresh} {rule.unit})."
    elif outcome == "department_lead_signoff":
        return f"Department lead signoff is sufficient for vendor contracts {months_phrase} duration (up to {thresh} months)."
    elif outcome == "legal_and_procurement_signoff":
        return f"Both Legal and Procurement signoff are required for vendor contracts {months_phrase} duration (exceeding {thresh} months)."
    elif outcome == "director_emergency_approval":
        return f"Director emergency approval is permitted for emergency purchases {val_phrase} (up to {thresh} {rule.unit})."
    elif outcome == "board_procurement_committee":
        return f"Board Procurement Committee approval is required for emergency purchases {val_phrase} (exceeding {thresh} {rule.unit})."
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
    "have", "has", "had", "will", "would", "shall", "should",
    "a", "an", "the", "of", "for", "under", "in", "on", "at", "by", "with", "about", "within",
    "after", "before", "to", "from", "than", "into", "through", "during", "and", "or", "as", "if",
    "not", "no", "per", "via", "without", "this", "that", "these", "those", "who", "whom", "which",
    "below", "above", "over", "between", "least", "most", "strictly", "fewer", "higher", "lower",
    "more", "less", "greater", "exact", "exactly", "including", "included",
    "off", "out", "up", "down",
    # Currency and units
    "inr", "rs", "rupee", "rupees", "usd", "eur", "gbp", "dollar", "dollars", "euro", "euros", "pound", "pounds",
    "day", "days", "month", "months", "hour", "hours", "mbps",
    # Decision / permission framing and query verbs
    "approve", "approval", "approved", "permits", "permit", "permitted", "permission", "permissible",
    "allows", "allow", "allowed", "allowable", "requires", "require", "required", "requirement", "requirements",
    "standard", "additional", "compliant", "compliance", "comply", "complies", "meet", "meets", "meeting",
    "suffice", "sufficient", "enough", "exceed", "exceeds", "exceeding", "eligible", "eligibility", "status",
    "qualify", "qualifies", "qualified", "applies", "apply", "applied",
    "trigger", "triggers", "triggered", "issue", "issues", "issued", "issuing",
    "use", "uses", "used", "using", "make", "makes", "made", "making",
    "need", "needs", "needed", "follow", "follows", "followed", "following",
    "sign", "signed", "signing", "signoff", "order", "orders",
    # Corporate roles and entities
    "director", "manager", "lead", "employee", "employees", "role", "roles", "policy", "corporate",
    "company", "asterline", "group", "threshold", "limit", "limits", "amount", "expense", "expenses",
    "spend", "spending", "claim", "claims", "reimbursement", "reimburse", "cost", "costing", "price", "direct",
    "functional", "department", "reporting",
    # Canonical condition field domain vocabulary
    "booking", "notice", "advance", "departure", "flight", "receipt", "tax", "invoice", "itemized",
    "incidental", "payment", "confirmation", "attached", "quote", "quotes", "single", "competitive",
    "three", "two", "multiple", "supplier", "purchase", "po", "emergency", "card", "credit",
    "monthly", "sick", "illness", "leave", "medical", "certificate", "self", "remote", "setup", "work",
    "internet", "speed", "remediation", "tuition", "assistance", "annual", "grant", "committee", "contract", "vendor", "agreement",
    "duration", "legal", "procurement", "board", "data", "classification", "tier",
    "confidential", "restricted", "encryption", "tokenized", "hsm", "aes", "privileged", "access",
    "review", "recertification", "suspension", "automatic", "cycle", "interval", "incident", "severity",
    "priority", "response", "time", "window", "sla", "breach", "escalation", "late", "exception", "exceptions",
    # Document tags
    "doc", "trv", "fin", "ppo", "prc", "sec",
    "c04", "b02", "r03", "p02", "e01", "a02", "i02", "v1", "v2", "v3",
    # Broad travel qualifiers
    "travel", "trip", "trips", "business",
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
    "annual_tuition_claim": {"tuition", "course", "education", "degree", "grant", "assistance"},
    "contract_duration_months": {"contract", "vendor", "agreement", "duration"},
    "privileged_role_review_interval_days": {"privileged", "access", "review", "recertification", "suspension"},
    "incident_severity_p1_response_hours": {"incident", "severity", "p1", "priority 1", "response", "sla"},
    "data_classification_tier": {"classification", "tier", "confidential", "restricted", "encryption"},
    "submission_delay_days": {"submit", "submission", "delay", "deadline"},
    "exception_amount": {"exception", "urgent", "deployment", "travel exception"},
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
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    q_lower = question.lower()

    # Rule out open-ended threshold definitions or cross-clause questions
    explanatory_markers = (
        "what is the maximum", "what is the threshold", "what is the limit",
        "what are the tiers", "into what four", "what four severity",
        "and what timeframe", "and who", "and when", "and within what"
        "and what timeframe", "and who", "and when", "and within what",
        "what are the rules", "explain ", "describe ", "how does", "how do", "overview of"
    )
    if any(em in q_lower for em in explanatory_markers):
        return AuthoritativeDecision(is_authoritative=False)
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    # Must contain a decision indicator
    decision_markers = (
        "can ", "could ", "is ", "are ", "does ", "do ", "may ", "require", "required",
        "permit", "permitted", "allow", "allowed", "eligible", "approval", "approve",
        "compliant", "exceed", "escalat"
    )
    if not any(m in q_lower for m in decision_markers):
        return AuthoritativeDecision(is_authoritative=False)
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    # Must have exactly one valid condition, and no ambiguous conditions
    # Check for ambiguous or conflicting condition inputs
    if any(c.get("is_ambiguous") for c in extracted_conditions):
        return AuthoritativeDecision(is_authoritative=False)
        ambig_reasons = [c.get("error") for c in extracted_conditions if c.get("is_ambiguous")]
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            reason="; ".join(filter(None, ambig_reasons)) or "Ambiguous condition inputs detected in query.",
        )

    valid_conds = [c for c in extracted_conditions if c.get("value") is not None and not c.get("is_ambiguous")]
    if len(valid_conds) != 1:
        return AuthoritativeDecision(is_authoritative=False)
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    target_cond = valid_conds[0]
    cond_field = target_cond["field"]
    cond_val = target_cond["value"]
    cond_unit = target_cond.get("unit", "")
    cond_comp = target_cond.get("comparison", "exact")

    # Find rules evaluated against this condition field with input_value
    evaluated_for_field = [
        r for r in rule_results
        if not r.is_error and r.input_value is not None and r.rule.condition_field == cond_field
    ]
    passing_rules = [r for r in evaluated_for_field if r.passed]

    # Evaluate boundary crossing dynamically against active authorized rules for this field
    active_field_rules = [r.rule for r in evaluated_for_field]
    if active_field_rules:
        boundary_analysis = analyze_numerical_boundary(
            active_rules=active_field_rules,
            comparison=cond_comp,
            value=cond_val,
            unit=cond_unit,
            min_value=target_cond.get("min_value"),
            max_value=target_cond.get("max_value"),
        )
        if boundary_analysis.status == "crossing_outcomes" or target_cond.get("is_boundary_crossing"):
            return AuthoritativeDecision(
                is_authoritative=False,
                is_decision_request=True,
                is_boundary_crossing=True,
                reason="Numerical condition crosses policy rule boundaries or has conflicting outcomes.",
            )
        if boundary_analysis.status == "cannot_evaluate" or target_cond.get("cannot_evaluate"):
            return AuthoritativeDecision(
                is_authoritative=False,
                is_decision_request=True,
                is_unresolved=True,
                reason=f"Numerical condition cannot be evaluated safely: {boundary_analysis.reason or target_cond.get('cannot_evaluate_reason')}",
            )
        if boundary_analysis.status != "single_outcome":
            return AuthoritativeDecision(is_authoritative=False)
    elif target_cond.get("is_boundary_crossing"):
            return AuthoritativeDecision(
                is_authoritative=False,
                is_decision_request=True,
                is_unresolved=True,
                reason="Condition does not fit completely within any single active rule outcome.",
            )
    else:
        if target_cond.get("is_boundary_crossing"):
            return AuthoritativeDecision(
                is_authoritative=False,
                is_decision_request=True,
                is_boundary_crossing=True,
                reason="Numerical condition crosses policy rule boundaries or has conflicting outcomes.",
            )
        if target_cond.get("cannot_evaluate"):
            return AuthoritativeDecision(
                is_authoritative=False,
                is_decision_request=True,
                is_unresolved=True,
                reason=f"Numerical condition cannot be evaluated safely: {target_cond.get('cannot_evaluate_reason')}",
            )
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            reason=f"Incomplete retrieval evidence: no active authorized rules found for condition field '{cond_field}'.",
        )

    if len(passing_rules) != 1:
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            reason=f"Condition cannot be resolved to a unique active rule ({len(passing_rules)} rules satisfied).",
        )

    passing_result = passing_rules[0]
    rule = passing_result.rule

    # 1. Subject match verification: Question must be about the matched rule's policy subject
    # 1. Unit verification: If rule has a unit, condition must supply compatible unit
    if rule.unit and (not cond_unit or not str(cond_unit).strip()):
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            rule=rule,
            rule_id=rule.rule_id,
            reason=f"Missing unit for condition '{cond_field}'. Expected '{rule.unit}'.",
        )
    if rule.unit and cond_unit and cond_unit.strip().lower() != rule.unit.strip().lower():
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            rule=rule,
            rule_id=rule.rule_id,
            reason=f"Incompatible unit '{cond_unit}' for rule {rule.rule_id}. Expected '{rule.unit}'.",
        )

    # 2. Subject match verification: Question must be about the matched rule's policy subject
    subject_terms = _RULE_FIELD_SUBJECT_TERMS.get(cond_field)
    if subject_terms and not any(st in q_lower for st in subject_terms):
        return AuthoritativeDecision(is_authoritative=False)
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    # 2. Verify rule's linked chunk is present in authorized chunks
    # 3. Verify rule's linked chunk is present in authorized chunks
    rule_chunk = None
    for c in authorized_chunks:
        cid = c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", None)
        if cid == rule.chunk_id:
            rule_chunk = c
            break

    if not rule_chunk:
        return AuthoritativeDecision(is_authoritative=False)
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            reason=f"Incomplete retrieval evidence: rule chunk '{rule.chunk_id}' is missing from authorized evidence.",
        )

    # 3. Unsupported requirements check: Question must not contain additional unsupported entities
    # 4. Unsupported requirements check: Question must not contain additional unsupported entities
    unsupported_terms = _find_unsupported_query_terms(question, rule_chunk)
    if unsupported_terms:
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            rule=rule,
            rule_id=rule.rule_id,
            reason=f"Query introduces unsupported terms [{', '.join(unsupported_terms)}] not established by rule evidence.",
            unsupported_terms=unsupported_terms,
        )

    answer_text = _format_canonical_decision_answer(rule, cond_val, cond_unit, comparison=cond_comp)

    return AuthoritativeDecision(
        is_authoritative=True,
        is_decision_request=True,
        rule=rule,
        answer=answer_text,
        citation_chunk_id=rule.chunk_id,
        rule_id=rule.rule_id,
        outcome=rule.outcome,
        reason=f"Authoritative deterministic decision based on canonical rule {rule.rule_id} ({rule.outcome}).",
    )
