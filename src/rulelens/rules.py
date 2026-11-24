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
from rulelens.grounding import request_meaning
from rulelens.ingestion import CorpusSnapshot, assert_snapshot_current, get_corpus_snapshot
from rulelens.retrieval import ALLOWED_ROLES_SET, is_chunk_eligible, parse_query_date


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
        if not numeric_threshold.is_finite():
            raise ValueError("Non-finite rule threshold.")

        if comp == "range":
            c_min = min_value if min_value is not None else (value[0] if isinstance(value, (list, tuple)) else value)
            c_max = max_value if max_value is not None else (value[1] if isinstance(value, (list, tuple)) else value)
            numeric_min = Decimal(str(c_min).strip().replace(",", ""))
            numeric_max = Decimal(str(c_max).strip().replace(",", ""))
            if not numeric_min.is_finite() or not numeric_max.is_finite():
                raise ValueError("Non-finite range.")
            if numeric_min > numeric_max:
                numeric_min, numeric_max = numeric_max, numeric_min
            if numeric_min < 0:
                raise ValueError(f"Negative values ({numeric_min}) are unsupported for policy rules.")
        else:
            clean_val_str = str(value).strip().replace(",", "")
            numeric_val = Decimal(clean_val_str)
            if not numeric_val.is_finite():
                raise ValueError("Non-finite condition.")
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
    corpus_snapshot: Optional[CorpusSnapshot] = None,
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
    snapshot = corpus_snapshot or get_corpus_snapshot(base_dir)
    assert_snapshot_current(snapshot)
    all_rules = load_rules(base_dir=snapshot.base_dir)
    rule_data = [{k: v for k, v in rule.to_dict().items() if k != "chunk_id"} for rule in all_rules]
    if rule_data != snapshot.rules:
        raise ValueError("Rules do not match the validated corpus snapshot.")
    canonical = {chunk["chunk_id"]: chunk for chunk in snapshot.chunks}

    # Set of chunk IDs in authorized evidence
    authorized_cids = {c["chunk_id"] for c in authorized_chunks}
    for chunk_id in authorized_cids:
        if chunk_id not in canonical:
            raise ValueError("Unknown rule evidence.")
    authorized_cids = {cid for cid in authorized_cids if is_chunk_eligible(canonical[cid], role, q_date)}
    assert_snapshot_current(snapshot)

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
_CURRENCY_MARKERS = r"inr|rs\.?|₹|usd|\$|eur|€|gbp|£"
_CURRENCY_PATTERN = re.compile(
    rf"(?<![\w.])(?P<sign>-)?\s*(?P<prefix>{_CURRENCY_MARKERS})?\s*"
    r"(?P<num>-?(?:[0-9][0-9,.]*(?:[eE][+-]?[0-9]+)?|nan|infinity|inf))"
    r"(?P<attached>[a-z]+)?"
    r"(?:\s+(?P<scale>k|lakhs?|crores?|million|billion|mn|cr)\b)?"
    r"(?:\s*(?P<suffix>inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?)\b)?",
    re.IGNORECASE,
)
_VALID_AMOUNT = re.compile(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3})(?:\.[0-9]+)?")
_CURRENCY_UNITS = {"inr":"INR", "rs":"INR", "rs.":"INR", "₹":"INR", "rupee":"INR", "rupees":"INR",
                   "usd":"USD", "$":"USD", "dollar":"USD", "dollars":"USD",
                   "eur":"EUR", "€":"EUR", "euro":"EUR", "euros":"EUR",
                   "gbp":"GBP", "£":"GBP", "pound":"GBP", "pounds":"GBP"}
_AMOUNT_SCALES = {"k":Decimal("1000"), "lakh":Decimal("100000"), "lakhs":Decimal("100000"),
                  "crore":Decimal("10000000"), "crores":Decimal("10000000")}


_UNIT_PATTERN = re.compile(
    r"(?<![\w.])(?P<num>-?(?:[0-9][0-9,.]*(?:[eE][+-]?[0-9]+)?|nan|infinity|inf))\s*"
    r"(?:(?P<kind>consecutive|business|working|calendar)\s*)?"
    r"(?P<unit>days?|months?|hours?|mbps|years?|weeks?|minutes?|seconds?)\b", re.IGNORECASE,
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


def _extract_currency_conditions(query: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    amounts = []
    unit_decision = bool(re.match(r"\s*(?:can|could|may|is|are|does|do)\b", query, re.I)
                         and re.search(r"\b(?:internet speed|contract duration|sick leave|flight booking|review interval|incident response)\b", query, re.I))
    date_spans = [match.span() for match in re.finditer(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", query)]
    for match in _CURRENCY_PATTERN.finditer(query):
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        prefix = (match.group("prefix") or "").lower()
        suffix = (match.group("suffix") or "").lower()
        attached = (match.group("attached") or "").lower()
        scale = (match.group("scale") or "").lower()
        if not prefix and not suffix and re.search(r"\b(?:priority|severity|tier)\s*$", query[:match.start()], re.I):
            continue
        if attached in _CURRENCY_UNITS:
            suffix, attached = attached, ""
        if not prefix and not suffix and re.match(r"\s*(?:consecutive\s+|working\s+|business\s+|calendar\s+)?(?:days?|months?|hours?|weeks?|years?|mbps|percent)\b", query[match.end():], re.I):
            continue
        if not prefix and not suffix and not attached and not scale and not unit_decision and not re.search(r"\b(?:amounts?|expenses?|spend|spending|costs?|price|claims?|purchase orders?|tuition|credit|cap|limit)\b", query, re.I):
            continue
        if not prefix and not suffix and re.match(r"\s*(?:and|to)\s*-?[0-9][0-9,.]*\s*(?:days?|months?|hours?|mbps)\b", query[match.end():], re.I):
            continue
        raw = match.group("num")
        if raw.endswith(","):
            raw = raw[:-1]
        if raw.endswith(".") and raw.count(".") == 1:
            raw = raw[:-1]
        negative = bool(match.group("sign")) or raw.startswith("-")
        digits = raw.lstrip("-")
        if digits.isdigit() and 2000 <= int(digits) <= 2099 and not (prefix or suffix or attached or scale):
            continue
        field = _determine_currency_field(query.lower(), match.group().lower()) or "unresolved_amount"
        units = {_CURRENCY_UNITS[item] for item in (prefix, suffix) if item in _CURRENCY_UNITS}
        unit = next(iter(units)) if len(units) == 1 else ""
        error = None
        value = None
        multiplier = attached or scale
        if (attached and scale) or (multiplier and multiplier not in _AMOUNT_SCALES) or not _VALID_AMOUNT.fullmatch(digits):
            error = "Invalid monetary expression. Provide a complete amount with valid grouping and a supported suffix."
        else:
            value = Decimal(digits.replace(",", "")) * _AMOUNT_SCALES.get(multiplier, Decimal(1))
            if negative:
                value = -value
                error = "Negative amounts cannot be evaluated."
        if len(units) > 1:
            error = "Conflicting currencies cannot be evaluated."
        amounts.append({"field":field, "value":value, "unit":unit,
                        "comparison":_detect_comparison_operator(query, match.start(), match.end()),
                        "min_value":None, "max_value":None, "is_boundary_crossing":False,
                        "cannot_evaluate":bool(error), "cannot_evaluate_reason":error,
                        "span":match.span()})
    conditions = []
    spans = []
    index = 0
    while index < len(amounts):
        amount = amounts[index]
        start, end = amount.pop("span")
        spans.append((start, end))
        if index + 1 < len(amounts):
            other = amounts[index + 1]
            other_start, other_end = other["span"]
            before = query[max(0, start - 12):start]
            between = query[end:other_start]
            if re.search(r"\b(?:between|from)\s*$", before, re.I) and re.fullmatch(r"\s*(?:and|to)\s*", between, re.I):
                spans.append((other_start, other_end))
                unit = amount["unit"] or other["unit"]
                error = amount["cannot_evaluate_reason"] or other["cannot_evaluate_reason"]
                if amount["unit"] and other["unit"] and amount["unit"] != other["unit"]:
                    error = "Incompatible units in the amount range."
                lower, upper = amount["value"], other["value"]
                if lower is None or upper is None or lower > upper:
                    error = error or "Provide the range in increasing order."
                amount.update(value=(lower, upper), unit=unit, comparison="range", min_value=lower, max_value=upper,
                              cannot_evaluate=bool(error), cannot_evaluate_reason=error)
                index += 1
        if not amount["unit"]:
            amount.update(cannot_evaluate=True, cannot_evaluate_reason="Specify the currency or measurement unit for the value.")
        conditions.append(amount)
        index += 1
    return conditions, spans


def _extract_unit_conditions(query: str, covered_spans: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    q_lower = query.lower()
    entries = []
    for match in _UNIT_PATTERN.finditer(query):
        if any(start <= match.start() < end for start, end in covered_spans):
            continue
        unit = match.group("unit").lower()
        unit = unit if unit == "mbps" else unit.rstrip("s") + "s"
        field = None
        if unit == "mbps":
            field = "internet_speed_mbps"
        elif any(k in q_lower for k in ("contract", "agreement")):
            field = "contract_duration_months"
        elif any(k in q_lower for k in ("advance", "booking", "departure", "flight")):
            field = "advance_booking_days"
        elif any(k in q_lower for k in ("submit", "submission", "delay", "expense claim", "reimbursement")):
            field = "submission_delay_days"
        elif any(k in q_lower for k in ("sick", "illness", "medical")):
            field = "consecutive_sick_days"
        elif any(k in q_lower for k in ("recertification", "access review", "privileged")):
            field = "privileged_role_review_interval_days"
        elif any(k in q_lower for k in ("incident", "p1", "severity", "sla")):
            field = "incident_severity_p1_response_hours"
        if field is None:
            continue
        raw = match.group("num")
        value = Decimal(raw.replace(",", "")) if _VALID_AMOUNT.fullmatch(raw.lstrip("-")) else None
        error = None
        if value is None or value < 0:
            error = "Provide a valid non-negative duration or speed with its unit."
        if match.group("kind") in ("business", "working"):
            error = "The structured rule does not define a working-day conversion. Please clarify the day unit."
        entry = {"field":field, "value":value, "unit":unit,
                 "comparison":_detect_comparison_operator(query, match.start(), match.end()),
                 "min_value":None, "max_value":None, "is_boundary_crossing":False,
                 "cannot_evaluate":bool(error), "cannot_evaluate_reason":error, "span":match.span()}
        entries.append(entry)
    conditions = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        start, end = entry.pop("span")
        lower = upper = None
        range_found = False
        before = query[:start]
        bare_range = re.search(r"\b(?:between|from)\s+(-?[0-9][0-9,.]*)\s+(?:and|to)\s*$", before, re.I)
        if bare_range:
            raw = bare_range.group(1)
            lower = Decimal(raw.replace(",", "")) if _VALID_AMOUNT.fullmatch(raw.lstrip("-")) else None
            upper = entry["value"]
            range_found = True
        elif index + 1 < len(entries) and re.search(r"\b(?:between|from)\s*$", before, re.I):
            other = entries[index + 1]
            if re.fullmatch(r"\s*(?:and|to)\s*", query[end:other["span"][0]], re.I):
                lower, upper = entry["value"], other["value"]
                range_found = True
                if entry["unit"] != other["unit"] or other["cannot_evaluate"]:
                    entry.update(cannot_evaluate=True, cannot_evaluate_reason="Clarify the units in the range.")
                index += 1
        if range_found:
            if lower is None or upper is None or lower < 0 or upper < lower:
                entry.update(cannot_evaluate=True, cannot_evaluate_reason="Provide a non-negative range in increasing order.")
            entry.update(value=(lower, upper), comparison="range", min_value=lower, max_value=upper)
        conditions.append(entry)
        index += 1
    return conditions


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

    money_conditions, money_spans = _extract_currency_conditions(query)
    raw_candidates.extend(money_conditions)
    covered_spans.extend(money_spans)

    raw_candidates.extend(_extract_unit_conditions(query, covered_spans))

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
    if any(c.get("cannot_evaluate") for c in conds):
        return ConditionExtractionResult(status="invalid", conditions=conds, reason="; ".join(c.get("cannot_evaluate_reason") or "Invalid condition." for c in conds if c.get("cannot_evaluate")))
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
        return f"The expense claim submitted {days_phrase} meets the {thresh}-day submission deadline. Other reimbursement requirements still apply."
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
        return f"The tuition claim {val_phrase} is within the standard tuition grant amount of {thresh} {rule.unit}. This amount check does not establish eligibility."
    elif outcome == "committee_review_required":
        return f"Tuition reimbursement committee review is required for claims {val_phrase} exceeding the standard {thresh} {rule.unit} grant."
    elif outcome == "single_quote_permitted":
        return f"A single supplier quote is permitted for purchase orders {val_phrase} (up to {thresh} {rule.unit})."
    elif outcome == "three_competitive_quotes_required":
        return f"Three competitive supplier quotes are required for purchase orders {val_phrase} (exceeding {thresh} {rule.unit})."
    elif outcome == "department_lead_signoff":
        return f"Department lead signoff is the required duration-based approval for vendor contracts {months_phrase} (up to {thresh} months). Other contract requirements still apply."
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
    "authorized", "otherwise", "already", "lakh", "lakhs", "crore", "crores",
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
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    q_lower = question.lower()

    # Rule out open-ended threshold definitions or cross-clause questions
    explanatory_markers = (
        "what is the maximum", "what is the threshold", "what is the limit",
        "what are the tiers", "into what four", "what four severity",
        "and what timeframe", "and who", "and when", "and within what",
        "what are the rules", "explain ", "describe ", "how does", "how do", "overview of"
    )
    if any(em in q_lower for em in explanatory_markers):
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    # Must contain a decision indicator
    decision_markers = (
        "can ", "could ", "is ", "are ", "does ", "do ", "may ", "require", "required",
        "permit", "permitted", "allow", "allowed", "eligible", "approval", "approve",
        "compliant", "exceed", "escalat"
    )
    if not any(m in q_lower for m in decision_markers):
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    # Must have exactly one valid condition, and no ambiguous conditions
    # Check for ambiguous or conflicting condition inputs
    if any(c.get("is_ambiguous") for c in extracted_conditions):
        ambig_reasons = [c.get("error") for c in extracted_conditions if c.get("is_ambiguous")]
        return AuthoritativeDecision(
            is_authoritative=False,
            is_decision_request=True,
            is_unresolved=True,
            reason="; ".join(filter(None, ambig_reasons)) or "Ambiguous condition inputs detected in query.",
        )

    valid_conds = [c for c in extracted_conditions if c.get("value") is not None and not c.get("is_ambiguous")]
    if len(valid_conds) != 1:
        if re.match(r"\s*(?:can|could|may)\b", q_lower) and (
            extracted_conditions or re.search(r"\b(?:travel expense|purchase order|tuition claim|internet speed|sick leave|flight booking)\b", q_lower)
        ):
            return AuthoritativeDecision(is_authoritative=False, is_decision_request=True, is_unresolved=True,
                                         reason="Provide one complete numerical condition, including its currency or unit.")
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=False)

    target_cond = valid_conds[0]
    cond_field = target_cond["field"]
    field_targets = {
        "expense_amount": "travel.approval", "advance_booking_days": "travel.advance_booking",
        "receipt_amount": "expense.receipts", "order_amount": "procurement.bids",
        "monthly_spend": "card.limit", "consecutive_sick_days": "leave.sick",
        "internet_speed_mbps": "remote.speed", "annual_tuition_claim": "tuition.allowance",
        "contract_duration_months": "contract.duration", "privileged_role_review_interval_days": "security.recertification",
        "incident_severity_p1_response_hours": "incident.response", "data_classification_tier": "security.encryption",
        "submission_delay_days": "expense.submission", "exception_amount": "travel.exceptions",
        "emergency_purchase_amount": "procurement.emergency",
    }
    targets = set(request_meaning(question)["targets"])
    if targets and targets != {field_targets.get(cond_field)}:
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=True, is_unresolved=True,
                                     reason="Please ask each policy decision separately so all requirements can be checked.")
    cond_val = target_cond["value"]
    cond_unit = target_cond.get("unit", "")
    cond_comp = target_cond.get("comparison", "exact")
    if re.search(r"\bnot\b(?!\s+(?:exceed|exceeds|exceeding|more\s+than|less\s+than|above|below|over|under)\b)", q_lower):
        return AuthoritativeDecision(is_authoritative=False, is_decision_request=True, is_unresolved=True,
                                     reason="Please clarify the negated decision before applying the numerical rule.")

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
