"""Public, non-sensitive policy topic catalog and deterministic topic matcher.

Provides high-level, non-sensitive topic definitions keyed by document ID
to identify policy areas that require role authorization checks before retrieval.
Contains no restricted clause text or sensitive operational specifics.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple


class PolicyTopicDefinition(NamedTuple):
    """Non-sensitive topic definition for a policy document."""

    doc_id: str
    topic_name: str
    required_concept_groups: Tuple[Set[str], ...]
    compound_phrases: Tuple[str, ...] = ()


# Approved catalog of public, non-sensitive policy topics keyed by document ID.
# Only covers high-level governance areas; contains NO clause text or confidential data.
PUBLIC_POLICY_TOPIC_CATALOG: Dict[str, List[PolicyTopicDefinition]] = {
    # DOC-SEC-004: Data Retention Policy
    # Covers storage media sanitization, decommissioning, and physical destruction.
    "DOC-SEC-004": [
        PolicyTopicDefinition(
            doc_id="DOC-SEC-004",
            topic_name="Storage media sanitization and disposal",
            required_concept_groups=(
                # Concept Group 1: Storage media terms
                {
                    "drive",
                    "drives",
                    "disk",
                    "disks",
                    "harddrive",
                    "harddrives",
                    "media",
                    "hardware",
                    "server",
                    "servers",
                    "storage",
                },
                # Concept Group 2: Sanitization / secure disposal terms
                {
                    "sanitize",
                    "sanitizing",
                    "sanitization",
                    "sanitise",
                    "sanitising",
                    "sanitisation",
                    "wipe",
                    "wiping",
                    "erasure",
                    "erase",
                    "erasing",
                    "decommission",
                    "decommissioned",
                    "decommissioning",
                    "disposal",
                    "dispose",
                    "disposing",
                    "destruction",
                    "shredding",
                    "derez",
                    "demagnetize",
                    "degauss",
                },
            ),
            compound_phrases=(
                "storage media",
                "storage drive",
                "storage drives",
                "storage disk",
                "storage disks",
                "drive sanitization",
                "disk wiping",
                "secure disposal",
                "secure erasure",
            ),
        ),
    ],
    # DOC-TRV-004: Travel Exceptions Policy
    "DOC-TRV-004": [
        PolicyTopicDefinition(
            doc_id="DOC-TRV-004",
            topic_name="Executive travel exceptions and escalation routes",
            required_concept_groups=(
                {"travel", "flight", "hotel", "trip", "booking"},
                {"exception", "escalation", "extraordinary", "exemption", "deviation"},
            ),
            compound_phrases=(
                "travel exception",
                "travel exceptions",
                "extraordinary travel",
            ),
        ),
    ],
    # DOC-FIN-003: Corporate Cards Policy
    "DOC-FIN-003": [
        PolicyTopicDefinition(
            doc_id="DOC-FIN-003",
            topic_name="Corporate credit card issuance and limit escalation",
            required_concept_groups=(
                {"card", "cards", "credit"},
                {"corporate", "issuance", "issued", "escalation", "monthly"},
            ),
            compound_phrases=(
                "corporate card",
                "corporate cards",
                "credit card",
                "credit cards",
                "card limit",
            ),
        ),
    ],
    # DOC-PPO-004: Employee Transfers Policy
    "DOC-PPO-004": [
        PolicyTopicDefinition(
            doc_id="DOC-PPO-004",
            topic_name="Internal employee transfers and eligibility",
            required_concept_groups=(
                {"transfer", "transfers"},
                {"internal", "tenure", "mobility", "relocation"},
            ),
            compound_phrases=(
                "employee transfer",
                "employee transfers",
                "internal transfer",
                "internal transfers",
            ),
        ),
    ],
    # DOC-PRC-002: Vendor Onboarding Policy
    "DOC-PRC-002": [
        PolicyTopicDefinition(
            doc_id="DOC-PRC-002",
            topic_name="Vendor onboarding and supplier code of conduct",
            required_concept_groups=(
                {"vendor", "vendors", "supplier", "suppliers"},
                {"onboard", "onboarding", "conduct", "compliance", "bribery"},
            ),
            compound_phrases=(
                "vendor onboarding",
                "supplier code",
                "code of conduct",
            ),
        ),
    ],
}


def _normalize_text(text: str) -> str:
    """Normalize text to lowercase with simple punctuation replaced by whitespace."""
    return re.sub(r"[^\w\s-]", " ", text.lower())


def _extract_stemmed_tokens(text: str) -> Set[str]:
    """Extract set of words and simple stem forms from normalized text."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stems = set(words)
    for w in words:
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                stems.add(w[:-len(suffix)])
                break
    return stems


def match_public_policy_topics(query: str) -> List[Tuple[str, int]]:
    """Match query against the non-sensitive public policy topic catalog.

    Deterministic matching rules:
    1. Text is normalized to lowercase and simple stem forms.
    2. For a topic to match, query must satisfy ALL required concept groups
       or match an explicit compound phrase.
    3. Generic words like 'storage', 'data', or 'policy' alone NEVER match a topic.
       For example, 'storage' alone without sanitization/disposal concepts produces no match.
    4. Returns a list of (doc_id, match_score) tuples, where higher score indicates stronger match.
    """
    if not query or not query.strip():
        return []

    norm_query = _normalize_text(query)
    query_tokens = _extract_stemmed_tokens(norm_query)

    matches: List[Tuple[str, int]] = []

    for doc_id, topics in PUBLIC_POLICY_TOPIC_CATALOG.items():
        for topic in topics:
            score = 0

            # Check explicit compound phrases first
            matched_phrases = 0
            for phrase in topic.compound_phrases:
                if phrase in norm_query:
                    matched_phrases += 1

            if matched_phrases > 0:
                score += matched_phrases * 25

            # Check concept group coverage
            all_groups_matched = True
            group_match_counts = []
            for group in topic.required_concept_groups:
                matched_in_group = query_tokens.intersection(group)
                if not matched_in_group:
                    # Also check simple stem overlap
                    matched_stem = False
                    for w in query_tokens:
                        for suffix in ("ing", "ed", "es", "s"):
                            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                                if w[:-len(suffix)] in group:
                                    matched_stem = True
                                    break
                        if matched_stem:
                            break
                    if not matched_stem:
                        all_groups_matched = False
                        break
                    else:
                        group_match_counts.append(1)
                else:
                    group_match_counts.append(len(matched_in_group))

            if all_groups_matched and len(group_match_counts) == len(topic.required_concept_groups):
                # Ensure generic-only terms do not qualify
                # For DOC-SEC-004, at least one concrete storage term and one disposal term
                score += sum(group_match_counts) * 15

            # Require minimum threshold score for topic relevance
            if score >= 30:
                matches.append((doc_id, score))

    return sorted(matches, key=lambda x: x[1], reverse=True)
