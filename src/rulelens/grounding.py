"""Requested policy facts and conservative source-span verification."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence


GROUNDING_VERSION = "1.1"

# These identify policy subjects, not answers. Values always come from the corpus.
FACT_TARGETS = (
    ("travel.meals", "TRV-C03", r"meals?|subsistence|per diem|food|dining"),
    ("travel.hotel", "TRV-C03", r"hotels?|lodging|nightly|accommodations?"),
    ("travel.air_class", "TRV-C02", r"economy|flight class|air travel standards|eight.*flight hours"),
    ("travel.approval", "TRV-C04", r"(?:manager|director|standard).*approv|travel.*(?:threshold|spending limit)|approv.*travel"),
    ("travel.disallowances", "TRV-C05", r"non.reimbursable|entertainment|alcohol|mini.?bar|traffic citations|late check.out|flight upgrades"),
    ("travel.portal", "TRV-B01", r"travel portal|corporate portal|reservations?.*book|book.*reservations?|third.party|personal bookings"),
    ("travel.advance_booking", "TRV-B02", r"advance.*(?:book|departure)|(?:flight|booking).*(?:days|notice|lead time)|booking window|booking lead time"),
    ("travel.changes", "TRV-B03", r"cancellations?|reschedul|itinerary|travel date modifications"),
    ("travel.exceptions", "TRV-E01", r"travel exception|extraordinary.*travel|exception amounts?"),
    ("travel.urgent_deployment", "TRV-E02", r"(?:urgent|emergency).*deployment|customer.site deployment"),
    ("travel.medical", "TRV-E03", r"medical accommodation|specialized seating"),
    ("expense.eligibility", "FIN-C01", r"claim eligibility|out.of.pocket|expense.*eligible|eligible.*reimburse"),
    ("expense.submission", "FIN-C02", r"submit.*(?:expense|claim|reimburse)|(?:expense|claim|reimburse).*(?:submit|submission|filing|deadline)|file reimbursement|incurring an expense"),
    ("expense.manager_review", "FIN-C03", r"manager.*(?:review|inspect)|supervising manager|review.*(?:claim|voucher)|review turnaround|approval turnaround"),
    ("expense.payment", "FIN-R01", r"payment cycle|disburs|bank deposit|reimbursement.*(?:paid|dates)|settled claims.*paid"),
    ("expense.exchange", "FIN-R02", r"foreign currency|exchange rate|converted|international.*(?:settled|settling)|reserve bank"),
    ("expense.receipts", "FIN-R03", r"receipt|tax invoice|incidental"),
    ("card.eligibility", "FIN-K01", r"card.*(?:issuance|issued|eligible)|eligible.*card|card issuance"),
    ("card.limit", "FIN-K02", r"(?:card|cardholder).*(?:limit|spend|escalation)|monthly.*credit limit"),
    ("card.reconciliation", "FIN-K03", r"reconcil|monthly.*statement|billing cycle"),
    ("audit.sampling", "FIN-A01", r"(?:audit|inspect).*(?:percent|percentage|sampling)|monthly audits|audit.*sample"),
    ("audit.discrepancy", "FIN-A02", r"duplicate voucher|inflated receipt|claim freezing|discrepanc|unauthorized purchase"),
    ("audit.retention", "FIN-A03", r"audit.*(?:working papers|sampling logs|findings|retention)|seven fiscal years"),
    ("leave.vacation", "PPO-L01", r"vacation|annual leave|paid annual|annual paid leave|monthly.*accrual"),
    ("leave.sick", "PPO-L02", r"sick|medical certificate|self.certification"),
    ("leave.parental", "PPO-L03", r"parental|caregivers?|maternity|paternity"),
    ("remote.eligibility", "PPO-R01", r"probation|remote.*(?:days per week|eligib)|remote work days|remote.*request"),
    ("remote.speed", "PPO-R02", r"internet|broadband|mbps|download bandwidth|network speed"),
    ("remote.hours", "PPO-R03", r"core.*hours|availability|reachable|accessible.*(?:chat|channel)|remote.*(?:hours|communication)"),
    ("tuition.eligibility", "PPO-B01", r"(?:service|tenure|eligible|eligibility|accredited|job.relevant).*(?:tuition|course|assistance)|tuition.*(?:eligible|eligibility|scope)"),
    ("tuition.allowance", "PPO-B02", r"(?:tuition|educational|learning).*(?:amount|allowance|claim|grant|approv)|claim tuition|annual.*assistance|tuition assistance.*(?:maximum|limit)"),
    ("tuition.completion", "PPO-B03", r"completion|course conclusion|passing grade|transcript"),
    ("mobility.tenure", "PPO-T01", r"(?:tenure|criteria|performance).*(?:transfer|mobility)|(?:transfer|mobility).*(?:tenure|criteria)|twelve.*current role"),
    ("mobility.authorization", "PPO-T02", r"(?:transfer|mobility).*(?:agreement|authorize|authorization|leaders|approval)|leaders.*transfer"),
    ("mobility.handover", "PPO-T03", r"handover|transition period|retain.*transferr"),
    ("procurement.requisition", "PRC-P01", r"requisition|originate|enterprise.*(?:erp|system)|raising a po"),
    ("procurement.bids", "PRC-P02", r"competitive.*(?:bid|quote)|single.*quote|bidding threshold|purchase order.*(?:amount|inr)|vendor quotes?"),
    ("procurement.delivery", "PRC-P03", r"inspection warranty|freight terms|delivery terms|warranty period"),
    ("vendor.diligence", "PRC-V01", r"due diligence|registration validation|tax compliance|credit screening"),
    ("vendor.security", "PRC-V02", r"supplier assessment|vendor.*(?:data|assessment|internal systems)|handle proprietary"),
    ("vendor.conduct", "PRC-V03", r"code of conduct|anti.bribery|fair labo[u]?r|compliance code"),
    ("contract.signatory", "PRC-C01", r"signator|delegation of authority|authority.*enter.*contract"),
    ("contract.duration", "PRC-C02", r"contract.*(?:duration|months|sign.off|approval)|sign.off.*contract"),
    ("contract.liability", "PRC-C03", r"liability|master service agreement"),
    ("procurement.emergency", "PRC-E01", r"emergency.*(?:purchase|procurement|spending)|director emergency"),
    ("procurement.sole_source", "PRC-E02", r"sole source|without competitive bidding|technical exclusivity"),
    ("procurement.ratification", "PRC-E03", r"ratification|retroactive documentation|emergency.*business days"),
    ("security.classification", "SEC-D01", r"classification (?:tiers?|levels?)|four.*tiers|categorize corporate information"),
    ("security.encryption", "SEC-D02", r"encrypt|aes|hardware security module|tokenized"),
    ("security.transfer", "SEC-D03", r"(?:transferr|transmission|document.transfer).*network|outside authorized|outside company|classification headers|handling and labeling"),
    ("security.least_privilege", "SEC-A01", r"least privilege|account.*provision|access provisioning|principle.*access"),
    ("security.recertification", "SEC-A02", r"recertif|privileged.*review|administrative.*review|automatic.*suspension"),
    ("security.separation", "SEC-A03", r"deployment permission|code development|production database|break.glass|deployment restriction"),
    ("incident.severity", "SEC-I01", r"severity.*(?:tier|categor|priorit)|priority tiers|incidents.*categoriz|four severity"),
    ("incident.response", "SEC-I02", r"(?:priority 1|p1|critical).*(?:incident|response|contain)|incident.*(?:response|containment|sla)"),
    ("incident.report", "SEC-I03", r"post.incident|post.mortem|incident closure|remediation report|ciso|chief information security officer"),
    ("retention.records", "SEC-R01", r"payroll|financial ledger|customer audit logs|connection telemetr|statutory retention"),
    ("retention.disposal", "SEC-R02", r"sanitiz|decommissioned|degauss|storage drives|nist"),
    ("retention.hold", "SEC-R03", r"litigation|legal disputes|retention disposal"),
)

QUERY_WORDS = set("""
what which who whom whose how when where why is are was were be been being do does did
can could may must should will would shall have has had a an the of for under in on at
by with about within after before to from than into through during and or as if not no
per via without this that these those my our their your its it i we they me us you
please tell explain describe give show much many long often frequently each every any
all some only own same alone up including exactly specific standard maximum minimum
cap caps limit limits threshold ceiling amount amounts rate revised current updated
policy policies rule rules procedure procedures guidelines corporate company asterline
services group employee employees staff personnel workers worker regular full time
provided granted receive receives entitled entitlement total annually daily yearly
hours days weeks months years date dates starting onward required requires require
mandatory need needs needed permitted permissible allowed allow allows eligible eligibility qualify
approval approve approved authorize authorized authority authorization sign off
reporting direct department manager director committee review financial finance
ordinary normal single individual independently sufficient enough over above below
exceed exceeds exceeding less more least most further additional otherwise already
business official domestic assignments assignment work travel travelling traveling india indian rs inr rupees
claim claims claiming reimbursement reimburse reimbursable expense expenses incurred
incurring paying paid taking taking applicable applies apply according governed governs
secure discounted fares advance departure notice lead time flight flights booking booked
system systems channels communication mandatory remain maintain active secure available
document documents documentation proof details purpose scope criteria check checks
v v1 v2 v3 versus against meet meets compliant compliance per diem lodging accommodation
meal meals subsistence hotel hotels night nightly metropolitan metro regional locations
city cities region location remote working payroll ledger record records preserve
preserved retain retained retention paid fully annual vacation caregivers caregiver
primary secondary maternity paternity parental continuous period monthly calendar
entitlements certified certified authorization amount exception exceptions urgent
""".split())


def request_meaning(question: str) -> Dict[str, Any]:
    q = question.lower()
    targets = sorted(key for key, _, pattern in FACT_TARGETS if re.search(pattern, q))
    if re.search(r"\btuition assistance\b", q) and not any(t.startswith("tuition.") for t in targets):
        targets.append("tuition.eligibility")
        targets.sort()
    # An exception approval is a different route from ordinary manager approval.
    if "travel.exceptions" in targets or "travel.urgent_deployment" in targets:
        targets = [t for t in targets if t != "travel.approval"]
    if "expense.manager_review" in targets and not re.search(r"employee.*submit|employee.*deadline|submission window|file reimbursement", q):
        targets = [t for t in targets if t != "expense.submission"]
    if "procurement.emergency" in targets:
        targets = [t for t in targets if t != "procurement.bids"]
    if any(not t.startswith("travel.") for t in targets) and not re.search(r"\btravel\b|\btrip\b", q):
        targets = [t for t in targets if t != "travel.approval"]
    slots = {}
    for name, choices in {
        "caregiver": {"primary": r"primary|maternity", "secondary": r"secondary|paternity"},
        "location": {"metropolitan": r"metropolitan|\bmetro\b", "regional": r"regional"},
        "classification": {"confidential": r"confidential", "restricted": r"restricted"},
        "records": {"payroll": r"payroll|financial ledger", "customer_logs": r"customer audit|connection telemetr"},
        "period": {"monthly": r"per\s+(?:(?:completed|calendar|full)\s+)*month|monthly|completed calendar month", "annual": r"per year|annual|each year"},
        "travel_scope": {"domestic": r"domestic|within india", "international": r"international"},
    }.items():
        selected = sorted(k for k, pattern in choices.items() if re.search(pattern, q))
        if selected:
            slots[name] = selected
    neutral = q
    if "travel.approval" in targets and re.search(r"maximum|limit|threshold|ceiling", q) and "manager" in q:
        neutral = re.sub(r"without\s+(?:additional\s+)?(?:department\s+)?director\s+(?:sign.off|approval)", "", neutral)
    neutral = re.sub(r"\bunder\s+(?:(?:the|current|corporate)\s+)*(?:travel\s+)?(?:policy|rules?)\s*(?:v\d+)?", "", neutral)
    negation = bool(re.search(r"\b(?:not|never|no|cannot|can't|without|neither|nor)\b", neutral))
    request_text = re.sub(r"^\s*(?:under|according to)[^,?]+,\s*", "", q)
    kind = "decision" if re.match(r"\s*(?:can|could|may|does|do|is|are)\b", request_text) else "lookup"
    authorities = sorted(set(re.findall(r"\b(?:cfo|ciso|director|manager|committee|board|vice president|department functional lead)\b", neutral)))
    # Lookup facts already specify their authorities; wording need not repeat them.
    if kind == "lookup":
        authorities = []
    return {"version": GROUNDING_VERSION, "targets": targets, "slots": slots,
            "kind": kind, "negation": negation, "authorities": authorities,
            "exception": bool(re.search(r"\b(?:exception|urgent|emergency|waiver|exempt)\b", neutral))}


def source_facts(chunks: Sequence[Dict[str, Any]], meaning: Dict[str, Any]) -> List[Dict[str, str]]:
    facts = []
    for target, clause_id, _ in FACT_TARGETS:
        if target not in meaning["targets"]:
            continue
        for chunk in chunks:
            if chunk["clause_id"] != clause_id:
                continue
            body = chunk["text"].split("\n", 1)[-1].strip()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
            if target == "travel.meals":
                sentences = [s for s in sentences if "meal" in s.lower()]
            elif target == "travel.hotel":
                sentences = [s for s in sentences if "hotel" in s.lower()]
            elif target in ("leave.parental", "security.encryption", "retention.records", "travel.air_class"):
                slot = {"leave.parental": "caregiver", "security.encryption": "classification",
                        "retention.records": "records", "travel.air_class": "travel_scope"}[target]
                selected = meaning["slots"].get(slot, [])
                if selected:
                    keywords = {"customer_logs": "customer audit", "payroll": "payroll"}
                    sentences = [s for s in sentences if any(keywords.get(v, v) in s.lower() for v in selected)]
            quote = " ".join(sentences)
            if quote:
                facts.append({"target": target, "chunk_id": chunk["chunk_id"], "quote": quote})
    return facts


def request_support(question: str, chunks: Sequence[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, str]], List[str]]:
    meaning = request_meaning(question)
    facts = source_facts(chunks, meaning)
    missing = sorted(set(meaning["targets"]) - {f["target"] for f in facts})
    versions = re.findall(r"\btravel\s+policy\s+v([0-9]+)\b", question, re.I)
    travel_chunks = [c for c in chunks if c["clause_id"].startswith("TRV-C")]
    if versions and not all(any(str(c["version"]).split(".")[0] == version for c in travel_chunks) for version in versions):
        missing.append("conflicting_version")
    source_words = set(re.findall(r"[a-z]+", " ".join(c["text"] for c in chunks).lower()))
    query_words = set(re.findall(r"[a-z]+", question.lower()))
    unknown = sorted(w for w in query_words if w not in QUERY_WORDS and w not in source_words
                     and w.rstrip("s") not in source_words)
    if not meaning["targets"]:
        missing.append("unknown_target")
    if meaning["kind"] == "decision":
        missing.append("unsupported_decision")
    if unknown:
        missing.append("unsupported_context")
    # Explicit subrequests must each have a known target.
    for part in re.split(r"\band\s+(?=(?:what|who|how|when|where|within|which)\b)|;", question, flags=re.I):
        if part.strip() and not request_meaning(part)["targets"]:
            missing.append("unsupported_subrequest")
    return meaning, facts, sorted(set(missing))


def validate_grounded_answer(question: str, answer: str, citations: Sequence[str], chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    meaning, facts, missing = request_support(question, chunks)
    if missing:
        raise ValueError("Requested facts are not fully supported.")
    available = [f for f in facts if f["chunk_id"] in citations]
    remaining = " ".join(line.removeprefix("> ").strip() for line in answer.strip().splitlines())
    claims = []
    # Consume complete facts only, preserving prerequisites in the same source span.
    while remaining:
        matches = [f for f in available if remaining.startswith(f["quote"])]
        if not matches:
            raise ValueError("Answer contains an unsupported or incomplete source span.")
        fact = max(matches, key=lambda f: len(f["quote"]))
        claims.extend(f for f in matches if f["quote"] == fact["quote"])
        remaining = remaining[len(fact["quote"]):].strip()
    if not claims or set(meaning["targets"]) != {f["target"] for f in claims}:
        raise ValueError("Answer does not cover every requested fact.")
    if set(citations) != {f["chunk_id"] for f in claims}:
        raise ValueError("Citations do not match the verified claims.")
    return claims


def render_grounded_answer(claims: Sequence[Dict[str, str]]) -> str:
    return "\n\n".join("> " + quote for quote in dict.fromkeys(c["quote"] for c in claims))


def fact_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    data = [(c["chunk_id"], c["content_hash"], c["version"]) for c in chunks]
    encoded = json.dumps([GROUNDING_VERSION, FACT_TARGETS, sorted(data)], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
