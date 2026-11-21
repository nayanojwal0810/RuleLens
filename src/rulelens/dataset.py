"""Deterministic enterprise policy dataset generator and validator for RuleLens.

Company: Asterline Services Group
Produces:
  - 20 connected policy PDFs (5 departments x 4 documents)
  - 5 structured rule CSV tables
  - roles.json (6 demo roles with explicit document permissions)
  - manifest.json (document inventory, clause IDs, and related rules)
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

COMPANY_NAME: str = "Asterline Services Group"

DEPARTMENTS: Tuple[str, ...] = (
    "Travel",
    "Finance",
    "People Operations",
    "Procurement",
    "Information Security",
)

ROLES: Tuple[str, ...] = (
    "employee",
    "manager",
    "hr",
    "finance",
    "security",
    "auditor",
)

CSV_FIELDNAMES: Tuple[str, ...] = (
    "rule_id",
    "doc_id",
    "clause_id",
    "condition_field",
    "operator",
    "threshold_value",
    "unit",
    "outcome",
    "effective_from",
    "effective_to",
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    doc_id: str
    clause_id: str
    condition_field: str
    operator: str
    threshold_value: str
    unit: str
    outcome: str
    effective_from: str
    effective_to: str


@dataclass(frozen=True)
class Clause:
    clause_id: str
    title: str
    text: str


@dataclass(frozen=True)
class PolicyDocument:
    doc_id: str
    filename: str
    title: str
    department: str
    version: str
    effective_from: str
    effective_to: str
    permitted_roles: Tuple[str, ...]
    clauses: Tuple[Clause, ...]


# ---------------------------------------------------------------------------
# Central Dataset Specification
# ---------------------------------------------------------------------------

DOCUMENTS: Tuple[PolicyDocument, ...] = (
    # --- Travel Department ---
    PolicyDocument(
        doc_id="DOC-TRV-001",
        filename="travel_policy_v1.pdf",
        title="Travel Policy v1",
        department="Travel",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2026-06-30",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="TRV-C01",
                title="Purpose and Scope",
                text="This policy establishes corporate travel guidelines for all Asterline Services Group personnel undertaking authorized business travel between January 1, 2026 and June 30, 2026. All business travel must be directly related to active commercial operations and approved in advance.",
            ),
            Clause(
                clause_id="TRV-C02",
                title="Air Travel Standards",
                text="Domestic air travel must be booked in economy class. International flights exceeding eight continuous flight hours are eligible for premium economy class subject to prior written recommendation from the reporting manager.",
            ),
            Clause(
                clause_id="TRV-C03",
                title="Lodging and Meals",
                text="Standard hotel accommodations are capped at INR 6,000 per night in metropolitan cities and INR 4,000 in regional locations. Daily meal subsistence allowance is capped at INR 1,500 per diem with itemized receipts.",
            ),
            Clause(
                clause_id="TRV-C04",
                title="Travel Expense Approval Thresholds",
                text="Travel expenses up to and including INR 40,000 follow standard approval by the direct reporting manager. Any travel expenses exceeding INR 40,000 require additional approval from the department director prior to reimbursement.",
            ),
            Clause(
                clause_id="TRV-C05",
                title="Non-Reimbursable Disallowances",
                text="Personal entertainment, alcoholic beverages, hotel minibar charges, traffic citations, late check-out fees, and flight upgrades purchased independently will not be reimbursed by Asterline Services Group.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-TRV-002",
        filename="travel_policy_v2.pdf",
        title="Travel Policy v2",
        department="Travel",
        version="2.0",
        effective_from="2026-07-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="TRV-C01",
                title="Purpose and Scope",
                text="This revised policy establishes updated corporate travel guidelines for all Asterline Services Group personnel undertaking authorized business travel from July 1, 2026 onward. This document formally supersedes Travel Policy v1.",
            ),
            Clause(
                clause_id="TRV-C02",
                title="Air Travel Standards",
                text="Domestic air travel must be booked in economy class. International flights exceeding eight continuous flight hours are eligible for premium economy class subject to prior written recommendation from the reporting manager.",
            ),
            Clause(
                clause_id="TRV-C03",
                title="Lodging and Meals",
                text="Standard hotel accommodations are capped at INR 7,500 per night in metropolitan cities and INR 5,000 in regional locations. Daily meal subsistence allowance is capped at INR 2,000 per diem with itemized receipts.",
            ),
            Clause(
                clause_id="TRV-C04",
                title="Travel Expense Approval Thresholds",
                text="Travel expenses up to and including INR 50,000 follow standard approval by the direct reporting manager. Any travel expenses exceeding INR 50,000 require additional approval from the department director prior to reimbursement.",
            ),
            Clause(
                clause_id="TRV-C05",
                title="Non-Reimbursable Disallowances",
                text="Personal entertainment, alcoholic beverages, hotel minibar charges, traffic citations, late check-out fees, and flight upgrades purchased independently will not be reimbursed by Asterline Services Group.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-TRV-003",
        filename="travel_booking.pdf",
        title="Travel Booking",
        department="Travel",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="TRV-B01",
                title="Central Travel Portal",
                text="All employee airline, railway, and hotel reservations must be booked exclusively through the designated Asterline Travel Portal. Direct personal bookings on commercial third-party websites are non-reimbursable.",
            ),
            Clause(
                clause_id="TRV-B02",
                title="Advance Booking Window",
                text="Domestic flight bookings must be made at least 14 days in advance of departure to secure corporate discounted fares. Bookings made fewer than 14 days before departure require expedited approval from the functional manager.",
            ),
            Clause(
                clause_id="TRV-B03",
                title="Cancellations and Rescheduling",
                text="Ticket cancellations or travel date modifications must be logged in the travel portal within 24 hours of itinerary change notification to minimize carrier penalty fees.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-TRV-004",
        filename="travel_exceptions.pdf",
        title="Travel Exceptions",
        department="Travel",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="TRV-E01",
                title="Travel Exception Approval Route",
                text="When extraordinary business exigencies require exceeding standard travel caps, exception amounts up to and including INR 100,000 require approval from the Department Vice President. Exception amounts exceeding INR 100,000 require approval from the Chief Financial Officer.",
            ),
            Clause(
                clause_id="TRV-E02",
                title="Emergency Client Deployments",
                text="Urgent customer-site deployments requested on less than 48 hours notice are exempt from advance booking windows upon simultaneous notification to Finance Operations.",
            ),
            Clause(
                clause_id="TRV-E03",
                title="Medical Accommodations",
                text="Medical accommodation requests requiring specialized seating or accommodations must be certified by People Operations and approved for enhanced travel allowance.",
            ),
        ),
    ),
    # --- Finance Department ---
    PolicyDocument(
        doc_id="DOC-FIN-001",
        filename="expense_claims.pdf",
        title="Expense Claims",
        department="Finance",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="FIN-C01",
                title="General Claim Eligibility",
                text="Employees incurring out-of-pocket expenses while conducting authorized company duties are entitled to full reimbursement upon submitting verified expense vouchers.",
            ),
            Clause(
                clause_id="FIN-C02",
                title="Claim Submission Window",
                text="Expense claims submitted within 30 days of expense incurrence are processed under regular reimbursement timelines. Claims submitted after 30 days require a late submission justification approved by Finance Operations.",
            ),
            Clause(
                clause_id="FIN-C03",
                title="Manager Verification SLA",
                text="Supervising managers must inspect supporting vouchers and approve or return subordinate claims within five working days of submission.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-FIN-002",
        filename="reimbursements.pdf",
        title="Reimbursements",
        department="Finance",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="FIN-R01",
                title="Payment Cycles",
                text="Approved claims are disbursed twice monthly via direct bank deposit on the 15th and the final business day of each calendar month.",
            ),
            Clause(
                clause_id="FIN-R02",
                title="Foreign Currency Claims",
                text="International expenditures are settled in Indian Rupees using the official Reserve Bank of India reference rate published on the date of transaction.",
            ),
            Clause(
                clause_id="FIN-R03",
                title="Receipt Verification Threshold",
                text="Incidental business expenses up to and including INR 1,000 do not require an itemized tax receipt provided digital payment confirmation is attached. Incidental expenses exceeding INR 1,000 strictly require an itemized tax invoice.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-FIN-003",
        filename="corporate_cards.pdf",
        title="Corporate Cards",
        department="Finance",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="FIN-K01",
                title="Card Issuance Criteria",
                text="Corporate credit cards are issued to designated managers and frequent business travelers following authorization by the Finance Controller.",
            ),
            Clause(
                clause_id="FIN-K02",
                title="Monthly Spend Limit",
                text="Corporate cardholders are allocated a standard monthly spending limit of INR 200,000. Monthly spending exceeding INR 200,000 requires card limit escalation and formal credit extension approved by Finance Operations.",
            ),
            Clause(
                clause_id="FIN-K03",
                title="Statement Reconciliation",
                text="Cardholders must submit expense reconciliations and itemized receipts within ten business days of monthly billing cycle statement generation.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-FIN-004",
        filename="expense_audits.pdf",
        title="Expense Audits",
        department="Finance",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("finance", "auditor"),
        clauses=(
            Clause(
                clause_id="FIN-A01",
                title="Random Audit Sampling",
                text="Internal Audit performs regular monthly audits sampling a minimum of 10 percent of all settled employee expense reimbursement claims.",
            ),
            Clause(
                clause_id="FIN-A02",
                title="Discrepancy Investigation",
                text="Suspected duplicate vouchers, inflated receipts, or unauthorized purchases trigger immediate claim freezing and formal referral to the Internal Audit Committee.",
            ),
            Clause(
                clause_id="FIN-A03",
                title="Audit Trail Retention",
                text="All financial audit working papers, sampling logs, and investigation findings must be archived in tamper-evident storage for seven fiscal years.",
            ),
        ),
    ),
    # --- People Operations Department ---
    PolicyDocument(
        doc_id="DOC-PPO-001",
        filename="leave_policy.pdf",
        title="Leave Policy",
        department="People Operations",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "hr", "auditor"),
        clauses=(
            Clause(
                clause_id="PPO-L01",
                title="Annual Leave Entitlement",
                text="Full-time employees accrue 20 working days of paid annual vacation per year, credited pro-rata at the rate of 1.67 days per completed calendar month.",
            ),
            Clause(
                clause_id="PPO-L02",
                title="Sick Leave Certification",
                text="Employees taking up to and including 3 consecutive sick leave days may submit self-certification upon return to work. Consecutive sick leave exceeding 3 days requires a medical certificate issued by a registered medical practitioner.",
            ),
            Clause(
                clause_id="PPO-L03",
                title="Parental Leave Provisions",
                text="Eligible primary caregivers receive 26 continuous weeks of fully paid maternity leave. Secondary caregivers receive 4 weeks of fully paid paternity leave.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PPO-002",
        filename="remote_work.pdf",
        title="Remote Work",
        department="People Operations",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "hr", "auditor"),
        clauses=(
            Clause(
                clause_id="PPO-R01",
                title="Remote Work Eligibility",
                text="Employees having completed their initial three-month probationary evaluation with satisfactory performance may request up to three remote work days per week.",
            ),
            Clause(
                clause_id="PPO-R02",
                title="Network Speed Standard",
                text="Remote personnel must maintain broadband internet connectivity with at least 50 mbps download bandwidth. Internet speeds below 50 mbps require connection remediation before remote work authorization is granted.",
            ),
            Clause(
                clause_id="PPO-R03",
                title="Core Availability Window",
                text="Remote employees must remain accessible via company chat and video conferencing during core business hours between 10:00 AM and 4:00 PM local time.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PPO-003",
        filename="learning_benefits.pdf",
        title="Learning Benefits",
        department="People Operations",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "hr", "auditor"),
        clauses=(
            Clause(
                clause_id="PPO-B01",
                title="Tuition Assistance Scope",
                text="Permanent employees with at least six months of continuous service are eligible to claim tuition assistance for accredited job-relevant degree or certification courses.",
            ),
            Clause(
                clause_id="PPO-B02",
                title="Annual Learning Allowance",
                text="Annual educational assistance up to and including INR 75,000 is granted under standard reporting manager approval. Educational assistance claims exceeding INR 75,000 require review and approval from the People Operations Learning Committee.",
            ),
            Clause(
                clause_id="PPO-B03",
                title="Completion Verification",
                text="Reimbursement disbursement requires submitting an official passing grade transcript or certificate of completion within 45 days of course conclusion.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PPO-004",
        filename="employee_transfers.pdf",
        title="Employee Transfers",
        department="People Operations",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "hr"),
        clauses=(
            Clause(
                clause_id="PPO-T01",
                title="Internal Mobility Tenancy",
                text="Employees must complete a minimum tenure of twelve consecutive months in their current role with satisfactory performance before applying for internal transfers.",
            ),
            Clause(
                clause_id="PPO-T02",
                title="Transfer Authorization Chain",
                text="Internal transfers require written agreement from the releasing department head, the acquiring department head, and People Operations HR Business Partner.",
            ),
            Clause(
                clause_id="PPO-T03",
                title="Transition Handover Period",
                text="The releasing department may retain the transferring employee for a handover transition period not exceeding thirty calendar days from offer acceptance.",
            ),
        ),
    ),
    # --- Procurement Department ---
    PolicyDocument(
        doc_id="DOC-PRC-001",
        filename="purchase_requests.pdf",
        title="Purchase Requests",
        department="Procurement",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="PRC-P01",
                title="Purchase Requisition Origination",
                text="All commercial purchases of hardware, software, licenses, or professional services must originate as an approved electronic Purchase Requisition in the enterprise ERP.",
            ),
            Clause(
                clause_id="PRC-P02",
                title="Competitive Bidding Threshold",
                text="Purchase orders up to and including INR 150,000 may be processed with a single qualified vendor quote. Purchase orders exceeding INR 150,000 strictly require at least three independent competitive bids submitted to Procurement.",
            ),
            Clause(
                clause_id="PRC-P03",
                title="Standard Delivery Terms",
                text="All purchase orders must incorporate standard commercial freight terms and a minimum 30-day inspection warranty period from receipt.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PRC-002",
        filename="vendor_onboarding.pdf",
        title="Vendor Onboarding",
        department="Procurement",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="PRC-V01",
                title="Vendor Due Diligence",
                text="Prospective suppliers must undergo statutory due diligence including corporate registration validation, tax compliance certificates, and credit screening.",
            ),
            Clause(
                clause_id="PRC-V02",
                title="Security Risk Assessment",
                text="Vendors handling proprietary company data or connecting to internal systems must complete an Information Security Supplier Assessment prior to contract signing.",
            ),
            Clause(
                clause_id="PRC-V03",
                title="Ethics and Compliance Code",
                text="All onboarded suppliers must execute the Asterline Supplier Code of Conduct affirming strict compliance with anti-bribery and fair labor regulations.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PRC-003",
        filename="contract_approval.pdf",
        title="Contract Approval",
        department="Procurement",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="PRC-C01",
                title="Signatory Authority",
                text="Commercial contracts may only be entered into by designated corporate signatories defined in the corporate Delegation of Authority matrix.",
            ),
            Clause(
                clause_id="PRC-C02",
                title="Contract Term Sign-Off Authority",
                text="Vendor contracts with duration up to and including 12 months require approval by the Department Functional Lead. Vendor contracts with duration exceeding 12 months require joint sign-off from Legal Counsel and Procurement.",
            ),
            Clause(
                clause_id="PRC-C03",
                title="Liability Limitation",
                text="Standard master service agreements must cap mutual commercial liability at twelve months of fees, unless an explicit waiver is granted by Legal Counsel.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-PRC-004",
        filename="purchase_exceptions.pdf",
        title="Purchase Exceptions",
        department="Procurement",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("manager", "finance", "auditor"),
        clauses=(
            Clause(
                clause_id="PRC-E01",
                title="Emergency Procurement Cap",
                text="In critical operational emergencies impacting business continuity, emergency purchases up to and including INR 300,000 may proceed under Director emergency approval. Emergency purchases exceeding INR 300,000 require authorization from the Board Procurement Committee.",
            ),
            Clause(
                clause_id="PRC-E02",
                title="Sole Source Justification",
                text="Procurement without competitive bidding requires an authenticated Sole Source Justification detailing proprietary technical exclusivity or market absence of alternatives.",
            ),
            Clause(
                clause_id="PRC-E03",
                title="Emergency Ratification",
                text="All emergency purchase orders must be formally submitted for retroactive documentation and invoice ratification within ten business days of issuance.",
            ),
        ),
    ),
    # --- Information Security Department ---
    PolicyDocument(
        doc_id="DOC-SEC-001",
        filename="data_classification.pdf",
        title="Data Classification",
        department="Information Security",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "security", "auditor"),
        clauses=(
            Clause(
                clause_id="SEC-D01",
                title="Classification Tiers",
                text="All Asterline digital and physical information assets are classified into four tiers: Public, Internal, Confidential, and Restricted based on business impact.",
            ),
            Clause(
                clause_id="SEC-D02",
                title="Tier Encryption Standards",
                text="Data classified as Confidential requires mandatory AES-256 encryption at rest and in transit. Data classified as Restricted requires hardware security module tokenized encryption and dual authorization for access.",
            ),
            Clause(
                clause_id="SEC-D03",
                title="Handling and Labeling",
                text="Confidential and Restricted documents must bear prominent visual classification headers and may not be transferred outside authorized networks without security approval.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-SEC-002",
        filename="access_requests.pdf",
        title="Access Requests",
        department="Information Security",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "security", "auditor"),
        clauses=(
            Clause(
                clause_id="SEC-A01",
                title="Least Privilege Principle",
                text="User accounts are provisioned solely with access permissions necessary to fulfill designated job responsibilities following the principle of least privilege.",
            ),
            Clause(
                clause_id="SEC-A02",
                title="Privileged Role Review Frequency",
                text="Privileged and administrative access rights must undergo recertification review at intervals up to and including 90 days. Privileged accounts exceeding 90 days without recertification are subject to automatic account suspension.",
            ),
            Clause(
                clause_id="SEC-A03",
                title="Role Separation",
                text="Personnel assigned code development duties may not hold administrative deployment permissions on production databases without a temporary break-glass waiver.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-SEC-003",
        filename="incident_reporting.pdf",
        title="Incident Reporting",
        department="Information Security",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("employee", "manager", "security", "auditor"),
        clauses=(
            Clause(
                clause_id="SEC-I01",
                title="Security Incident Severity",
                text="Security incidents are triaged into Priority 1 (Critical), Priority 2 (Major), Priority 3 (Moderate), and Priority 4 (Minor) according to system impact.",
            ),
            Clause(
                clause_id="SEC-I02",
                title="Priority 1 Response SLA",
                text="Priority 1 critical incidents involving confirmed system compromise must be responded to and contained within 1 hour of alert generation. Response times exceeding 1 hour require immediate SLA breach escalation to executive leadership.",
            ),
            Clause(
                clause_id="SEC-I03",
                title="Incident Post-Mortem",
                text="A comprehensive post-incident analysis and remediation report must be published to the Chief Information Security Officer within 72 hours of incident closure.",
            ),
        ),
    ),
    PolicyDocument(
        doc_id="DOC-SEC-004",
        filename="data_retention.pdf",
        title="Data Retention",
        department="Information Security",
        version="1.0",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
        permitted_roles=("security", "auditor"),
        clauses=(
            Clause(
                clause_id="SEC-R01",
                title="Statutory Retention Periods",
                text="Employee payroll and financial ledger records must be preserved for seven calendar years. Customer audit logs and connection telemetries are retained for two years.",
            ),
            Clause(
                clause_id="SEC-R02",
                title="Sanitization and Disposal",
                text="Decommissioned storage drives must be purged in compliance with NIST SP 800-88 standards or physically degaussed before disposal.",
            ),
            Clause(
                clause_id="SEC-R03",
                title="Litigation Hold Protocol",
                text="Information assets associated with active or anticipated legal disputes must be placed on litigation hold and exempted from automated retention disposal.",
            ),
        ),
    ),
)


RULES: Tuple[Rule, ...] = (
    # --- Travel Rules (data/source/rules/travel_rules.csv) ---
    Rule(
        rule_id="R-TRV-001",
        doc_id="DOC-TRV-001",
        clause_id="TRV-C04",
        condition_field="expense_amount",
        operator="<=",
        threshold_value="40000",
        unit="INR",
        outcome="standard_approval",
        effective_from="2026-01-01",
        effective_to="2026-06-30",
    ),
    Rule(
        rule_id="R-TRV-002",
        doc_id="DOC-TRV-001",
        clause_id="TRV-C04",
        condition_field="expense_amount",
        operator=">",
        threshold_value="40000",
        unit="INR",
        outcome="additional_approval",
        effective_from="2026-01-01",
        effective_to="2026-06-30",
    ),
    Rule(
        rule_id="R-TRV-003",
        doc_id="DOC-TRV-002",
        clause_id="TRV-C04",
        condition_field="expense_amount",
        operator="<=",
        threshold_value="50000",
        unit="INR",
        outcome="standard_approval",
        effective_from="2026-07-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-TRV-004",
        doc_id="DOC-TRV-002",
        clause_id="TRV-C04",
        condition_field="expense_amount",
        operator=">",
        threshold_value="50000",
        unit="INR",
        outcome="additional_approval",
        effective_from="2026-07-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-TRV-005",
        doc_id="DOC-TRV-003",
        clause_id="TRV-B02",
        condition_field="advance_booking_days",
        operator=">=",
        threshold_value="14",
        unit="days",
        outcome="standard_booking",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-TRV-006",
        doc_id="DOC-TRV-003",
        clause_id="TRV-B02",
        condition_field="advance_booking_days",
        operator="<",
        threshold_value="14",
        unit="days",
        outcome="expedited_approval_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-TRV-007",
        doc_id="DOC-TRV-004",
        clause_id="TRV-E01",
        condition_field="exception_amount",
        operator="<=",
        threshold_value="100000",
        unit="INR",
        outcome="vp_approval",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-TRV-008",
        doc_id="DOC-TRV-004",
        clause_id="TRV-E01",
        condition_field="exception_amount",
        operator=">",
        threshold_value="100000",
        unit="INR",
        outcome="cfo_approval",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    # --- Finance Rules (data/source/rules/finance_rules.csv) ---
    Rule(
        rule_id="R-FIN-001",
        doc_id="DOC-FIN-001",
        clause_id="FIN-C02",
        condition_field="submission_delay_days",
        operator="<=",
        threshold_value="30",
        unit="days",
        outcome="eligible_reimbursement",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-FIN-002",
        doc_id="DOC-FIN-001",
        clause_id="FIN-C02",
        condition_field="submission_delay_days",
        operator=">",
        threshold_value="30",
        unit="days",
        outcome="requires_late_exception",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-FIN-003",
        doc_id="DOC-FIN-002",
        clause_id="FIN-R03",
        condition_field="receipt_amount",
        operator="<=",
        threshold_value="1000",
        unit="INR",
        outcome="no_receipt_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-FIN-004",
        doc_id="DOC-FIN-002",
        clause_id="FIN-R03",
        condition_field="receipt_amount",
        operator=">",
        threshold_value="1000",
        unit="INR",
        outcome="itemized_receipt_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-FIN-005",
        doc_id="DOC-FIN-003",
        clause_id="FIN-K02",
        condition_field="monthly_spend",
        operator="<=",
        threshold_value="200000",
        unit="INR",
        outcome="within_standard_card_limit",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-FIN-006",
        doc_id="DOC-FIN-003",
        clause_id="FIN-K02",
        condition_field="monthly_spend",
        operator=">",
        threshold_value="200000",
        unit="INR",
        outcome="card_limit_escalation_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    # --- People Operations Rules (data/source/rules/people_rules.csv) ---
    Rule(
        rule_id="R-PPO-001",
        doc_id="DOC-PPO-001",
        clause_id="PPO-L02",
        condition_field="consecutive_sick_days",
        operator="<=",
        threshold_value="3",
        unit="days",
        outcome="self_certification",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PPO-002",
        doc_id="DOC-PPO-001",
        clause_id="PPO-L02",
        condition_field="consecutive_sick_days",
        operator=">",
        threshold_value="3",
        unit="days",
        outcome="medical_certificate_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PPO-003",
        doc_id="DOC-PPO-002",
        clause_id="PPO-R02",
        condition_field="internet_speed_mbps",
        operator=">=",
        threshold_value="50",
        unit="mbps",
        outcome="remote_setup_compliant",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PPO-004",
        doc_id="DOC-PPO-002",
        clause_id="PPO-R02",
        condition_field="internet_speed_mbps",
        operator="<",
        threshold_value="50",
        unit="mbps",
        outcome="remediation_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PPO-005",
        doc_id="DOC-PPO-003",
        clause_id="PPO-B02",
        condition_field="annual_tuition_claim",
        operator="<=",
        threshold_value="75000",
        unit="INR",
        outcome="standard_tuition_grant",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PPO-006",
        doc_id="DOC-PPO-003",
        clause_id="PPO-B02",
        condition_field="annual_tuition_claim",
        operator=">",
        threshold_value="75000",
        unit="INR",
        outcome="committee_review_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    # --- Procurement Rules (data/source/rules/procurement_rules.csv) ---
    Rule(
        rule_id="R-PRC-001",
        doc_id="DOC-PRC-001",
        clause_id="PRC-P02",
        condition_field="order_amount",
        operator="<=",
        threshold_value="150000",
        unit="INR",
        outcome="single_quote_permitted",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PRC-002",
        doc_id="DOC-PRC-001",
        clause_id="PRC-P02",
        condition_field="order_amount",
        operator=">",
        threshold_value="150000",
        unit="INR",
        outcome="three_competitive_quotes_required",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PRC-003",
        doc_id="DOC-PRC-003",
        clause_id="PRC-C02",
        condition_field="contract_duration_months",
        operator="<=",
        threshold_value="12",
        unit="months",
        outcome="department_lead_signoff",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PRC-004",
        doc_id="DOC-PRC-003",
        clause_id="PRC-C02",
        condition_field="contract_duration_months",
        operator=">",
        threshold_value="12",
        unit="months",
        outcome="legal_and_procurement_signoff",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PRC-005",
        doc_id="DOC-PRC-004",
        clause_id="PRC-E01",
        condition_field="emergency_purchase_amount",
        operator="<=",
        threshold_value="300000",
        unit="INR",
        outcome="director_emergency_approval",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-PRC-006",
        doc_id="DOC-PRC-004",
        clause_id="PRC-E01",
        condition_field="emergency_purchase_amount",
        operator=">",
        threshold_value="300000",
        unit="INR",
        outcome="board_procurement_committee",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    # --- Information Security Rules (data/source/rules/security_rules.csv) ---
    Rule(
        rule_id="R-SEC-001",
        doc_id="DOC-SEC-001",
        clause_id="SEC-D02",
        condition_field="data_classification_tier",
        operator="==",
        threshold_value="Confidential",
        unit="tier",
        outcome="aes_256_mandatory",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-SEC-002",
        doc_id="DOC-SEC-001",
        clause_id="SEC-D02",
        condition_field="data_classification_tier",
        operator="==",
        threshold_value="Restricted",
        unit="tier",
        outcome="hsm_tokenized_encryption",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-SEC-003",
        doc_id="DOC-SEC-002",
        clause_id="SEC-A02",
        condition_field="privileged_role_review_interval_days",
        operator="<=",
        threshold_value="90",
        unit="days",
        outcome="access_recertification_compliant",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-SEC-004",
        doc_id="DOC-SEC-002",
        clause_id="SEC-A02",
        condition_field="privileged_role_review_interval_days",
        operator=">",
        threshold_value="90",
        unit="days",
        outcome="automatic_account_suspension",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-SEC-005",
        doc_id="DOC-SEC-003",
        clause_id="SEC-I02",
        condition_field="incident_severity_p1_response_hours",
        operator="<=",
        threshold_value="1",
        unit="hours",
        outcome="sla_met",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
    Rule(
        rule_id="R-SEC-006",
        doc_id="DOC-SEC-003",
        clause_id="SEC-I02",
        condition_field="incident_severity_p1_response_hours",
        operator=">",
        threshold_value="1",
        unit="hours",
        outcome="sla_breach_escalation",
        effective_from="2026-01-01",
        effective_to="2099-12-31",
    ),
)


# Mapping department to rules CSV filename
DEPARTMENT_RULES_MAPPING: Dict[str, str] = {
    "Travel": "travel_rules.csv",
    "Finance": "finance_rules.csv",
    "People Operations": "people_rules.csv",
    "Procurement": "procurement_rules.csv",
    "Information Security": "security_rules.csv",
}

# Substrings in policy clause texts supporting specific rule outcomes
OUTCOME_TEXT_SUPPORT: Dict[str, Tuple[str, ...]] = {
    "standard_approval": ("standard approval",),
    "additional_approval": ("additional approval",),
    "standard_booking": ("advance", "fares"),
    "expedited_approval_required": ("expedited approval",),
    "vp_approval": ("Vice President",),
    "cfo_approval": ("Chief Financial Officer",),
    "eligible_reimbursement": ("regular reimbursement",),
    "requires_late_exception": ("late submission",),
    "no_receipt_required": ("do not require an itemized",),
    "itemized_receipt_required": ("itemized tax invoice",),
    "within_standard_card_limit": ("standard monthly spending limit",),
    "card_limit_escalation_required": ("card limit escalation",),
    "self_certification": ("self-certification",),
    "medical_certificate_required": ("medical certificate",),
    "remote_setup_compliant": ("remote work",),
    "remediation_required": ("remediation",),
    "standard_tuition_grant": ("standard reporting manager approval",),
    "committee_review_required": ("Learning Committee",),
    "single_quote_permitted": ("single qualified vendor quote",),
    "three_competitive_quotes_required": ("three independent competitive bids",),
    "department_lead_signoff": ("Department Functional Lead",),
    "legal_and_procurement_signoff": ("Legal Counsel and Procurement",),
    "director_emergency_approval": ("Director emergency approval",),
    "board_procurement_committee": ("Board Procurement Committee",),
    "aes_256_mandatory": ("AES-256",),
    "hsm_tokenized_encryption": ("hardware security module", "tokenized encryption"),
    "access_recertification_compliant": ("recertification review",),
    "automatic_account_suspension": ("automatic account suspension",),
    "sla_met": ("within 1 hour",),
    "sla_breach_escalation": ("breach escalation",),
}


# ---------------------------------------------------------------------------
# Deterministic PDF Canvas & Generation
# ---------------------------------------------------------------------------

class DeterministicNumberedCanvas(canvas.Canvas):
    """Two-pass canvas for deterministic PDF creation with exact page numbering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)
        self._doc.info.producer = "RuleLens Deterministic Generator"
        self._doc.info.creator = "RuleLens"
        self._doc.info.author = COMPANY_NAME
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def _draw_page_decorations(self, total_pages: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Top running header
        self.drawString(54, 750, f"{COMPANY_NAME} — Corporate Policy Documentation")
        self.drawRightString(612 - 54, 750, "CONFIDENTIAL & PROPRIETARY")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 744, 612 - 54, 744)

        # Bottom footer
        self.line(54, 48, 612 - 54, 48)
        self.drawString(54, 36, f"{COMPANY_NAME} • Internal Compliance Reference")
        self.drawRightString(612 - 54, 36, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def generate_pdf(doc: PolicyDocument, output_file: Path) -> None:
    """Build a deterministic PDF for a policy document."""
    pdf_doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=60,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor="#1A365D",
        spaceAfter=4,
    )
    meta_label_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor="#2D3748",
    )
    meta_val_style = ParagraphStyle(
        "DocMetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor="#4A5568",
        spaceAfter=3,
    )
    clause_title_style = ParagraphStyle(
        "ClauseTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor="#2B6CB0",
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    clause_body_style = ParagraphStyle(
        "ClauseBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor="#2D3748",
        spaceAfter=6,
    )

    story: List[Any] = []

    # Header block
    story.append(Paragraph(COMPANY_NAME.upper(), meta_label_style))
    story.append(Paragraph(doc.title, title_style))
    story.append(Spacer(1, 4))

    # Metadata block
    roles_str = ", ".join(doc.permitted_roles)
    meta_lines = (
        f"<b>Document ID:</b> {doc.doc_id} &nbsp;|&nbsp; <b>Version:</b> {doc.version} &nbsp;|&nbsp; <b>Department:</b> {doc.department}",
        f"<b>Effective Interval:</b> {doc.effective_from} to {doc.effective_to}",
        f"<b>Permitted Roles:</b> {roles_str}",
    )
    for line in meta_lines:
        story.append(Paragraph(line, meta_val_style))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=10))

    # Numbered clauses
    for clause in doc.clauses:
        story.append(
            Paragraph(f"[{clause.clause_id}] {clause.title}", clause_title_style)
        )
        story.append(Paragraph(clause.text, clause_body_style))

    pdf_doc.build(story, canvasmaker=DeterministicNumberedCanvas)


# ---------------------------------------------------------------------------
# Structured CSV Rules Generation
# ---------------------------------------------------------------------------

def generate_csv_rules(rules: Sequence[Rule], output_file: Path) -> None:
    """Write rules to CSV with deterministic line endings and strict field order."""
    with open(output_file, mode="w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for rule in rules:
            writer.writerow(asdict(rule))


# ---------------------------------------------------------------------------
# Roles Specification & Generation
# ---------------------------------------------------------------------------

def build_roles_data() -> Dict[str, Any]:
    """Generate the 6 demo roles and their explicit document permissions."""
    role_descriptions = {
        "employee": "Standard employee role with access to general workplace, travel, and conduct policies.",
        "manager": "Team manager role with operational approval and departmental transfer authority.",
        "hr": "People Operations specialist with access to employee welfare and personnel governance policies.",
        "finance": "Finance specialist with operational access to travel, expense, card, and procurement governance.",
        "security": "Information Security officer with access to data protection, access controls, and retention rules.",
        "auditor": "Compliance auditor with explicit read access to auditable financial, operational, and security policies.",
    }

    roles_dict: Dict[str, Any] = {}
    for role_name in ROLES:
        allowed_docs = [
            doc.doc_id
            for doc in DOCUMENTS
            if role_name in doc.permitted_roles
        ]
        roles_dict[role_name] = {
            "role": role_name,
            "description": role_descriptions[role_name],
            "document_count": len(allowed_docs),
            "allowed_documents": sorted(allowed_docs),
        }

    return {
        "company": COMPANY_NAME,
        "roles": roles_dict,
    }


# ---------------------------------------------------------------------------
# Manifest Generation
# ---------------------------------------------------------------------------

def build_manifest_data() -> Dict[str, Any]:
    """Generate manifest recording document inventory, versions, roles, and rules."""
    doc_entries = []
    for doc in DOCUMENTS:
        doc_rules = [r.rule_id for r in RULES if r.doc_id == doc.doc_id]
        doc_entries.append({
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "title": doc.title,
            "department": doc.department,
            "version": doc.version,
            "effective_from": doc.effective_from,
            "effective_to": doc.effective_to,
            "permitted_roles": list(doc.permitted_roles),
            "clause_ids": [c.clause_id for c in doc.clauses],
            "related_rule_ids": sorted(doc_rules),
        })

    return {
        "company": COMPANY_NAME,
        "document_count": len(DOCUMENTS),
        "rule_count": len(RULES),
        "roles": list(ROLES),
        "documents": doc_entries,
    }


# ---------------------------------------------------------------------------
# Dataset Generator & Validator
# ---------------------------------------------------------------------------

def generate_dataset(root_dir: Path | str | None = None) -> Dict[str, str]:
    """Deterministically generate all 20 PDFs, 5 CSV tables, roles.json, and manifest.json.

    Returns:
        Dictionary mapping relative file paths to their SHA-256 hex digests.
    """
    base = Path(root_dir) if root_dir else Path.cwd()
    data_dir = base / "data" / "source"
    policies_dir = data_dir / "policies"
    rules_dir = data_dir / "rules"

    policies_dir.mkdir(parents=True, exist_ok=True)
    rules_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 20 Policy PDFs
    for doc in DOCUMENTS:
        pdf_path = policies_dir / doc.filename
        generate_pdf(doc, pdf_path)

    # 2. Generate 5 CSV Rule Tables
    for dept, filename in DEPARTMENT_RULES_MAPPING.items():
        dept_doc_ids = {d.doc_id for d in DOCUMENTS if d.department == dept}
        dept_rules = [r for r in RULES if r.doc_id in dept_doc_ids]
        csv_path = rules_dir / filename
        generate_csv_rules(dept_rules, csv_path)

    # 3. Generate roles.json
    roles_path = data_dir / "roles.json"
    roles_data = build_roles_data()
    with open(roles_path, mode="w", encoding="utf-8", newline="\n") as f:
        json.dump(roles_data, f, indent=2, sort_keys=True)
        f.write("\n")

    # 4. Generate manifest.json
    manifest_path = data_dir / "manifest.json"
    manifest_data = build_manifest_data()
    with open(manifest_path, mode="w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)
        f.write("\n")

    # Calculate SHA-256 for all generated files
    hashes: Dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            rel_str = str(path.relative_to(base)).replace("\\", "/")
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                hasher.update(f.read())
            hashes[rel_str] = hasher.hexdigest()

    return hashes


def validate_dataset(root_dir: Path | str | None = None) -> bool:
    """Validate all generated files, structures, constraints, and relationships.

    Returns:
        True if all validations pass, raises ValueError otherwise.
    """
    base = Path(root_dir) if root_dir else Path.cwd()
    data_dir = base / "data" / "source"
    policies_dir = data_dir / "policies"
    rules_dir = data_dir / "rules"
    roles_path = data_dir / "roles.json"
    manifest_path = data_dir / "manifest.json"

    # Check existence of source components
    if not policies_dir.exists() or not rules_dir.exists():
        raise ValueError("Data directories missing.")
    if not roles_path.exists() or not manifest_path.exists():
        raise ValueError("roles.json or manifest.json missing.")

    # Helper: Validate ISO date string (YYYY-MM-DD)
    def _validate_iso_date(d_str: str, ctx: str) -> None:
        try:
            datetime.date.fromisoformat(d_str)
        except Exception as err:
            raise ValueError(f"Invalid ISO date '{d_str}' in {ctx}: {err}")

    # Check ISO dates and no duplicate clause IDs on canonical DOCUMENTS
    for doc in DOCUMENTS:
        _validate_iso_date(doc.effective_from, f"document {doc.doc_id} effective_from")
        _validate_iso_date(doc.effective_to, f"document {doc.doc_id} effective_to")
        if doc.effective_from > doc.effective_to:
            raise ValueError(f"Document {doc.doc_id} effective_from > effective_to")
        clause_ids = [c.clause_id for c in doc.clauses]
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError(f"Duplicate logical clause IDs found inside document {doc.doc_id}: {clause_ids}")

    # 1. Validate PDFs (count, filenames, non-empty, and text extraction via PyMuPDF)
    pdf_files = list(policies_dir.glob("*.pdf"))
    if len(pdf_files) != 20:
        raise ValueError(f"Expected exactly 20 PDFs, found {len(pdf_files)}")

    expected_filenames = {doc.filename for doc in DOCUMENTS}
    found_filenames = {p.name for p in pdf_files}
    if expected_filenames != found_filenames:
        raise ValueError(f"Filename mismatch: {expected_filenames ^ found_filenames}")

    doc_map = {d.doc_id: d for d in DOCUMENTS}

    for doc in DOCUMENTS:
        pdf_path = policies_dir / doc.filename
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF missing or empty: {doc.filename}")

        try:
            reader = pymupdf.open(pdf_path)
            if len(reader) < 1:
                raise ValueError(f"PDF has no pages: {doc.filename}")
            full_text = " ".join(" ".join(page.get_text().split()) for page in reader)
            reader.close()
        except Exception as e:
            raise ValueError(f"Failed to open or extract text from PDF {doc.filename}: {e}")

        # Confirm company name and document ID in extracted text
        if COMPANY_NAME not in full_text:
            raise ValueError(f"Company name '{COMPANY_NAME}' missing in PDF text: {doc.filename}")
        if doc.doc_id not in full_text:
            raise ValueError(f"Document ID '{doc.doc_id}' missing in PDF text: {doc.filename}")

        # Confirm each clause ID and full statement in extracted text
        for clause in doc.clauses:
            if clause.clause_id not in full_text:
                raise ValueError(f"Clause ID '{clause.clause_id}' missing in PDF text: {doc.filename}")
            norm_clause_text = " ".join(clause.text.split())
            if norm_clause_text not in full_text:
                raise ValueError(
                    f"Full statement for clause '{clause.clause_id}' missing in PDF text: {doc.filename}"
                )

    # 2. Validate CSV Tables (count, headers, exact match with canonical RULES)
    csv_files = list(rules_dir.glob("*.csv"))
    expected_csvs = set(DEPARTMENT_RULES_MAPPING.values())
    found_csvs = {f.name for f in csv_files}
    if found_csvs != expected_csvs:
        raise ValueError(f"CSV files mismatch: expected {expected_csvs}, got {found_csvs}")

    all_csv_rules_by_id: Dict[str, Dict[str, str]] = {}
    for dept, filename in DEPARTMENT_RULES_MAPPING.items():
        csv_path = rules_dir / filename
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(CSV_FIELDNAMES):
                raise ValueError(f"CSV header mismatch in {filename}: {reader.fieldnames}")
            rows = list(reader)
            if len(rows) < 4:
                raise ValueError(f"Table {filename} has fewer than 4 rules ({len(rows)})")
            for row in rows:
                rid = row["rule_id"]
                if rid in all_csv_rules_by_id:
                    raise ValueError(f"Duplicate rule_id across CSV tables: {rid}")
                all_csv_rules_by_id[rid] = row

    if len(all_csv_rules_by_id) != len(RULES):
        raise ValueError(f"Rule count mismatch: expected {len(RULES)}, got {len(all_csv_rules_by_id)}")

    canonical_rules_by_id = {r.rule_id: r for r in RULES}
    for rid, canon_rule in canonical_rules_by_id.items():
        if rid not in all_csv_rules_by_id:
            raise ValueError(f"Canonical rule {rid} missing from CSV tables")
        csv_row = all_csv_rules_by_id[rid]

        # Verify exact field values against canonical rule
        for field in CSV_FIELDNAMES:
            expected_val = getattr(canon_rule, field)
            actual_val = csv_row[field]
            if actual_val != expected_val:
                raise ValueError(
                    f"Rule {rid} field '{field}' mismatch: expected '{expected_val}', got '{actual_val}'"
                )

        _validate_iso_date(csv_row["effective_from"], f"rule {rid} effective_from")
        _validate_iso_date(csv_row["effective_to"], f"rule {rid} effective_to")
        if csv_row["effective_from"] > csv_row["effective_to"]:
            raise ValueError(f"Rule {rid} effective_from > effective_to")

    # 3. Validate Rule Textual Support in Clauses (Numerical and Categorical)
    for rule in RULES:
        doc = doc_map[rule.doc_id]
        matching_clause = next((c for c in doc.clauses if c.clause_id == rule.clause_id), None)
        if matching_clause is None:
            raise ValueError(f"Rule {rule.rule_id} references nonexistent clause {rule.clause_id} in {doc.doc_id}")

        clause_text = matching_clause.text
        val = rule.threshold_value

        # Check numerical or categorical threshold representation in clause
        try:
            int_val = int(val)
            comma_val = f"{int_val:,}"
            has_val = (val in clause_text) or (comma_val in clause_text)
        except ValueError:
            has_val = val in clause_text

        if not has_val:
            raise ValueError(
                f"Rule {rule.rule_id} threshold value '{val}' not found in clause {rule.clause_id} text"
            )

        # Check outcome support in clause text
        expected_phrases = OUTCOME_TEXT_SUPPORT.get(rule.outcome)
        if expected_phrases:
            if not any(phrase.lower() in clause_text.lower() for phrase in expected_phrases):
                raise ValueError(
                    f"Rule {rule.rule_id} outcome '{rule.outcome}' not supported in clause {rule.clause_id} text"
                )

    # 4. Check for Conflicting / Overlapping Conditions in Same Document, Field, and Effective Period
    for doc in DOCUMENTS:
        doc_rules = [r for r in RULES if r.doc_id == doc.doc_id]
        by_field: Dict[str, List[Rule]] = {}
        for r in doc_rules:
            by_field.setdefault(r.condition_field, []).append(r)

        for field, field_rules in by_field.items():
            for i in range(len(field_rules)):
                for j in range(i + 1, len(field_rules)):
                    r1 = field_rules[i]
                    r2 = field_rules[j]
                    # Check active date interval overlap
                    overlap = max(r1.effective_from, r2.effective_from) <= min(r1.effective_to, r2.effective_to)
                    if overlap:
                        if r1.operator == r2.operator and r1.threshold_value == r2.threshold_value:
                            raise ValueError(
                                f"Conflicting identical conditions in {doc.doc_id} for field '{field}': "
                                f"{r1.rule_id} and {r2.rule_id}"
                            )

    # 5. Validate Roles (exact match against canonical permissions and document_count)
    with open(roles_path, mode="r", encoding="utf-8") as f:
        roles_json = json.load(f)

    if set(roles_json.get("roles", {}).keys()) != set(ROLES):
        raise ValueError(
            f"Roles mismatch in roles.json: expected {set(ROLES)}, got {set(roles_json.get('roles', {}).keys())}"
        )

    for role in ROLES:
        role_data = roles_json["roles"][role]
        canonical_allowed = sorted([d.doc_id for d in DOCUMENTS if role in d.permitted_roles])
        actual_allowed = sorted(role_data.get("allowed_documents", []))
        if actual_allowed != canonical_allowed:
            raise ValueError(
                f"Role '{role}' allowed_documents mismatch: expected {canonical_allowed}, got {actual_allowed}"
            )
        doc_count = role_data.get("document_count")
        if doc_count != len(canonical_allowed):
            raise ValueError(
                f"Role '{role}' document_count ({doc_count}) != actual allowed count ({len(canonical_allowed)})"
            )

    # 6. Validate Manifest Metadata against Canonical Specification
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        manifest_json = json.load(f)

    if manifest_json.get("company") != COMPANY_NAME:
        raise ValueError("Manifest company mismatch.")
    if manifest_json.get("document_count") != len(DOCUMENTS):
        raise ValueError(f"Manifest document_count mismatch: expected {len(DOCUMENTS)}, got {manifest_json.get('document_count')}")
    if manifest_json.get("rule_count") != len(RULES):
        raise ValueError(f"Manifest rule_count mismatch: expected {len(RULES)}, got {manifest_json.get('rule_count')}")
    if manifest_json.get("roles") != list(ROLES):
        raise ValueError(f"Manifest roles mismatch: expected {list(ROLES)}, got {manifest_json.get('roles')}")

    manifest_docs = {d["doc_id"]: d for d in manifest_json.get("documents", [])}
    if len(manifest_docs) != len(DOCUMENTS):
        raise ValueError("Manifest documents count mismatch.")

    for doc in DOCUMENTS:
        if doc.doc_id not in manifest_docs:
            raise ValueError(f"Document {doc.doc_id} missing from manifest")
        m_doc = manifest_docs[doc.doc_id]
        if m_doc.get("filename") != doc.filename:
            raise ValueError(f"Manifest filename mismatch for {doc.doc_id}")
        if m_doc.get("title") != doc.title:
            raise ValueError(f"Manifest title mismatch for {doc.doc_id}")
        if m_doc.get("department") != doc.department:
            raise ValueError(f"Manifest department mismatch for {doc.doc_id}")
        if m_doc.get("version") != doc.version:
            raise ValueError(f"Manifest version mismatch for {doc.doc_id}")
        if m_doc.get("effective_from") != doc.effective_from:
            raise ValueError(f"Manifest effective_from mismatch for {doc.doc_id}")
        if m_doc.get("effective_to") != doc.effective_to:
            raise ValueError(f"Manifest effective_to mismatch for {doc.doc_id}")
        if sorted(m_doc.get("permitted_roles", [])) != sorted(doc.permitted_roles):
            raise ValueError(f"Manifest permitted_roles mismatch for {doc.doc_id}")
        if m_doc.get("clause_ids") != [c.clause_id for c in doc.clauses]:
            raise ValueError(f"Manifest clause_ids mismatch for {doc.doc_id}")
        expected_rule_ids = sorted([r.rule_id for r in RULES if r.doc_id == doc.doc_id])
        if sorted(m_doc.get("related_rule_ids", [])) != expected_rule_ids:
            raise ValueError(f"Manifest related_rule_ids mismatch for {doc.doc_id}")

    # 7. Travel v1 vs v2 Checks
    v1 = doc_map["DOC-TRV-001"]
    v2 = doc_map["DOC-TRV-002"]
    if v1.effective_to >= v2.effective_from:
        raise ValueError("Travel Policy v1 and v2 active dates overlap.")

    v1_clauses = {c.clause_id for c in v1.clauses}
    v2_clauses = {c.clause_id for c in v2.clauses}
    if v1_clauses != v2_clauses:
        raise ValueError("Travel Policy v1 and v2 must retain identical logical clause IDs.")

    v1_rules = {r["rule_id"]: r for r in all_csv_rules_by_id.values() if r["doc_id"] == "DOC-TRV-001" and r["clause_id"] == "TRV-C04"}
    v2_rules = {r["rule_id"]: r for r in all_csv_rules_by_id.values() if r["doc_id"] == "DOC-TRV-002" and r["clause_id"] == "TRV-C04"}

    v1_std = [r for r in v1_rules.values() if r["outcome"] == "standard_approval"]
    v2_std = [r for r in v2_rules.values() if r["outcome"] == "standard_approval"]

    if not v1_std or v1_std[0]["threshold_value"] != "40000":
        raise ValueError("Travel Policy v1 standard approval threshold must be 40000.")
    if not v2_std or v2_std[0]["threshold_value"] != "50000":
        raise ValueError("Travel Policy v2 standard approval threshold must be 50000.")

    return True


# ---------------------------------------------------------------------------
# CLI Command Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI interface for dataset generation and validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rulelens.dataset [generate|validate]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "generate":
        hashes = generate_dataset()
        print(f"Dataset generated successfully ({len(hashes)} files created/verified).")
    elif cmd == "validate":
        try:
            validate_dataset()
            print("Dataset validation passed successfully.")
        except Exception as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}. Use 'generate' or 'validate'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
