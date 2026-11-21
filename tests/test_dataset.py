"""Offline unit and integration tests for RuleLens enterprise policy dataset."""

import hashlib
import json
import socket
import tempfile
from pathlib import Path

import pymupdf
import pytest

from rulelens.dataset import (
    DOCUMENTS,
    ROLES,
    RULES,
    generate_dataset,
    validate_dataset,
)


@pytest.fixture(scope="module")
def dataset_root():
    """Ensure dataset is generated and return repository root."""
    root = Path(__file__).resolve().parent.parent
    # Ensure dataset files exist
    generate_dataset(root)
    return root


def test_dataset_counts(dataset_root):
    """Verify exactly 20 PDFs, 5 CSV tables, roles.json, and manifest.json exist."""
    policies_dir = dataset_root / "data" / "source" / "policies"
    rules_dir = dataset_root / "data" / "source" / "rules"
    roles_file = dataset_root / "data" / "source" / "roles.json"
    manifest_file = dataset_root / "data" / "source" / "manifest.json"

    assert policies_dir.exists()
    assert rules_dir.exists()
    assert roles_file.exists()
    assert manifest_file.exists()

    pdfs = list(policies_dir.glob("*.pdf"))
    csvs = list(rules_dir.glob("*.csv"))

    assert len(pdfs) == 20
    assert len(csvs) == 5


def test_six_roles_and_permissions(dataset_root):
    """Verify exactly six valid roles and explicit access restrictions."""
    roles_file = dataset_root / "data" / "source" / "roles.json"
    with open(roles_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    roles = data["roles"]
    assert set(roles.keys()) == set(ROLES)
    assert len(roles) == 6

    # Employee has access to general policies but not restricted ones
    emp_docs = set(roles["employee"]["allowed_documents"])
    assert "DOC-TRV-001" in emp_docs
    assert "DOC-TRV-002" in emp_docs
    assert "DOC-PPO-001" in emp_docs
    assert "DOC-TRV-004" not in emp_docs  # Travel Exceptions is manager/finance/auditor
    assert "DOC-FIN-003" not in emp_docs  # Corporate Cards is manager/finance/auditor
    assert "DOC-SEC-004" not in emp_docs  # Data Retention is security/auditor

    # Auditor access is explicit and restricted (not blanket unrestricted)
    auditor_docs = set(roles["auditor"]["allowed_documents"])
    assert "DOC-PPO-004" not in auditor_docs  # Internal employee transfers


def test_unique_ids_and_stable_clauses():
    """Verify unique document, rule, and clause IDs, preserving stable clause IDs across versions."""
    doc_ids = [d.doc_id for d in DOCUMENTS]
    assert len(doc_ids) == len(set(doc_ids)) == 20

    rule_ids = [r.rule_id for r in RULES]
    assert len(rule_ids) == len(set(rule_ids)) == len(RULES)

    # Stable clause IDs across Travel Policy v1 and v2
    v1 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-001")
    v2 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-002")

    v1_clause_ids = [c.clause_id for c in v1.clauses]
    v2_clause_ids = [c.clause_id for c in v2.clauses]
    assert v1_clause_ids == v2_clause_ids
    assert "TRV-C04" in v1_clause_ids


def test_valid_document_to_rule_references():
    """Verify every rule references an existing document and clause with valid dates."""
    doc_map = {d.doc_id: d for d in DOCUMENTS}

    for rule in RULES:
        assert rule.doc_id in doc_map
        doc = doc_map[rule.doc_id]
        doc_clause_ids = {c.clause_id for c in doc.clauses}
        assert rule.clause_id in doc_clause_ids

        # Date interval checks
        assert rule.effective_from <= rule.effective_to
        assert rule.effective_from >= doc.effective_from
        assert rule.effective_to <= doc.effective_to


def test_no_overlapping_travel_policy_versions():
    """Verify Travel Policy v1 and v2 have mutually exclusive active date intervals."""
    v1 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-001")
    v2 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-002")

    assert v1.effective_from == "2026-01-01"
    assert v1.effective_to == "2026-06-30"
    assert v2.effective_from == "2026-07-01"
    assert v1.effective_to < v2.effective_from


def test_travel_policy_approval_thresholds_and_outcomes():
    """Verify Travel Policy v1 and v2 standard vs additional approval thresholds."""
    v1_rules = [r for r in RULES if r.doc_id == "DOC-TRV-001" and r.clause_id == "TRV-C04"]
    v2_rules = [r for r in RULES if r.doc_id == "DOC-TRV-002" and r.clause_id == "TRV-C04"]

    assert len(v1_rules) == 2
    assert len(v2_rules) == 2

    # v1: <= 40000 is standard, > 40000 is additional
    v1_std = next(r for r in v1_rules if r.operator == "<=")
    v1_add = next(r for r in v1_rules if r.operator == ">")
    assert v1_std.threshold_value == "40000"
    assert v1_std.outcome == "standard_approval"
    assert v1_add.threshold_value == "40000"
    assert v1_add.outcome == "additional_approval"

    # v2: <= 50000 is standard, > 50000 is additional
    v2_std = next(r for r in v2_rules if r.operator == "<=")
    v2_add = next(r for r in v2_rules if r.operator == ">")
    assert v2_std.threshold_value == "50000"
    assert v2_std.outcome == "standard_approval"
    assert v2_add.threshold_value == "50000"
    assert v2_add.outcome == "additional_approval"


def test_no_conflicting_active_rules():
    """Verify no two active rules for the same document and condition field conflict."""
    for doc in DOCUMENTS:
        doc_rules = [r for r in RULES if r.doc_id == doc.doc_id]
        by_field = {}
        for r in doc_rules:
            by_field.setdefault(r.condition_field, []).append(r)

        for field, rules in by_field.items():
            # For each field, combinations of (operator, threshold_value) must be unique
            condition_signatures = [(r.operator, r.threshold_value) for r in rules]
            assert len(condition_signatures) == len(set(condition_signatures)), (
                f"Conflicting identical conditions in {doc.doc_id} for field {field}"
            )
            # Outcomes must be defined and meaningful
            for r in rules:
                assert r.outcome


def test_pdf_text_extraction_sample_10_documents(dataset_root):
    """Use PyMuPDF to extract text from a sample of 10 documents and confirm key clauses appear."""
    policies_dir = dataset_root / "data" / "source" / "policies"
    sample_docs = DOCUMENTS[:10]

    for doc in sample_docs:
        pdf_path = policies_dir / doc.filename
        assert pdf_path.exists()

        pdf_reader = pymupdf.open(pdf_path)
        assert len(pdf_reader) >= 1

        all_text = " ".join(page.get_text() for page in pdf_reader)
        pdf_reader.close()

        # Check company name and document ID
        assert "Asterline Services Group" in all_text
        assert doc.doc_id in all_text

        # Check every clause ID appears in extracted text
        for clause in doc.clauses:
            assert clause.clause_id in all_text, f"Clause {clause.clause_id} missing in {doc.filename}"

    # Specific checks on Travel Policy v1 vs v2 text
    v1_text = " ".join(page.get_text() for page in pymupdf.open(policies_dir / "travel_policy_v1.pdf"))
    v2_text = " ".join(page.get_text() for page in pymupdf.open(policies_dir / "travel_policy_v2.pdf"))

    assert "40,000" in v1_text
    assert "50,000" in v2_text


def test_deterministic_generation_file_hashes():
    """Verify running generation twice yields identical byte-level SHA-256 hashes."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        hashes1 = generate_dataset(Path(tmp1))
        hashes2 = generate_dataset(Path(tmp2))

        assert len(hashes1) == len(hashes2)
        assert len(hashes1) == 27  # 20 PDFs + 5 CSVs + roles.json + manifest.json

        for rel_path, h1 in hashes1.items():
            assert rel_path in hashes2
            h2 = hashes2[rel_path]
            assert h1 == h2, f"Hash divergence detected for {rel_path}: {h1} != {h2}"


def test_dataset_generation_offline(monkeypatch):
    """Verify dataset generation and validation run completely offline without external network access."""
    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline dataset operation")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    with tempfile.TemporaryDirectory() as tmp:
        hashes = generate_dataset(Path(tmp))
        assert len(hashes) == 27
        is_valid = validate_dataset(Path(tmp))
        assert is_valid is True
