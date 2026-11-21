"""Offline unit and integration tests for RuleLens enterprise policy dataset.

Validates the committed dataset, boundary conditions, determinism,
and negative test cases against corruptions.
"""

import csv
import datetime
import json
import shutil
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
    """Return repository root without regenerating the committed dataset."""
    return Path(__file__).resolve().parent.parent


def copy_source_dataset(dataset_root: Path, target_dir: Path) -> Path:
    """Helper to copy the committed source dataset into a temporary directory."""
    src_data = dataset_root / "data" / "source"
    dest_data = target_dir / "data" / "source"
    shutil.copytree(src_data, dest_data)
    return target_dir


# ---------------------------------------------------------------------------
# Committed Dataset Integrity Tests
# ---------------------------------------------------------------------------

def test_committed_dataset_passes_validation(dataset_root):
    """Verify the existing committed dataset passes full strengthened validation."""
    assert validate_dataset(dataset_root) is True


def test_committed_dataset_counts(dataset_root):
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


def test_six_roles_and_counts(dataset_root):
    """Verify exactly six valid roles, explicit permissions, and exact document counts in roles.json."""
    roles_file = dataset_root / "data" / "source" / "roles.json"
    with open(roles_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    roles = data["roles"]
    assert set(roles.keys()) == set(ROLES)
    assert len(roles) == 6

    # Canonical document counts per role
    expected_counts = {
        "employee": 12,
        "manager": 18,
        "hr": 4,
        "finance": 12,
        "security": 4,
        "auditor": 19,
    }

    for role_name, expected_count in expected_counts.items():
        role_entry = roles[role_name]
        actual_docs = role_entry["allowed_documents"]
        stated_count = role_entry["document_count"]

        assert len(actual_docs) == expected_count, (
            f"Role {role_name} count mismatch: expected {expected_count}, got {len(actual_docs)}"
        )
        assert stated_count == expected_count, (
            f"Role {role_name} stated document_count mismatch: {stated_count} != {expected_count}"
        )

    # Permission restrictions check
    emp_docs = set(roles["employee"]["allowed_documents"])
    assert "DOC-TRV-001" in emp_docs
    assert "DOC-TRV-002" in emp_docs
    assert "DOC-TRV-004" not in emp_docs  # Travel Exceptions
    assert "DOC-FIN-003" not in emp_docs  # Corporate Cards
    assert "DOC-SEC-004" not in emp_docs  # Data Retention

    auditor_docs = set(roles["auditor"]["allowed_documents"])
    assert "DOC-PPO-004" not in auditor_docs  # Employee Transfers (confidential HR deliberations)


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


def test_valid_iso_dates_and_intervals():
    """Verify all document and rule date strings are valid ISO dates and form valid intervals."""
    for doc in DOCUMENTS:
        d1 = datetime.date.fromisoformat(doc.effective_from)
        d2 = datetime.date.fromisoformat(doc.effective_to)
        assert d1 <= d2

    for rule in RULES:
        r1 = datetime.date.fromisoformat(rule.effective_from)
        r2 = datetime.date.fromisoformat(rule.effective_to)
        assert r1 <= r2


def test_no_overlapping_travel_policy_versions():
    """Verify Travel Policy v1 and v2 have mutually exclusive active date intervals."""
    v1 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-001")
    v2 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-002")

    assert v1.effective_from == "2026-01-01"
    assert v1.effective_to == "2026-06-30"
    assert v2.effective_from == "2026-07-01"
    assert v1.effective_to < v2.effective_from


def test_travel_policy_boundary_conditions():
    """Test boundary amount conditions and historical vs current date evaluations."""
    v1 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-001")
    v2 = next(d for d in DOCUMENTS if d.doc_id == "DOC-TRV-002")

    v1_from = datetime.date.fromisoformat(v1.effective_from)
    v1_to = datetime.date.fromisoformat(v1.effective_to)
    v2_from = datetime.date.fromisoformat(v2.effective_from)

    # Date boundary checks
    date_march = datetime.date(2026, 3, 15)
    date_v1_end = datetime.date(2026, 6, 30)
    date_v2_start = datetime.date(2026, 7, 1)

    assert v1_from <= date_march <= v1_to  # Historical query resolves to v1
    assert v1_from <= date_v1_end <= v1_to  # v1 inclusive end
    assert not (v1_from <= date_v2_start <= v1_to)  # v1 expired on July 1
    assert date_v2_start >= v2_from  # Current query resolves to v2

    # Amount threshold boundary evaluations
    v1_threshold = 40000
    v2_threshold = 50000

    def evaluate_v1(amount: int) -> str:
        return "standard_approval" if amount <= v1_threshold else "additional_approval"

    def evaluate_v2(amount: int) -> str:
        return "standard_approval" if amount <= v2_threshold else "additional_approval"

    # v1 boundary amounts
    assert evaluate_v1(39999) == "standard_approval"
    assert evaluate_v1(40000) == "standard_approval"  # exact boundary
    assert evaluate_v1(40001) == "additional_approval"

    # v2 boundary amounts
    assert evaluate_v2(49999) == "standard_approval"
    assert evaluate_v2(50000) == "standard_approval"  # exact boundary
    assert evaluate_v2(50001) == "additional_approval"

    # Policy update invalidation: INR 45,000 required additional approval in v1, but standard in v2
    amount_diff = 45000
    assert evaluate_v1(amount_diff) == "additional_approval"
    assert evaluate_v2(amount_diff) == "standard_approval"


def test_pdf_text_extraction_all_documents(dataset_root):
    """Verify PyMuPDF can extract text from all 20 committed PDFs and confirm clause statements."""
    policies_dir = dataset_root / "data" / "source" / "policies"

    for doc in DOCUMENTS:
        pdf_path = policies_dir / doc.filename
        reader = pymupdf.open(pdf_path)
        assert len(reader) >= 1

        all_text = " ".join(" ".join(page.get_text().split()) for page in reader)
        reader.close()

        assert "Asterline Services Group" in all_text
        assert doc.doc_id in all_text

        for clause in doc.clauses:
            assert clause.clause_id in all_text
            norm_clause_text = " ".join(clause.text.split())
            assert norm_clause_text in all_text


# ---------------------------------------------------------------------------
# Fresh Generation & Determinism Tests (in temporary directories only)
# ---------------------------------------------------------------------------

def test_deterministic_generation_file_hashes():
    """Verify running generation twice in temporary directories yields identical SHA-256 hashes."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        hashes1 = generate_dataset(Path(tmp1))
        hashes2 = generate_dataset(Path(tmp2))

        assert len(hashes1) == len(hashes2) == 27

        for rel_path, h1 in hashes1.items():
            assert rel_path in hashes2
            h2 = hashes2[rel_path]
            assert h1 == h2, f"Hash divergence detected for {rel_path}: {h1} != {h2}"


def test_dataset_generation_offline(monkeypatch):
    """Verify dataset generation and validation run completely offline in a temporary directory."""
    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline dataset operation")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    with tempfile.TemporaryDirectory() as tmp:
        hashes = generate_dataset(Path(tmp))
        assert len(hashes) == 27
        is_valid = validate_dataset(Path(tmp))
        assert is_valid is True


# ---------------------------------------------------------------------------
# Negative Tests: Deliberate Corruptions Must Fail Validation
# ---------------------------------------------------------------------------

def test_negative_corrupt_csv_rule(dataset_root, tmp_path):
    """Confirm validation fails when a CSV rule field or threshold is corrupted."""
    copy_source_dataset(dataset_root, tmp_path)
    csv_file = tmp_path / "data" / "source" / "rules" / "travel_rules.csv"

    # Read and corrupt threshold_value of R-TRV-001
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows[0]["threshold_value"] = "99999"  # Corrupted threshold

    with open(csv_file, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="mismatch"):
        validate_dataset(tmp_path)


def test_negative_corrupt_role_permission(dataset_root, tmp_path):
    """Confirm validation fails when an unauthorized document is added to a role in roles.json."""
    copy_source_dataset(dataset_root, tmp_path)
    roles_file = tmp_path / "data" / "source" / "roles.json"

    with open(roles_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Corrupt employee role by adding unauthorized DOC-SEC-004
    data["roles"]["employee"]["allowed_documents"].append("DOC-SEC-004")
    data["roles"]["employee"]["allowed_documents"].sort()

    with open(roles_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ValueError, match="allowed_documents mismatch|document_count"):
        validate_dataset(tmp_path)


def test_negative_corrupt_role_document_count(dataset_root, tmp_path):
    """Confirm validation fails when document_count does not equal allowed_documents length."""
    copy_source_dataset(dataset_root, tmp_path)
    roles_file = tmp_path / "data" / "source" / "roles.json"

    with open(roles_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["roles"]["employee"]["document_count"] = 99  # Corrupted count

    with open(roles_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ValueError, match="document_count"):
        validate_dataset(tmp_path)


def test_negative_corrupt_manifest_entry(dataset_root, tmp_path):
    """Confirm validation fails when manifest metadata is altered."""
    copy_source_dataset(dataset_root, tmp_path)
    manifest_file = tmp_path / "data" / "source" / "manifest.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["documents"][0]["title"] = "Corrupted Title"

    with open(manifest_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ValueError, match="Manifest title mismatch"):
        validate_dataset(tmp_path)


def test_negative_corrupt_pdf_file(dataset_root, tmp_path):
    """Confirm validation fails when a PDF file is truncated or corrupted."""
    copy_source_dataset(dataset_root, tmp_path)
    pdf_file = tmp_path / "data" / "source" / "policies" / "travel_policy_v1.pdf"

    # Truncate PDF to invalid bytes
    with open(pdf_file, "wb") as f:
        f.write(b"NOT A VALID PDF FILE")

    with pytest.raises(ValueError, match="Failed to open or extract text from PDF"):
        validate_dataset(tmp_path)


def test_negative_corrupt_iso_date(dataset_root, tmp_path):
    """Confirm validation fails when a non-ISO date string is introduced."""
    copy_source_dataset(dataset_root, tmp_path)
    csv_file = tmp_path / "data" / "source" / "rules" / "travel_rules.csv"

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows[0]["effective_from"] = "2026/01/01"  # Invalid non-ISO format

    with open(csv_file, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="mismatch|Invalid ISO date"):
        validate_dataset(tmp_path)
