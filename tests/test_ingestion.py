"""Offline unit and integration tests for RuleLens ingestion pipeline."""

import hashlib
import json
import shutil
import socket
import tempfile
from pathlib import Path

import pymupdf
import pytest

from rulelens.ingestion import (
    build_corpus,
    validate_corpus,
    calculate_source_fingerprint,
    calculate_processed_fingerprint,
)


@pytest.fixture(scope="module")
def repo_root():
    """Return repository root without automatically rebuilding processed files."""
    return Path(__file__).resolve().parent.parent


def copy_source_and_processed(repo_root: Path, target_dir: Path) -> Path:
    """Helper to copy committed source and processed files into a temporary directory."""
    src_data = repo_root / "data" / "source"
    dest_data = target_dir / "data" / "source"
    shutil.copytree(src_data, dest_data)

    proc_data = repo_root / "data" / "processed"
    dest_proc = target_dir / "data" / "processed"
    shutil.copytree(proc_data, dest_proc)
    return target_dir


# ---------------------------------------------------------------------------
# Committed Processed Data Tests
# ---------------------------------------------------------------------------

def test_committed_processed_corpus_validation(repo_root):
    """Verify that existing committed processed corpus passes validation."""
    assert validate_corpus(repo_root) is True


def test_processed_documents_and_chunk_count(repo_root):
    """Verify exactly 64 chunks across exactly 20 distinct documents."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    assert chunks_path.exists()

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    assert len(chunks) == 64
    doc_ids = {c["doc_id"] for c in chunks}
    assert len(doc_ids) == 20


def test_processed_rules_count_and_types(repo_root):
    """Verify exactly 32 rules with proper numerical and categorical threshold types."""
    rules_path = repo_root / "data" / "processed" / "rules.jsonl"
    assert rules_path.exists()

    rules = []
    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))

    assert len(rules) == 32

    for rule in rules:
        val = rule["threshold_value"]
        if rule["rule_id"] in ("R-SEC-001", "R-SEC-002"):
            # Categorical rules
            assert isinstance(val, str)
            assert val in ("Confidential", "Restricted")
        else:
            # Numerical rules
            assert isinstance(val, (int, float))
            assert val > 0


def test_six_roles_represented_and_permissions(repo_root):
    """Verify all 6 roles are represented and restricted roles are not leaked."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    all_roles = set()

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            all_roles.update(chunk["allowed_roles"])

            # Permission boundary tests
            if chunk["doc_id"] in ("DOC-TRV-004", "DOC-FIN-003", "DOC-SEC-004"):
                assert "employee" not in chunk["allowed_roles"], (
                    f"Unauthorized role 'employee' found in restricted chunk {chunk['chunk_id']}"
                )

            if chunk["doc_id"] == "DOC-PPO-004":
                assert "auditor" not in chunk["allowed_roles"], (
                    f"Unauthorized role 'auditor' found in confidential HR chunk {chunk['chunk_id']}"
                )

    assert all_roles == {"employee", "manager", "hr", "finance", "security", "auditor"}


def test_unique_chunk_ids(repo_root):
    """Verify that all 64 chunk IDs are unique."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    chunk_ids = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk_ids.append(json.loads(line)["chunk_id"])

    assert len(chunk_ids) == 64
    assert len(set(chunk_ids)) == 64


def test_stable_travel_v1_v2_clause_identities(repo_root):
    """Verify Travel Policy v1 and v2 chunks maintain distinct document and version identities."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    v1_chunks = []
    v2_chunks = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["doc_id"] == "DOC-TRV-001":
                v1_chunks.append(c)
            elif c["doc_id"] == "DOC-TRV-002":
                v2_chunks.append(c)

    assert len(v1_chunks) == 5
    assert len(v2_chunks) == 5

    # Clause IDs are logically identical
    assert [c["clause_id"] for c in v1_chunks] == [c["clause_id"] for c in v2_chunks]

    # Verify versions and thresholds differ
    v1_c04 = next(c for c in v1_chunks if c["clause_id"] == "TRV-C04")
    v2_c04 = next(c for c in v2_chunks if c["clause_id"] == "TRV-C04")

    assert v1_c04["version"] == "1.0"
    assert v2_c04["version"] == "2.0"
    assert "40,000" in v1_c04["text"]
    assert "50,000" in v2_c04["text"]


def test_accurate_original_pdf_page_references_sample_10(repo_root):
    """Directly verify at least 10 chunks against actual PDF pages."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    policies_dir = repo_root / "data" / "source" / "policies"

    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    # Sample 10 chunks across different documents
    sample_chunks = all_chunks[::6][:10]
    assert len(sample_chunks) >= 10

    for chunk in sample_chunks:
        pdf_file = policies_dir / chunk["source_file"]
        assert pdf_file.exists()

        pdf_doc = pymupdf.open(pdf_file)
        page_idx = chunk["source_page"] - 1  # Convert 1-based to 0-based
        assert 0 <= page_idx < len(pdf_doc)

        page_text = " ".join(pdf_doc[page_idx].get_text().split())
        pdf_doc.close()

        # Confirm clause ID and key phrase from text is on that page
        assert chunk["clause_id"] in page_text
        assert "Asterline Services Group" in page_text


def test_text_and_content_hash_consistency(repo_root):
    """Verify all 64 chunks have valid SHA-256 content hashes matching their text."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            expected_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            assert chunk["content_hash"] == expected_hash


def test_rule_to_chunk_linking(repo_root):
    """Verify all 32 rules link to valid chunks with compatible date intervals."""
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    rules_path = repo_root / "data" / "processed" / "rules.jsonl"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_by_id = {c["chunk_id"]: c for c in (json.loads(l) for l in f if l.strip())}

    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rule = json.loads(line)
            chunk_key = f"{rule['doc_id']}_{rule['clause_id']}"
            assert chunk_key in chunks_by_id

            chunk = chunks_by_id[chunk_key]
            assert rule["effective_from"] >= chunk["effective_from"]
            assert rule["effective_to"] <= chunk["effective_to"]


# ---------------------------------------------------------------------------
# Fresh Build & Determinism Tests
# ---------------------------------------------------------------------------

def test_repeated_builds_produce_identical_hashes(repo_root):
    """Verify independent builds in two temporary directories yield identical file hashes."""
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        dir1 = Path(t1)
        dir2 = Path(t2)

        # Copy source dataset
        shutil.copytree(repo_root / "data" / "source", dir1 / "data" / "source")
        shutil.copytree(repo_root / "data" / "source", dir2 / "data" / "source")

        m1 = build_corpus(dir1)
        m2 = build_corpus(dir2)

        assert m1["source_fingerprint"] == m2["source_fingerprint"]
        assert m1["processed_fingerprint"] == m2["processed_fingerprint"]

        # Check raw byte SHA-256 for all 3 processed files
        for filename in ("chunks.jsonl", "rules.jsonl", "corpus_manifest.json"):
            h1 = hashlib.sha256((dir1 / "data" / "processed" / filename).read_bytes()).hexdigest()
            h2 = hashlib.sha256((dir2 / "data" / "processed" / filename).read_bytes()).hexdigest()
            assert h1 == h2, f"Hash divergence for {filename}"


def test_ingestion_offline(repo_root, monkeypatch, tmp_path):
    """Verify build and validation work completely offline with blocked network sockets."""
    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during offline ingestion operation")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    shutil.copytree(repo_root / "data" / "source", tmp_path / "data" / "source")

    manifest = build_corpus(tmp_path)
    assert manifest["chunk_count"] == 64
    assert validate_corpus(tmp_path) is True


# ---------------------------------------------------------------------------
# Negative Tests: Corruptions Must Fail Validation
# ---------------------------------------------------------------------------

def test_negative_corrupt_chunk_text(repo_root, tmp_path):
    """Validation must fail when chunk text is modified without updating content_hash."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"

    lines = chunks_file.read_text(encoding="utf-8").splitlines()
    chunk0 = json.loads(lines[0])
    chunk0["text"] = chunk0["text"] + " TAMPERED"
    lines[0] = json.dumps(chunk0, sort_keys=True)
    chunks_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Content hash mismatch"):
        validate_corpus(tmp_path)


def test_negative_corrupt_rule_link(repo_root, tmp_path):
    """Validation must fail when a rule links to a nonexistent chunk."""
    copy_source_and_processed(repo_root, tmp_path)
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    lines = rules_file.read_text(encoding="utf-8").splitlines()
    rule0 = json.loads(lines[0])
    rule0["clause_id"] = "NONEXISTENT-CLAUSE"
    lines[0] = json.dumps(rule0, sort_keys=True)
    rules_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nonexistent chunk"):
        validate_corpus(tmp_path)


def test_negative_corrupt_manifest_fingerprint(repo_root, tmp_path):
    """Validation must fail when corpus manifest processed fingerprint is altered."""
    copy_source_and_processed(repo_root, tmp_path)
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    data["processed_fingerprint"] = "0" * 64
    manifest_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Processed corpus fingerprint mismatch"):
        validate_corpus(tmp_path)


def test_negative_corrupt_source_fingerprint(repo_root, tmp_path):
    """Validation must fail when source dataset files are out of sync with stored fingerprint."""
    copy_source_and_processed(repo_root, tmp_path)
    # Tamper with a source CSV file
    csv_file = tmp_path / "data" / "source" / "rules" / "travel_rules.csv"
    csv_file.write_text(csv_file.read_text(encoding="utf-8") + "# comment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Source dataset fingerprint mismatch"):
        validate_corpus(tmp_path)


def test_negative_corrupted_pdf_build_fails(repo_root, tmp_path):
    """Build must fail safely without creating outputs if a source PDF is corrupted."""
    shutil.copytree(repo_root / "data" / "source", tmp_path / "data" / "source")
    # Corrupt a PDF
    pdf_file = tmp_path / "data" / "source" / "policies" / "travel_policy_v1.pdf"
    pdf_file.write_bytes(b"CORRUPTED_NOT_A_PDF")

    with pytest.raises(Exception):
        build_corpus(tmp_path)
