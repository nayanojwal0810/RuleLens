"""Offline unit and integration tests for RuleLens ingestion pipeline.

Validates the committed processed corpus, determinism, failure recovery,
and negative test cases against corruptions.
"""

import hashlib
import json
import os
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
    get_corpus_snapshot,
    assert_snapshot_current,
    validate_policy_versions,
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


# Committed Processed Data Tests
def test_committed_processed_corpus_validation(repo_root):
    """Verify that existing committed processed corpus passes strengthened validation."""
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

        # Confirm clause ID and company name appear on that page
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


# Fresh Build & Determinism Tests
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


# Negative Tests: Corruptions & Failure Handling
def test_negative_unauthorized_role_added_with_recomputed_hash(repo_root, tmp_path):
    """Validation must reject unauthorized role even when processed fingerprint is recomputed."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Find Travel Exceptions chunk and inject unauthorized role 'employee'
    for c in chunks:
        if c["doc_id"] == "DOC-TRV-004":
            c["allowed_roles"].append("employee")
            c["allowed_roles"].sort()
            break

    # Recompute processed fingerprint
    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    chunks_file.write_text("\n".join(json.dumps(c, sort_keys=True) for c in chunks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allowed_roles mismatch|Unauthorized role"):
        validate_corpus(tmp_path)


def test_negative_wrong_document_version_metadata(repo_root, tmp_path):
    """Validation must reject altered document metadata even with recomputed fingerprint."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    chunks[0]["version"] = "9.9"  # Corrupt version

    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    chunks_file.write_text("\n".join(json.dumps(c, sort_keys=True) for c in chunks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="version mismatch"):
        validate_corpus(tmp_path)


def test_negative_incorrect_source_page(repo_root, tmp_path):
    """Validation must reject incorrect source_page even with recomputed fingerprint."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    chunks[0]["source_page"] = 42  # Wrong source page

    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    chunks_file.write_text("\n".join(json.dumps(c, sort_keys=True) for c in chunks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_page mismatch"):
        validate_corpus(tmp_path)


def test_negative_text_changed_with_recomputed_hashes(repo_root, tmp_path):
    """Validation must reject altered chunk text even when content hash and fingerprint are recomputed."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Alter text and legitimately recompute content hash
    chunks[0]["text"] += "\nUnauthorized policy modification."
    chunks[0]["content_hash"] = hashlib.sha256(chunks[0]["text"].encode("utf-8")).hexdigest()

    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    chunks_file.write_text("\n".join(json.dumps(c, sort_keys=True) for c in chunks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="text mismatch with re-extracted PDF text"):
        validate_corpus(tmp_path)


def test_negative_rule_threshold_or_outcome_changed_with_recomputed_fingerprint(repo_root, tmp_path):
    """Validation must reject altered rule threshold or outcome even when fingerprint is recomputed."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Alter rule threshold
    rules[0]["threshold_value"] = 999999

    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    rules_file.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rules) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mismatch"):
        validate_corpus(tmp_path)


def test_negative_missing_or_substituted_chunk_ids(repo_root, tmp_path):
    """Validation must reject missing or substituted chunk IDs."""
    copy_source_and_processed(repo_root, tmp_path)
    chunks_file = tmp_path / "data" / "processed" / "chunks.jsonl"
    manifest_file = tmp_path / "data" / "processed" / "corpus_manifest.json"
    rules_file = tmp_path / "data" / "processed" / "rules.jsonl"

    chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rules = [json.loads(l) for l in rules_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Substitute chunk_id
    chunks[0]["chunk_id"] = "DOC-TRV-001_TRV-C99"

    proc_fp = calculate_processed_fingerprint(chunks, rules)
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["processed_fingerprint"] = proc_fp

    chunks_file.write_text("\n".join(json.dumps(c, sort_keys=True) for c in chunks) + "\n", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="format|Unexpected or nonexistent chunk_id|Chunk ID set mismatch"):
        validate_corpus(tmp_path)


def test_replacement_failure_during_second_output_restores_previous_state(repo_root, tmp_path, monkeypatch):
    """A replacement error during the second file update must restore exact previous files byte-for-byte."""
    copy_source_and_processed(repo_root, tmp_path)
    processed_dir = tmp_path / "data" / "processed"

    # Record previous SHA-256 hashes of all three files
    initial_hashes = {
        name: hashlib.sha256((processed_dir / name).read_bytes()).hexdigest()
        for name in ("chunks.jsonl", "rules.jsonl", "corpus_manifest.json")
    }

    original_replace = os.replace

    def mock_replace(src, dst):
        # Fail when replacing the second file: rules.jsonl
        if Path(dst).name == "rules.jsonl":
            raise OSError("Injected disk/filesystem error on rules.jsonl replacement")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    with pytest.raises(RuntimeError, match="Corpus build replacement failed; previous output state restored"):
        build_corpus(tmp_path)

    # Verify all three files exist and remain byte-for-byte identical to initial state
    for name, expected_hash in initial_hashes.items():
        assert (processed_dir / name).exists()
        actual_hash = hashlib.sha256((processed_dir / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"File {name} was not properly restored!"

    # Verify no temporary staging files remain in processed_dir
    stray_files = [p.name for p in processed_dir.glob("*.tmp")]
    assert len(stray_files) == 0, f"Found leftover temporary files: {stray_files}"


def test_extraction_failure_preserves_existing_outputs(repo_root, tmp_path):
    """An extraction error before replacement must leave existing outputs untouched."""
    copy_source_and_processed(repo_root, tmp_path)
    processed_dir = tmp_path / "data" / "processed"

    initial_hashes = {
        name: hashlib.sha256((processed_dir / name).read_bytes()).hexdigest()
        for name in ("chunks.jsonl", "rules.jsonl", "corpus_manifest.json")
    }

    # Corrupt a source PDF
    pdf_file = tmp_path / "data" / "source" / "policies" / "travel_policy_v1.pdf"
    pdf_file.write_bytes(b"CORRUPTED_PDF_BYTES")

    with pytest.raises(Exception):
        build_corpus(tmp_path)

    # Verify previous outputs are completely intact
    for name, expected_hash in initial_hashes.items():
        actual_hash = hashlib.sha256((processed_dir / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"File {name} was modified despite extraction failure!"


def test_snapshot_reuses_validation_but_not_mutable_payloads(repo_root, tmp_path, monkeypatch):
    import rulelens.ingestion as ingestion

    copy_source_and_processed(repo_root, tmp_path)
    calls = []
    original = ingestion.validate_corpus

    def counted_validation(base):
        calls.append(base)
        return original(base)

    monkeypatch.setattr(ingestion, "validate_corpus", counted_validation)
    snapshot = get_corpus_snapshot(tmp_path)
    assert get_corpus_snapshot(tmp_path) is snapshot
    assert len(calls) == 1
    edited = snapshot.chunks
    edited[0]["text"] = "Forged text"
    assert snapshot.chunks[0]["text"] != "Forged text"


@pytest.mark.parametrize("filename", ["rules.jsonl", "chunks.jsonl", "corpus_manifest.json"])
def test_snapshot_detects_warm_file_changes(repo_root, tmp_path, filename):
    copy_source_and_processed(repo_root, tmp_path)
    snapshot = get_corpus_snapshot(tmp_path)
    path = tmp_path / "data/processed" / filename
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        assert_snapshot_current(snapshot)


def test_recomputed_fingerprints_do_not_validate_changed_rules(repo_root, tmp_path):
    copy_source_and_processed(repo_root, tmp_path)
    get_corpus_snapshot(tmp_path)
    processed = tmp_path / "data/processed"
    chunks = [json.loads(line) for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    rules = [json.loads(line) for line in (processed / "rules.jsonl").read_text(encoding="utf-8").splitlines()]
    rules[0]["threshold_value"] = 99999
    manifest = json.loads((processed / "corpus_manifest.json").read_text(encoding="utf-8"))
    manifest["processed_fingerprint"] = calculate_processed_fingerprint(chunks, rules)
    (processed / "rules.jsonl").write_text("\n".join(json.dumps(rule) for rule in rules), encoding="utf-8")
    (processed / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        get_corpus_snapshot(tmp_path)


def test_snapshot_detects_changes_during_validation(repo_root, tmp_path, monkeypatch):
    import rulelens.ingestion as ingestion

    copy_source_and_processed(repo_root, tmp_path)
    original = ingestion.validate_corpus

    def changing_validation(base):
        original(base)
        path = base / "data/processed/rules.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    monkeypatch.setattr(ingestion, "validate_corpus", changing_validation)
    with pytest.raises(ValueError, match="changed"):
        get_corpus_snapshot(tmp_path)


def test_conflicting_versions_are_rejected():
    records = [
        {"doc_id": "travel_old", "title": "Travel Policy v1", "effective_from": "2026-01-01", "effective_to": "2026-07-01"},
        {"doc_id": "travel_new", "title": "Travel Policy v2", "effective_from": "2026-07-01", "effective_to": "2099-12-31"},
    ]
    with pytest.raises(ValueError, match="Conflicting active policy versions"):
        validate_policy_versions(records)
    records[0]["effective_to"] = "2026-06-30"
    validate_policy_versions(records)
