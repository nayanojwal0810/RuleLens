"""Deterministic policy document ingestion pipeline for RuleLens.

Converts the 20 policy PDFs and 5 CSV rule tables into structured,
reproducible, validated JSONL and manifest records for hybrid retrieval.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pymupdf

from rulelens.config import EMBEDDING_MODEL
from rulelens.dataset import (
    DOCUMENTS,
    ROLES,
    RULES,
    DEPARTMENT_RULES_MAPPING,
    validate_dataset,
)

CHUNKING_STRATEGY: str = "clause_level"
PROCESSING_SCHEMA_VERSION: str = "1.0.0"

APPROVED_OPERATORS: Set[str] = {"<", "<=", ">", ">=", "=="}

CHUNK_REQUIRED_FIELDS: Tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "clause_id",
    "title",
    "department",
    "version",
    "effective_from",
    "effective_to",
    "allowed_roles",
    "source_file",
    "source_page",
    "text",
    "content_hash",
)

RULE_REQUIRED_FIELDS: Tuple[str, ...] = (
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
class CorpusSnapshot:
    base_dir: Path
    file_hashes: Tuple[Tuple[str, str], ...]
    chunks_json: str
    rules_json: str
    manifest_json: str

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return json.loads(self.chunks_json)

    @property
    def rules(self) -> List[Dict[str, Any]]:
        return json.loads(self.rules_json)

    @property
    def fingerprint(self) -> str:
        return json.loads(self.manifest_json)["processed_fingerprint"]


_CORPUS_SNAPSHOTS: Dict[Path, CorpusSnapshot] = {}


def corpus_file_hashes(base_dir: Path) -> Tuple[Tuple[str, str], ...]:
    files = [p for p in (base_dir / "data/source").rglob("*") if p.is_file()]
    files += [base_dir / "data/processed" / name for name in
              ("chunks.jsonl", "rules.jsonl", "corpus_manifest.json")]
    return tuple((p.relative_to(base_dir).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(files))


def assert_snapshot_current(snapshot: CorpusSnapshot) -> None:
    if corpus_file_hashes(snapshot.base_dir) != snapshot.file_hashes:
        raise ValueError("Corpus changed during the request.")


def validate_policy_versions(chunks: Sequence[Dict[str, Any]]) -> None:
    families: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for chunk in chunks:
        family = re.sub(r"\bv\d+(?:\.\d+)*\b", "", chunk["title"].lower()).strip()
        families.setdefault(family, {})[chunk["doc_id"]] = chunk
    for documents in families.values():
        records = list(documents.values())
        for index, left in enumerate(records):
            for right in records[index + 1:]:
                if max(left["effective_from"], right["effective_from"]) <= min(
                    left.get("effective_to") or "9999-12-31",
                    right.get("effective_to") or "9999-12-31",
                ):
                    raise ValueError("Conflicting active policy versions.")


def get_corpus_snapshot(base_dir: Optional[Path | str] = None) -> CorpusSnapshot:
    base = (Path(base_dir) if base_dir is not None else Path.cwd()).resolve()
    before = corpus_file_hashes(base)
    cached = _CORPUS_SNAPSHOTS.get(base)
    if cached is not None and cached.file_hashes == before:
        return cached
    validate_corpus(base)
    processed = base / "data/processed"
    chunks_text = (processed / "chunks.jsonl").read_text(encoding="utf-8")
    rules_text = (processed / "rules.jsonl").read_text(encoding="utf-8")
    manifest_text = (processed / "corpus_manifest.json").read_text(encoding="utf-8")
    chunks = [json.loads(line) for line in chunks_text.splitlines() if line.strip()]
    rules = [json.loads(line) for line in rules_text.splitlines() if line.strip()]
    manifest = json.loads(manifest_text)
    validate_policy_versions(chunks)
    if calculate_processed_fingerprint(chunks, rules) != manifest["processed_fingerprint"]:
        raise ValueError("Snapshot fingerprint mismatch.")
    snapshot = CorpusSnapshot(base, before, json.dumps(chunks), json.dumps(rules), manifest_text)
    assert_snapshot_current(snapshot)
    if len(_CORPUS_SNAPSHOTS) >= 4:
        _CORPUS_SNAPSHOTS.pop(next(iter(_CORPUS_SNAPSHOTS)))
    _CORPUS_SNAPSHOTS[base] = snapshot
    return snapshot


# Fingerprint Calculations
def calculate_source_fingerprint(source_dir: Path) -> str:
    """Compute SHA-256 fingerprint over all approved source files in sorted order."""
    hasher = hashlib.sha256()
    source_files = sorted(
        [p for p in source_dir.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(source_dir)).replace("\\", "/"),
    )

    for path in source_files:
        rel_path = str(path.relative_to(source_dir)).replace("\\", "/")
        hasher.update(rel_path.encode("utf-8"))
        with open(path, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()


def calculate_processed_fingerprint(
    chunks: Sequence[Dict[str, Any]], rules: Sequence[Dict[str, Any]]
) -> str:
    """Compute SHA-256 fingerprint over deterministic chunk and rule records."""
    hasher = hashlib.sha256()

    for chunk in chunks:
        serialized = json.dumps(chunk, sort_keys=True, separators=(",", ":"))
        hasher.update(serialized.encode("utf-8"))

    for rule in rules:
        serialized = json.dumps(rule, sort_keys=True, separators=(",", ":"))
        hasher.update(serialized.encode("utf-8"))

    return hasher.hexdigest()


# PDF Extraction and Chunking
def extract_pdf_chunks(
    policies_dir: Path, manifest_docs: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract clause-level chunks from all policy PDFs using PyMuPDF.

    Extracts exactly one chunk per policy clause, combining the clause heading
    and policy statement, with whitespace normalized and headers/footers excluded.
    """
    manifest_by_filename = {d["filename"]: d for d in manifest_docs}
    chunks: List[Dict[str, Any]] = []

    for doc_entry in manifest_docs:
        filename = doc_entry["filename"]
        pdf_path = policies_dir / filename
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise ValueError(f"Missing or empty PDF: {filename}")

        doc_id = doc_entry["doc_id"]
        expected_clause_ids = doc_entry["clause_ids"]

        pdf_doc = pymupdf.open(pdf_path)
        extracted_clauses: List[Dict[str, Any]] = []

        try:
            for page_idx, page in enumerate(pdf_doc, start=1):
                blocks = page.get_text("blocks")
                for b_idx, block in enumerate(blocks):
                    block_text = block[4].strip()
                    # Detect clause heading: [CLAUSE-ID] Clause Title
                    match = re.match(r"^\[([A-Z0-9]+-[A-Z0-9]+)\]\s*(.*)", block_text)
                    if match:
                        cid = match.group(1)
                        heading = block_text
                        # Statement is the subsequent text block
                        if b_idx + 1 < len(blocks):
                            body_raw = blocks[b_idx + 1][4].strip()
                            if body_raw.startswith("["):
                                raise ValueError(
                                    f"Malformed statement for clause {cid} in {filename}: unexpected header {body_raw}"
                                )
                            norm_body = " ".join(body_raw.split())
                        else:
                            raise ValueError(f"Missing statement block for clause {cid} in {filename}")

                        chunk_text = f"{heading}\n{norm_body}"
                        extracted_clauses.append({
                            "clause_id": cid,
                            "source_page": page_idx,
                            "text": chunk_text,
                        })
        finally:
            pdf_doc.close()

        found_cids = [c["clause_id"] for c in extracted_clauses]
        if found_cids != expected_clause_ids:
            raise ValueError(
                f"Clause extraction mismatch in {filename}: found {found_cids}, expected {expected_clause_ids}"
            )

        for clause_info in extracted_clauses:
            cid = clause_info["clause_id"]
            chunk_text = clause_info["text"]
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = f"{doc_id}_{cid}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "clause_id": cid,
                "title": doc_entry["title"],
                "department": doc_entry["department"],
                "version": doc_entry["version"],
                "effective_from": doc_entry["effective_from"],
                "effective_to": doc_entry["effective_to"],
                "allowed_roles": list(doc_entry["permitted_roles"]),
                "source_file": filename,
                "source_page": clause_info["source_page"],
                "text": chunk_text,
                "content_hash": content_hash,
            })

    return chunks


# CSV Rule Processing
def parse_threshold_value(val_str: str) -> int | float | str:
    """Parse threshold string into typed integer, float, or string."""
    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str


def parse_csv_rules(
    rules_dir: Path, chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Parse and validate all structured rule CSVs into typed dictionaries."""
    rules: List[Dict[str, Any]] = []

    # Read files in deterministic department order
    for dept, filename in DEPARTMENT_RULES_MAPPING.items():
        csv_path = rules_dir / filename
        if not csv_path.exists():
            raise ValueError(f"Missing CSV rule table: {filename}")

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rule_id = row["rule_id"]
                doc_id = row["doc_id"]
                clause_id = row["clause_id"]
                operator = row["operator"]

                if operator not in APPROVED_OPERATORS:
                    raise ValueError(f"Unapproved operator '{operator}' in rule {rule_id}")

                chunk_key = f"{doc_id}_{clause_id}"
                if chunk_key not in chunks_by_id:
                    raise ValueError(
                        f"Rule {rule_id} references nonexistent chunk {chunk_key} ({doc_id}, {clause_id})"
                    )

                chunk = chunks_by_id[chunk_key]
                # Validate date interval compatibility with chunk
                if row["effective_from"] < chunk["effective_from"] or row["effective_to"] > chunk["effective_to"]:
                    raise ValueError(
                        f"Rule {rule_id} dates [{row['effective_from']}, {row['effective_to']}] "
                        f"exceed chunk dates [{chunk['effective_from']}, {chunk['effective_to']}]"
                    )

                typed_threshold = parse_threshold_value(row["threshold_value"])

                rules.append({
                    "rule_id": rule_id,
                    "doc_id": doc_id,
                    "clause_id": clause_id,
                    "condition_field": row["condition_field"],
                    "operator": operator,
                    "threshold_value": typed_threshold,
                    "unit": row["unit"],
                    "outcome": row["outcome"],
                    "effective_from": row["effective_from"],
                    "effective_to": row["effective_to"],
                })

    return rules


# Corpus Building and Validation Pipeline
def build_corpus(base_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Execute the full ingestion build pipeline atomically."""
    base = Path(base_dir) if base_dir else Path.cwd()
    source_dir = base / "data" / "source"
    policies_dir = source_dir / "policies"
    rules_dir = source_dir / "rules"
    manifest_path = source_dir / "manifest.json"
    roles_path = source_dir / "roles.json"

    processed_dir = base / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate source dataset first
    validate_dataset(base)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_docs = manifest_data["documents"]

    # 2. Extract PDF chunks
    chunks = extract_pdf_chunks(policies_dir, manifest_docs)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    if len(chunks) != 64:
        raise ValueError(f"Expected exactly 64 chunks, got {len(chunks)}")

    # 3. Parse CSV rules
    rules = parse_csv_rules(rules_dir, chunks_by_id)
    if len(rules) != 32:
        raise ValueError(f"Expected exactly 32 rules, got {len(rules)}")

    # 4. Calculate fingerprints
    source_fp = calculate_source_fingerprint(source_dir)
    processed_fp = calculate_processed_fingerprint(chunks, rules)

    corpus_manifest = {
        "chunk_count": len(chunks),
        "chunking_strategy": CHUNKING_STRATEGY,
        "document_count": len(manifest_docs),
        "embedding_model": EMBEDDING_MODEL,
        "processed_fingerprint": processed_fp,
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "rule_count": len(rules),
        "source_fingerprint": source_fp,
    }

    # 5. Safe output replacement using temporary staging directory and rollback
    target_filenames = ("chunks.jsonl", "rules.jsonl", "corpus_manifest.json")
    backups: Dict[str, bytes] = {}
    for name in target_filenames:
        target_path = processed_dir / name
        if target_path.exists():
            backups[name] = target_path.read_bytes()

    with tempfile.TemporaryDirectory(dir=processed_dir) as staging_str:
        staging_dir = Path(staging_str)
        staged_chunks = staging_dir / "chunks.jsonl"
        staged_rules = staging_dir / "rules.jsonl"
        staged_manifest = staging_dir / "corpus_manifest.json"

        with open(staged_chunks, "w", encoding="utf-8", newline="\n") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, sort_keys=True) + "\n")

        with open(staged_rules, "w", encoding="utf-8", newline="\n") as f:
            for rule in rules:
                f.write(json.dumps(rule, sort_keys=True) + "\n")

        with open(staged_manifest, "w", encoding="utf-8", newline="\n") as f:
            json.dump(corpus_manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        replaced: List[str] = []
        try:
            for name in target_filenames:
                src_path = staging_dir / name
                dst_path = processed_dir / name
                os.replace(src_path, dst_path)
                replaced.append(name)
        except Exception as exc:
            # Restore previous complete output set on handled replacement error
            for name, content in backups.items():
                (processed_dir / name).write_bytes(content)
            for name in replaced:
                if name not in backups:
                    p = processed_dir / name
                    if p.exists():
                        p.unlink()
            raise RuntimeError(
                f"Corpus build replacement failed; previous output state restored: {exc}"
            ) from exc

    return corpus_manifest


def validate_corpus(base_dir: Optional[Path | str] = None) -> bool:
    """Validate existing processed corpus without rebuilding files.

    Performs comprehensive verification against the authoritative source dataset:
    - Verifies source manifest, roles, and source fingerprint.
    - Re-extracts expected clause records from PDFs with PyMuPDF and verifies exact
      text, source_page, and metadata match for all chunks.
    - Parses canonical CSV rules and verifies exact field match for all rules.
    - Requires exact expected chunk and rule ID sets without duplicates or omissions.
    - Verifies stored content hashes and processed corpus fingerprint.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    source_dir = base / "data" / "source"
    policies_dir = source_dir / "policies"
    rules_dir = source_dir / "rules"
    manifest_path = source_dir / "manifest.json"
    roles_path = source_dir / "roles.json"
    processed_dir = base / "data" / "processed"

    # 1. Validate source dataset first
    validate_dataset(base)

    chunks_file = processed_dir / "chunks.jsonl"
    rules_file = processed_dir / "rules.jsonl"
    manifest_file = processed_dir / "corpus_manifest.json"

    for required_file in (chunks_file, rules_file, manifest_file):
        if not required_file.exists() or required_file.stat().st_size == 0:
            raise ValueError(f"Processed file missing or empty: {required_file.name}")

    # 2. Validate corpus manifest
    with open(manifest_file, "r", encoding="utf-8") as f:
        corpus_manifest = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)
    manifest_docs = source_manifest["documents"]
    doc_map = {d["doc_id"]: d for d in manifest_docs}

    with open(roles_path, "r", encoding="utf-8") as f:
        roles_data = json.load(f)["roles"]

    if corpus_manifest.get("document_count") != len(manifest_docs):
        raise ValueError(f"Document count mismatch in manifest: {corpus_manifest.get('document_count')}")
    if corpus_manifest.get("chunk_count") != 64:
        raise ValueError(f"Chunk count mismatch in manifest: {corpus_manifest.get('chunk_count')}")
    if corpus_manifest.get("rule_count") != 32:
        raise ValueError(f"Rule count mismatch in manifest: {corpus_manifest.get('rule_count')}")
    if corpus_manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError(f"Embedding model mismatch: {corpus_manifest.get('embedding_model')}")
    if corpus_manifest.get("chunking_strategy") != CHUNKING_STRATEGY:
        raise ValueError(f"Chunking strategy mismatch: {corpus_manifest.get('chunking_strategy')}")
    if corpus_manifest.get("processing_schema_version") != PROCESSING_SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: {corpus_manifest.get('processing_schema_version')}")

    # Check source fingerprint against actual source files
    expected_source_fp = calculate_source_fingerprint(source_dir)
    if corpus_manifest.get("source_fingerprint") != expected_source_fp:
        raise ValueError("Source dataset fingerprint mismatch (source dataset was modified or out of sync).")

    # 3. Re-extract ground-truth expected chunks from source PDFs
    expected_chunks = extract_pdf_chunks(policies_dir, manifest_docs)
    expected_chunks_by_id = {c["chunk_id"]: c for c in expected_chunks}

    # 4. Validate chunks from chunks.jsonl
    chunks: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            chunk = json.loads(line_str)

            for req_field in CHUNK_REQUIRED_FIELDS:
                if req_field not in chunk:
                    raise ValueError(f"Chunk line {line_no} missing required field: {req_field}")

            cid = chunk["chunk_id"]
            if cid in chunk_ids:
                raise ValueError(f"Duplicate chunk_id found in chunks.jsonl: {cid}")
            chunk_ids.append(cid)

            # Check chunk_id format
            expected_cid = f"{chunk['doc_id']}_{chunk['clause_id']}"
            if cid != expected_cid:
                raise ValueError(f"Chunk ID '{cid}' does not match expected format '{expected_cid}'")

            if cid not in expected_chunks_by_id:
                raise ValueError(f"Unexpected or nonexistent chunk_id: {cid}")

            expected_c = expected_chunks_by_id[cid]
            doc_entry = doc_map.get(chunk["doc_id"])
            if doc_entry is None:
                raise ValueError(f"Chunk {cid} references nonexistent doc_id: {chunk['doc_id']}")

            if chunk["clause_id"] not in doc_entry["clause_ids"]:
                raise ValueError(f"Clause {chunk['clause_id']} does not belong to document {chunk['doc_id']}")

            # Verify document metadata
            if chunk["title"] != doc_entry["title"]:
                raise ValueError(f"Chunk {cid} title mismatch: '{chunk['title']}' != '{doc_entry['title']}'")
            if chunk["department"] != doc_entry["department"]:
                raise ValueError(f"Chunk {cid} department mismatch: '{chunk['department']}' != '{doc_entry['department']}'")
            if chunk["version"] != doc_entry["version"]:
                raise ValueError(f"Chunk {cid} version mismatch: '{chunk['version']}' != '{doc_entry['version']}'")
            if chunk["effective_from"] != doc_entry["effective_from"]:
                raise ValueError(f"Chunk {cid} effective_from mismatch")
            if chunk["effective_to"] != doc_entry["effective_to"]:
                raise ValueError(f"Chunk {cid} effective_to mismatch")
            if chunk["source_file"] != doc_entry["filename"]:
                raise ValueError(f"Chunk {cid} source_file mismatch")

            # Verify allowed_roles against manifest and roles.json
            if sorted(chunk["allowed_roles"]) != sorted(doc_entry["permitted_roles"]):
                raise ValueError(f"Chunk {cid} allowed_roles mismatch with manifest permitted_roles")
            for r in chunk["allowed_roles"]:
                if r not in roles_data or chunk["doc_id"] not in roles_data[r]["allowed_documents"]:
                    raise ValueError(f"Unauthorized role '{r}' for chunk {cid}")

            # Verify text, content hash, and source page against actual PDF re-extraction
            if chunk["source_page"] != expected_c["source_page"]:
                raise ValueError(
                    f"Chunk {cid} source_page mismatch: {chunk['source_page']} != {expected_c['source_page']}"
                )
            if chunk["text"] != expected_c["text"]:
                raise ValueError(f"Chunk {cid} text mismatch with re-extracted PDF text")

            computed_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            if chunk["content_hash"] != computed_hash:
                raise ValueError(f"Content hash mismatch in chunk {cid}")
            if chunk["content_hash"] != expected_c["content_hash"]:
                raise ValueError(f"Content hash does not match expected hash for chunk {cid}")

            chunks.append(chunk)

    # Require exact expected chunk ID set
    expected_chunk_ids_set = set(expected_chunks_by_id.keys())
    found_chunk_ids_set = set(chunk_ids)
    if found_chunk_ids_set != expected_chunk_ids_set:
        raise ValueError(
            f"Chunk ID set mismatch: missing {expected_chunk_ids_set - found_chunk_ids_set}, "
            f"extra {found_chunk_ids_set - expected_chunk_ids_set}"
        )
    if len(chunks) != 64:
        raise ValueError(f"Expected 64 chunks in {chunks_file.name}, found {len(chunks)}")

    # 5. Parse canonical rules and validate rules.jsonl
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    canonical_rules = parse_csv_rules(rules_dir, chunks_by_id)
    canonical_rules_by_id = {r["rule_id"]: r for r in canonical_rules}

    rules: List[Dict[str, Any]] = []
    rule_ids: List[str] = []

    with open(rules_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            rule = json.loads(line_str)

            for req_field in RULE_REQUIRED_FIELDS:
                if req_field not in rule:
                    raise ValueError(f"Rule line {line_no} missing required field: {req_field}")

            rid = rule["rule_id"]
            if rid in rule_ids:
                raise ValueError(f"Duplicate rule_id found in rules.jsonl: {rid}")
            rule_ids.append(rid)

            if rid not in canonical_rules_by_id:
                raise ValueError(f"Unexpected or nonexistent rule_id: {rid}")

            canon = canonical_rules_by_id[rid]

            # Verify every field against canonical rule
            for field in RULE_REQUIRED_FIELDS:
                actual_val = rule[field]
                expected_val = canon[field]
                if actual_val != expected_val:
                    raise ValueError(
                        f"Rule {rid} field '{field}' mismatch: expected '{expected_val}', got '{actual_val}'"
                    )
                if type(actual_val) is not type(expected_val):
                    raise ValueError(
                        f"Rule {rid} field '{field}' type mismatch: expected {type(expected_val)}, got {type(actual_val)}"
                    )

            rules.append(rule)

    # Require exact expected rule ID set
    expected_rule_ids_set = set(canonical_rules_by_id.keys())
    found_rule_ids_set = set(rule_ids)
    if found_rule_ids_set != expected_rule_ids_set:
        raise ValueError(
            f"Rule ID set mismatch: missing {expected_rule_ids_set - found_rule_ids_set}, "
            f"extra {found_rule_ids_set - expected_rule_ids_set}"
        )
    if len(rules) != 32:
        raise ValueError(f"Expected 32 rules in {rules_file.name}, found {len(rules)}")

    # 6. Verify processed fingerprint
    expected_proc_fp = calculate_processed_fingerprint(chunks, rules)
    if corpus_manifest.get("processed_fingerprint") != expected_proc_fp:
        raise ValueError("Processed corpus fingerprint mismatch (records modified or corrupted).")

    return True


# CLI Command Entrypoint
def main() -> None:
    """CLI interface for corpus building and validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rulelens.ingestion [build|validate]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "build":
        try:
            manifest = build_corpus()
            print(
                f"Corpus build completed successfully:\n"
                f"  Documents: {manifest['document_count']}\n"
                f"  Chunks:    {manifest['chunk_count']}\n"
                f"  Rules:     {manifest['rule_count']}\n"
                f"  Source FP: {manifest['source_fingerprint'][:16]}...\n"
                f"  Proc FP:   {manifest['processed_fingerprint'][:16]}..."
            )
        except Exception as e:
            print(f"Build failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "validate":
        try:
            validate_corpus()
            print("Corpus validation passed successfully.")
        except Exception as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}. Use 'build' or 'validate'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
