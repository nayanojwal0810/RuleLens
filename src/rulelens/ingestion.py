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


# ---------------------------------------------------------------------------
# Fingerprint Calculations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PDF Extraction and Chunking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CSV Rule Processing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Corpus Building and Validation Pipeline
# ---------------------------------------------------------------------------

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

    # 5. Atomic file writing using temporary files
    chunks_tmp = processed_dir / "chunks.jsonl.tmp"
    rules_tmp = processed_dir / "rules.jsonl.tmp"
    manifest_tmp = processed_dir / "corpus_manifest.json.tmp"

    try:
        with open(chunks_tmp, "w", encoding="utf-8", newline="\n") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, sort_keys=True) + "\n")

        with open(rules_tmp, "w", encoding="utf-8", newline="\n") as f:
            for rule in rules:
                f.write(json.dumps(rule, sort_keys=True) + "\n")

        with open(manifest_tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(corpus_manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        # Atomic replacements
        os.replace(chunks_tmp, processed_dir / "chunks.jsonl")
        os.replace(rules_tmp, processed_dir / "rules.jsonl")
        os.replace(manifest_tmp, processed_dir / "corpus_manifest.json")
    finally:
        for tmp_file in (chunks_tmp, rules_tmp, manifest_tmp):
            if tmp_file.exists():
                tmp_file.unlink()

    return corpus_manifest


def validate_corpus(base_dir: Optional[Path | str] = None) -> bool:
    """Validate existing processed corpus without rebuilding files."""
    base = Path(base_dir) if base_dir else Path.cwd()
    source_dir = base / "data" / "source"
    processed_dir = base / "data" / "processed"

    chunks_file = processed_dir / "chunks.jsonl"
    rules_file = processed_dir / "rules.jsonl"
    manifest_file = processed_dir / "corpus_manifest.json"

    for required_file in (chunks_file, rules_file, manifest_file):
        if not required_file.exists() or required_file.stat().st_size == 0:
            raise ValueError(f"Processed file missing or empty: {required_file.name}")

    # 1. Validate corpus manifest
    with open(manifest_file, "r", encoding="utf-8") as f:
        corpus_manifest = json.load(f)

    if corpus_manifest.get("document_count") != 20:
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

    # Check source fingerprint
    expected_source_fp = calculate_source_fingerprint(source_dir)
    if corpus_manifest.get("source_fingerprint") != expected_source_fp:
        raise ValueError("Source dataset fingerprint mismatch (source dataset was modified or out of sync).")

    # 2. Validate chunks
    chunks: List[Dict[str, Any]] = []
    chunk_ids: Set[str] = set()

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
                raise ValueError(f"Duplicate chunk_id found: {cid}")
            chunk_ids.add(cid)

            # Verify content hash matches actual text
            computed_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            if chunk["content_hash"] != computed_hash:
                raise ValueError(f"Content hash mismatch in chunk {cid}")

            # Check 1-based source page
            if not isinstance(chunk["source_page"], int) or chunk["source_page"] < 1:
                raise ValueError(f"Invalid source_page in chunk {cid}: {chunk['source_page']}")

            # Validate ISO dates
            datetime.date.fromisoformat(chunk["effective_from"])
            datetime.date.fromisoformat(chunk["effective_to"])
            if chunk["effective_from"] > chunk["effective_to"]:
                raise ValueError(f"Invalid date interval in chunk {cid}")

            # Check roles is a valid non-empty list of known roles
            if not isinstance(chunk["allowed_roles"], list) or not chunk["allowed_roles"]:
                raise ValueError(f"Invalid allowed_roles in chunk {cid}")
            for role in chunk["allowed_roles"]:
                if role not in ROLES:
                    raise ValueError(f"Unknown role '{role}' in chunk {cid}")

            chunks.append(chunk)

    if len(chunks) != 64:
        raise ValueError(f"Expected 64 chunks in {chunks_file.name}, found {len(chunks)}")

    # Distinct document count
    docs_represented = {c["doc_id"] for c in chunks}
    if len(docs_represented) != 20:
        raise ValueError(f"Expected 20 documents represented in chunks, found {len(docs_represented)}")

    # 3. Validate rules
    rules: List[Dict[str, Any]] = []
    rule_ids: Set[str] = set()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

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
                raise ValueError(f"Duplicate rule_id found: {rid}")
            rule_ids.add(rid)

            if rule["operator"] not in APPROVED_OPERATORS:
                raise ValueError(f"Unapproved operator in rule {rid}: {rule['operator']}")

            # Verify threshold typing
            val = rule["threshold_value"]
            if not isinstance(val, (int, float, str)):
                raise ValueError(f"Invalid threshold type in rule {rid}: {type(val)}")

            # Link to chunk
            chunk_key = f"{rule['doc_id']}_{rule['clause_id']}"
            if chunk_key not in chunks_by_id:
                raise ValueError(f"Rule {rid} links to nonexistent chunk: {chunk_key}")

            chunk = chunks_by_id[chunk_key]
            if rule["effective_from"] < chunk["effective_from"] or rule["effective_to"] > chunk["effective_to"]:
                raise ValueError(f"Rule {rid} effective dates exceed chunk {chunk_key} bounds")

            rules.append(rule)

    if len(rules) != 32:
        raise ValueError(f"Expected 32 rules in {rules_file.name}, found {len(rules)}")

    # 4. Verify processed fingerprint
    expected_proc_fp = calculate_processed_fingerprint(chunks, rules)
    if corpus_manifest.get("processed_fingerprint") != expected_proc_fp:
        raise ValueError("Processed corpus fingerprint mismatch (records modified or corrupted).")

    return True


# ---------------------------------------------------------------------------
# CLI Command Entrypoint
# ---------------------------------------------------------------------------

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
