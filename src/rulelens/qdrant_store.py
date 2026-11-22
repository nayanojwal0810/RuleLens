"""Qdrant Cloud vector database integration for RuleLens.

Provides deterministic collection synchronization, server-side role and temporal
filtering, collection verification, and dense vector search over the 64 policy chunks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from qdrant_client import QdrantClient, models

try:
    from qdrant_client.local.qdrant_local import QdrantLocal
except ImportError:
    QdrantLocal = None

from rulelens.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    get_env_var,
    is_invalid_qdrant_url,
)
from rulelens.dataset import ROLES
from rulelens.ingestion import validate_corpus
from rulelens.retrieval import (
    DEFAULT_CANDIDATE_DEPTH,
    EMBEDDING_DIMENSION,
    BGEEmbedder,
    load_validated_corpus,
    parse_query_date,
)

ALLOWED_ROLES_SET: Set[str] = set(ROLES)

# Deterministic namespace for chunk UUIDv5 point IDs
POINT_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "rulelens.asterline.internal")

REQUIRED_PAYLOAD_INDEXES: List[Tuple[str, models.PayloadSchemaType]] = [
    ("allowed_roles", models.PayloadSchemaType.KEYWORD),
    ("corpus_fingerprint", models.PayloadSchemaType.KEYWORD),
    ("effective_from_rfc3339", models.PayloadSchemaType.DATETIME),
    ("effective_to_rfc3339", models.PayloadSchemaType.DATETIME),
]


def is_in_memory_or_local_client(client: QdrantClient) -> bool:
    """Return True if the client is using local in-memory storage rather than a server."""
    inner = getattr(client, "_client", None)
    if QdrantLocal is not None and isinstance(inner, QdrantLocal):
        return True
    return inner.__class__.__name__ == "QdrantLocal"


def check_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> Dict[str, Any]:
    """Check payload indexes on collection for a real server client.

    Returns dict with status and details. If on a real server and required indexes
    are missing or of incorrect type, raises ValueError.
    For local in-memory clients, reports that indexes are handled by local mock.
    """
    if is_in_memory_or_local_client(client):
        return {"status": "in_memory_client", "indexes": {}}

    col_info = client.get_collection(collection_name)
    schema = getattr(col_info, "payload_schema", {}) or {}

    missing_or_bad = []
    verified_indexes = {}
    for field_name, expected_type in REQUIRED_PAYLOAD_INDEXES:
        if field_name not in schema:
            missing_or_bad.append(f"Missing index on field '{field_name}'")
        else:
            field_info = schema[field_name]
            dtype = getattr(field_info, "data_type", None)
            if dtype is None and isinstance(field_info, dict):
                dtype = field_info.get("data_type")
            expected_str = expected_type.value if hasattr(expected_type, "value") else str(expected_type).lower()
            actual_str = dtype.value if hasattr(dtype, "value") else str(dtype).lower()
            if actual_str.lower() != expected_str.lower():
                missing_or_bad.append(
                    f"Index on '{field_name}' has type '{actual_str}'; expected '{expected_str}'"
                )
            else:
                verified_indexes[field_name] = actual_str

    if missing_or_bad:
        raise ValueError(
            f"Collection '{collection_name}' has incomplete payload indexes: {', '.join(missing_or_bad)}"
        )

    return {"status": "verified", "indexes": verified_indexes}


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Ensure required payload indexes exist on a server collection with correct types.

    1. Inspects existing payload_schema.
    2. If an index exists with the correct type, keeps it.
    3. If an index exists with the wrong type, raises ValueError.
    4. If an index is missing, creates it.
    5. Propagates unexpected creation errors clearly.
    6. Verifies all required indexes after creation.
    """
    if is_in_memory_or_local_client(client):
        return

    col_info = client.get_collection(collection_name)
    schema = getattr(col_info, "payload_schema", {}) or {}

    for field_name, expected_type in REQUIRED_PAYLOAD_INDEXES:
        expected_str = expected_type.value if hasattr(expected_type, "value") else str(expected_type).lower()
        if field_name in schema:
            field_info = schema[field_name]
            dtype = getattr(field_info, "data_type", None)
            if dtype is None and isinstance(field_info, dict):
                dtype = field_info.get("data_type")
            actual_str = dtype.value if hasattr(dtype, "value") else str(dtype).lower()
            if actual_str.lower() != expected_str.lower():
                raise ValueError(
                    f"Collection '{collection_name}' has index on '{field_name}' with wrong type '{actual_str}'; "
                    f"expected '{expected_str}'."
                )
            # Valid index already exists; keep it
            continue

        # Missing index: create it
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=expected_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create payload index on '{field_name}' in collection '{collection_name}': {exc}"
            ) from exc

    # Final verification of required payload indexes
    check_payload_indexes(client, collection_name)


def get_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUIDv5 string point ID from a chunk ID."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> QdrantClient:
    """Instantiate a QdrantClient using explicit parameters or environment variables.

    Raises:
        ValueError: If QDRANT_URL is not set or empty, or points to the Qdrant marketing website.
    """
    target_url = url or get_env_var(QDRANT_URL)
    target_key = api_key or get_env_var(QDRANT_API_KEY)

    if not target_url or not target_url.strip():
        raise ValueError("Missing required QDRANT_URL. Set the environment variable or provide url parameter.")

    if is_invalid_qdrant_url(target_url):
        raise ValueError(
            f"Invalid QDRANT_URL '{target_url}': This is the marketing website, not an active Qdrant cluster endpoint. "
            "Please provide a valid cluster URL (e.g. 'https://<cluster-id>.cloud.qdrant.io:6333')."
        )

    return QdrantClient(
        url=target_url.strip(),
        api_key=target_key.strip() if target_key else None,
        timeout=timeout,
    )


def prepare_chunk_payload(chunk: Dict[str, Any], corpus_fingerprint: str) -> Dict[str, Any]:
    """Prepare payload dictionary for a chunk including ISO and RFC 3339 datetime fields."""
    eff_from = chunk["effective_from"]
    eff_to = chunk.get("effective_to") or "2099-12-31"

    eff_from_rfc3339 = f"{eff_from}T00:00:00Z"
    eff_to_rfc3339 = f"{eff_to}T23:59:59Z"

    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "clause_id": chunk["clause_id"],
        "title": chunk["title"],
        "department": chunk["department"],
        "version": chunk["version"],
        "effective_from": eff_from,
        "effective_to": eff_to,
        "effective_from_rfc3339": eff_from_rfc3339,
        "effective_to_rfc3339": eff_to_rfc3339,
        "allowed_roles": list(chunk["allowed_roles"]),
        "source_file": chunk["source_file"],
        "source_page": int(chunk["source_page"]),
        "content_hash": chunk["content_hash"],
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_model": EMBEDDING_MODEL,
    }


def ensure_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    expected_fingerprint: Optional[str] = None,
) -> None:
    """Ensure the target collection exists with compatible vector and indexing configuration.

    Raises:
        ValueError: If an existing collection has incompatible vector dimensions, distance,
            or conflicting corpus fingerprint.
    """
    collections = client.get_collections().collections
    existing_names = {col.name for col in collections}

    if collection_name in existing_names:
        col_info = client.get_collection(collection_name)
        v_config = col_info.config.params.vectors

        if isinstance(v_config, dict):
            size = v_config.get("size")
            distance = v_config.get("distance")
        else:
            size = getattr(v_config, "size", None)
            distance = getattr(v_config, "distance", None)

        if size != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Collection '{collection_name}' has vector dimension {size}; expected {EMBEDDING_DIMENSION}."
            )

        if distance not in (models.Distance.COSINE, "Cosine", "cosine"):
            raise ValueError(
                f"Collection '{collection_name}' has distance metric {distance}; expected Cosine."
            )

        if expected_fingerprint and col_info.points_count and col_info.points_count > 0:
            sample_points, _ = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
            )
            if sample_points and sample_points[0].payload:
                stored_fp = sample_points[0].payload.get("corpus_fingerprint")
                if stored_fp and stored_fp != expected_fingerprint:
                    raise ValueError(
                        f"Collection '{collection_name}' contains incompatible corpus fingerprint '{stored_fp}'; "
                        f"expected '{expected_fingerprint}'."
                    )
        return

    # Create new collection with approved parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    ensure_payload_indexes(client, collection_name)


def build_qdrant_filter(
    role: str,
    query_date: Union[str, datetime.date],
    corpus_fingerprint: str,
) -> models.Filter:
    """Build server-side Qdrant filter for role, corpus fingerprint, and inclusive dates."""
    if not corpus_fingerprint or not corpus_fingerprint.strip():
        raise ValueError("Missing required corpus_fingerprint for Qdrant filtering.")
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    q_date = parse_query_date(query_date)
    query_dt_str = f"{q_date.isoformat()}T12:00:00Z"

    return models.Filter(
        must=[
            models.FieldCondition(
                key="allowed_roles",
                match=models.MatchValue(value=role),
            ),
            models.FieldCondition(
                key="corpus_fingerprint",
                match=models.MatchValue(value=corpus_fingerprint.strip()),
            ),
            models.FieldCondition(
                key="effective_from_rfc3339",
                range=models.DatetimeRange(
                    lte=query_dt_str,
                ),
            ),
            models.FieldCondition(
                key="effective_to_rfc3339",
                range=models.DatetimeRange(
                    gte=query_dt_str,
                ),
            ),
        ]
    )


def validate_existing_collection_before_sync(
    client: QdrantClient,
    collection_name: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    expected_fingerprint: str,
) -> Set[str]:
    """Inspect all existing points in an existing collection before any upsert.

    Returns set of verified existing chunk_ids.
    Raises ValueError if any point is corrupted, invalid, unknown, or duplicate.
    """
    all_records: List[Any] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_records.extend(records)
        if next_offset is None or not records:
            break
        offset = next_offset

    seen_cids: Set[str] = set()
    for point in all_records:
        if not point.payload:
            raise ValueError(f"Existing point '{point.id}' has empty payload.")

        cid = point.payload.get("chunk_id")
        if not cid:
            raise ValueError(f"Existing point '{point.id}' missing chunk_id in payload.")

        if cid in seen_cids:
            raise ValueError(f"Existing collection contains duplicate point for chunk '{cid}'.")
        seen_cids.add(cid)

        if cid not in chunks_by_id:
            raise ValueError(f"Existing collection contains unknown chunk '{cid}'.")

        expected_chunk = chunks_by_id[cid]
        expected_pid = get_point_id(cid)
        if str(point.id) != expected_pid:
            raise ValueError(
                f"Existing point for chunk '{cid}' has unexpected ID '{point.id}'; expected '{expected_pid}'."
            )

        stored_fp = point.payload.get("corpus_fingerprint")
        if stored_fp != expected_fingerprint:
            raise ValueError(
                f"Existing point '{cid}' has incompatible corpus_fingerprint '{stored_fp}'; expected '{expected_fingerprint}'."
            )

        stored_model = point.payload.get("embedding_model")
        if stored_model != EMBEDDING_MODEL:
            raise ValueError(
                f"Existing point '{cid}' has incompatible embedding_model '{stored_model}'; expected '{EMBEDDING_MODEL}'."
            )

        for field in (
            "doc_id",
            "clause_id",
            "title",
            "department",
            "version",
            "effective_from",
            "effective_to",
            "source_file",
            "source_page",
            "content_hash",
        ):
            stored_val = point.payload.get(field)
            expected_val = expected_chunk.get(field)
            if stored_val != expected_val:
                raise ValueError(
                    f"Existing point '{cid}' has mismatched field '{field}': stored '{stored_val}', expected '{expected_val}'."
                )

        stored_roles = sorted(point.payload.get("allowed_roles") or [])
        expected_roles = sorted(expected_chunk.get("allowed_roles") or [])
        if stored_roles != expected_roles:
            raise ValueError(
                f"Existing point '{cid}' has mismatched allowed_roles: stored {stored_roles}, expected {expected_roles}."
            )

        expected_from_rfc = f"{expected_chunk['effective_from']}T00:00:00Z"
        expected_to_rfc = f"{expected_chunk.get('effective_to') or '2099-12-31'}T23:59:59Z"
        if point.payload.get("effective_from_rfc3339") != expected_from_rfc:
            raise ValueError(
                f"Existing point '{cid}' has mismatched effective_from_rfc3339: stored '{point.payload.get('effective_from_rfc3339')}', expected '{expected_from_rfc}'."
            )
        if point.payload.get("effective_to_rfc3339") != expected_to_rfc:
            raise ValueError(
                f"Existing point '{cid}' has mismatched effective_to_rfc3339: stored '{point.payload.get('effective_to_rfc3339')}', expected '{expected_to_rfc}'."
            )

    return seen_cids


def sync_corpus(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    embedder: Optional[BGEEmbedder] = None,
    base_dir: Optional[Union[Path, str]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Synchronize the 64 approved policy chunks into Qdrant collection.

    Verifies source/corpus integrity, generates 384-d normalized embeddings,
    validates existing collection compatibility across all existing points,
    uploads points idempotently, and runs full verification.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    base = Path(base_dir) if base_dir else Path.cwd()

    # 1. Validate corpus integrity before any cloud upload
    validate_corpus(base)
    chunks = load_validated_corpus(base)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    manifest_path = base / "data" / "processed" / "corpus_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    corpus_fp = manifest["processed_fingerprint"]

    # 2. Check compatibility of existing collection before uploading
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        col_info = client.get_collection(collection_name)
        v_config = col_info.config.params.vectors
        if isinstance(v_config, dict):
            size = v_config.get("size")
            distance = v_config.get("distance")
        else:
            size = getattr(v_config, "size", None)
            distance = getattr(v_config, "distance", None)

        if size != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Existing collection '{collection_name}' has vector dimension {size}; expected {EMBEDDING_DIMENSION}."
            )
        if distance not in (models.Distance.COSINE, "Cosine", "cosine"):
            raise ValueError(
                f"Existing collection '{collection_name}' has distance metric {distance}; expected Cosine."
            )

        # For real server: ensure/create missing indexes with strict type check
        ensure_payload_indexes(client, collection_name)

        # Validate EVERY existing point before ANY upsert!
        existing_cids = validate_existing_collection_before_sync(
            client=client,
            collection_name=collection_name,
            chunks_by_id=chunks_by_id,
            expected_fingerprint=corpus_fp,
        )
    else:
        ensure_collection(client, collection_name=collection_name, expected_fingerprint=corpus_fp)
        existing_cids = set()

    # Determine chunks that need uploading (chunks not already verified in collection)
    chunks_to_upsert = [c for c in chunks if c["chunk_id"] not in existing_cids]

    if chunks_to_upsert:
        active_embedder = embedder or BGEEmbedder()
        texts_to_embed = [c["text"] for c in chunks_to_upsert]
        vectors = active_embedder.embed_texts(texts_to_embed)

        if len(vectors) != len(chunks_to_upsert):
            raise ValueError(
                f"Generated {len(vectors)} embeddings for {len(chunks_to_upsert)} chunks to upload. "
                "Vector count must exactly match chunk count."
            )

        points: List[models.PointStruct] = []
        for chunk, vector in zip(chunks_to_upsert, vectors):
            pid = get_point_id(chunk["chunk_id"])
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(f"Vector for {chunk['chunk_id']} has dimension {len(vector)}; expected 384.")
            if not all(math.isfinite(x) for x in vector):
                raise ValueError(f"Vector for {chunk['chunk_id']} contains non-finite values.")

            payload = prepare_chunk_payload(chunk, corpus_fingerprint=corpus_fp)
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vector,
                    payload=payload,
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

    # 3. Verify collection after sync
    verify_result = verify_collection(
        client=client,
        collection_name=collection_name,
        expected_fingerprint=corpus_fp,
        base_dir=base,
    )
    if verify_result.get("status") != "verified":
        raise ValueError(f"Collection verification failed after sync: {verify_result}")

    info = client.get_collection(collection_name)
    count = info.points_count

    return {
        "collection_name": collection_name,
        "points_synced": len(chunks_to_upsert),
        "total_points_in_collection": count,
        "corpus_fingerprint": corpus_fp,
        "embedding_model": EMBEDDING_MODEL,
        "status": "synchronized",
    }


def verify_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    expected_fingerprint: Optional[str] = None,
    base_dir: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Verify that an existing Qdrant collection matches configuration and approved chunks.

    Performs read-only validation:
    1. Collection exists.
    2. Vector dimension is 384.
    3. Distance metric is Cosine.
    4. Exactly 64 unique points exist.
    5. Every point has its deterministic UUIDv5 point ID.
    6. No unexpected or duplicate chunk IDs exist.
    7. Every point has the expected corpus fingerprint.
    8. Every point has the expected embedding-model identifier.
    9. Stored payload matches canonical processed chunk across all 15 fields.
    10. Verifies required payload indexes on real servers (strictly read-only).
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    if expected_fingerprint is None:
        manifest_path = base / "data" / "processed" / "corpus_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_fingerprint = manifest["processed_fingerprint"]

    # 1. Verify collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant cluster.")

    col_info = client.get_collection(collection_name)
    v_config = col_info.config.params.vectors

    if isinstance(v_config, dict):
        size = v_config.get("size")
        distance = v_config.get("distance")
    else:
        size = getattr(v_config, "size", None)
        distance = getattr(v_config, "distance", None)

    # 2. Vector dimension
    if size != EMBEDDING_DIMENSION:
        raise ValueError(f"Vector size {size} != expected {EMBEDDING_DIMENSION}")

    # 3. Distance metric
    if distance not in (models.Distance.COSINE, "Cosine", "cosine"):
        raise ValueError(f"Distance metric {distance} != Cosine")

    # 4. Scroll all points with pagination to ensure all 64 are retrieved
    all_records: List[Any] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_records.extend(records)
        if next_offset is None or not records:
            break
        offset = next_offset

    if len(all_records) != 64:
        raise ValueError(f"Collection '{collection_name}' has {len(all_records)} points; expected exactly 64.")

    chunks = load_validated_corpus(base)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    expected_cids = set(chunks_by_id.keys())

    stored_chunk_ids: Set[str] = set()

    for record in all_records:
        if not record.payload:
            raise ValueError(f"Point {record.id} has empty payload.")

        cid = record.payload.get("chunk_id")
        if not cid:
            raise ValueError(f"Point {record.id} missing chunk_id in payload.")

        if cid in stored_chunk_ids:
            raise ValueError(f"Duplicate chunk ID '{cid}' found in collection points.")
        stored_chunk_ids.add(cid)

        # 5. Deterministic UUIDv5 ID
        expected_point_id = get_point_id(cid)
        if str(record.id) != expected_point_id:
            raise ValueError(
                f"Point ID mismatch for chunk '{cid}': stored '{record.id}', expected '{expected_point_id}'."
            )

        # 6. Corpus fingerprint
        stored_fp = record.payload.get("corpus_fingerprint")
        if stored_fp != expected_fingerprint:
            raise ValueError(
                f"Corpus fingerprint mismatch for chunk '{cid}': stored '{stored_fp}', expected '{expected_fingerprint}'."
            )

        # 7. Embedding model identifier
        stored_model = record.payload.get("embedding_model")
        if stored_model != EMBEDDING_MODEL:
            raise ValueError(
                f"Embedding model mismatch for chunk '{cid}': stored '{stored_model}', expected '{EMBEDDING_MODEL}'."
            )

        # 8. Compare all 15 chunk payload fields against canonical metadata
        canonical = chunks_by_id[cid]
        for field in (
            "doc_id",
            "clause_id",
            "title",
            "department",
            "version",
            "effective_from",
            "source_file",
            "content_hash",
        ):
            stored_val = record.payload.get(field)
            expected_val = canonical.get(field)
            if stored_val != expected_val:
                raise ValueError(
                    f"Payload mismatch for chunk '{cid}', field '{field}': stored '{stored_val}', expected '{expected_val}'."
                )

        if int(record.payload.get("source_page", -1)) != int(canonical["source_page"]):
            raise ValueError(f"Payload source_page mismatch for chunk '{cid}'.")

        eff_to_stored = record.payload.get("effective_to")
        eff_to_canon = canonical.get("effective_to") or "2099-12-31"
        if (eff_to_stored or "2099-12-31") != eff_to_canon:
            raise ValueError(
                f"Payload effective_to mismatch for chunk '{cid}': stored '{eff_to_stored}', expected '{eff_to_canon}'."
            )

        exp_rfc_from = f"{canonical['effective_from']}T00:00:00Z"
        exp_rfc_to = f"{eff_to_canon}T23:59:59Z"
        if record.payload.get("effective_from_rfc3339") != exp_rfc_from:
            raise ValueError(f"Payload effective_from_rfc3339 mismatch for chunk '{cid}'.")
        if record.payload.get("effective_to_rfc3339") != exp_rfc_to:
            raise ValueError(f"Payload effective_to_rfc3339 mismatch for chunk '{cid}'.")

        stored_roles = sorted(record.payload.get("allowed_roles", []))
        canonical_roles = sorted(canonical.get("allowed_roles", []))
        if stored_roles != canonical_roles:
            raise ValueError(
                f"Payload allowed_roles mismatch for chunk '{cid}': stored {stored_roles}, expected {canonical_roles}."
            )

    # 9. Exact chunk ID set match
    if stored_chunk_ids != expected_cids:
        missing = expected_cids - stored_chunk_ids
        extra = stored_chunk_ids - expected_cids
        raise ValueError(f"Mismatch in stored chunk IDs. Missing: {missing}, Extra: {extra}")

    # 10. Check payload indexes strictly read-only
    index_check = check_payload_indexes(client, collection_name)

    return {
        "collection_name": collection_name,
        "point_count": len(stored_chunk_ids),
        "points_count": len(stored_chunk_ids),
        "corpus_fingerprint": expected_fingerprint,
        "vector_dimension": size,
        "vector_size": size,
        "distance": distance,
        "payload_indexes": index_check,
        "status": "verified",
    }


def search_qdrant(
    client: QdrantClient,
    query_vector: List[float],
    role: str,
    query_date: Union[str, datetime.date],
    corpus_fingerprint: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = DEFAULT_CANDIDATE_DEPTH,
) -> List[Tuple[str, float]]:
    """Execute dense vector search against Qdrant with server-side role and date filtering.

    Applies mandatory server-side filter and post-query verification safeguards.

    Returns:
        List of (chunk_id, similarity_score) tuples sorted by descending similarity.
    """
    if not corpus_fingerprint or not corpus_fingerprint.strip():
        raise ValueError("corpus_fingerprint must be a non-empty string.")
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"Invalid top_k={top_k}. Must be a positive integer.")

    if not isinstance(query_vector, (list, tuple)) or len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"query_vector must have dimension {EMBEDDING_DIMENSION}, got {len(query_vector) if hasattr(query_vector, '__len__') else type(query_vector)}"
        )
    if not all(math.isfinite(x) for x in query_vector):
        raise ValueError("query_vector contains non-finite values.")

    validated_date = parse_query_date(query_date)

    q_filter = build_qdrant_filter(
        role=role,
        query_date=validated_date,
        corpus_fingerprint=corpus_fingerprint.strip(),
    )

    query_res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
    )

    results: List[Tuple[str, float]] = []
    q_str = validated_date.isoformat()
    expected_fp = corpus_fingerprint.strip()

    for point in query_res.points:
        if point.payload and "chunk_id" in point.payload:
            payload = point.payload
            # Verify returned payload fingerprint matches requested fingerprint before accepting result
            pt_fp = payload.get("corpus_fingerprint")
            if pt_fp != expected_fp:
                continue

            # Client-side safeguard against server-side filtering discrepancies
            allowed = payload.get("allowed_roles", [])
            if role not in allowed:
                continue
            eff_from = payload.get("effective_from", "")
            eff_to = payload.get("effective_to") or "2099-12-31"
            if not (eff_from <= q_str <= eff_to):
                continue
            results.append((payload["chunk_id"], float(point.score)))

    return results


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for sync and verify commands."""
    parser = argparse.ArgumentParser(description="RuleLens Qdrant Cloud Storage Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sync", help="Synchronize the 64 policy chunks to Qdrant collection")
    subparsers.add_parser("verify", help="Verify the Qdrant collection configuration and point data")

    parsed_args = parser.parse_args(args)

    try:
        client = get_qdrant_client()
    except ValueError:
        print("Error: Qdrant client initialization failed. Please check configuration.", file=sys.stderr)
        sys.exit(1)

    try:
        if parsed_args.command == "sync":
            print(f"Synchronizing corpus to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
            res = sync_corpus(client)
            print("Sync complete:", res)
        elif parsed_args.command == "verify":
            print(f"Verifying Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
            res = verify_collection(client)
            print("Verification complete:", res)
    except Exception:
        print("Operation failed: An unexpected infrastructure error occurred during Qdrant operation.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
