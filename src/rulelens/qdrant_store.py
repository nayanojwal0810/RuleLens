"""Qdrant Cloud vector database integration for RuleLens.

Provides deterministic collection synchronization, server-side role and temporal
filtering, collection verification, and dense vector search over the 64 policy chunks.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from qdrant_client import QdrantClient, models

from rulelens.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    get_env_var,
)
from rulelens.ingestion import validate_corpus
from rulelens.retrieval import (
    DEFAULT_CANDIDATE_DEPTH,
    EMBEDDING_DIMENSION,
    BGEEmbedder,
    load_validated_corpus,
    parse_query_date,
)

# Deterministic namespace for chunk UUIDv5 point IDs
POINT_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "rulelens.asterline.internal")


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
        ValueError: If QDRANT_URL is not set or empty.
    """
    target_url = url or get_env_var(QDRANT_URL)
    target_key = api_key or get_env_var(QDRANT_API_KEY)

    if not target_url or not target_url.strip():
        raise ValueError("Missing required QDRANT_URL. Set the environment variable or provide url parameter.")

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

        # Inspect vector configuration
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

        # Check existing fingerprint compatibility if collection already has points
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

    # Create payload indexes for server-side filtering
    try:
        client.create_payload_index(collection_name, "allowed_roles", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "corpus_fingerprint", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "effective_from_rfc3339", models.PayloadSchemaType.DATETIME)
        client.create_payload_index(collection_name, "effective_to_rfc3339", models.PayloadSchemaType.DATETIME)
    except Exception:
        # Some local/in-memory test drivers warn or ignore payload indexes
        pass


def build_qdrant_filter(
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
    corpus_fingerprint: Optional[str] = None,
) -> models.Filter:
    """Build server-side Qdrant filter for role, corpus fingerprint, and inclusive dates."""
    q_date = parse_query_date(query_date)
    query_dt_str = f"{q_date.isoformat()}T12:00:00Z"

    must_conditions: List[models.FieldCondition] = [
        models.FieldCondition(
            key="allowed_roles",
            match=models.MatchValue(value=role),
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

    if corpus_fingerprint:
        must_conditions.append(
            models.FieldCondition(
                key="corpus_fingerprint",
                match=models.MatchValue(value=corpus_fingerprint),
            )
        )

    return models.Filter(must=must_conditions)


def sync_corpus(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    embedder: Optional[BGEEmbedder] = None,
    base_dir: Optional[Union[Path, str]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Synchronize the 64 approved policy chunks into Qdrant collection.

    Verifies source/corpus integrity, generates 384-d normalized embeddings,
    and uploads points idempotently.
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    # 1. Validate corpus integrity before any cloud upload
    validate_corpus(base)
    chunks = load_validated_corpus(base)

    import json
    with open(base / "data" / "processed" / "corpus_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    corpus_fp = manifest["processed_fingerprint"]

    # 2. Embed all 64 document passages without query instructions
    active_embedder = embedder or BGEEmbedder()
    texts = [c["text"] for c in chunks]
    vectors = active_embedder.embed_texts(texts)

    if len(vectors) != 64:
        raise ValueError(f"Expected 64 vectors, got {len(vectors)}")

    # 3. Ensure collection exists and is compatible
    ensure_collection(client, collection_name=collection_name, expected_fingerprint=corpus_fp)

    # 4. Construct point structs with deterministic UUIDv5 IDs
    points: List[models.PointStruct] = []
    point_ids_set: Set[str] = set()

    for chunk, vector in zip(chunks, vectors):
        pid = get_point_id(chunk["chunk_id"])
        point_ids_set.add(pid)

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

    if len(point_ids_set) != 64:
        raise ValueError(f"Expected 64 unique point IDs, got {len(point_ids_set)}")

    # 5. Idempotent upsert in batches
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )

    # 6. Verify point count after sync
    info = client.get_collection(collection_name)
    count = info.points_count

    return {
        "collection_name": collection_name,
        "points_synced": len(points),
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

    Does not modify data.
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    if expected_fingerprint is None:
        import json
        manifest_path = base / "data" / "processed" / "corpus_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_fingerprint = manifest["processed_fingerprint"]

    # Verify collection exists
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

    if size != EMBEDDING_DIMENSION:
        raise ValueError(f"Vector size {size} != expected {EMBEDDING_DIMENSION}")

    if distance not in (models.Distance.COSINE, "Cosine", "cosine"):
        raise ValueError(f"Distance metric {distance} != Cosine")

    if col_info.points_count != 64:
        raise ValueError(f"Collection '{collection_name}' has {col_info.points_count} points; expected exactly 64.")

    # Scroll all points and check stored chunk IDs and fingerprint
    scroll_records, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    stored_chunk_ids: Set[str] = set()
    for record in scroll_records:
        if not record.payload:
            raise ValueError(f"Point {record.id} has empty payload.")
        cid = record.payload.get("chunk_id")
        if not cid:
            raise ValueError(f"Point {record.id} missing chunk_id in payload.")
        stored_chunk_ids.add(cid)

        fp = record.payload.get("corpus_fingerprint")
        if fp != expected_fingerprint:
            raise ValueError(f"Point {record.id} has fingerprint '{fp}'; expected '{expected_fingerprint}'.")

    chunks = load_validated_corpus(base)
    expected_cids = {c["chunk_id"] for c in chunks}

    if stored_chunk_ids != expected_cids:
        missing = expected_cids - stored_chunk_ids
        extra = stored_chunk_ids - expected_cids
        raise ValueError(f"Mismatch in stored chunk IDs. Missing: {missing}, Extra: {extra}")

    return {
        "collection_name": collection_name,
        "point_count": len(stored_chunk_ids),
        "points_count": len(stored_chunk_ids),
        "corpus_fingerprint": expected_fingerprint,
        "vector_dimension": size,
        "vector_size": size,
        "distance": distance,
        "status": "verified",
    }


def search_qdrant(
    client: QdrantClient,
    query_vector: List[float],
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
    corpus_fingerprint: Optional[str] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = DEFAULT_CANDIDATE_DEPTH,
) -> List[Tuple[str, float]]:
    """Execute dense vector search against Qdrant with server-side role and date filtering.

    Returns:
        List of (chunk_id, similarity_score) tuples sorted by descending similarity.
    """
    q_filter = build_qdrant_filter(
        role=role,
        query_date=query_date,
        corpus_fingerprint=corpus_fingerprint,
    )

    query_res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
    )

    results: List[Tuple[str, float]] = []
    for point in query_res.points:
        if point.payload and "chunk_id" in point.payload:
            results.append((point.payload["chunk_id"], float(point.score)))

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
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
    except Exception as exc:
        print(f"Operation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
