"""Offline deterministic tests for RuleLens Stage D2 Qdrant vector database integration.

Verifies:
- Deterministic UUIDv5 point ID generation across all 64 corpus chunks.
- Missing QDRANT_URL credential validation.
- Payload preparation including RFC 3339 timestamps for server-side datetime filtering.
- Collection schema management (size=384, Distance.COSINE, payload indexes).
- Incompatible collection detection (vector size, distance metric, corpus fingerprint).
- Idempotent corpus synchronization (64 points) using in-memory QdrantClient.
- Collection verification and integrity checks.
- Server-side role and temporal filtering in search_qdrant.
- Strict error propagation without silent fallback.
- CLI entrypoints for sync and verify.
"""

from __future__ import annotations

import datetime
import socket
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient, models

from rulelens.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
)
from rulelens.dataset import ROLES
from rulelens.qdrant_store import (
    POINT_NAMESPACE,
    ensure_collection,
    get_point_id,
    get_qdrant_client,
    main,
    prepare_chunk_payload,
    search_qdrant,
    sync_corpus,
    verify_collection,
)
from rulelens.retrieval import (
    EMBEDDING_DIMENSION,
    BGEEmbedder,
    load_validated_corpus,
)


@pytest.fixture(autouse=True)
def guard_network(monkeypatch):
    """Enforce that unit tests make zero external network calls."""

    def guarded_connect(*args, **kwargs):
        raise AssertionError("External network connection attempted during offline unit tests")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_chunks(repo_root: Path) -> List[Dict[str, Any]]:
    """Load the 64 validated corpus chunks."""
    return load_validated_corpus(repo_root)


class FakeSentenceTransformerModel:
    """Deterministic fake model for testing BGEEmbedder without downloading weights."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def encode(
        self,
        sentences: List[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        import hashlib
        import numpy as np

        embeddings = []
        for s in sentences:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            vec = [(b / 255.0) - 0.5 for b in h[:16]]
            full_vec = (vec * (self.dimension // len(vec) + 1))[: self.dimension]
            arr = np.array(full_vec, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            embeddings.append(arr)
        return np.array(embeddings)


# ==============================================================================
# 1. Point IDs & Payload Preparation
# ==============================================================================


def test_get_point_id_deterministic_and_unique(sample_chunks: List[Dict[str, Any]]):
    """get_point_id must generate deterministic UUIDv5 strings that are distinct for all 64 chunks."""
    point_ids = set()
    for chunk in sample_chunks:
        cid = chunk["chunk_id"]
        pid1 = get_point_id(cid)
        pid2 = get_point_id(cid)
        # Deterministic
        assert pid1 == pid2
        # Valid UUID
        parsed_uuid = uuid.UUID(pid1)
        assert parsed_uuid.version == 5
        point_ids.add(pid1)

    assert len(point_ids) == 64


def test_prepare_chunk_payload_format(sample_chunks: List[Dict[str, Any]]):
    """Payload preparation must produce all required fields including RFC 3339 timestamps."""
    sample = sample_chunks[0]
    payload = prepare_chunk_payload(sample, corpus_fingerprint="test_fp_123")

    assert payload["chunk_id"] == sample["chunk_id"]
    assert payload["doc_id"] == sample["doc_id"]
    assert payload["clause_id"] == sample["clause_id"]
    assert payload["title"] == sample["title"]
    assert payload["department"] == sample["department"]
    assert payload["version"] == sample["version"]
    assert payload["effective_from"] == sample["effective_from"]
    assert payload["effective_from_rfc3339"] == f"{sample['effective_from']}T00:00:00Z"
    assert "effective_to_rfc3339" in payload
    assert payload["allowed_roles"] == list(sample["allowed_roles"])
    assert payload["source_file"] == sample["source_file"]
    assert payload["source_page"] == int(sample["source_page"])
    assert payload["content_hash"] == sample["content_hash"]
    assert payload["corpus_fingerprint"] == "test_fp_123"


# ==============================================================================
# 2. Client Initialization & Error Handling
# ==============================================================================


def test_get_qdrant_client_missing_url(monkeypatch):
    """get_qdrant_client must raise ValueError when QDRANT_URL is not provided or empty."""
    monkeypatch.delenv(QDRANT_URL, raising=False)
    with pytest.raises(ValueError, match="Missing required QDRANT_URL"):
        get_qdrant_client(url=None)

    with pytest.raises(ValueError, match="Missing required QDRANT_URL"):
        get_qdrant_client(url="   ")


def test_get_qdrant_client_rejects_marketing_url():
    """get_qdrant_client must reject https://qdrant.io marketing URL."""
    with pytest.raises(ValueError, match="marketing website"):
        get_qdrant_client(url="https://qdrant.io")

    with pytest.raises(ValueError, match="marketing website"):
        get_qdrant_client(url="http://qdrant.io/")


def test_get_qdrant_client_initialization():
    """get_qdrant_client must instantiate QdrantClient with provided URL and key."""
    client = get_qdrant_client(url="http://localhost:6333", api_key="secret-key")
    assert isinstance(client, QdrantClient)


# ==============================================================================
# 3. Collection Management & Incompatible Configuration Detection
# ==============================================================================


def test_ensure_collection_in_memory():
    """ensure_collection must create the collection with size=384 and Cosine distance."""
    client = QdrantClient(":memory:")
    col_name = "test_collection"

    ensure_collection(client, collection_name=col_name)

    col_info = client.get_collection(col_name)
    assert col_info.config.params.vectors.size == 384
    assert col_info.config.params.vectors.distance == models.Distance.COSINE

    # Calling again must be idempotent
    ensure_collection(client, collection_name=col_name)


def test_ensure_collection_rejects_incompatible_vector_size():
    """ensure_collection must raise ValueError if existing collection has wrong vector size."""
    client = QdrantClient(":memory:")
    col_name = "bad_size_collection"

    client.create_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="expected 384"):
        ensure_collection(client, collection_name=col_name)


def test_ensure_collection_rejects_incompatible_distance_metric():
    """ensure_collection must raise ValueError if existing collection has wrong distance metric."""
    client = QdrantClient(":memory:")
    col_name = "bad_dist_collection"

    client.create_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.DOT),
    )

    with pytest.raises(ValueError, match="expected Cosine"):
        ensure_collection(client, collection_name=col_name)


def test_ensure_collection_rejects_conflicting_corpus_fingerprint():
    """ensure_collection must raise ValueError if existing populated collection has conflicting fingerprint."""
    client = QdrantClient(":memory:")
    col_name = "conflicting_fp_collection"

    client.create_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    # Upsert a dummy point with an old fingerprint
    client.upsert(
        collection_name=col_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0] * 384,
                payload={"corpus_fingerprint": "old_fp_value"},
            )
        ],
    )

    with pytest.raises(ValueError, match="incompatible corpus fingerprint"):
        ensure_collection(client, collection_name=col_name, expected_fingerprint="new_fp_value")


# ==============================================================================
# 4. Corpus Synchronization & Verification (In-Memory)
# ==============================================================================


def test_sync_corpus_and_verify_collection_in_memory(repo_root: Path):
    """sync_corpus must populate 64 points and verify_collection must succeed."""
    client = QdrantClient(":memory:")
    col_name = "rulelens_test_sync"
    fake_embedder = BGEEmbedder(model_instance=FakeSentenceTransformerModel(dimension=384))

    # Initial sync
    sync_result = sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
        batch_size=32,
    )

    assert sync_result["points_synced"] == 64
    assert sync_result["total_points_in_collection"] == 64
    assert sync_result["collection_name"] == col_name

    # Verification
    verify_result = verify_collection(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
    )
    assert verify_result["status"] == "verified"
    assert verify_result["points_count"] == 64
    assert verify_result["vector_size"] == 384
    assert verify_result["distance"] == models.Distance.COSINE

    # Second sync must be idempotent (count stays 64)
    sync_result_2 = sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
        batch_size=32,
    )
    assert sync_result_2["total_points_in_collection"] == 64


def test_verify_collection_detects_incomplete_points(repo_root: Path):
    """verify_collection must raise ValueError if points count != 64."""
    client = QdrantClient(":memory:")
    col_name = "incomplete_collection"

    client.create_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    # Only 1 point
    client.upsert(
        collection_name=col_name,
        points=[models.PointStruct(id=str(uuid.uuid4()), vector=[0.0] * 384, payload={})],
    )

    with pytest.raises(ValueError, match="expected exactly 64"):
        verify_collection(client=client, base_dir=repo_root, collection_name=col_name)


# ==============================================================================
# 5. Server-Side Filtering (Roles & Dates)
# ==============================================================================


def test_search_qdrant_server_side_filtering(repo_root: Path, sample_chunks: List[Dict[str, Any]]):
    """search_qdrant must filter by role and temporal validity server-side."""
    client = QdrantClient(":memory:")
    col_name = "test_search_filters"
    fake_embedder = BGEEmbedder(model_instance=FakeSentenceTransformerModel(dimension=384))

    sync_result = sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
    )
    fp = sync_result["corpus_fingerprint"]

    query_vec = fake_embedder.embed_query("travel exceptions approval process")

    # 1. Role: Employee must NOT see DOC-TRV-004_TRV-E01 (Travel Exceptions is manager+ only)
    employee_results = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    employee_cids = {cid for cid, score in employee_results}
    assert "DOC-TRV-004_TRV-E01" not in employee_cids

    # 2. Role: Manager MUST be allowed to retrieve DOC-TRV-004_TRV-E01
    manager_results = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="manager",
        query_date="2026-11-22",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    manager_cids = {cid for cid, score in manager_results}
    assert "DOC-TRV-004_TRV-E01" in manager_cids

    # 3. Temporal: Querying historical date 2026-03-15 must not retrieve chunks effective only after 2026-07-01
    v2_query_vec = fake_embedder.embed_query("Asterline travel accommodation standards")
    hist_results = search_qdrant(
        client=client,
        query_vector=v2_query_vec,
        role="employee",
        query_date="2026-03-15",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    # Check effective dates of all returned results against corpus
    lookup = {c["chunk_id"]: c for c in sample_chunks}
    for cid, score in hist_results:
        chunk = lookup[cid]
        eff_from = chunk["effective_from"]
        eff_to = chunk.get("effective_to") or "2099-12-31"
        assert eff_from <= "2026-03-15"
        assert "2026-03-15" <= eff_to


def test_search_qdrant_mandatory_auditor_filter(repo_root: Path):
    """Mandatory test case: Auditor cannot retrieve Employee Transfers (DOC-PCO-003)."""
    client = QdrantClient(":memory:")
    col_name = "test_auditor_filter"
    fake_embedder = BGEEmbedder(model_instance=FakeSentenceTransformerModel(dimension=384))

    sync_result = sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
    )
    fp = sync_result["corpus_fingerprint"]

    query_vec = fake_embedder.embed_query("employee transfer relocation procedures")
    results = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="auditor",
        query_date="2026-11-22",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    result_cids = {cid for cid, score in results}
    # DOC-PCO-003 is Employee Transfers (accessible to employee, manager, hr, but NOT auditor)
    for cid in result_cids:
        assert not cid.startswith("DOC-PCO-003"), f"Auditor must not access DOC-PCO-003: {cid}"


def test_search_qdrant_mandatory_date_boundaries(repo_root: Path):
    """Mandatory test cases:
    - 2026-06-30: Travel Policy v1 is eligible; v2 is not.
    - 2026-07-01: Travel Policy v2 is eligible; v1 is not.
    - 2026-11-22: Expired Travel Policy v1 must not appear.
    """
    client = QdrantClient(":memory:")
    col_name = "test_date_boundaries"
    fake_embedder = BGEEmbedder(model_instance=FakeSentenceTransformerModel(dimension=384))

    sync_result = sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
    )
    fp = sync_result["corpus_fingerprint"]
    query_vec = fake_embedder.embed_query("business travel expense allowance")

    # Date 1: 2026-06-30
    res_june = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="employee",
        query_date="2026-06-30",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    cids_june = {cid for cid, score in res_june}
    assert any(cid.startswith("DOC-TRV-001") for cid in cids_june), "v1 must be eligible on 2026-06-30"
    assert not any(cid.startswith("DOC-TRV-002") for cid in cids_june), "v2 must not be eligible on 2026-06-30"

    # Date 2: 2026-07-01
    res_july = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="employee",
        query_date="2026-07-01",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    cids_july = {cid for cid, score in res_july}
    assert any(cid.startswith("DOC-TRV-002") for cid in cids_july), "v2 must be eligible on 2026-07-01"
    assert not any(cid.startswith("DOC-TRV-001") for cid in cids_july), "v1 must not be eligible on 2026-07-01"

    # Date 3: 2026-11-22
    res_nov = search_qdrant(
        client=client,
        query_vector=query_vec,
        role="employee",
        query_date="2026-11-22",
        corpus_fingerprint=fp,
        collection_name=col_name,
        top_k=64,
    )
    cids_nov = {cid for cid, score in res_nov}
    assert any(cid.startswith("DOC-TRV-002") for cid in cids_nov), "v2 must be eligible on 2026-11-22"
    assert not any(cid.startswith("DOC-TRV-001") for cid in cids_nov), "v1 must not appear on 2026-11-22"


def test_search_qdrant_rejects_empty_fingerprint():
    """search_qdrant must reject empty corpus_fingerprint with ValueError."""
    mock_client = MagicMock()
    with pytest.raises(ValueError, match="corpus_fingerprint must be a non-empty string"):
        search_qdrant(
            client=mock_client,
            query_vector=[0.1] * 384,
            role="employee",
            query_date="2026-11-22",
            corpus_fingerprint="",
        )


def test_search_qdrant_no_silent_fallback(repo_root: Path):
    """search_qdrant must propagate exceptions if Qdrant call fails (no silent fallback)."""
    mock_client = MagicMock()
    mock_client.query_points.side_effect = RuntimeError("Qdrant cluster unavailable")

    with pytest.raises(RuntimeError, match="Qdrant cluster unavailable"):
        search_qdrant(
            client=mock_client,
            query_vector=[0.1] * 384,
            role="employee",
            query_date="2026-11-22",
            corpus_fingerprint="test_fp_123",
        )


def test_verify_collection_detects_payload_field_mismatch(repo_root: Path):
    """verify_collection must raise ValueError if a point payload field is corrupted."""
    client = QdrantClient(":memory:")
    col_name = "test_tampered_payload"
    fake_embedder = BGEEmbedder(model_instance=FakeSentenceTransformerModel(dimension=384))

    sync_corpus(
        client=client,
        base_dir=repo_root,
        collection_name=col_name,
        embedder=fake_embedder,
    )

    # Tamper with point 0 payload
    points, _ = client.scroll(collection_name=col_name, limit=1, with_payload=True, with_vectors=True)
    p = points[0]
    p.payload["department"] = "CORRUPTED_DEPARTMENT"
    client.upsert(
        collection_name=col_name,
        points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)],
    )

    with pytest.raises(ValueError, match="Payload mismatch"):
        verify_collection(client=client, base_dir=repo_root, collection_name=col_name)


# ==============================================================================
# 6. CLI Entrypoints
# ==============================================================================


def test_cli_missing_credentials_fails_cleanly(monkeypatch):
    """CLI sync and verify without QDRANT_URL must return non-zero exit code with clear error message."""
    monkeypatch.delenv(QDRANT_URL, raising=False)

    with pytest.raises(SystemExit) as exc1:
        main(["sync"])
    assert exc1.value.code != 0

    with pytest.raises(SystemExit) as exc2:
        main(["verify"])
    assert exc2.value.code != 0
