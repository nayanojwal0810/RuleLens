"""Hybrid retrieval engine for RuleLens.

Combines BM25 lexical search and BGE-small dense embeddings using
Reciprocal Rank Fusion (RRF) with role-based and temporal eligibility filtering.
Supports local CPU dense retrieval and Qdrant Cloud vector search.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from rulelens.config import EMBEDDING_MODEL, GENERATION_EVIDENCE_LIMIT
from rulelens.dataset import ROLES
from rulelens.ingestion import calculate_processed_fingerprint, validate_corpus

# Default parameters
RRF_K: int = 60
DEFAULT_CANDIDATE_DEPTH: int = 20
DEFAULT_TOP_K: int = GENERATION_EVIDENCE_LIMIT  # 4
EMBEDDING_DIMENSION: int = 384

# Approved enterprise roles for validation
ALLOWED_ROLES_SET: Set[str] = set(ROLES)

# Official BGE query instruction for retrieval
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# Tokenization regex: lowercase word and number tokens
WORD_TOKEN_PATTERN: re.Pattern = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words and numbers.

    Retains numbers (e.g. '40000', '50000') and normalizes comma-formatted numbers.
    Does not use external stopword resources.
    """
    if not text or not text.strip():
        return []
    # Normalize digit grouping commas (e.g. 50,000 -> 50000) so numerical search succeeds
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return WORD_TOKEN_PATTERN.findall(normalized)


def parse_query_date(query_date: Optional[Union[str, datetime.date]] = None) -> datetime.date:
    """Parse and validate an ISO query date, defaulting to today if None.

    Raises:
        ValueError: If query_date is a malformed string, explicit empty string, or invalid type.
    """
    if query_date is None:
        return datetime.date.today()
    if isinstance(query_date, datetime.date):
        return query_date
    if isinstance(query_date, str):
        cleaned = query_date.strip()
        if not cleaned:
            raise ValueError("Malformed query date: explicit empty string provided. Only None defaults to today.")
        try:
            return datetime.date.fromisoformat(cleaned)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Malformed query date '{query_date}'. Expected ISO format YYYY-MM-DD.") from exc
    raise ValueError(f"Malformed query date '{query_date}'. Expected ISO string or datetime.date.")


def is_chunk_eligible(
    chunk: Dict[str, Any],
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
) -> bool:
    """Check whether a chunk is accessible to the specified role on the query date.

    Args:
        chunk: The chunk metadata dictionary.
        role: The requesting employee role (must be one of ROLES).
        query_date: The effective date for the policy query (defaults to current date).

    Returns:
        True if the chunk is accessible to the role and active on query_date.

    Raises:
        ValueError: If role is unknown or query_date is malformed.
    """
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    q_date = parse_query_date(query_date)

    # 1. Role access check
    allowed_roles = chunk.get("allowed_roles", [])
    if role not in allowed_roles:
        return False

    # 2. Date boundary check
    eff_from_str = chunk.get("effective_from")
    if not eff_from_str:
        return False
    try:
        eff_from = datetime.date.fromisoformat(eff_from_str)
    except ValueError:
        return False

    if q_date < eff_from:
        return False

    eff_to_str = chunk.get("effective_to")
    if eff_to_str:
        try:
            eff_to = datetime.date.fromisoformat(eff_to_str)
            if q_date > eff_to:
                return False
        except ValueError:
            return False

    return True


def filter_eligible_chunks(
    chunks: Sequence[Dict[str, Any]],
    role: str,
    query_date: Optional[Union[str, datetime.date]] = None,
) -> List[Dict[str, Any]]:
    """Filter a collection of chunks to those eligible for the given role and query date."""
    validated_date = parse_query_date(query_date)
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    return [c for c in chunks if is_chunk_eligible(c, role, validated_date)]


@dataclass(frozen=True)
class RetrievalResult:
    """Clean representation of a retrieved policy chunk with rank and citation metadata."""

    chunk_id: str
    text: str
    doc_id: str
    clause_id: str
    title: str
    department: str
    version: str
    source_file: str
    source_page: int
    effective_from: str
    effective_to: Optional[str]
    allowed_roles: List[str]
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    rrf_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Candidate:
    """Intermediate candidate result from a single retrieval method."""

    chunk_id: str
    score: float
    rank: int = 1
    chunk: Optional[Dict[str, Any]] = None


class BM25Index:
    """In-memory BM25 index over policy chunks using rank-bm25."""

    def __init__(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """Initialize BM25 index over the provided chunks."""
        self.chunks = list(chunks)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.corpus_token_sets = [set(tokens) for tokens in self.corpus_tokens]
        if self.chunks:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = DEFAULT_CANDIDATE_DEPTH) -> List[Candidate]:
        """Search the BM25 index.

        Args:
            query: The search query string.
            top_k: Maximum number of candidate results to return.

        Returns:
            Ranked Candidate list with 1-based ranks and scores.
            Empty queries, empty index, or queries with no matches return an empty list.
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"Invalid top_k={top_k}. Must be a non-negative integer.")
        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with chunks and keep only chunks containing at least one query term
        scored_pairs: List[Tuple[float, str, Dict[str, Any]]] = []
        for idx, (score, chunk) in enumerate(zip(scores, self.chunks)):
            if any(t in self.corpus_token_sets[idx] for t in query_tokens):
                scored_pairs.append((float(score), chunk["chunk_id"], chunk))

        if not scored_pairs:
            return []

        # Sort by (-score, chunk_id) for deterministic tie-breaking
        scored_pairs.sort(key=lambda item: (-item[0], item[1]))

        # Limit to top_k and assign 1-based ranks
        candidates: List[Candidate] = []
        for rank_idx, (score, chunk_id, chunk) in enumerate(scored_pairs[:top_k], start=1):
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank_idx,
                    chunk=chunk,
                )
            )
        return candidates


class BGEEmbedder:
    """Lightweight SentenceTransformers wrapper for BAAI/bge-small-en-v1.5.

    Loads the model lazily on first embedding request.
    Uses CPU by default and generates normalized 384-dimensional embeddings.
    Applies the BGE query instruction to search questions but not passage texts.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        model_instance: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = model_instance

    @property
    def model(self) -> Any:
        """Lazily load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Batch-embed a sequence of passage text strings without query instruction.

        Returns:
            List of 384-dimensional normalized float vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        dim = embeddings.shape[1] if hasattr(embeddings, "shape") and len(embeddings.shape) > 1 else len(embeddings[0])
        if dim != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {dim}")

        raw_vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else [list(vec) for vec in embeddings]
        validated_vectors: List[List[float]] = []
        for vec in raw_vectors:
            if len(vec) != EMBEDDING_DIMENSION:
                raise ValueError(f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(vec)}")
            if not all(math.isfinite(x) for x in vec):
                raise ValueError("Embedding vector contains non-finite values (NaN or Inf)")
            validated_vectors.append([float(x) for x in vec])

        return validated_vectors

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string with the BGE query instruction.

        Appends the official BGE query instruction exactly once before embedding.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only")

        clean = query.strip()
        if clean.startswith(BGE_QUERY_INSTRUCTION):
            prefixed = clean
        else:
            prefixed = f"{BGE_QUERY_INSTRUCTION}{clean}"

        return self.embed_texts([prefixed])[0]


def normalize_dense_candidates(
    raw_candidates: Sequence[Union[Candidate, Dict[str, Any], Tuple[str, float]]],
    eligible_ids: Set[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> List[Candidate]:
    """Normalize raw dense candidates across local or Qdrant search results.

    1. Validate chunk ID and finite similarity score.
    2. Reject unknown chunks not in canonical corpus.
    3. Apply permission and date eligibility.
    4. Deduplicate by chunk ID, keeping the best score.
    5. Sort by descending similarity with chunk-ID tie-breaking.
    6. Assign consecutive ranks starting at 1.
    7. Restrict the list to the requested candidate depth.
    """
    best_scores: Dict[str, float] = {}

    for item in raw_candidates:
        if isinstance(item, Candidate):
            cid = item.chunk_id
            score = float(item.score)
        elif isinstance(item, dict):
            cid = str(item["chunk_id"])
            score = float(item.get("score", 0.0))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            cid = str(item[0])
            score = float(item[1])
        else:
            raise TypeError(f"Unsupported dense candidate type: {type(item)}")

        if not math.isfinite(score):
            raise ValueError(f"Non-finite similarity score for chunk '{cid}': {score}")

        if cid not in chunks_by_id:
            raise KeyError(f"Unknown chunk '{cid}' not found in canonical corpus")

        if cid not in eligible_ids:
            continue

        if cid not in best_scores or score > best_scores[cid]:
            best_scores[cid] = score

    # Sort descending by similarity score, tie-break by chunk_id
    sorted_items = sorted(best_scores.items(), key=lambda x: (-x[1], x[0]))

    candidates: List[Candidate] = []
    for rank_idx, (cid, score) in enumerate(sorted_items[:candidate_depth], start=1):
        candidates.append(
            Candidate(
                chunk_id=cid,
                score=score,
                rank=rank_idx,
                chunk=chunks_by_id[cid],
            )
        )
    return candidates


def reciprocal_rank_fusion(
    bm25_candidates: Sequence[Union[Candidate, Dict[str, Any]]],
    dense_candidates: Sequence[Union[Candidate, Dict[str, Any]]],
    chunks_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = RRF_K,
    limit: int = DEFAULT_TOP_K,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> List[RetrievalResult]:
    """Combine lexical and dense candidates using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(doc) = sum(1 / (k + rank)) across available retrieval methods.

    Args:
        bm25_candidates: Ranked BM25 candidate list.
        dense_candidates: Ranked dense candidate list.
        chunks_lookup: Mapping from chunk_id to full chunk dictionary for metadata citation.
        k: RRF constant (default 60).
        limit: Maximum final results to return (default 4).
        candidate_depth: Maximum candidates considered per method (default 20).

    Returns:
        Top `limit` RetrievalResult objects sorted by (-rrf_score, chunk_id).
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Invalid RRF constant k={k}. Must be a positive integer.")
    if not isinstance(limit, int) or limit < 0:
        raise ValueError(f"Invalid limit={limit}. Must be a non-negative integer.")
    if not isinstance(candidate_depth, int) or candidate_depth <= 0:
        raise ValueError(f"Invalid candidate_depth={candidate_depth}. Must be a positive integer.")

    lookup: Dict[str, Dict[str, Any]] = dict(chunks_lookup or {})

    def process_method_candidates(
        raw_list: Sequence[Union[Candidate, Dict[str, Any]]],
        method_name: str,
    ) -> List[Tuple[str, float, int]]:
        processed: List[Tuple[str, float, int]] = []
        seen_chunks: Set[str] = set()
        ranks_assigned: Dict[int, str] = {}

        for idx, item in enumerate(raw_list[:candidate_depth], start=1):
            if isinstance(item, Candidate):
                cid = item.chunk_id
                score = float(item.score)
                rank = item.rank if item.rank is not None else idx
            elif isinstance(item, dict):
                cid = str(item["chunk_id"])
                score = float(item.get("score", 0.0))
                rank = int(item.get("rank", idx))
            else:
                raise TypeError(f"Unsupported candidate type in {method_name}: {type(item)}")

            if rank < 1:
                raise ValueError(f"Invalid rank {rank} for chunk '{cid}' in {method_name}. Ranks must be >= 1.")

            # Reject duplicate ranks assigned to different chunks within the method
            if rank in ranks_assigned and ranks_assigned[rank] != cid:
                raise ValueError(
                    f"Duplicate rank {rank} assigned to different chunks in {method_name}: "
                    f"'{ranks_assigned[rank]}' and '{cid}'."
                )
            ranks_assigned[rank] = cid

            # Count each chunk only once per retrieval method
            if cid in seen_chunks:
                continue
            seen_chunks.add(cid)

            processed.append((cid, score, rank))

        return processed

    bm25_processed = process_method_candidates(bm25_candidates, "bm25")
    dense_processed = process_method_candidates(dense_candidates, "dense")

    scores_by_id: Dict[str, float] = {}
    bm25_info: Dict[str, Tuple[int, float]] = {}
    dense_info: Dict[str, Tuple[int, float]] = {}

    for cid, score, rank in bm25_processed:
        scores_by_id[cid] = scores_by_id.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in bm25_info:
            bm25_info[cid] = (rank, score)

    for cid, score, rank in dense_processed:
        scores_by_id[cid] = scores_by_id.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in dense_info:
            dense_info[cid] = (rank, score)

    # Sort chunks by (-rrf_score, chunk_id) for deterministic tie-breaking
    sorted_chunk_ids = sorted(scores_by_id.keys(), key=lambda cid: (-scores_by_id[cid], cid))

    results: List[RetrievalResult] = []
    for cid in sorted_chunk_ids[:limit]:
        chunk = lookup.get(cid)
        if not chunk:
            raise KeyError(f"Chunk metadata for '{cid}' not found in canonical lookup")

        b_rank, b_score = bm25_info.get(cid, (None, None))
        d_rank, d_score = dense_info.get(cid, (None, None))

        results.append(
            RetrievalResult(
                chunk_id=cid,
                text=chunk["text"],
                doc_id=chunk["doc_id"],
                clause_id=chunk["clause_id"],
                title=chunk["title"],
                department=chunk["department"],
                version=chunk["version"],
                source_file=chunk["source_file"],
                source_page=int(chunk["source_page"]),
                effective_from=chunk["effective_from"],
                effective_to=chunk.get("effective_to"),
                allowed_roles=list(chunk["allowed_roles"]),
                bm25_rank=b_rank,
                bm25_score=b_score,
                dense_rank=d_rank,
                dense_score=d_score,
                rrf_score=scores_by_id[cid],
            )
        )

    return results


def load_validated_corpus(base_dir: Optional[Union[Path, str]] = None) -> List[Dict[str, Any]]:
    """Load the 64 processed policy chunks from chunks.jsonl after Stage C corpus validation."""
    base = Path(base_dir) if base_dir else Path.cwd()
    processed_dir = base / "data" / "processed"
    chunks_path = processed_dir / "chunks.jsonl"
    rules_path = processed_dir / "rules.jsonl"
    manifest_path = processed_dir / "corpus_manifest.json"

    if not chunks_path.exists() or not rules_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing required processed files in {processed_dir}")

    # Call Stage C validation to verify all source documents, PyMuPDF extraction,
    # permissions, manifest metadata, and rule integrity
    validate_corpus(base)

    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if len(chunks) != 64:
        raise ValueError(f"Expected exactly 64 chunks in processed corpus, got {len(chunks)}")

    return chunks


class HybridRetriever:
    """Hybrid retriever coordinating BM25, local dense, Qdrant Cloud, and RRF fusion."""

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        base_dir: Optional[Union[Path, str]] = None,
        embedder: Optional[BGEEmbedder] = None,
        corpus_manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.chunks = chunks if chunks is not None else load_validated_corpus(self.base_dir)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.embedder = embedder or BGEEmbedder()
        self.corpus_manifest = corpus_manifest
        if self.corpus_manifest is None:
            manifest_path = self.base_dir / "data" / "processed" / "corpus_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    self.corpus_manifest = json.load(f)

        # Bounded in-memory cache for BM25Index per distinct eligible chunk set
        self._bm25_cache: Dict[Tuple[str, ...], BM25Index] = {}
        self._bm25_cache_max_size: int = 32

        # Cached document embeddings for local dense retrieval
        self._dense_embeddings_cache: Optional[Dict[str, List[float]]] = None

    def get_bm25_index(self, eligible_chunks: Sequence[Dict[str, Any]]) -> BM25Index:
        """Retrieve or construct a cached BM25Index for the exact eligible chunk set."""
        cache_key = tuple(c["chunk_id"] for c in eligible_chunks)
        if cache_key not in self._bm25_cache:
            if len(self._bm25_cache) >= self._bm25_cache_max_size:
                # FIFO eviction
                self._bm25_cache.pop(next(iter(self._bm25_cache)))
            self._bm25_cache[cache_key] = BM25Index(eligible_chunks)
        return self._bm25_cache[cache_key]

    def get_dense_embeddings(self) -> Dict[str, List[float]]:
        """Compute or retrieve cached 384-d normalized embeddings for all 64 corpus chunks."""
        if self._dense_embeddings_cache is None:
            texts = [c["text"] for c in self.chunks]
            cids = [c["chunk_id"] for c in self.chunks]
            vectors = self.embedder.embed_texts(texts)
            if len(vectors) != len(self.chunks):
                raise ValueError(f"Expected {len(self.chunks)} document vectors, got {len(vectors)}")
            cache: Dict[str, List[float]] = {}
            for cid, vec in zip(cids, vectors):
                if len(vec) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Document vector for '{cid}' has dimension {len(vec)}; expected {EMBEDDING_DIMENSION}."
                    )
                if not all(math.isfinite(x) for x in vec):
                    raise ValueError(f"Document vector for '{cid}' contains non-finite values.")
                cache[cid] = vec
            self._dense_embeddings_cache = cache
        return self._dense_embeddings_cache

    def dense_search(
        self,
        query: str,
        eligible_ids: Set[str],
        top_k: int = DEFAULT_CANDIDATE_DEPTH,
    ) -> List[Candidate]:
        """Execute local dense retrieval using BGE-small embeddings and cosine similarity."""
        query_vec = self.embedder.embed_query(query)
        if len(query_vec) != EMBEDDING_DIMENSION:
            raise ValueError(f"Query vector has dimension {len(query_vec)}; expected {EMBEDDING_DIMENSION}.")
        if not all(math.isfinite(x) for x in query_vec):
            raise ValueError("Query vector contains non-finite values.")

        doc_embeddings = self.get_dense_embeddings()

        raw_candidates: List[Tuple[str, float]] = []
        for cid in eligible_ids:
            doc_vec = doc_embeddings.get(cid)
            if doc_vec is None:
                continue
            # Cosine similarity for normalized vectors is the dot product
            sim = sum(q * d for q, d in zip(query_vec, doc_vec))
            if not math.isfinite(sim):
                raise ValueError(f"Non-finite similarity score for chunk '{cid}': {sim}")
            raw_candidates.append((cid, float(sim)))

        return normalize_dense_candidates(
            raw_candidates=raw_candidates,
            eligible_ids=eligible_ids,
            chunks_by_id=self.chunks_by_id,
            candidate_depth=top_k,
        )

    def search(
        self,
        query: str,
        role: str,
        query_date: Optional[Union[str, datetime.date]] = None,
        dense_candidates: Optional[Sequence[Union[Candidate, Dict[str, Any]]]] = None,
        mode: str = "hybrid",
        dense_backend: str = "local",
        qdrant_client: Optional[Any] = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    ) -> List[RetrievalResult]:
        """Execute hybrid search with pre-filtering for role and policy effective date.

        Args:
            query: The user search query string.
            role: Employee role requesting retrieval (must be one of ROLES).
            query_date: Policy effective date (defaults to today).
            dense_candidates: Optional dense candidates (e.g. from vector search or mock).
            mode: Retrieval mode: 'bm25', 'dense', or 'hybrid' (defaults to 'hybrid').
            dense_backend: Dense retrieval provider: 'local' (default) or 'qdrant'.
            qdrant_client: Optional QdrantClient instance for 'qdrant' dense backend.
            top_k: Number of final RRF results to return (default 4).
            candidate_depth: Number of candidate results to take from each retriever (default 20).

        Returns:
            List of up to top_k RetrievalResult objects, filtered by permission and date.
        """
        # 1. Validate top_k and candidate_depth
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"Invalid top_k={top_k}. Must be a non-negative integer.")
        if not isinstance(candidate_depth, int) or candidate_depth <= 0:
            raise ValueError(f"Invalid candidate_depth={candidate_depth}. Must be a positive integer.")

        # 2. Validate role against approved enterprise roles
        if role not in ALLOWED_ROLES_SET:
            raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

        # 3. Parse and validate query_date
        validated_date = parse_query_date(query_date)

        # 4. Validate retrieval mode (must be 'bm25', 'dense', or 'hybrid')
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"Unknown retrieval mode '{mode}'. Supported modes: 'bm25', 'dense', 'hybrid'.")

        # 5. Validate dense backend (must be 'local' or 'qdrant')
        if dense_backend not in ("local", "qdrant"):
            raise ValueError(f"Unknown dense backend '{dense_backend}'. Supported backends: 'local', 'qdrant'.")

        # 6. Check if query is empty or whitespace-only (ONLY AFTER all inputs above are validated)
        if not query or not query.strip():
            return []

        # 7. Pre-filter eligible chunks using validated role and date
        eligible_chunks = filter_eligible_chunks(self.chunks, role, validated_date)
        if not eligible_chunks:
            return []

        eligible_ids: Set[str] = {c["chunk_id"] for c in eligible_chunks}

        # 8. BM25 Search (using cached index for eligible chunk set)
        bm25_candidates: List[Candidate] = []
        if mode in ("bm25", "hybrid"):
            bm25_index = self.get_bm25_index(eligible_chunks)
            bm25_candidates = bm25_index.search(query, top_k=candidate_depth)

        # 9. Dense Search
        final_dense_candidates: List[Candidate] = []
        if mode in ("dense", "hybrid"):
            if dense_candidates is not None:
                final_dense_candidates = normalize_dense_candidates(
                    raw_candidates=dense_candidates,
                    eligible_ids=eligible_ids,
                    chunks_by_id=self.chunks_by_id,
                    candidate_depth=candidate_depth,
                )
            else:
                if dense_backend == "qdrant":
                    if qdrant_client is None:
                        from rulelens.qdrant_store import get_qdrant_client

                        qdrant_client = get_qdrant_client()
                    from rulelens.qdrant_store import search_qdrant

                    query_vec = self.embedder.embed_query(query)
                    fp = self.corpus_manifest.get("processed_fingerprint", "") if self.corpus_manifest else ""
                    if not fp:
                        raise ValueError("Missing required corpus_fingerprint for Qdrant retrieval.")
                    raw_qdrant = search_qdrant(
                        client=qdrant_client,
                        query_vector=query_vec,
                        role=role,
                        query_date=validated_date,
                        corpus_fingerprint=fp,
                        top_k=candidate_depth,
                    )
                    final_dense_candidates = normalize_dense_candidates(
                        raw_candidates=raw_qdrant,
                        eligible_ids=eligible_ids,
                        chunks_by_id=self.chunks_by_id,
                        candidate_depth=candidate_depth,
                    )
                else:
                    final_dense_candidates = self.dense_search(
                        query=query,
                        eligible_ids=eligible_ids,
                        top_k=candidate_depth,
                    )

        # 10. Return results based on selected mode
        if mode == "bm25":
            results: List[RetrievalResult] = []
            for cand in bm25_candidates[:top_k]:
                c = self.chunks_by_id[cand.chunk_id]
                results.append(
                    RetrievalResult(
                        chunk_id=cand.chunk_id,
                        text=c["text"],
                        doc_id=c["doc_id"],
                        clause_id=c["clause_id"],
                        title=c["title"],
                        department=c["department"],
                        version=c["version"],
                        source_file=c["source_file"],
                        source_page=int(c["source_page"]),
                        effective_from=c["effective_from"],
                        effective_to=c.get("effective_to"),
                        allowed_roles=list(c["allowed_roles"]),
                        bm25_rank=cand.rank,
                        bm25_score=cand.score,
                        dense_rank=None,
                        dense_score=None,
                        rrf_score=1.0 / (RRF_K + cand.rank),
                    )
                )
            return results

        if mode == "dense":
            results = []
            for cand in final_dense_candidates[:top_k]:
                c = self.chunks_by_id[cand.chunk_id]
                results.append(
                    RetrievalResult(
                        chunk_id=cand.chunk_id,
                        text=c["text"],
                        doc_id=c["doc_id"],
                        clause_id=c["clause_id"],
                        title=c["title"],
                        department=c["department"],
                        version=c["version"],
                        source_file=c["source_file"],
                        source_page=int(c["source_page"]),
                        effective_from=c["effective_from"],
                        effective_to=c.get("effective_to"),
                        allowed_roles=list(c["allowed_roles"]),
                        bm25_rank=None,
                        bm25_score=None,
                        dense_rank=cand.rank,
                        dense_score=cand.score,
                        rrf_score=1.0 / (RRF_K + cand.rank),
                    )
                )
            return results

        # Hybrid mode (BM25 + Dense + RRF)
        return reciprocal_rank_fusion(
            bm25_candidates=bm25_candidates,
            dense_candidates=final_dense_candidates,
            chunks_lookup=self.chunks_by_id,
            k=RRF_K,
            limit=top_k,
            candidate_depth=candidate_depth,
        )
