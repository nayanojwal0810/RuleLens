"""Hybrid retrieval engine for RuleLens.

Combines BM25 lexical search and BGE-small dense embeddings using
Reciprocal Rank Fusion (RRF) with role-based and temporal eligibility filtering.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from rulelens.config import EMBEDDING_MODEL, GENERATION_EVIDENCE_LIMIT
from rulelens.dataset import ROLES
from rulelens.ingestion import calculate_processed_fingerprint

# Default parameters
RRF_K: int = 60
DEFAULT_CANDIDATE_DEPTH: int = 20
DEFAULT_TOP_K: int = GENERATION_EVIDENCE_LIMIT  # 4
EMBEDDING_DIMENSION: int = 384

# Approved enterprise roles for validation
ALLOWED_ROLES_SET: Set[str] = set(ROLES)

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
        ValueError: If query_date is a malformed string or invalid type.
    """
    if query_date is None:
        return datetime.date.today()
    if isinstance(query_date, datetime.date):
        return query_date
    if isinstance(query_date, str):
        cleaned = query_date.strip()
        if not cleaned:
            return datetime.date.today()
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
    # Will raise ValueError on unknown role or malformed date
    parse_query_date(query_date)
    if role not in ALLOWED_ROLES_SET:
        raise ValueError(f"Unknown role '{role}'. Allowed roles are: {sorted(ALLOWED_ROLES_SET)}")

    return [c for c in chunks if is_chunk_eligible(c, role, query_date)]


@dataclass(frozen=True)
class RetrievalResult:
    """Clean representation of a retrieved policy chunk with rank and citation metadata."""

    chunk_id: str
    text: str
    doc_id: str
    clause_id: str
    title: str
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
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.corpus_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.corpus_token_sets = [set(tokens) for tokens in self.corpus_tokens]
        self._bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = DEFAULT_CANDIDATE_DEPTH) -> List[Candidate]:
        """Search the BM25 index.

        Args:
            query: The search query string.
            top_k: Maximum number of candidate results to return.

        Returns:
            Ranked Candidate list with 1-based ranks and scores.
            Empty queries or queries with no matches return an empty list.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
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
        """Batch-embed a sequence of text strings.

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

        return embeddings.tolist() if hasattr(embeddings, "tolist") else [list(vec) for vec in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string into a 384-dimensional normalized vector."""
        if not query or not query.strip():
            # Return a zero vector or embed empty string
            pass
        return self.embed_texts([query])[0]


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
    lookup: Dict[str, Dict[str, Any]] = dict(chunks_lookup or {})

    # Helper to normalize Candidate objects
    def normalize_list(raw_list: Sequence[Union[Candidate, Dict[str, Any]]]) -> List[Tuple[str, float, int, Optional[Dict[str, Any]]]]:
        normalized = []
        for idx, item in enumerate(raw_list[:candidate_depth], start=1):
            if isinstance(item, Candidate):
                cid = item.chunk_id
                score = item.score
                rank = item.rank if item.rank is not None else idx
                chunk_dict = item.chunk
            elif isinstance(item, dict):
                cid = item["chunk_id"]
                score = float(item.get("score", 0.0))
                rank = int(item.get("rank", idx))
                chunk_dict = item.get("chunk", item)
            else:
                raise TypeError(f"Unsupported candidate type: {type(item)}")

            if chunk_dict and cid not in lookup:
                lookup[cid] = chunk_dict
            normalized.append((cid, score, rank, chunk_dict))
        return normalized

    bm25_norm = normalize_list(bm25_candidates)
    dense_norm = normalize_list(dense_candidates)

    scores_by_id: Dict[str, float] = {}
    bm25_info: Dict[str, Tuple[int, float]] = {}
    dense_info: Dict[str, Tuple[int, float]] = {}

    for cid, score, rank, chunk_dict in bm25_norm:
        if cid not in scores_by_id:
            scores_by_id[cid] = 0.0
        scores_by_id[cid] += 1.0 / (k + rank)
        if cid not in bm25_info:
            bm25_info[cid] = (rank, score)

    for cid, score, rank, chunk_dict in dense_norm:
        if cid not in scores_by_id:
            scores_by_id[cid] = 0.0
        scores_by_id[cid] += 1.0 / (k + rank)
        if cid not in dense_info:
            dense_info[cid] = (rank, score)

    # Sort chunks by (-rrf_score, chunk_id) for deterministic tie-breaking
    sorted_chunk_ids = sorted(scores_by_id.keys(), key=lambda cid: (-scores_by_id[cid], cid))

    results: List[RetrievalResult] = []
    for cid in sorted_chunk_ids[:limit]:
        chunk = lookup.get(cid)
        if not chunk:
            raise KeyError(f"Chunk metadata for '{cid}' not found in candidate lookup")

        b_rank, b_score = bm25_info.get(cid, (None, None))
        d_rank, d_score = dense_info.get(cid, (None, None))

        results.append(
            RetrievalResult(
                chunk_id=cid,
                text=chunk["text"],
                doc_id=chunk["doc_id"],
                clause_id=chunk["clause_id"],
                title=chunk["title"],
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
    """Load the 64 processed policy chunks from chunks.jsonl after verifying fingerprint integrity."""
    base = Path(base_dir) if base_dir else Path.cwd()
    processed_dir = base / "data" / "processed"
    chunks_path = processed_dir / "chunks.jsonl"
    rules_path = processed_dir / "rules.jsonl"
    manifest_path = processed_dir / "corpus_manifest.json"

    if not chunks_path.exists() or not rules_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing required processed files in {processed_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    rules: List[Dict[str, Any]] = []
    with open(rules_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))

    # Verify corpus fingerprint integrity
    computed_fp = calculate_processed_fingerprint(chunks, rules)
    if computed_fp != manifest.get("processed_fingerprint"):
        raise ValueError(
            f"Corpus fingerprint mismatch: computed {computed_fp} != manifest {manifest.get('processed_fingerprint')}"
        )

    if len(chunks) != 64:
        raise ValueError(f"Expected exactly 64 chunks in processed corpus, got {len(chunks)}")

    return chunks


class HybridRetriever:
    """Offline hybrid retriever coordinating BM25, permission filtering, and RRF fusion."""

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        base_dir: Optional[Union[Path, str]] = None,
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_validated_corpus(base_dir)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.embedder = embedder or BGEEmbedder()

    def search(
        self,
        query: str,
        role: str,
        query_date: Optional[Union[str, datetime.date]] = None,
        dense_candidates: Optional[Sequence[Union[Candidate, Dict[str, Any]]]] = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    ) -> List[RetrievalResult]:
        """Execute hybrid search with pre-filtering for role and policy effective date.

        Args:
            query: The user search query string.
            role: Employee role requesting retrieval (must be one of ROLES).
            query_date: Policy effective date (defaults to today).
            dense_candidates: Optional dense candidates (e.g. from vector search or mock).
                If None, only BM25 candidates are used (does not fabricate dense results).
            top_k: Number of final RRF results to return (default 4).
            candidate_depth: Number of candidate results to take from each retriever (default 20).

        Returns:
            List of up to top_k RetrievalResult objects, filtered by permission and date.
        """
        # 1. Pre-filter eligible chunks before retrieval and ranking
        eligible_chunks = filter_eligible_chunks(self.chunks, role, query_date)
        if not eligible_chunks:
            return []

        eligible_ids: Set[str] = {c["chunk_id"] for c in eligible_chunks}

        # 2. BM25 search on eligible chunks
        bm25_index = BM25Index(eligible_chunks)
        bm25_candidates = bm25_index.search(query, top_k=candidate_depth)

        # 3. Dense candidates handling
        eligible_dense_candidates: List[Union[Candidate, Dict[str, Any]]] = []
        if dense_candidates is not None:
            for item in dense_candidates:
                cid = item.chunk_id if isinstance(item, Candidate) else item["chunk_id"]
                if cid in eligible_ids:
                    eligible_dense_candidates.append(item)

        # 4. RRF fusion
        return reciprocal_rank_fusion(
            bm25_candidates=bm25_candidates,
            dense_candidates=eligible_dense_candidates,
            chunks_lookup=self.chunks_by_id,
            k=RRF_K,
            limit=top_k,
            candidate_depth=candidate_depth,
        )
