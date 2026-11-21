# RuleLens

Enterprise Policy Intelligence Engine with Hybrid Retrieval and Guarded Semantic Caching.

## Project Purpose
RuleLens is an enterprise policy question-answering system designed to deliver verified, clause-grounded policy answers with role-based access control and safe semantic cache reuse. When uncertain or when policy evidence is insufficient, RuleLens safely refuses unsupported questions rather than generating unverified responses.

> **Status:** Stage D2 Complete (Complete Hybrid Retrieval & Development Evaluation). BM25 lexical search, BGE-small dense embeddings, Reciprocal Rank Fusion, local vector search, and Qdrant Cloud integration implemented and evaluated.

## Architecture

```text
User Query + Role + Effective Date
       │
       ▼
 ┌────────────────────────────────────────────────────────┐
 │ Deterministic Pre-Filtering                           │
 │ (Role-Based Permissions & Temporal Version Boundaries)  │
 └───────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ┌──────────────┐                 ┌──────────────┐
 │ BM25 Lexical │                 │ BGE-small-v1 │
 │ Search Engine│                 │ Dense Vector │
 └──────┬───────┘                 └──────┬───────┘
        │                                 │
        └────────────────┬────────────────┘
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Reciprocal Rank Fusion (RRF, k=60)                     │
 └───────────────────────┬────────────────────────────────┘
                         ▼
        Top-K Clause-Grounded Evidence Chunks
```

- **Hybrid Retrieval:** Dense embeddings (`BAAI/bge-small-en-v1.5`, 384 dimensions) combined with lexical search (BM25) fused via Reciprocal Rank Fusion (RRF, $k=60$) over role-authorized, date-active documents.
- **Dual Dense Backends:** High-performance local in-memory cosine similarity and remote Qdrant Cloud vector store with server-side payload filters.
- **Rule Resolution:** Deterministic Python evaluation of structured conditions (amounts, approval thresholds, dates) linked to policy clauses.
- **Guarded Semantic Cache:** Vector similarity caching with verification gates (role scope, active policy version, rule outcome, and evidence hash matching).
- **Dataset:** Deterministic enterprise policy system (20 PDFs and 5 CSV rule tables across 6 employee roles: `employee`, `manager`, `hr`, `finance`, `security`, `auditor`).

## Installation & Setup

### Prerequisites
- Python >= 3.11, < 3.12

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/nayanojwal0810/RuleLens.git
cd RuleLens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with development dependencies
python -m pip install -e ".[dev]"
```

### Environment Configuration
Copy `.env.example` to `.env` and provide your credentials (optional for offline local operation):
```bash
cp .env.example .env
```
Supported variables:
- `GROQ_API_KEY`: API key for Groq LLM generation.
- `QDRANT_URL`: Cluster URL for Qdrant Cloud (e.g. `https://xyz.cloud.qdrant.io:6333`).
- `QDRANT_API_KEY`: API key for Qdrant Cloud.

## CLI Commands

### 1. Validate Dataset Integrity
Verify canonical policy PDFs and CSV rule matrices:
```bash
python -m rulelens.dataset validate
```

### 2. Validate Processed Corpus
Verify PyMuPDF clause extraction, chunking, and content hashes:
```bash
python -m rulelens.ingestion validate
```

### 3. Run Offline Test Suite
Run 101 unit tests completely offline with network sockets blocked:
```bash
python -m pytest -q
```

### 4. Qdrant Cloud Collection Management
Synchronize the 64 policy chunks to Qdrant:
```bash
python -m rulelens.qdrant_store sync
```
Verify Qdrant collection configuration and point data:
```bash
python -m rulelens.qdrant_store verify
```

### 5. Run Retrieval Evaluation Harness
Evaluate BM25, Dense, and Hybrid retrieval modes against the 30-question development dataset:
```bash
python -m rulelens.retrieval_eval
```
Results and summaries are saved to `local_eval/development_results.json` and `local_eval/development_summary.md`.

## Development Evaluation Results

Evaluated over 30 questions (6 per department across all 5 departments) with top-5 cutoff:

| Retrieval Method | Recall@5 | Hit@5 | MRR@5 | Median Latency (ms) | P95 Latency (ms) | Role Violations | Version Violations |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BM25 (Lexical) | 1.0000 | 1.0000 | 0.9250 | 0.61 ms | 3.85 ms | 0 | 0 |
| BGE Dense (Local) | 1.0000 | 1.0000 | 0.9750 | 39.72 ms | 87.88 ms | 0 | 0 |
| **Hybrid (BM25 + Dense + RRF)** | 1.0000 | 1.0000 | 0.9500 | 44.37 ms | 112.00 ms | 0 | 0 |

> Zero role violations and zero temporal version violations observed across all modes.
