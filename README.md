# RuleLens

Enterprise Policy Intelligence Engine with Hybrid Retrieval and Guarded Semantic Caching.

## Project Purpose

RuleLens is an enterprise policy question-answering system designed to deliver verified, clause-grounded policy answers with role-based access control and safe semantic cache reuse. When uncertain or when policy evidence is insufficient, RuleLens safely refuses unsupported questions rather than generating unverified responses.

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

- **Hybrid as Primary Retrieval Architecture:** RuleLens uses Hybrid Retrieval as its standard mode, combining lexical precision (BM25) with semantic recall (`BAAI/bge-small-en-v1.5`, 384-dimensional normalized embeddings) fused via Reciprocal Rank Fusion (RRF, $k=60$).
- **Supported Retrieval Modes:**
  - `hybrid` (default): Fuses BM25 lexical candidates and BGE dense vector candidates via RRF.
  - `bm25`: Fast lexical-only retrieval based on Okapi BM25 scoring.
  - `dense`: Dense semantic vector retrieval using cosine similarity.
- **Dual Dense Backends:**
  - `local`: High-performance local in-memory cosine similarity over cached document embeddings (100% offline).
  - `qdrant`: Vector database retrieval utilizing server-side payload filtering for permissions, dates, and corpus fingerprints.
- **Security & Version Integrity:** Role-based permissions (across `employee`, `manager`, `hr`, `finance`, `security`, `auditor`) and temporal version windows are enforced via deterministic pre-filtering before scoring.

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
Copy `.env.example` to `.env` to configure optional external services:
```bash
cp .env.example .env
```
Supported variables:
- `GROQ_API_KEY`: API key for Groq LLM generation.
- `QDRANT_URL`: Cluster endpoint URL for Qdrant Cloud (e.g. `https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333`). Note: The marketing website `https://qdrant.io` is rejected.
- `QDRANT_API_KEY`: API key for Qdrant Cloud.

Local `.env` values are automatically loaded when environment variables are not set. Explicitly configured environment variables take precedence.

## CLI Commands

### 1. Validate Dataset Integrity
Verify canonical policy PDFs and CSV rule matrices:
```bash
python -m rulelens.dataset validate
```

### 2. Validate Processed Corpus
Verify clause extraction, chunking, and content hashes:
```bash
python -m rulelens.ingestion validate
```

### 3. Validate Development Evaluation Dataset
Verify the public 50-question evaluation dataset without loading the embedding model:
```bash
python -m rulelens.retrieval_eval validate --questions data/evaluation/dev_questions.json
```

### 4. Run Offline Test Suite
Run unit tests completely offline with network sockets blocked:
```bash
python -m pytest -q
```

### 5. Qdrant Collection Management
Synchronize the 64 policy chunks to Qdrant:
```bash
python -m rulelens.qdrant_store sync
```

Verify Qdrant collection configuration, point UUIDs, and payload fields:
```bash
python -m rulelens.qdrant_store verify
```

### 6. Run Development Retrieval Evaluation
Run evaluation across BM25, Dense, and Hybrid modes:
```bash
# Using local dense backend (default)
python -m rulelens.retrieval_eval run --questions data/evaluation/dev_questions.json --backend local

# Using Qdrant Cloud backend (requires active cluster credentials)
python -m rulelens.retrieval_eval run --questions data/evaluation/dev_questions.json --backend qdrant
```
Detailed results and markdown summaries are generated in `local_eval/development_results.json` and `local_eval/development_summary.md` (ignored by version control).
