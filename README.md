# RuleLens

Enterprise Policy Intelligence Engine with Hybrid Retrieval and Grounded Answer Generation.

## Project Purpose

RuleLens is an enterprise policy question-answering system designed to deliver verified, clause-grounded policy answers with role-based access control, deterministic rule execution, and strict citation validation. When uncertain or when policy evidence is insufficient, RuleLens safely refuses unsupported questions rather than generating unverified responses.

## Implementation Status & Roadmap

### Implemented Functionality (Stages A–E)
- **Corpus & Rule Ingestion**: 20 canonical policy documents across 5 departments, 64 validated clauses, and 32 structured rules with SHA-256 fingerprint verification.
- **Hybrid Retrieval**: Okapi BM25 lexical search combined with `BAAI/bge-small-en-v1.5` dense embeddings (384-dimensional) via Reciprocal Rank Fusion (RRF, $k=60$).
- **Dual Vector Backends**: 100% offline local in-memory cosine similarity and Qdrant Cloud vector search with server-side payload filtering.
- **Role & Temporal Filtering**: Deterministic pre-retrieval filtering enforcing 6 employee roles and active temporal version windows.
- **Restricted Topic Detection**: Non-sensitive public policy topic catalog ensuring role-restricted queries (such as storage media disposal) fail closed before retrieval.
- **Authoritative Rule Decisions**: Deterministic Python rule evaluation (`Decimal` comparisons) for single-condition decision queries without LLM hallucination.
- **Grounded Answer Generation**: `openai/gpt-oss-20b` via the Groq SDK with strict Pydantic JSON schema structured output.
- **Citation & Safety Verification**: Verifies every citation against supplied evidence; rejects fabricated or unauthorized references.
- **Development Evaluations**: Comprehensive retrieval and generation evaluation harnesses supporting configurable cutoffs ($K=4$, $K=5$).

### Planned / Not Yet Implemented
- **Stage F**: Guarded Semantic Caching with role-safe cache reuse and cluster invalidation.
- **Stage G**: Interactive Streamlit web application.
- **Final Evaluation**: Independent end-to-end benchmark evaluation.

> **Scope Note:** RuleLens is an enterprise policy intelligence prototype. It does not provide production-grade enterprise authentication, identity management, or security hardening, nor does it claim infallible natural language understanding.

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
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ┌──────────────┐                 ┌──────────────┐
 │ Authoritative│                 │ Safe Refusal │
 │ Rule Engine  │                 │ (Early Gate) │
 └──────┬───────┘                 └──────────────┘
        │                                 │
        └────────────────┬────────────────┘
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Groq Generation (openai/gpt-oss-20b)                   │
 │ Strict JSON Schema + Citation & Rule Validation       │
 └────────────────────────────────────────────────────────┘
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
- **Deterministic Rule Engine:** Evaluates 32 canonical structured policy rules in Python for boundary conditions and numeric comparisons (`<`, `<=`, `>`, `>=`, `==`) to eliminate arithmetic hallucinations. For concrete single-condition decision queries, authoritative decisions are returned directly with linked citations without calling the LLM.
- **Evidence-Grounded Generation:** Uses `openai/gpt-oss-20b` via the Groq SDK with strict Pydantic JSON schema structured output.
- **Logical Answer Statuses (Model Layer):**
  - `answered`: Query is directly and unambiguously supported by authorized policy evidence.
  - `insufficient`: Evidence is missing, out-of-scope, or insufficient to answer the query safely.
  - `restricted`: Requesting role lacks authorization to access the target policy area.
- **Application Result Layer:**
  - In addition to the three logical model statuses, the application layer (`GenerationResult`) reports `status="operational_error"` for infrastructure, provider, or validation errors (e.g. rate limits, network timeouts, invalid citations, or integrity failures).
- **Strict Citation Validation:** Every citation is verified against authorized evidence chunks; fabricated, unretrieved, unauthorized, or expired chunk IDs are rejected. When evidence is insufficient or the role lacks permission, the system safely refuses without generating unverified claims.
- **Evaluation Distinction:** Gold-evidence generation evaluation measures model grounding and citation validity given verified gold chunks; it does not measure end-to-end retrieval recall.

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

### 7. Evidence-Grounded Answer Generation
Ask policy questions with role-based access control, temporal version windows, and deterministic rule evaluation:
```bash
python -m rulelens.answering --question "How many days in advance must domestic flights be booked?" --role employee --date 2026-11-22 --backend local
```

### 8. Run Generation Evaluation
Evaluate grounded answer generation against the development evaluation dataset:
```bash
# Evaluate development questions (e.g., balanced 30-question subset)
python -m rulelens.generation_eval --questions data/evaluation/dev_questions.json --limit 30 --backend local

# Run 5-question smoke test
python -m rulelens.generation_eval --smoke --backend local
```

> **Development Evaluation Notes:**
> - **Evaluation Modes:** Evaluates answerable questions in `gold_evidence_generation` mode with strict authorization, temporal validity, verbatim quote, and 4-chunk limit checks before calling Groq. Evaluates diagnostic probes (`application_diagnostic` mode) through the end-to-end `answer_question` pipeline using the selected retrieval backend (`local` or `qdrant`).
> - **Provider Quotas and Pacing:** When evaluating live against the Groq API (`openai/gpt-oss-20b`), sequential requests are paced (2.5s) to stay within per-minute TPM/RPM quotas. If an HTTP 429 rate limit is encountered, the evaluation harness stops gracefully without data loss, marking the run as partial and persisting all completed results to `local_eval/generation_development_results.json` and `local_eval/generation_development_summary.md`.
> - **Offline Verification:** Full generation evaluation workflows and fail-closed safety behaviors are verified offline with mocked LLM wrappers via `python -m pytest -q`.
