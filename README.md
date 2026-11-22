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
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ┌──────────────┐                 ┌──────────────┐
 │ Deterministic│                 │ Safe Refusal │
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
- **Deterministic Rule Engine:** Evaluates 32 canonical structured policy rules in Python for boundary conditions and numeric comparisons (`<`, `<=`, `>`, `>=`, `==`) to eliminate LLM arithmetic hallucinations.
- **Evidence-Grounded Generation:** Uses `openai/gpt-oss-20b` via the Groq SDK with strict Pydantic JSON schema structured output.
- **5 Structured Answer Fields:**
  - `answer`: Clause-grounded textual answer or explicit refusal explanation.
  - `status`: One of `ANSWERED`, `INSUFFICIENT_EVIDENCE`, `ROLE_RESTRICTED`, `POLICY_CONFLICT`, `OPERATIONAL_ERROR`.
  - `citations`: List of valid, retrieved chunk IDs directly supporting the answer.
  - `referenced_rule_ids`: List of deterministic rule IDs applied and verified.
  - `reason`: Explanation of the status classification and evidence basis.
- **Strict Citation Validation & Refusal:** Every citation is verified against authorized evidence chunks; fabricated, unretrieved, unauthorized, or expired chunk IDs are rejected. When evidence is insufficient or the role lacks permission, the system safely refuses without generating unverified claims.

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
