# RuleLens

Enterprise Policy Intelligence Engine with Hybrid Retrieval and Guarded Semantic Caching.

## Project Purpose
RuleLens is an enterprise policy question-answering system designed to deliver verified, clause-grounded policy answers with role-based access control and safe semantic cache reuse. When uncertain or when policy evidence is insufficient, RuleLens safely refuses unsupported questions rather than generating unverified responses.

> **Status:** Implementation in progress (Stage A: Repository Foundation). Features, benchmarks, and deployment are currently being developed.

## Planned Architecture
- **Hybrid Retrieval:** Dense embeddings (`BAAI/bge-small-en-v1.5`) combined with lexical search (BM25) fused via Reciprocal Rank Fusion (RRF) over role-authorized documents.
- **Rule Resolution:** Deterministic Python evaluation of structured conditions (amounts, approval thresholds, dates) linked to policy clauses.
- **Guarded Semantic Cache:** Vector similarity caching in Qdrant with verification gates (role scope, active policy version, rule outcome, and evidence hash matching) before reusing answers.
- **Generation:** Groq LLM generation (`openai/gpt-oss-20b`) producing structured Pydantic-validated responses with strict citation-to-evidence checks.
- **Dataset:** Deterministic fictional enterprise policy system (20 PDFs and 5 CSV rule tables across 6 employee roles).
- **Deployment Target:** Streamlit Community Cloud with Qdrant Cloud.

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

### Running Tests
Run offline unit tests:
```bash
python -m pytest -q
```
