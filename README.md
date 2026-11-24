# RuleLens

RuleLens answers questions about a fictional company's policies using role and date
filters, hybrid retrieval, deterministic rules and verified source quotations.

The sample corpus contains 20 policy PDFs, five rule tables, 64 clause chunks and
32 rules. The roles are demonstration inputs, not a production identity system.
Semantic caching is disabled by default. Final performance results are not yet
published.

## Installation

Use Python 3.11 and an activated virtual environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS. Install the base package:

```bash
python -m pip install -e .
```

Dense and hybrid retrieval additionally require the semantic extra. It installs
the embedding libraries; the model weights are downloaded on first use.

```bash
python -m pip install -e ".[semantic]"
```

Set `GROQ_API_KEY`, `QDRANT_URL` and `QDRANT_API_KEY` in a local `.env`, using
`.env.example` as the template. Keep existing credentials when updating the project.
The `.env` file is excluded from Git.

## Usage

Validate the included corpus without calling external services:

```bash
python -m rulelens.ingestion validate
```

An amount-based decision can run with the base install and lexical retrieval:

```python
from rulelens.answering import answer_question

result = answer_question(
    "Can my manager approve a travel expense of INR 50,000?",
    role="employee",
    query_date="2026-11-22",
    mode="bm25",
)
print(result.answer)
```

With the semantic extra and API configuration available:

```bash
python -m rulelens.qdrant_store sync
python -m rulelens.qdrant_store verify
python -m rulelens.answering --question "What is the daily meal allowance?" --role employee --date 2026-11-22 --backend qdrant
```

Generated answers must match complete authorized source spans. Ambiguous inputs,
unsupported requests and conflicting policy contexts can require clarification or
produce a refusal.

## Tests

The test install omits the semantic extra and model weights:

```bash
python -m pip install -r requirements_test.txt
python -m pip install --no-build-isolation -e .
python -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp
```

Tests block outbound connections and use deterministic embedding doubles. They
check functionality and safeguards; they do not measure live answer quality or
cache performance. CI runs the public suite on Windows and Linux.

The development question banks are in `data/evaluation/`. Retrieval questions
include role/date context, expected answer or refusal routes, and source-backed
facts with material prerequisites. These are development cases, not unseen final
evaluation results. Structured gold values require source review; matching a quote
or passing schema validation alone does not prove that an answer is correct.
