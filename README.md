# RAG Evaluation Pipeline — RAGAS + Gemini

> Measure your RAG system rigorously: faithfulness, relevancy, recall, and precision — all scored automatically using Gemini as judge. **This is what separates senior AI engineers from juniors.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.6-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)

**Prerequisite:** [Project 1 — RAG Chatbot](../rag-chatbot-langchain/) must exist at `../rag-chatbot-langchain/`

## What This Does

```
Your hand-crafted Q&A pairs + your PDF
        ↓
Run each question through the Project 1 RAG chain
        ↓
RAGAS scores: faithfulness / answer_relevancy / context_recall / context_precision
        ↓
results/baseline_YYYYMMDD.csv
        ↓
Change ONE variable (prompt / chunk size / k)  →  Re-run  →  Compare delta
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/rag-evaluation-pipeline
cd rag-evaluation-pipeline

cp .env.example .env
# Same GOOGLE_API_KEY as Project 1

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Step 1: Fill in eval_datasets/qa_pairs.json with real Q&A for your PDF
# Step 2: Run baseline
python run_evaluation.py eval --pdf path/to/your.pdf --run-name baseline

# Step 3: Improve something (prompt, chunk size, k) then re-run
python run_evaluation.py eval --pdf path/to/your.pdf --run-name v2_tighter_prompt

# Step 4: Compare
python run_evaluation.py compare results/baseline_*.csv results/v2_*.csv
```

## RAGAS Metrics

| Metric | Measures | Low Score Means |
|--------|----------|-----------------|
| `faithfulness` | Is every claim in the answer grounded in context? | LLM is hallucinating |
| `answer_relevancy` | Does the answer address the question? | LLM is being evasive |
| `context_recall` | Was the right information retrieved? | Retrieval is missing chunks |
| `context_precision` | Were retrieved chunks useful? | Too much noise retrieved |

**Target:** All metrics ≥ 0.75 before shipping to production.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Evaluation | RAGAS 0.2.6 | Industry-standard RAG eval framework |
| Judge LLM | Gemini 2.0 Flash | Free, accurate scoring |
| Embeddings | text-embedding-004 | Consistent with Project 1 |
| Tracking | LangSmith | Per-question trace for debugging |
| Reporting | pandas + matplotlib | CSV results + comparison charts |

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/how_it_works.md](docs/how_it_works.md) | Deep-dive into each RAGAS metric with fix strategies |
| [docs/interview_prep.md](docs/interview_prep.md) | Interview Q&A — metrics, design decisions, production connection |
| [docs/architecture.md](docs/architecture.md) | Pipeline diagram and eval loop workflow |

## Evaluation Results

| Run | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|-----|-------------|-----------------|---------------|------------------|
| Baseline | — | — | — | — |
| v2 (improved prompt) | — | — | — | — |

*Populate after running your first evaluation.*

## Lessons Learned

- *Fill in after building.*

---

*Part of the [AI Engineer Portfolio](https://github.com/YOUR_USERNAME) — Project 2 of 5.*
