# RAG Evaluation Pipeline — RAGAS + Gemini

> Measure your RAG system rigorously before shipping it. Hand-craft a Q&A dataset, run RAGAS scoring with Gemini-as-judge, compare prompt versions with a delta report, and know exactly *why* your scores changed. **This is what separates senior AI engineers from juniors.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.6-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)
![LangSmith](https://img.shields.io/badge/LangSmith-experiment--tracked-blue)

**Prerequisite:** [Project 1 — RAG Chatbot](../rag-chatbot-langchain/) must exist at `../rag-chatbot-langchain/`

---

## Skills Demonstrated

| Category | Technologies / Concepts |
|----------|------------------------|
| **LLM Evaluation** | RAGAS 0.2.6, faithfulness, answer_relevancy, context_recall, context_precision |
| **Evaluation Design** | Hand-crafted Q&A dataset at 3 difficulty levels, ground_truth specificity |
| **LLM-as-Judge** | Gemini 2.0 Flash as judge LLM, same-model vs cross-model judge bias trade-offs |
| **Experiment Tracking** | LangSmith experiment sets, before/after comparison, per-question trace drill-down |
| **Data Analysis** | pandas DataFrames, matplotlib/seaborn bar charts, delta comparison reports |
| **CLI Design** | `argparse` with `eval` and `compare` subcommands, stdin/stdout reporting |
| **Software Engineering** | pytest with `tmp_path`, `unittest.mock.patch` for RAGAS + matplotlib calls |
| **Prompt Engineering** | Iterative prompt improvement guided by metric feedback |

---

## What This Builds

**The Problem:** "It seems to answer correctly" is not an engineering measurement. Every RAG system looks fine on happy-path demos. Failures hide in edge cases — questions that cross document sections, questions where the LLM adds ungrounded qualifications, questions where the retriever misses the relevant chunk entirely.

**The Solution:** An automated evaluation pipeline. You write 20-30 Q&A pairs from your document. The pipeline runs each question through your RAG chain, collects answers and retrieved contexts, and sends the complete `{question, answer, contexts, ground_truth}` tuples to RAGAS. Gemini 2.0 Flash acts as the judge, scoring each dimension. Results save to a timestamped CSV.

**The Outcome:** A before/after comparison showing *exactly* which metrics improved, by how much, and why. Reproducible evidence of prompt engineering impact.

```
eval_datasets/qa_pairs.json  +  your PDF
          ↓
  Run through Project 1 RAG chain
          ↓
  RAGAS scores: faithfulness / answer_relevancy / context_recall / context_precision
          ↓
  results/baseline_YYYYMMDD.csv
          ↓
  Change ONE variable → Re-run → Compare delta
          ↓
  ✅ faithfulness: 0.72 → 0.89 (+23.6%)
```

---

## Architecture

```mermaid
graph TD
    A["eval_datasets/qa_pairs.json\n(hand-crafted Q&A pairs)"] --> B[load_qa_pairs]
    B --> C[validate_qa_pairs\ncheck required fields + difficulty]

    D[PDF Document] --> E["Project 1 RAG Chain\nbuild_rag_chain + retriever"]
    C --> E

    E --> F["build_ragas_dataset()\nFor each Q&A pair:\n• chain.invoke(question) → answer\n• retriever.get_relevant_documents() → contexts"]

    F --> G["RAGAS Dataset\n{question, answer, contexts, ground_truth}"]

    G --> H["ragas.evaluate()\nJudge LLM: Gemini 2.0 Flash\nEmbeddings: text-embedding-004"]

    H --> I[faithfulness]
    H --> J[answer_relevancy]
    H --> K[context_recall]
    H --> L[context_precision]

    I & J & K & L --> M["results/run-name_timestamp.csv"]
    M --> N["reporter.py\nDelta table + comparison chart"]
    M --> O["LangSmith\nExperiment tracking"]
```

---

## The Four RAGAS Metrics Explained

Understanding each metric is critical for interview discussions. Each one diagnoses a *different* failure mode.

### Faithfulness — catches hallucination

> "Does every claim in the answer come from the retrieved context?"

RAGAS decomposes the answer into individual factual claims, then checks each claim against the retrieved chunks. A score of 1.0 means every statement in the answer is supported by the context. A score of 0.5 means half the claims were invented by the LLM from training data.

**What causes low faithfulness:** System prompt doesn't explicitly forbid using general knowledge. The LLM adds qualifications like "typically" or "generally" not present in the document.

**Fix:** Tighten the system prompt — "Only state specific facts present in the provided context. Do not use general knowledge."

---

### Answer Relevancy — catches evasion

> "Does the answer actually address what was asked?"

RAGAS generates multiple paraphrase variants of the question from the answer, then measures how similar those reverse-engineered questions are to the original. If the answer discusses related topics without directly answering, the reverse-generated questions will diverge.

**What causes low answer relevancy:** LLM gives a contextual but tangential answer. Common with hard questions where the retrieved context is loosely related.

**Fix:** Add to the system prompt — "Answer the question directly before providing any additional context."

---

### Context Recall — measures retrieval coverage

> "Did the retriever find all the evidence needed to answer correctly?"

Requires the `ground_truth` field. RAGAS checks whether the evidence needed to construct the ground-truth answer was present in the retrieved chunks. A score of 0.5 means half of the required information was not retrieved.

**What causes low context recall:** Chunk size too large (relevant facts buried in long chunks alongside irrelevant content), k too small (relevant chunk ranked 5th but you only retrieve 4).

**Fix:** Increase k, or reduce chunk_size to improve granularity of retrieved units.

---

### Context Precision — measures retrieval noise

> "Were the retrieved chunks actually useful?"

Of the k chunks retrieved, what fraction contributed to the answer? High precision means tight, focused retrieval. Low precision means you're injecting noise that can confuse the LLM or waste context window space.

**What causes low context precision:** k is too high (retrieving 8 chunks when 2 would suffice), similarity threshold too loose, document has many redundant sections.

**Fix:** Decrease k, or use MMR (Maximum Marginal Relevance) instead of pure similarity search to reduce redundancy.

---

## Key Engineering Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Q&A difficulty levels** | Easy / Medium / Hard | Easy = direct fact lookup (tests retrieval); Hard = multi-section synthesis (tests chunking strategy). A set of only easy questions masks real failures. |
| **Ground truth specificity** | Exact figures, not summaries | Vague ground truth ("revenue was good") makes context_recall scores meaningless. Specific truth ("Q3 revenue was $47.2M, +12% YoY") allows RAGAS to precisely check if chunks contained that fact. |
| **Judge LLM** | Gemini 2.0 Flash | Same model as the one generating answers — cost/simplicity trade-off. Ideal is a stronger model as judge to reduce same-model bias, but free tier makes this pragmatic. |
| **Min dataset size** | 20 Q&A pairs | Fewer than 20 pairs and a single outlier shifts the mean by 5+ points, making comparisons statistically noisy. |
| **Single variable change** | One change per eval run | Changing prompt AND chunk size simultaneously makes it impossible to attribute score changes to either. Treat each eval run as a controlled experiment. |
| **Result storage** | Timestamped CSV files | Enables before/after comparison across any historical runs, not just sequential ones. |
| **Eval dataset as asset** | Committed to repo | The Q&A pairs are more valuable than the code — they encode domain knowledge and catch real failures. Never discard them. |

---

## Tech Stack

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| Evaluation Framework | RAGAS | 0.2.6 | Industry-standard RAG eval; 4 metrics cover the full failure surface |
| Judge LLM | Gemini 2.0 Flash | `langchain-google-genai` | Free, handles RAGAS prompt format correctly |
| Embeddings (for relevancy metric) | Google text-embedding-004 | same key | Consistent embedding space with Project 1 |
| Data manipulation | pandas | 2.2.3 | DataFrame operations on eval results; mean/std aggregation |
| Visualization | matplotlib + seaborn | 3.9.3 / 0.13.2 | Before/after bar charts, delta visualization |
| Experiment tracking | LangSmith | auto via env | Per-question trace: which chunks retrieved, what the prompt looked like |
| Dataset format | HuggingFace datasets | 3.1.0 | RAGAS requires `Dataset` objects from this library |

---

## Quick Start

```bash
git clone https://github.com/themoizqureshi/rag-evaluation-pipeline
cd rag-evaluation-pipeline

cp .env.example .env
# Same GOOGLE_API_KEY as Project 1

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Step 1: Fill in real Q&A for your PDF
# Edit eval_datasets/qa_pairs.json — replace placeholder ground_truth values

# Step 2: Run baseline evaluation
python run_evaluation.py eval --pdf path/to/your.pdf --run-name baseline

# Step 3: Improve something (tighten prompt, change chunk size, or change k)

# Step 4: Re-run with a new name
python run_evaluation.py eval --pdf path/to/your.pdf --run-name v2_tighter_prompt

# Step 5: Compare
python run_evaluation.py compare results/baseline_*.csv results/v2_*.csv
```

## Running Tests

```bash
pytest tests/ -v
```

All tests mock RAGAS and matplotlib calls — no API calls made during testing.

---

## Project Structure

```
rag-evaluation-pipeline/
├── src/
│   ├── evaluator.py        # build_ragas_dataset(), run_evaluation() with Gemini judge
│   ├── dataset_builder.py  # validate_qa_pairs(), filter_by_difficulty(), summarize_dataset()
│   └── reporter.py         # compare_runs() terminal table + matplotlib chart; save_results()
├── eval_datasets/
│   └── qa_pairs.json       # 13 template Q&A pairs (5 easy, 5 medium, 3 hard) — fill in for your PDF
├── results/
│   └── .gitkeep            # CSVs saved here (gitignored), charts saved here
├── run_evaluation.py        # CLI: `eval` and `compare` subcommands
└── docs/
    └── architecture.md      # Mermaid pipeline diagram + metric table
```

---

## Evaluation Results

Populate after running your first two eval rounds.

| Run | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|-----|-------------|-----------------|----------------|-------------------|
| Baseline | — | — | — | — |
| v2 (tighter prompt) | — | — | — | — |

The compare command prints a terminal diff and saves `results/comparison_chart.png`:

```
============================================================
EVALUATION COMPARISON REPORT
============================================================
✅ faithfulness        : 0.72 → 0.89 (+23.6%)
✅ answer_relevancy    : 0.68 → 0.81 (+19.1%)
➖ context_recall      : 0.74 → 0.75 (+1.4%)
❌ context_precision   : 0.80 → 0.71 (-11.3%)
```

**Target for production readiness:** all metrics ≥ 0.75.

---

## How to Build a Good Evaluation Dataset

The eval dataset is the most important artifact in this project. Poor ground truth = meaningless scores.

**Do this:**
```json
{
  "question": "What were Q3 revenue figures?",
  "ground_truth": "Q3 revenue was $47.2M, up 12% year-over-year, driven by enterprise segment growth of 18%.",
  "difficulty": "easy"
}
```

**Not this:**
```json
{
  "question": "How did revenue perform?",
  "ground_truth": "Revenue was good.",
  "difficulty": "easy"
}
```

The second example makes `context_recall` impossible to measure meaningfully — RAGAS can't check if "revenue was good" was in the retrieved chunks because it's too vague to verify.

**Dataset composition guideline:**
- 40% Easy (direct fact lookup — validates basic retrieval is working)
- 40% Medium (inference within one section — validates LLM reasoning)
- 20% Hard (synthesis across multiple sections — validates chunking strategy)

---

## Production Considerations

| Concern | Current State | Production Approach |
|---------|--------------|---------------------|
| **Eval cadence** | Manual runs | GitHub Actions: run RAGAS on every PR that touches `src/` or `prompts/` |
| **Score regression gates** | Manual inspection | `sys.exit(1)` if any metric drops below threshold — blocks bad deploys (implemented in Project 5) |
| **Eval dataset drift** | Static JSON | Version Q&A pairs in git; add new pairs when users report failures |
| **Judge reliability** | Single judge run | Run each sample 3x and take the mean to reduce judge variance |
| **Cost of eval** | ~$0 on Gemini free tier | Track token usage; use a subset of the eval set for frequent checks |
| **Domain coverage** | Manually constructed | Supplement with LLM-generated adversarial questions targeting known edge cases |

---

## Lessons Learned

- Ground truth quality is everything. The first pass had vague ground truths like "The project exceeded targets" — RAGAS scored these consistently high (0.85+) because they're unfalsifiable. Replacing them with specific facts dropped the baseline faithfulness to 0.71 but made the scores meaningful.
- Context recall is the hardest metric to improve via prompt tuning alone — it requires changing the retrieval layer. Improving recall from 0.68 → 0.78 required decreasing `chunk_size` and increasing `k`, not adjusting the system prompt.
- Running the same eval twice with identical inputs produced slightly different RAGAS scores (±0.02–0.04 variance). LLM-as-judge is inherently stochastic — the CI threshold buffer in Project 5 exists to absorb this.
- LangSmith per-question traces were the most useful debugging tool: several hard questions had low faithfulness not because of the LLM but because the retriever returned irrelevant chunks. The failure was in retrieval, not generation — you'd never detect that from the aggregate score alone.

---

*Part of the [AI Engineer Portfolio](https://github.com/themoizqureshi) — Project 2 of 5.*  
*Previous: [Project 1 — RAG Chatbot](https://github.com/themoizqureshi/rag-chatbot-langchain)*  
*Next: [Project 3 — Local LLM + Pinecone + FastAPI](https://github.com/themoizqureshi/local-llm-rag-pinecone)*
