# How It Works — RAG Evaluation Pipeline

## The Core Problem This Solves

Project 1 gives you a working RAG chatbot. But how do you know if it's *good*? "It seems to answer correctly" is not an engineering answer.

This project adds a rigorous **measurement framework** — the difference between a demo and a production system. You need numbers, not vibes.

---

## What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework that uses an LLM to score your RAG system's outputs. Instead of manually reading every answer, you let Gemini judge whether each answer is:

1. **Faithful** — grounded in the retrieved context
2. **Relevant** — actually answers the question asked
3. **Well-retrieved** — the right chunks were fetched

The key insight: you don't need human labelers. The LLM can act as a judge at scale.

---

## The Four Metrics Explained

### Faithfulness (catches hallucination)

**Formula:** What fraction of claims in the answer are supported by the context?

RAGAS breaks the answer into individual factual claims, then checks each one against the retrieved chunks. If Gemini says "The revenue was $50M" but the retrieved chunks only mention "$45M", faithfulness drops.

- Score of 1.0: every claim is supported by the context
- Score of 0.5: half the claims are unsupported (hallucinated)
- **Low score fix:** Tighten the system prompt — add "Only state facts present in the context"

### Answer Relevancy (catches evasion)

**Formula:** How closely does the answer relate to the question?

RAGAS generates several variations of the question from the answer, then checks if they're similar to the original question. If the answer is tangential, the reverse-engineered questions will differ significantly.

- Score of 1.0: answer directly addresses the question
- Score of 0.5: answer talks around the topic without answering
- **Low score fix:** Add "Answer the question directly" to the system prompt

### Context Recall (measures retrieval coverage)

**Formula:** What fraction of the ground-truth answer can be attributed to retrieved chunks?

This requires your `ground_truth` field. RAGAS checks whether the evidence needed to produce the ground-truth answer was actually retrieved.

- Score of 1.0: all necessary evidence was retrieved
- Score of 0.5: half the required context was missed
- **Low score fix:** Increase `k` (retrieve more chunks), or reduce chunk size

### Context Precision (measures retrieval noise)

**Formula:** Of the retrieved chunks, what fraction were actually useful?

Retrieves k chunks and checks how many were relevant to answering the question. High precision means tight, focused retrieval.

- Score of 1.0: all retrieved chunks were relevant
- Score of 0.5: half the retrieved chunks were noise
- **Low score fix:** Decrease `k`, or use MMR (maximum marginal relevance) instead of similarity search

---

## The Evaluation Loop

### Step 1: Build Your Evaluation Dataset

**File:** `eval_datasets/qa_pairs.json`

This is the most important step. You hand-write 20-30 Q&A pairs:
```json
{
  "question": "What were the Q3 revenue figures?",
  "ground_truth": "Q3 revenue was $47.2M, up 12% year-over-year.",
  "difficulty": "easy"
}
```

Three difficulty levels serve different purposes:
- **Easy** (direct fact lookup): If these fail, your retrieval is broken
- **Medium** (inference required): Tests whether the LLM can reason within context
- **Hard** (multi-section synthesis): Tests whether retrieval finds all relevant chunks

**The ground_truth matters enormously.** If your ground truth is vague ("Revenue was good"), context_recall scores become meaningless. Be specific and verifiable.

### Step 2: Run the Baseline

```bash
python run_evaluation.py eval --pdf your_doc.pdf --run-name baseline
```

The pipeline:
1. Loads `qa_pairs.json`
2. Validates all pairs have required fields
3. Builds the RAG chain from Project 1 on your PDF
4. For each question: runs the chain (gets answer) + runs the retriever (gets contexts)
5. Packages into a RAGAS `Dataset` object: `{question, answer, contexts, ground_truth}`
6. Calls `ragas.evaluate()` — Gemini judges each sample
7. Saves to `results/baseline_YYYYMMDD_HHMMSS.csv`

### Step 3: Diagnose

Open the results CSV. Look for rows where individual metric scores are low. Ask:
- Low `faithfulness` on questions 3, 7, 12? → Check what context was retrieved for those questions (LangSmith traces help here)
- Low `context_recall` across all questions? → Your chunks might be too large (too coarse) or k too small
- Low `answer_relevancy` on hard questions? → The LLM may be confused by injected context from multiple sections

### Step 4: Change ONE Variable

Common experiments:
- Increase `k` from 4 to 6 → should improve context_recall
- Tighten system prompt → should improve faithfulness
- Reduce chunk_size from 1000 to 600 → should improve precision

**Only change one at a time.** If you change both prompt and chunk size together, you can't attribute the score change to either.

### Step 5: Compare

```bash
python run_evaluation.py compare results/baseline_*.csv results/v2_*.csv
```

The reporter prints:
```
✅ faithfulness        : 0.72 → 0.89 (+23.6%)
✅ answer_relevancy    : 0.68 → 0.81 (+19.1%)
➖ context_recall      : 0.74 → 0.75 (+1.4%)
❌ context_precision   : 0.80 → 0.71 (-11.3%)
```

And saves a bar chart to `results/comparison_chart.png`.

---

## Why the Ground Truth Dataset is Your Most Valuable Asset

In production, the eval dataset is often more valuable than the code. A carefully crafted, diverse set of 25+ Q&A pairs with verified answers:

- Catches regressions when you change the prompt (run eval before every deploy)
- Guides future improvements (low-scoring questions tell you exactly what's broken)
- Provides concrete numbers for stakeholder communication ("faithfulness improved from 0.72 to 0.89")
- Demonstrates rigor in interviews — most candidates only do manual testing
