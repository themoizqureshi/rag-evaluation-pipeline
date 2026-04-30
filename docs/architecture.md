# Architecture — RAG Evaluation Pipeline

## Pipeline Overview

```mermaid
graph TD
    A["eval_datasets/qa_pairs.json\n(hand-crafted Q&A)"] --> B[load_qa_pairs]
    B --> C[validate_qa_pairs]

    D["PDF Document"] --> E["Project 1 RAG Chain\n(Gemini 2.0 Flash)"]
    C --> E

    E --> F["build_ragas_dataset()\nFor each question:\n• run chain → answer\n• run retriever → contexts"]

    F --> G["RAGAS Dataset\n{question, answer, contexts, ground_truth}"]

    G --> H["RAGAS evaluate()\nJudge LLM: Gemini 2.0 Flash"]

    H --> I["faithfulness score"]
    H --> J["answer_relevancy score"]
    H --> K["context_recall score"]
    H --> L["context_precision score"]

    I & J & K & L --> M["results/run_name_timestamp.csv"]
    M --> N["reporter.py\nDelta comparison chart"]

    O["Improved prompt / chunk size"] --> E
    M --> P["LangSmith\nExperiment tracking"]
```

## What Each RAGAS Metric Measures

| Metric | Question it answers | What a low score means |
|--------|---------------------|------------------------|
| `faithfulness` | Is the answer supported by the retrieved context? | LLM is hallucinating — adding info not in chunks |
| `answer_relevancy` | Does the answer address the question? | LLM is giving evasive or off-topic answers |
| `context_recall` | Did retrieval find the chunks needed? | Wrong chunks retrieved — tune chunk size or k |
| `context_precision` | Are the retrieved chunks relevant? | Too much noise retrieved — reduce k or add filters |

## Eval Loop Workflow

```
1. Run baseline eval    →  results/baseline_YYYYMMDD.csv
2. Inspect low scores   →  Which metric failed? Which questions?
3. Change ONE variable  →  prompt, chunk_size, k, or model
4. Run improved eval    →  results/v2_YYYYMMDD.csv
5. Compare              →  python run_evaluation.py compare baseline.csv v2.csv
6. Repeat               →  Until all metrics ≥ 0.75
```

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Judge LLM | Gemini 2.0 Flash | Free, handles RAGAS prompt format correctly |
| Embeddings | text-embedding-004 | Same as Project 1 — consistent embedding space |
| Min dataset size | 20 Q&A pairs | Statistical minimum for meaningful averages |
| Difficulty split | Easy / Medium / Hard | Hard questions (multi-section synthesis) expose real failures |
| Result storage | Timestamped CSVs | Enables before/after comparison across multiple runs |
