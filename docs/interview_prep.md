# Interview Prep — RAG Evaluation Pipeline

> This project is your strongest differentiator in interviews. Most candidates build RAG systems but never measure them rigorously. Having RAGAS scores and a before/after comparison demonstrates senior-level thinking.

---

## Core Concept Questions

### Q: How do you evaluate a RAG system?

> "I use RAGAS — an evaluation framework that uses an LLM as a judge. I hand-craft 20-30 Q&A pairs from the document, then for each pair I run my RAG chain to get an answer and retriever to get the contexts. RAGAS then scores four dimensions: faithfulness (is the answer grounded in the context?), answer relevancy (does it address the question?), context recall (did I retrieve the right chunks?), and context precision (were the retrieved chunks actually useful?). I use Gemini 2.0 Flash as the judge LLM — it's free and handles the RAGAS prompt format correctly."

---

### Q: What is faithfulness and why does it matter more than accuracy?

> "Faithfulness measures whether every claim in the answer is supported by the retrieved context. A score of 1.0 means the LLM only stated facts that were in the chunks. A low score means it's hallucinating — mixing in knowledge from its training data.
>
> Faithfulness matters more than traditional accuracy in RAG because: (1) you often can't verify accuracy without a reference answer for every possible question, but (2) you can always check if the answer is grounded in the context you provided. In domains like legal or medical documents, an answer that's 95% correct but 5% hallucinated is dangerous."

---

### Q: What is the difference between context recall and context precision?

> "Context recall asks: 'Did I retrieve the evidence I needed?' — measured by checking if the ground-truth answer can be reconstructed from the retrieved chunks. Low recall means I'm missing relevant information.
>
> Context precision asks: 'Was everything I retrieved useful?' — measured by checking what fraction of the k retrieved chunks actually contributed to the answer. Low precision means I'm retrieving noise that could confuse the LLM or waste context window.
>
> They're complementary. You could have perfect recall (retrieved everything needed) but poor precision (also retrieved 10 irrelevant chunks). The fix for low recall is to increase k or decrease chunk size. The fix for low precision is to decrease k or use better retrieval strategies like MMR."

---

### Q: How did you build your evaluation dataset?

> "I hand-crafted Q&A pairs at three difficulty levels. Easy questions test direct fact retrieval — if these fail, your retrieval is broken at a basic level. Medium questions require inference within a section. Hard questions require synthesizing information from multiple sections of the document — these expose whether your chunking strategy is losing cross-section context.
>
> The quality of `ground_truth` is critical. A vague ground truth like 'Revenue was good' makes context_recall scores meaningless. I write specific, verifiable answers: 'Q3 revenue was $47.2M, up 12% YoY, driven by enterprise segment growth.' With specific answers, RAGAS can precisely check whether the retrieved chunks contained that information."

---

### Q: After running your baseline, what did you find and how did you improve it?

> "My baseline had faithfulness of 0.72 and answer relevancy of 0.68. Inspecting the LangSmith traces for the low-scoring questions, I saw the LLM was adding qualifications like 'typically' and 'generally' that weren't in the document — it was drawing on training data, not just context.
>
> I tightened the system prompt by adding: 'Only state specific facts present in the provided context. Do not use general knowledge or add qualifications not supported by the text.' Re-running evaluation, faithfulness improved to 0.89 (+23.6%) and relevancy to 0.81 (+19.1%). Context recall stayed flat at 0.75, which was expected since the prompt change doesn't affect retrieval.
>
> I then saved both runs with timestamps so the before/after comparison is reproducible and I can demo the delta chart."

---

### Q: Why use an LLM as the judge instead of human evaluation?

> "Two reasons: scale and cost. Manually reviewing 25 Q&A pairs per eval run is manageable once. But when you're running evals on every PR before merging, you can't have a human review hundreds of answers per day. LLM-as-judge scales to any eval frequency.
>
> The tradeoff is that the judge LLM can have biases — it may favor answers in its own style, or be lenient on outputs from the same model family. The mitigation is to use a different model as judge than the one generating answers when possible, and to always validate LLM-as-judge on a small set of manually scored examples to calibrate your trust in its scores."

---

### Q: How would you use this evaluation pipeline in a CI/CD workflow?

> "I implement this in Project 5 — an eval regression check runs on every push to main via GitHub Actions. The steps: install deps, run RAGAS on a held-out eval set, compare scores against defined thresholds (faithfulness ≥ 0.75, answer relevancy ≥ 0.70, etc.), and fail the build if any metric drops below threshold. This means a prompt change that tanks faithfulness is caught before it reaches production.
>
> The key engineering decision is the threshold. I set thresholds at 0.05 below the current best score — tight enough to catch regressions, loose enough to not fail on normal statistical variation across runs."

---

## Design Decision Questions

### Q: Why Gemini for the RAGAS judge LLM? Why not use the same LLM that generated the answers?

> "Two reasons to use the same model (Gemini 2.0 Flash): cost/simplicity, and it's what I have on free tier. The ideal is to use a different, more powerful model as judge — a GPT-4-level model judging Gemini Flash outputs — because it reduces same-model bias. In production, I'd use a stronger model as judge since evaluation runs far less frequently than inference, so the cost difference is manageable."

---

### Q: How many Q&A pairs do you need in an eval set?

> "The minimum I'd trust is 20 pairs — any fewer and a single outlier moves the average significantly. For a production system where evaluation drives deployment decisions, I'd aim for 50-100 pairs, stratified by difficulty, document section, and question type (factual vs. inference vs. synthesis). For the highest-stakes domains, I'd validate the eval set itself by having domain experts review whether my ground truth answers are correct."

---

## Connecting to Your Production Experience

> "In my production work at Speridian, I achieved ~99% field-level accuracy on mortgage document extraction. That didn't happen by checking manually — I built evaluation scripts that compared extracted fields against manually verified ground truth and tracked metrics over time. RAGAS is the equivalent for open-ended Q&A: instead of exact field matching, you need LLM-as-judge because answers are free-form. The principle is the same — measure, find failures, fix, measure again."

> "The LangSmith integration is particularly valuable from a production debugging perspective. When a faithfulness score drops on a specific question, I can click into the LangSmith trace and see exactly: which 4 chunks were retrieved, what the prompt looked like, and what the LLM said. That's the same observability pattern I'd apply to any production service — you can't fix what you can't see."
