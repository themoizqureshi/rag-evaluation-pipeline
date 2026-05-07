"""
Streamlit UI for the RAG Evaluation Pipeline.

Two modes:
  Demo  — loads pre-computed results from results/sample_*.csv (no API calls needed)
  Live  — upload your own PDF, run a real RAGAS evaluation (needs OPENROUTER_API_KEY)

Run with: streamlit run app.py
"""

import json
import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Evaluation Pipeline", page_icon="📊", layout="wide")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_recall": 0.65,
    "context_precision": 0.65,
}
SAMPLE_V1 = "results/sample_v1_baseline.csv"
SAMPLE_V2 = "results/sample_v2_improved.csv"
SAMPLE_DOC = "eval_datasets/sample_document.txt"
SAMPLE_QA = "eval_datasets/sample_qa.json"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Mode")
    mode = st.radio("", ["Demo (pre-computed)", "Live (your PDF)"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**RAGAS Metrics**")
    for m, t in THRESHOLDS.items():
        st.markdown(f"**{m.replace('_', ' ').title()}** ≥ {t}")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("RAGAS · LangChain · ChromaDB")
    st.markdown("BAAI/bge-small-en-v1.5 (local)")
    st.markdown("Gemini 2.0 Flash via OpenRouter")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 RAG Evaluation Pipeline")
st.caption("RAGAS scoring with LLM-as-judge across 4 dimensions — baseline vs. strict-grounding prompt")


# ── Demo mode ─────────────────────────────────────────────────────────────────
if mode == "Demo (pre-computed)":
    st.info(
        "**Demo mode** — showing pre-computed evaluation results for the bundled RAG overview document. "
        "Switch to **Live** in the sidebar to run against your own PDF.",
        icon="ℹ️",
    )

    if not os.path.exists(SAMPLE_V1) or not os.path.exists(SAMPLE_V2):
        st.error("Pre-computed results not found. Run the live evaluation once first.")
        st.stop()

    v1 = pd.read_csv(SAMPLE_V1)
    v2 = pd.read_csv(SAMPLE_V2)

    tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Per-Question Breakdown", "Regression Gate"])

    with tab1:
        st.subheader("v1 Baseline vs. v2 Strict Prompt")
        st.caption(
            "Single variable changed: system prompt strictness. "
            "All other params identical (model: gemini-2.0-flash, chunk_size: 1000, k: 4)."
        )
        cols = st.columns(4)
        for i, metric in enumerate(METRICS):
            v1_mean = v1[metric].mean()
            v2_mean = v2[metric].mean()
            delta = v2_mean - v1_mean
            cols[i].metric(
                label=metric.replace("_", " ").title(),
                value=f"{v2_mean:.3f}",
                delta=f"{delta:+.3f} vs baseline",
            )

        st.markdown("---")
        chart_data = pd.DataFrame(
            {
                "v1 Baseline": [v1[m].mean() for m in METRICS],
                "v2 Strict": [v2[m].mean() for m in METRICS],
            },
            index=[m.replace("_", " ").title() for m in METRICS],
        )
        st.bar_chart(chart_data, color=["#94a3b8", "#3b82f6"])

    with tab2:
        version = st.selectbox("Version", ["v1 Baseline", "v2 Strict Prompt"])
        df = v1 if version == "v1 Baseline" else v2
        st.dataframe(
            df[["question", "answer"] + METRICS].style.format({m: "{:.3f}" for m in METRICS}),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Regression Gate — v2 vs Thresholds")
        all_pass = True
        for metric in METRICS:
            actual = v2[metric].mean()
            threshold = THRESHOLDS[metric]
            passed = actual >= threshold
            if not passed:
                all_pass = False
            status = "✅ PASS" if passed else "❌ FAIL"
            gap = actual - threshold
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"{status} &nbsp; **{metric.replace('_', ' ').title()}**: `{actual:.3f}` (threshold ≥ {threshold}, gap: `{gap:+.3f}`)")
        st.markdown("")
        if all_pass:
            st.success("All quality gates passed. Safe to deploy.")
        else:
            st.error("Quality regression detected. Inspect failing questions above.")


# ── Live mode ─────────────────────────────────────────────────────────────────
else:
    has_key = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not has_key:
        st.warning("Set `OPENROUTER_API_KEY` in `.env` to run live evaluation.", icon="⚠️")

    st.markdown("### Step 1 — Choose document")
    doc_choice = st.radio("", ["Use bundled sample document", "Upload my own PDF"], label_visibility="collapsed")

    pdf_path = None
    tmp_file = None

    if doc_choice == "Use bundled sample document":
        if os.path.exists(SAMPLE_DOC):
            st.success(f"Using: `{SAMPLE_DOC}` (RAG overview, ~500 words)")
            pdf_path = SAMPLE_DOC
        else:
            st.error(f"Bundled document not found at `{SAMPLE_DOC}`.")
    else:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(uploaded.getvalue())
            tmp_file.flush()
            pdf_path = tmp_file.name
            st.success(f"Uploaded: {uploaded.name}")

    st.markdown("### Step 2 — Load Q&A pairs")
    qa_choice = st.radio("", ["Use bundled sample Q&A (5 questions)", "Paste custom Q&A JSON"], label_visibility="collapsed")

    qa_pairs = None
    if qa_choice == "Use bundled sample Q&A (5 questions)":
        if os.path.exists(SAMPLE_QA):
            with open(SAMPLE_QA) as f:
                qa_pairs = json.load(f)
            st.success(f"Loaded {len(qa_pairs)} Q&A pairs from `{SAMPLE_QA}`")
            with st.expander("Preview Q&A pairs"):
                for i, pair in enumerate(qa_pairs, 1):
                    st.markdown(f"**Q{i}:** {pair['question']}")
                    st.markdown(f"*Ground truth:* {pair['ground_truth']}")
                    st.markdown("---")
    else:
        st.caption(
            'JSON array of objects with "question" and "ground_truth" keys. '
            'See `eval_datasets/sample_qa.json` for the format.'
        )
        raw = st.text_area("Paste Q&A JSON", height=200, placeholder='[{"question": "...", "ground_truth": "..."}]')
        if raw.strip():
            try:
                qa_pairs = json.loads(raw)
                st.success(f"Parsed {len(qa_pairs)} Q&A pairs")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.markdown("### Step 3 — Choose prompt version")
    prompt_version = st.selectbox("Prompt", ["v1 (baseline)", "v2 (strict grounding)"])
    pv = "v1" if "v1" in prompt_version else "v2"

    st.markdown("### Step 4 — Run evaluation")
    run_name = st.text_input("Run name", value=f"my_eval_{pv}")

    if st.button("▶️ Run RAGAS Evaluation", type="primary", disabled=not (pdf_path and qa_pairs and has_key)):
        with st.spinner("Ingesting document..."):
            from src.rag_chain import ingest_document, build_rag_chain
            vs = ingest_document(pdf_path, persist_dir=f"./eval_chroma_db_{run_name}")
            chain, retriever = build_rag_chain(vs, prompt_version=pv)

        with st.spinner(f"Running RAGAS on {len(qa_pairs)} questions (this takes 1–3 min)..."):
            from src.evaluator import build_ragas_dataset, run_evaluation, save_results
            dataset = build_ragas_dataset(qa_pairs, chain, retriever)
            results_df = run_evaluation(dataset)
            results_path = save_results(results_df, run_name)

        if tmp_file:
            os.unlink(tmp_file.name)

        st.success(f"Evaluation complete! Results saved to `{results_path}`")
        st.subheader("Results")
        cols = st.columns(4)
        for i, metric in enumerate(METRICS):
            cols[i].metric(metric.replace("_", " ").title(), f"{results_df[metric].mean():.3f}")
        st.dataframe(results_df[["question", "answer"] + METRICS], use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            results_df.to_csv(index=False),
            file_name=f"{run_name}.csv",
            mime="text/csv",
        )
