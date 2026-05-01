"""
RAGAS evaluation module.

RAGAS metrics:
- faithfulness:        Is the answer grounded in the retrieved context? (catches hallucination)
- answer_relevancy:    Does the answer actually address the question? (catches evasion)
- context_recall:      Did we retrieve the chunks needed to answer? (measures retrieval coverage)
- context_precision:   Are the retrieved chunks relevant? (measures retrieval noise)

All metrics: 0.0 (worst) → 1.0 (best). Aim for > 0.75 on all four before shipping.

Gemini 2.0 Flash is used as the judge LLM — it evaluates each answer and awards scores.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import os

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0):
    """Return chat LLM — OpenRouter (if key set) else Gemini direct."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)


METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def build_ragas_dataset(
    qa_pairs: List[Dict],
    chain,
    retriever,
) -> Dataset:
    """
    Run each Q&A pair through the RAG chain and collect the four RAGAS fields:
    question, answer (from chain), contexts (retrieved chunks), ground_truth (hand-written).
    """
    data: Dict[str, List] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        logger.info(f"[{i+1}/{len(qa_pairs)}] {question[:60]}")

        answer = chain.invoke(question)
        docs = retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(pair["ground_truth"])

    logger.info(f"Built RAGAS dataset with {len(data['question'])} samples")
    return Dataset.from_dict(data)


def run_evaluation(dataset: Dataset) -> pd.DataFrame:
    """
    Score the dataset with RAGAS using Gemini 2.0 Flash as judge.

    RAGAS calls Gemini ~2-3 times per sample per metric, so for 20 samples
    expect ~100-200 LLM calls. This stays well within Gemini's free tier.
    """
    logger.info("Running RAGAS evaluation (this takes a few minutes)...")

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=_get_llm(temperature=0),
        embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )

    df = result.to_pandas()
    means = df[METRIC_NAMES].mean().round(3).to_dict()
    logger.info(f"Mean scores: {means}")
    return df


def save_results(df: pd.DataFrame, run_name: str = "baseline") -> str:
    """Save results CSV with timestamp so multiple runs can be compared."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{run_name}_{timestamp}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Results saved → {path}")
    return path


def load_qa_pairs(path: str = "eval_datasets/qa_pairs.json") -> List[Dict]:
    """Load the hand-crafted Q&A evaluation set."""
    with open(path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} Q&A pairs from {path}")
    return pairs
