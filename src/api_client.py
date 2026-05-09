"""
HTTP client for the rag-chatbot-langchain /chat and /ingest APIs.

Stage 5a integration: evaluation now measures the live deployed API
rather than a local chain. Set RAG_API_URL in .env to enable.

Fallback: if RAG_API_URL is not set, callers should fall back to
the local rag_chain.py chain (backwards compatible).
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0  # RAG calls can be slow with streaming disabled


def get_api_url() -> Optional[str]:
    """Return the RAG API base URL from env, or None if not configured."""
    url = os.getenv("RAG_API_URL", "").rstrip("/")
    return url if url else None


def ingest_via_api(pdf_path: str, api_url: str) -> dict:
    """Upload a PDF to the RAG API's /ingest endpoint and return the response."""
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        resp = httpx.post(f"{api_url}/ingest", files=files, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def chat_via_api(question: str, api_url: str) -> dict:
    """
    Call /chat and return {"answer": str, "sources": [...], "cost": {...}}.

    The /chat endpoint returns a ChatResponse Pydantic model. We parse the
    answer and contexts from it so the evaluator can feed them to RAGAS.
    """
    resp = httpx.post(
        f"{api_url}/chat",
        json={"question": question},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def build_ragas_dataset_from_api(
    qa_pairs: list[dict],
    api_url: str,
    pdf_path: Optional[str] = None,
) -> "datasets.Dataset":
    """
    Build a RAGAS dataset by calling the live /chat API for each question.

    If pdf_path is provided, ingests it first via /ingest.
    Returns a Dataset with question, answer, contexts, ground_truth columns.
    """
    from datasets import Dataset

    if pdf_path:
        logger.info(f"Ingesting {pdf_path} via API at {api_url}")
        ingest_result = ingest_via_api(pdf_path, api_url)
        logger.info(f"Ingested: {ingest_result.get('chunks', '?')} chunks")

    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        logger.info(f"[{i+1}/{len(qa_pairs)}] {question[:60]}")

        response = chat_via_api(question, api_url)

        # Extract answer string
        answer = response.get("answer", "")

        # Extract context strings from sources
        sources = response.get("sources", [])
        contexts = [s.get("content", s) if isinstance(s, dict) else str(s) for s in sources]
        if not contexts:
            contexts = [answer]  # fallback: use answer as its own context

        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(pair["ground_truth"])

    logger.info(f"Built RAGAS dataset via API — {len(data['question'])} samples")
    return Dataset.from_dict(data)
