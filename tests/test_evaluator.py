"""Tests for the evaluation pipeline (no API calls)."""

import pytest
from unittest.mock import MagicMock, patch
from src.dataset_builder import validate_qa_pairs, filter_by_difficulty, summarize_dataset


# ── dataset_builder tests ──

VALID_PAIRS = [
    {"question": "What is RAG?", "ground_truth": "RAG is retrieval augmented generation.", "difficulty": "easy"},
    {"question": "How does chunking work?", "ground_truth": "Text is split into overlapping chunks.", "difficulty": "medium"},
    {"question": "What are the trade-offs of k=4?", "ground_truth": "k=4 balances context size and precision.", "difficulty": "hard"},
]


def test_validate_qa_pairs_valid():
    errors = validate_qa_pairs(VALID_PAIRS)
    assert errors == []


def test_validate_qa_pairs_missing_field():
    bad = [{"question": "What?", "difficulty": "easy"}]  # missing ground_truth
    errors = validate_qa_pairs(bad)
    assert any("ground_truth" in e for e in errors)


def test_validate_qa_pairs_invalid_difficulty():
    bad = [{"question": "Q?", "ground_truth": "A.", "difficulty": "extreme"}]
    errors = validate_qa_pairs(bad)
    assert len(errors) == 1


def test_validate_qa_pairs_empty_question():
    bad = [{"question": "  ", "ground_truth": "A.", "difficulty": "easy"}]
    errors = validate_qa_pairs(bad)
    assert any("empty question" in e for e in errors)


def test_filter_by_difficulty():
    easy = filter_by_difficulty(VALID_PAIRS, "easy")
    assert len(easy) == 1
    assert easy[0]["difficulty"] == "easy"


def test_summarize_dataset():
    summary = summarize_dataset(VALID_PAIRS)
    assert summary["total"] == 3
    assert summary["by_difficulty"]["easy"] == 1
    assert summary["by_difficulty"]["medium"] == 1
    assert summary["by_difficulty"]["hard"] == 1


# ── reporter tests ──

def test_compare_runs_returns_dataframe(tmp_path):
    """compare_runs should return a DataFrame with delta columns."""
    import pandas as pd
    from src.reporter import compare_runs

    # Create two fake result CSVs
    data = {
        "faithfulness": [0.7, 0.8, 0.6],
        "answer_relevancy": [0.65, 0.75, 0.70],
        "context_recall": [0.60, 0.70, 0.65],
        "context_precision": [0.55, 0.65, 0.60],
    }
    baseline_path = tmp_path / "baseline.csv"
    improved_path = tmp_path / "improved.csv"

    df_b = pd.DataFrame(data)
    df_i = pd.DataFrame({k: [v * 1.1 for v in vals] for k, vals in data.items()})

    df_b.to_csv(baseline_path, index=False)
    df_i.to_csv(improved_path, index=False)

    with patch("src.reporter._save_chart"):  # skip matplotlib in tests
        result = compare_runs(str(baseline_path), str(improved_path))

    assert "delta" in result.columns
    assert len(result) == 4  # one row per metric
    assert all(result["delta"] > 0)  # improved > baseline
