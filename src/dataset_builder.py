"""
Utilities for building and validating the evaluation dataset.

A good eval set is the foundation of your entire evaluation pipeline.
20-30 Q&A pairs across three difficulty levels is the minimum viable set.
"""

import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def validate_qa_pairs(pairs: List[Dict]) -> List[str]:
    """
    Check that every Q&A pair has required fields and non-empty values.
    Returns a list of error messages (empty = all valid).
    """
    required_fields = {"question", "ground_truth", "difficulty"}
    valid_difficulties = {"easy", "medium", "hard"}
    errors = []

    for i, pair in enumerate(pairs):
        missing = required_fields - set(pair.keys())
        if missing:
            errors.append(f"Pair {i}: missing fields {missing}")
            continue

        if not pair["question"].strip():
            errors.append(f"Pair {i}: empty question")
        if not pair["ground_truth"].strip():
            errors.append(f"Pair {i}: empty ground_truth")
        if pair["difficulty"] not in valid_difficulties:
            errors.append(f"Pair {i}: difficulty must be one of {valid_difficulties}")

    return errors


def filter_by_difficulty(
    pairs: List[Dict], difficulty: str
) -> List[Dict]:
    """Return only pairs of a specific difficulty level."""
    return [p for p in pairs if p.get("difficulty") == difficulty]


def summarize_dataset(pairs: List[Dict]) -> Dict:
    """Print a summary of the evaluation dataset."""
    total = len(pairs)
    by_difficulty = {
        "easy": len(filter_by_difficulty(pairs, "easy")),
        "medium": len(filter_by_difficulty(pairs, "medium")),
        "hard": len(filter_by_difficulty(pairs, "hard")),
    }
    summary = {"total": total, "by_difficulty": by_difficulty}
    logger.info(f"Dataset summary: {summary}")
    return summary


def save_qa_pairs(pairs: List[Dict], path: str) -> None:
    """Save Q&A pairs to JSON."""
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved {len(pairs)} pairs → {path}")
