"""
Comparison reporter: takes two eval result CSVs and produces a delta report + chart.

Use this after changing your prompt, chunk size, or model to measure the impact.
Always change ONE variable at a time — if you change both chunk size and the prompt
together you can't attribute the score change to either.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def compare_runs(baseline_path: str, improved_path: str) -> pd.DataFrame:
    """
    Compare two eval runs and print a terminal delta report.
    Also saves a comparison chart to results/comparison_chart.png.

    Args:
        baseline_path: Path to the baseline results CSV (your first run)
        improved_path: Path to the improved results CSV (after changes)

    Returns:
        DataFrame with per-metric baseline, improved, delta, and delta_pct columns
    """
    baseline = pd.read_csv(baseline_path)
    improved = pd.read_csv(improved_path)

    rows = []
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON REPORT")
    print(f"  Baseline : {baseline_path}")
    print(f"  Improved : {improved_path}")
    print("=" * 60)

    for metric in METRICS:
        b_mean = baseline[metric].mean()
        i_mean = improved[metric].mean()
        delta = i_mean - b_mean
        delta_pct = (delta / b_mean) * 100 if b_mean > 0 else 0.0
        icon = "✅" if delta > 0.01 else ("❌" if delta < -0.01 else "➖")
        print(f"{icon} {metric:25s}: {b_mean:.3f} → {i_mean:.3f} ({delta_pct:+.1f}%)")
        rows.append({
            "metric": metric,
            "baseline": round(b_mean, 3),
            "improved": round(i_mean, 3),
            "delta": round(delta, 3),
            "delta_pct": round(delta_pct, 1),
        })

    print("=" * 60)
    comparison_df = pd.DataFrame(rows)

    _save_chart(comparison_df)
    return comparison_df


def print_single_run_summary(results_path: str) -> None:
    """Print a summary of a single evaluation run."""
    df = pd.read_csv(results_path)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS SUMMARY")
    print(f"  File: {results_path}")
    print(f"  Samples: {len(df)}")
    print("=" * 50)

    for metric in METRICS:
        if metric in df.columns:
            mean = df[metric].mean()
            status = "✅" if mean >= 0.75 else ("⚠️ " if mean >= 0.60 else "❌")
            print(f"{status} {metric:25s}: {mean:.3f}")

    print("=" * 50)


def _save_chart(comparison_df: pd.DataFrame) -> None:
    """Save a comparison bar chart to results/comparison_chart.png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = comparison_df["metric"].tolist()
    x = range(len(metrics))

    # Left: side-by-side bars
    axes[0].bar([i - 0.2 for i in x], comparison_df["baseline"], 0.4,
                label="Baseline", color="steelblue")
    axes[0].bar([i + 0.2 for i in x], comparison_df["improved"], 0.4,
                label="Improved", color="seagreen")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metrics, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Baseline vs Improved")
    axes[0].legend()
    axes[0].axhline(y=0.75, color="red", linestyle="--", alpha=0.5, label="Target 0.75")

    # Right: delta bars
    deltas = comparison_df["delta"].tolist()
    colors = ["seagreen" if d >= 0 else "tomato" for d in deltas]
    axes[1].bar(metrics, deltas, color=colors)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_xticklabels(metrics, rotation=15, ha="right")
    axes[1].set_title("Delta (Improved − Baseline)")

    plt.tight_layout()
    chart_path = "results/comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved → {chart_path}")
