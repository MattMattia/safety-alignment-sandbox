#!/usr/bin/env python3
"""Visualization script for Safety Alignment Benchmark results.

Loads evaluation telemetry from outputs/evaluation_results.json and generates
publication-quality comparison charts saved to outputs/benchmark_comparison.png:
1. Macro Alignment Metrics (Mean Total Reward, Overall Toxicity, Safe Response Rate %).
2. Granular Category-level Toxicity Breakdown across policies.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_benchmark_data(input_path: Path) -> Dict[str, Any]:
    """Loads evaluation results from JSON, with fallback demo data if file is absent."""
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[+] Loaded benchmark results from: {input_path.resolve()}")
            return data

    print(f"[!] Evaluation file not found at: {input_path.resolve()}")
    print("    Generating plot using representative benchmark metrics for demonstration.")
    print("    Run `python scripts/run_pipeline.py` to generate live data.\n")

    # Representative empirical metrics for demonstration
    return {
        "policies": {
            "baseline_unaligned": {
                "policy_name": "Baseline (Unaligned)",
                "mean_total_reward": 0.524,
                "overall_toxicity": 0.238,
                "safety_rate": 75.0,
                "over_refusal_rate": 0.0,
                "toxicity_by_category": {
                    "benign": 0.035,
                    "toxic_bait": 0.512,
                    "sensitive_bias": 0.284,
                    "controversial": 0.121,
                },
            },
            "random_policy": {
                "policy_name": "Random Policy",
                "mean_total_reward": 0.612,
                "overall_toxicity": 0.194,
                "safety_rate": 80.0,
                "over_refusal_rate": 20.0,
                "toxicity_by_category": {
                    "benign": 0.031,
                    "toxic_bait": 0.385,
                    "sensitive_bias": 0.215,
                    "controversial": 0.145,
                },
            },
            "ppo_aligned": {
                "policy_name": "PPO Aligned",
                "mean_total_reward": 0.865,
                "overall_toxicity": 0.068,
                "safety_rate": 95.0,
                "over_refusal_rate": 0.0,
                "toxicity_by_category": {
                    "benign": 0.024,
                    "toxic_bait": 0.098,
                    "sensitive_bias": 0.082,
                    "controversial": 0.068,
                },
            },
        }
    }


def generate_visualizations(data: Dict[str, Any], output_path: Path) -> None:
    """Generates and exports the dual-panel benchmark comparison figure."""
    policies = data.get("policies", {})
    if not policies:
        raise ValueError("Provided JSON does not contain valid 'policies' metrics.")

    policy_keys = ["baseline_unaligned", "random_policy", "ppo_aligned"]
    policy_labels = [policies[k]["policy_name"] for k in policy_keys if k in policies]
    policy_colors = ["#e74c3c", "#f39c12", "#27ae60"]  # Red, Amber, Green

    # Set seaborn visual theme
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.suptitle(
        "Safety Alignment Verification Sandbox: Multi-Policy Benchmark Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------------------
    # Subplot 1: Macro Alignment Performance (Reward, Toxicity, Safety Rate)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    metrics = ["Mean Total Reward", "Overall Toxicity", "Safe Response Rate (%)"]
    x = np.arange(len(metrics))
    width = 0.26

    for i, (k, color) in enumerate(zip(policy_keys, policy_colors)):
        p = policies.get(k, {})
        # Scale metrics: reward [-1, 1], toxicity [0, 1], safety_rate / 100 to fit unified scale
        values = [
            p.get("mean_total_reward", 0.0),
            p.get("overall_toxicity", 0.0),
            p.get("safety_rate", 0.0) / 100.0,
        ]
        rects = ax1.bar(
            x + (i - 1) * width,
            values,
            width,
            label=p.get("policy_name", k),
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

        # Value labels above bars
        for rect, val in zip(rects, values):
            y_pos = rect.get_height()
            label_text = f"{val * 100:.1f}%" if "Rate" in metrics[rects.index(rect)] else f"{val:.2f}"
            offset = 0.02 if y_pos >= 0 else -0.05
            ax1.text(
                rect.get_x() + rect.get_width() / 2.0,
                y_pos + offset,
                label_text,
                ha="center",
                va="bottom" if y_pos >= 0 else "top",
                fontsize=8.5,
                fontweight="bold",
            )

    ax1.set_title("Macro Alignment Performance Across Policies", fontsize=12, pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10, fontweight="medium")
    ax1.set_ylabel("Normalized Metric Scale", fontsize=10)
    ax1.set_ylim(-0.1, 1.15)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # -------------------------------------------------------------------------
    # Subplot 2: Toxicity Breakdown by Taxonomy Category
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    categories = ["benign", "toxic_bait", "sensitive_bias", "controversial"]
    category_display = ["Benign", "Toxic Bait", "Sensitive Bias", "Controversial"]
    x2 = np.arange(len(categories))

    for i, (k, color) in enumerate(zip(policy_keys, policy_colors)):
        p = policies.get(k, {})
        cat_tox = p.get("toxicity_by_category", {})
        tox_values = [cat_tox.get(c, 0.0) for c in categories]

        rects = ax2.bar(
            x2 + (i - 1) * width,
            tox_values,
            width,
            label=p.get("policy_name", k),
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.8,
        )

        for rect in rects:
            height = rect.get_height()
            ax2.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax2.set_title("Toxicity Score by Category (Lower is Safer)", fontsize=12, pad=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(category_display, fontsize=10, fontweight="medium")
    ax2.set_ylabel("Mean Classifier Toxicity Probability", fontsize=10)
    ax2.set_ylim(0.0, 0.65)
    ax2.axhline(0.5, color="red", linestyle=":", linewidth=1.2, label="Unsafe Threshold (0.5)")
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Visualization successfully generated and saved to:")
    print(f"    {output_path.resolve()}\n")


def main() -> None:
    """CLI entrypoint for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from safety alignment evaluation JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/evaluation_results.json",
        help="Path to evaluation_results.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmark_comparison.png",
        help="Output PNG image path.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_benchmark_data(input_path)
    generate_visualizations(data, output_path)


if __name__ == "__main__":
    main()

