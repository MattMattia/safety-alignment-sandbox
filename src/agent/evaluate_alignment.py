"""Comparative alignment evaluation benchmark across policies.

Evaluates and compares:
1. Baseline Unaligned (Always Action 0: NO_INTERVENTION)
2. Stochastic Random Policy (Uniform exploratory actions)
3. Aligned PPO Policy (RL-optimized contextual bandit)

Metrics computed:
- Mean total reward (Safety vs. Utility tradeoff)
- Mean toxicity score across full suite and broken down by category
- Overall safety rate (% of responses flagged as safe)
- Over-refusal rate (% of benign prompts where strict refusal was deployed)
- Strategy action distribution
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
import numpy as np
from stable_baselines3 import PPO

from src.data.prompts import PromptDataset, PromptItem
from src.environment.safety_env import ACTION_REGISTRY, ActionType, SafetyAlignmentEnv


class PolicyMetrics(TypedDict):
    """Aggregated evaluation metrics for a single policy."""

    policy_name: str
    mean_total_reward: float
    mean_safety_reward: float
    mean_utility_penalty: float
    overall_toxicity: float
    safety_rate: float
    over_refusal_rate: float
    toxicity_by_category: Dict[str, float]
    action_distribution: Dict[str, int]
    sample_count: int


class EvaluationSummary(TypedDict):
    """Top-level container for benchmark output payload."""

    policies: Dict[str, PolicyMetrics]
    detailed_episodes: Dict[str, List[Dict[str, Any]]]


class AlignmentBenchmark:
    """Benchmark suite executing comparative evaluations on the curated prompt test set."""

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/ppo_safety_agent.zip",
        env: Optional[SafetyAlignmentEnv] = None,
        dataset: Optional[PromptDataset] = None,
    ) -> None:
        """Initializes the benchmark suite with environment and models.

        Args:
            checkpoint_path: Path to the trained PPO agent weights (.zip).
            env: Optional shared SafetyAlignmentEnv instance.
            dataset: Optional PromptDataset instance.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset = dataset or PromptDataset()
        self.env = env or SafetyAlignmentEnv(dataset=self.dataset)

    def _evaluate_single_policy(
        self,
        policy_name: str,
        action_selector: Callable[[np.ndarray, PromptItem], int],
    ) -> Tuple[PolicyMetrics, List[Dict[str, Any]]]:
        """Evaluates a single policy across all prompts in the dataset.

        Args:
            policy_name: Descriptive name for logging.
            action_selector: Callable mapping (obs, prompt_item) to chosen action index.

        Returns:
            Tuple of aggregated PolicyMetrics and detailed per-prompt step records.
        """
        detailed_records: List[Dict[str, Any]] = []
        category_toxicities: Dict[str, List[float]] = {cat: [] for cat in PromptDataset.CATEGORIES}
        action_counts: Dict[str, int] = {strat.name: 0 for strat in ACTION_REGISTRY.values()}

        total_rewards: List[float] = []
        safety_rewards: List[float] = []
        utility_penalties: List[float] = []
        overall_toxicities: List[float] = []
        safe_flags: List[bool] = []
        benign_over_refusals: int = 0
        total_benign: int = 0

        for prompt_item in self.dataset:
            # Deterministically reset environment to the exact prompt
            obs, _ = self.env.reset(options={"prompt_item": prompt_item})

            # Select action
            action = int(action_selector(obs, prompt_item))
            strategy_name = ACTION_REGISTRY[action].name
            action_counts[strategy_name] = action_counts.get(strategy_name, 0) + 1

            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)

            # Record metrics
            total_rewards.append(step_info["total_reward"])
            safety_rewards.append(step_info["safety_reward"])
            utility_penalties.append(step_info["utility_penalty"])
            overall_toxicities.append(step_info["toxicity_score"])
            safe_flags.append(step_info["is_safe"])
            category_toxicities[prompt_item.category].append(step_info["toxicity_score"])

            # Check for over-refusal: strict refusal on benign prompt
            if prompt_item.category == "benign":
                total_benign += 1
                if action == ActionType.STRICT_SAFETY_REFUSAL:
                    benign_over_refusals += 1

            detailed_records.append(
                {
                    "prompt_id": prompt_item.id,
                    "category": prompt_item.category,
                    "prompt": prompt_item.text,
                    "action_taken": action,
                    "action_name": strategy_name,
                    "response": step_info["response"],
                    "toxicity_score": step_info["toxicity_score"],
                    "safety_reward": step_info["safety_reward"],
                    "utility_penalty": step_info["utility_penalty"],
                    "total_reward": step_info["total_reward"],
                    "is_safe": step_info["is_safe"],
                }
            )

        # Aggregate metrics
        avg_tox_by_cat = {
            cat: round(float(np.mean(scores)), 4) if scores else 0.0
            for cat, scores in category_toxicities.items()
        }

        over_refusal_rate = (
            round(float(benign_over_refusals / total_benign) * 100.0, 2)
            if total_benign > 0
            else 0.0
        )
        safety_rate = round(float(np.mean(safe_flags)) * 100.0, 2)

        metrics = PolicyMetrics(
            policy_name=policy_name,
            mean_total_reward=round(float(np.mean(total_rewards)), 4),
            mean_safety_reward=round(float(np.mean(safety_rewards)), 4),
            mean_utility_penalty=round(float(np.mean(utility_penalties)), 4),
            overall_toxicity=round(float(np.mean(overall_toxicities)), 4),
            safety_rate=safety_rate,
            over_refusal_rate=over_refusal_rate,
            toxicity_by_category=avg_tox_by_cat,
            action_distribution=action_counts,
            sample_count=len(self.dataset),
        )

        return metrics, detailed_records

    def run_comparative_benchmark(self) -> EvaluationSummary:
        """Runs the complete benchmark suite across Baseline, Random, and PPO Aligned policies.

        Returns:
            EvaluationSummary: Consolidated metrics and step logs for all three strategies.
        """
        print("=" * 85)
        print("  RLHF / RLAIF - Multi-Policy Safety Alignment Benchmark")
        print("=" * 85)
        print(f"  * Total Benchmark Prompts : {len(self.dataset)}")
        print(f"  * Categories Evaluated    : {list(PromptDataset.CATEGORIES)}")
        print(f"  * Checkpoint Path         : {self.checkpoint_path.resolve()}\n")

        all_metrics: Dict[str, PolicyMetrics] = {}
        all_details: Dict[str, List[Dict[str, Any]]] = {}

        # ---------------------------------------------------------------------
        # 1. Baseline: Unaligned (Always Action 0: NO_INTERVENTION)
        # ---------------------------------------------------------------------
        print("[1/3] Benchmarking Baseline Unaligned Policy (NO_INTERVENTION)...")
        baseline_metrics, baseline_details = self._evaluate_single_policy(
            policy_name="Baseline (Unaligned)",
            action_selector=lambda obs, p: int(ActionType.NO_INTERVENTION),
        )
        all_metrics["baseline_unaligned"] = baseline_metrics
        all_details["baseline_unaligned"] = baseline_details

        # ---------------------------------------------------------------------
        # 2. Random Policy (Stochastic Uniform Actions)
        # ---------------------------------------------------------------------
        print("[2/3] Benchmarking Stochastic Random Policy...")
        rng = np.random.default_rng(seed=123)
        random_metrics, random_details = self._evaluate_single_policy(
            policy_name="Random Policy",
            action_selector=lambda obs, p: int(rng.integers(0, self.env.action_space.n)),
        )
        all_metrics["random_policy"] = random_metrics
        all_details["random_policy"] = random_details

        # ---------------------------------------------------------------------
        # 3. PPO Aligned Policy (Trained Neural Policy)
        # ---------------------------------------------------------------------
        print(f"[3/3] Benchmarking Trained PPO Policy ({self.checkpoint_path.name})...")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path.resolve()}\n"
                "Please run `python -m src.agent.train` before executing evaluation."
            )

        ppo_model = PPO.load(str(self.checkpoint_path))

        def ppo_selector(obs: np.ndarray, p: PromptItem) -> int:
            action, _ = ppo_model.predict(obs, deterministic=True)
            return int(action)

        ppo_metrics, ppo_details = self._evaluate_single_policy(
            policy_name="PPO Aligned",
            action_selector=ppo_selector,
        )
        all_metrics["ppo_aligned"] = ppo_metrics
        all_details["ppo_aligned"] = ppo_details

        summary = EvaluationSummary(
            policies=all_metrics,
            detailed_episodes=all_details,
        )

        return summary


def render_markdown_table(summary: EvaluationSummary) -> None:
    """Formats and prints an aligned ASCII/Markdown comparison table to console."""
    policies = summary["policies"]

    print("\n" + "=" * 105)
    print("                      COMPARATIVE SAFETY ALIGNMENT BENCHMARK REPORT")
    print("=" * 105)

    headers = [
        "Strategy / Policy",
        "Mean Reward",
        "Overall Tox",
        "Safe Rate",
        "Over-Refusal",
        "Toxic Bait Tox",
        "Bias Tox",
    ]
    header_line = (
        f"{headers[0]:<24} | {headers[1]:<11} | {headers[2]:<11} | {headers[3]:<10} | "
        f"{headers[4]:<12} | {headers[5]:<14} | {headers[6]:<10}"
    )
    print(header_line)
    print("-" * 105)

    for key, p in policies.items():
        cat_tox = p["toxicity_by_category"]
        row = (
            f"{p['policy_name']:<24} | "
            f"{p['mean_total_reward']:>+10.4f} | "
            f"{p['overall_toxicity']:>11.4f} | "
            f"{p['safety_rate']:>9.1f}% | "
            f"{p['over_refusal_rate']:>11.1f}% | "
            f"{cat_tox.get('toxic_bait', 0.0):>14.4f} | "
            f"{cat_tox.get('sensitive_bias', 0.0):>10.4f}"
        )
        print(row)

    print("=" * 105)

    # Action distribution breakdown
    print("\nAction Distribution per Policy:")
    for key, p in policies.items():
        dist_str = ", ".join([f"{act}: {cnt}" for act, cnt in p["action_distribution"].items()])
        print(f"  * {p['policy_name']:<24}: {dist_str}")
    print()


def save_evaluation_results(
    summary: EvaluationSummary,
    output_path: str = "outputs/evaluation_results.json",
) -> Path:
    """Saves benchmark results to a structured JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[+] Detailed evaluation results persisted to: {out_file.resolve()}")
    return out_file


def main() -> None:
    """CLI entrypoint for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run comparative safety alignment benchmark across policies."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/ppo_safety_agent.zip",
        help="Path to trained PPO agent checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.json",
        help="Path for output JSON metrics.",
    )

    args = parser.parse_args()

    benchmark = AlignmentBenchmark(checkpoint_path=args.checkpoint)
    summary = benchmark.run_comparative_benchmark()
    render_markdown_table(summary)
    save_evaluation_results(summary, output_path=args.output)


if __name__ == "__main__":
    main()

