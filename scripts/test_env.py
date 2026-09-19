#!/usr/bin/env python3
"""Verification script for SafetyAlignmentEnv Gymnasium environment.

Initializes the environment, executes multiple simulation episodes using random actions,
and outputs structured diagnostic telemetry for safety alignment verification.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.safety_env import SafetyAlignmentEnv


def print_banner(title: str, char: str = "=", length: int = 80) -> None:
    """Prints a styled decorative header."""
    print(char * length)
    print(f"  {title}")
    print(char * length)


def run_simulation(num_episodes: int = 3, seed: int = 42) -> None:
    """Runs simulation episodes in the safety alignment environment.

    Args:
        num_episodes: Number of contextual bandit episodes to simulate.
        seed: Random seed for deterministic reproducibility.
    """
    print_banner("RLHF/RLAIF Safety Alignment Sandbox - Environment Test")

    print("\n[+] Instantiating SafetyAlignmentEnv...")
    env = SafetyAlignmentEnv()
    print(f"    - Observation Space : {env.observation_space}")
    print(f"    - Action Space      : {env.action_space} (n={env.action_space.n})")
    print(f"    - Embedding Dim     : {env.embedding_dim}")
    print(f"    - Utility Weight    : {env.utility_weight}")
    print(f"    - Device Target     : {env.generator.device.type.upper()}")

    print(f"\n[+] Executing {num_episodes} Simulation Episodes (Random Policy)...\n")

    for ep in range(1, num_episodes + 1):
        # Reset environment
        obs, reset_info = env.reset(seed=seed + ep)

        # Sample an action from the discrete action space
        action = env.action_space.sample()

        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # Output detailed telemetry
        print("=" * 80)
        print(f" Episode {ep}/{num_episodes} | Category: [{step_info['category'].upper()}]")
        print("=" * 80)
        print(f" Prompt ID       : {step_info['prompt_id']}")
        print(f" Input Prompt    : {step_info['prompt']}")
        print(f" Action Chosen   : [{step_info['action_taken']}] {step_info['action_name']}")
        print(f" Model Response  : \"{step_info['response']}\"")
        print("-" * 80)
        print(" Evaluation & Reward Breakdown:")
        print(f"   * Toxicity Probability : {step_info['toxicity_score']:>7.4f}  (in [0.0, 1.0])")
        print(f"   * Safety Reward (R_s)  : {step_info['safety_reward']:>7.4f}  (in [-1.0, 1.0])")
        print(f"   * Utility Penalty      : {step_info['utility_penalty']:>7.4f}  (over-refusal / vacuity)")
        print(f"   * Total Reward (R_tot) : {step_info['total_reward']:>7.4f}  (R_s - lambda * penalty)")
        print(f"   * Safe Content Flag    : {'[SAFE]' if step_info['is_safe'] else '[UNSAFE]'}")
        print(f"   * Terminated           : {terminated}")
        print()

    print_banner("[SUCCESS] Environment verification completed successfully!")


if __name__ == "__main__":
    run_simulation(num_episodes=3, seed=100)

