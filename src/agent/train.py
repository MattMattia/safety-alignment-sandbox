"""PPO Agent Training Pipeline for Safety Alignment.

Trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3 on the
SafetyAlignmentEnv contextual bandit environment. The policy learns to select
optimal alignment and steering interventions given semantic prompt embeddings.
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.safety_env import SafetyAlignmentEnv


class AlignmentTrainingCallback(BaseCallback):
    """Callback for monitoring training dynamics, mean rewards, and policy entropy."""

    def __init__(self, check_freq: int = 64, verbose: int = 1) -> None:
        """Initializes the alignment training callback.

        Args:
            check_freq: Frequency of rollout steps at which metrics are evaluated.
            verbose: Verbosity level (0 or 1).
        """
        super().__init__(verbose)
        self.check_freq: int = check_freq
        self.rewards_history: list[float] = []

    def _on_step(self) -> bool:
        """Collects step rewards and logs rolling performance statistics."""
        # Retrieve reward from the latest environment step
        if len(self.locals.get("rewards", [])) > 0:
            step_reward = float(self.locals["rewards"][0])
            self.rewards_history.append(step_reward)

        if self.n_calls % self.check_freq == 0 and len(self.rewards_history) > 0:
            recent_rewards = self.rewards_history[-self.check_freq :]
            mean_reward = float(np.mean(recent_rewards))
            std_reward = float(np.std(recent_rewards))
            min_reward = float(np.min(recent_rewards))
            max_reward = float(np.max(recent_rewards))

            if self.verbose > 0:
                print(
                    f"  [Step {self.n_calls:5d}] Rolling Mean Reward: {mean_reward:>+6.3f} "
                    f"(±{std_reward:5.3f}) | Range: [{min_reward:>+6.3f}, {max_reward:>+6.3f}]"
                )

        return True


def train_safety_agent(
    total_timesteps: int = 1500,
    save_dir: str = "models/checkpoints",
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 64,
    batch_size: int = 32,
    ent_coef: float = 0.05,
) -> PPO:
    """Trains a PPO policy on the SafetyAlignmentEnv.

    Args:
        total_timesteps: Total environment interaction steps to execute.
        save_dir: Output directory path for trained checkpoint weights.
        seed: Random seed for environment sampling and neural network initialisation.
        learning_rate: Optimizer learning rate.
        n_steps: Horizon length for PPO rollout buffer collection before update.
        batch_size: Minibatch size for policy surrogate gradient optimization.
        ent_coef: Entropy bonus coefficient encouraging exploratory diversity.

    Returns:
        PPO: The trained policy model instance.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = save_path / "ppo_safety_agent.zip"

    print("=" * 80)
    print("  RLHF / RLAIF - PPO Safety Agent Training")
    print("=" * 80)
    print(f"  * Total Timesteps : {total_timesteps}")
    print(f"  * Rollout Horizon : {n_steps} steps (Batch: {batch_size})")
    print(f"  * Learning Rate   : {learning_rate}")
    print(f"  * Entropy Coef    : {ent_coef}")
    print(f"  * Random Seed     : {seed}")
    print(f"  * Target Output   : {checkpoint_file.resolve()}\n")

    # Instantiate raw environment and wrap with Monitor for episodic metrics
    raw_env = SafetyAlignmentEnv()
    env = Monitor(raw_env)

    # Configure PPO agent with MlpPolicy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
    )

    callback = AlignmentTrainingCallback(check_freq=n_steps, verbose=1)

    print("[+] Commencing policy optimization...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    # Save trained model weights
    model.save(str(checkpoint_file))
    print(f"\n[+] Policy training finished successfully!")
    print(f"    Saved checkpoint to: {checkpoint_file.resolve()}")
    print("=" * 80)

    # Close environment to release resources
    env.close()

    return model


def main() -> None:
    """CLI entrypoint for standalone training execution."""
    parser = argparse.ArgumentParser(
        description="Train PPO safety alignment agent on Gymnasium sandbox."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1500,
        help="Total training steps (default: 1500).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy bonus coefficient (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to persist checkpoints (default: 'models/checkpoints').",
    )

    args = parser.parse_args()
    train_safety_agent(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
        ent_coef=args.ent_coef,
    )


if __name__ == "__main__":
    main()

