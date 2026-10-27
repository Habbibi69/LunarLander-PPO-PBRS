"""
Evaluation script for trained RL agents on Gymnasium LunarLander-v3.

Usage:
  python src/evaluate.py --model-path models/ppo_baseline_smoke.zip --episodes 50 --output-prefix eval_smoke
  python src/evaluate.py --model-path models/ppo_baseline_300k.zip --episodes 100 --output-prefix eval_baseline_300k
"""

import argparse
import csv
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate_agent(
    model_path: str = None,
    env_id: str = "LunarLander-v3",
    num_episodes: int = 50,
    base_seed: int = 1000,
    deterministic: bool = True,
    output_prefix: str = "eval_results",
):
    print("=" * 80)
    print(f"EVALUATING AGENT ON {env_id}")
    print("=" * 80)
    print(f"  - Model Path:       {model_path if model_path else 'RANDOM BASELINE'}")
    print(f"  - Episodes:         {num_episodes}")
    print(f"  - Base Seed:        {base_seed} (episodes use base_seed + i)")
    print(f"  - Deterministic:    {deterministic}")
    print("-" * 80)

    # Load model if provided
    model = None
    if model_path:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = PPO.load(str(model_file), device="cpu")
        print(f"  Successfully loaded model from {model_file.name}")
    else:
        print("  No model supplied: evaluating UNTRAINED RANDOM agent.")

    env = gym.make(env_id)

    episode_records = []
    returns = []
    lengths = []
    success_count = 0
    crash_count = 0
    timeout_count = 0

    for ep_idx in range(num_episodes):
        ep_seed = base_seed + ep_idx
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0
        ep_length = 0
        last_step_reward = 0.0
        both_legs_touched = False

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(action)
            else:
                action = int(env.action_space.sample())

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            last_step_reward = reward

            # Track if both legs were in ground contact
            if next_obs[6] > 0.5 and next_obs[7] > 0.5:
                both_legs_touched = True

            obs = next_obs
            done = terminated or truncated

        # Determine ground-truth flight outcome
        if terminated:
            # Landing terminal reward is +100, crash penalty is -100
            if ep_return >= 200.0 or (last_step_reward >= 50.0 and both_legs_touched):
                outcome = "SUCCESS"
                success_count += 1
            else:
                outcome = "CRASH"
                crash_count += 1
        elif truncated:
            outcome = "TIMEOUT"
            timeout_count += 1
        else:
            outcome = "UNKNOWN"

        returns.append(ep_return)
        lengths.append(ep_length)

        record = {
            "episode": ep_idx + 1,
            "seed": ep_seed,
            "return": round(ep_return, 2),
            "length": ep_length,
            "last_step_reward": round(last_step_reward, 2),
            "outcome": outcome,
        }
        episode_records.append(record)

    env.close()

    # Calculate statistics
    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)

    mean_return = float(np.mean(returns_arr))
    std_return = float(np.std(returns_arr))
    median_return = float(np.median(returns_arr))
    min_return = float(np.min(returns_arr))
    max_return = float(np.max(returns_arr))
    mean_length = float(np.mean(lengths_arr))

    success_rate = (success_count / num_episodes) * 100
    crash_rate = (crash_count / num_episodes) * 100
    timeout_rate = (timeout_count / num_episodes) * 100

    summary = {
        "model_path": str(model_path) if model_path else "random_agent",
        "num_episodes": num_episodes,
        "base_seed": base_seed,
        "mean_return": round(mean_return, 2),
        "std_return": round(std_return, 2),
        "median_return": round(median_return, 2),
        "min_return": round(min_return, 2),
        "max_return": round(max_return, 2),
        "mean_length": round(mean_length, 1),
        "success_rate_pct": round(success_rate, 2),
        "crash_rate_pct": round(crash_rate, 2),
        "timeout_rate_pct": round(timeout_rate, 2),
        "success_count": success_count,
        "crash_count": crash_count,
        "timeout_count": timeout_count,
    }

    # Save results
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{output_prefix}_summary.json"
    csv_path = results_dir / f"{output_prefix}_episodes.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "seed", "return", "length", "last_step_reward", "outcome"])
        writer.writeheader()
        writer.writerows(episode_records)

    # Print clean results table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Episodes Tested:      {num_episodes}")
    print(f"  Mean Return:          {mean_return:+.2f} +/- {std_return:.2f}")
    print(f"  Median Return:        {median_return:+.2f}")
    print(f"  Min / Max Return:     {min_return:+.2f} / {max_return:+.2f}")
    print(f"  Mean Episode Length:  {mean_length:.1f} steps")
    print(f"  Landing Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"  Crash Rate:           {crash_rate:.1f}% ({crash_count}/{num_episodes})")
    print(f"  Timeout Rate:         {timeout_rate:.1f}% ({timeout_count}/{num_episodes})")
    print(f"  Summary JSON:         {json_path}")
    print(f"  Per-Episode CSV:      {csv_path}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL agent on LunarLander-v3")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model .zip file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for evaluation")
    parser.add_argument("--output-prefix", type=str, default="eval_results", help="Prefix for output result files")
    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model_path,
        num_episodes=args.episodes,
        base_seed=args.seed,
        output_prefix=args.output_prefix,
    )
