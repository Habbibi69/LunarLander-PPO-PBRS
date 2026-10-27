"""
Training script for PPO on Gymnasium LunarLander-v3.

Usage:
  python src/train.py --config configs/baseline_config.json --timesteps 10000 --output-name ppo_baseline_smoke
  python src/train.py --config configs/baseline_config.json --timesteps 300000 --output-name ppo_baseline_300k
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train(config_path: str, timesteps: int = None, output_name: str = None, seed_override: int = None):
    cfg = load_config(config_path)

    seed = seed_override if seed_override is not None else cfg.get("seed", 42)
    set_seed(seed)

    env_id = cfg.get("env_id", "LunarLander-v3")
    total_timesteps = timesteps if timesteps is not None else cfg.get("main_timesteps", 300000)
    name = output_name if output_name is not None else f"{cfg.get('experiment_name', 'ppo')}_{total_timesteps}"

    # Directory layout
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    logs_dir = project_root / "logs" / name
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING PPO TRAINING: {name}")
    print("=" * 80)
    print(f"  - Environment:      {env_id}")
    print(f"  - Total Timesteps:  {total_timesteps:,}")
    print(f"  - Seed:             {seed}")
    print(f"  - Device:           {cfg.get('device', 'cpu')}")
    print(f"  - Model Output:     {models_dir / f'{name}.zip'}")
    print(f"  - Monitor Log Dir:  {logs_dir}")
    print("-" * 80)

    # Initialize environment wrapped with Monitor for tracking returns & episode lengths
    raw_env = gym.make(env_id)
    monitor_csv = str(logs_dir / "monitor.csv")
    env = Monitor(raw_env, filename=monitor_csv)
    env.reset(seed=seed)

    hparams = cfg.get("hyperparameters", {})
    policy = cfg.get("policy", "MlpPolicy")

    model = PPO(
        policy=policy,
        env=env,
        learning_rate=hparams.get("learning_rate", 3e-4),
        n_steps=hparams.get("n_steps", 2048),
        batch_size=hparams.get("batch_size", 64),
        n_epochs=hparams.get("n_epochs", 10),
        gamma=hparams.get("gamma", 0.99),
        gae_lambda=hparams.get("gae_lambda", 0.95),
        clip_range=hparams.get("clip_range", 0.2),
        ent_coef=hparams.get("ent_coef", 0.01),
        vf_coef=hparams.get("vf_coef", 0.5),
        max_grad_norm=hparams.get("max_grad_norm", 0.5),
        seed=seed,
        device=cfg.get("device", "cpu"),
        verbose=1,
    )

    # Periodic checkpoint callback (every 50k steps or total_timesteps)
    save_freq = max(10000, total_timesteps // 5)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(models_dir / f"checkpoints_{name}"),
        name_prefix="ppo_ckpt",
        verbose=1,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    elapsed_time = time.time() - start_time

    # Save final model
    final_model_path = models_dir / f"{name}.zip"
    model.save(str(final_model_path))

    # Save metadata
    fps = total_timesteps / elapsed_time if elapsed_time > 0 else 0
    metadata = {
        "experiment_name": name,
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_seconds": round(elapsed_time, 2),
        "timesteps_per_second": round(fps, 2),
        "hyperparameters": hparams,
        "saved_model_path": str(final_model_path),
        "monitor_log": monitor_csv,
    }
    with open(logs_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()

    print("=" * 80)
    print("TRAINING RUN COMPLETE")
    print("=" * 80)
    print(f"  - Elapsed Time:     {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"  - Throughput:       {fps:.1f} steps/second")
    print(f"  - Model Saved To:   {final_model_path}")
    print(f"  - Logs Saved To:    {logs_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--config", type=str, default="configs/baseline_config.json", help="Path to config file")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--output-name", type=str, default=None, help="Model and log artifact name")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    train(args.config, timesteps=args.timesteps, output_name=args.output_name, seed_override=args.seed)
